#include "registration/PointCloudRegistration.hpp"
#include "registration/HeadSurfaceReconstructor.hpp"

#include <open3d/Open3D.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace reg = open3d::pipelines::registration;
using Cloud = open3d::geometry::PointCloud;

namespace {
struct Prepared { std::shared_ptr<Cloud> cloud; std::shared_ptr<reg::Feature> feature; };
Prepared prepare(const Cloud& input, double voxel) {
    auto cloud = input.VoxelDownSample(voxel);
    if (!cloud || cloud->IsEmpty()) throw std::runtime_error("downsampled point cloud is empty");
    cloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel * 2.0, 30));
    auto feature = reg::ComputeFPFHFeature(*cloud,
        open3d::geometry::KDTreeSearchParamHybrid(voxel * 5.0, 100));
    return {std::move(cloud), std::move(feature)};
}

std::string lowerExtension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension;
}

void scalePointCloud(Cloud& cloud, double scale) {
    if (scale == 1.0) return;
    for (auto& point : cloud.points_) point *= scale;
}

bool readTarget(const std::string& path, Cloud& target) {
    if (lowerExtension(path) == ".stl") return false;
    return open3d::io::ReadPointCloud(path, target) && !target.IsEmpty();
}

struct DistanceSummary {
    double mean{0.0};
    double rmse{0.0};
    double median{0.0};
    double p95{0.0};
    double maximum{0.0};
};

DistanceSummary summarizeDistances(std::vector<double> values) {
    DistanceSummary summary;
    values.erase(std::remove_if(values.begin(), values.end(),
        [](double value) { return !std::isfinite(value); }), values.end());
    if (values.empty()) return summary;
    double sum = 0.0;
    double squares = 0.0;
    for (const double value : values) {
        sum += value;
        squares += value * value;
    }
    std::sort(values.begin(), values.end());
    const auto at = [&values](double percentile) {
        return values[static_cast<std::size_t>(std::llround(
            percentile * static_cast<double>(values.size() - 1)))];
    };
    summary.mean = sum / static_cast<double>(values.size());
    summary.rmse = std::sqrt(squares / static_cast<double>(values.size()));
    summary.median = at(0.50);
    summary.p95 = at(0.95);
    summary.maximum = values.back();
    return summary;
}

DistanceSummary sourceToTargetQuality(
    const Cloud& source,
    const Cloud& target,
    const Eigen::Matrix4d& transformation) {
    Cloud aligned = source;
    aligned.Transform(transformation);
    return summarizeDistances(aligned.ComputePointCloudDistance(target));
}

void calculateSymmetricMetrics(Cloud& aligned, Cloud& target, RegistrationResult& result) {
    const auto forward = summarizeDistances(aligned.ComputePointCloudDistance(target));
    const auto backward = summarizeDistances(target.ComputePointCloudDistance(aligned));
    result.sourceToTargetMeanMm = forward.mean;
    result.sourceToTargetRmseMm = forward.rmse;
    result.sourceToTargetMedianMm = forward.median;
    result.sourceToTargetP95Mm = forward.p95;
    result.sourceToTargetMaxMm = forward.maximum;
    result.targetToSourceMeanMm = backward.mean;
    result.targetToSourceRmseMm = backward.rmse;
    result.targetToSourceP95Mm = backward.p95;
    result.symmetricMeanMm = 0.5 * (result.sourceToTargetMeanMm + result.targetToSourceMeanMm);
    result.symmetricRmseMm = std::sqrt(0.5 * (result.sourceToTargetRmseMm * result.sourceToTargetRmseMm +
                                               result.targetToSourceRmseMm * result.targetToSourceRmseMm));
    result.symmetricMedianMm = 0.5 * (result.sourceToTargetMedianMm + backward.median);
    result.symmetricP95Mm = 0.5 * (result.sourceToTargetP95Mm + result.targetToSourceP95Mm);
    result.symmetricMaxMm = std::max(result.sourceToTargetMaxMm, backward.maximum);
}
}

PointCloudRegistration::PointCloudRegistration(RegistrationOptions options) : options_(options) {}

StlPointCloudGenerationResult PointCloudRegistration::prepareRegistrationCloudFromStl(
    const std::string& stlPath,
    const std::string& outputPly,
    double modelUnitScale) {
    HeadSurfaceReconstructor::Options options;
    options.modelUnitScale = modelUnitScale;
    return prepareRegistrationCloudFromStl(stlPath, outputPly, options);
}

StlPointCloudGenerationResult PointCloudRegistration::prepareRegistrationCloudFromStl(
    const std::string& stlPath,
    const std::string& outputPly,
    const HeadSurfaceReconstructor::Options& reconstructionOptions) {
    StlPointCloudGenerationResult result;
    result.outputPly = outputPly;
    try {
        if (lowerExtension(stlPath) != ".stl")
            throw std::runtime_error("model input is not an STL file: " + stlPath);
        if (lowerExtension(outputPly) != ".ply")
            throw std::runtime_error("generated model point cloud must use a .ply path: " + outputPly);
        if (!(reconstructionOptions.modelUnitScale > 0.0) ||
            !std::isfinite(reconstructionOptions.modelUnitScale))
            throw std::runtime_error("STL model unit scale must be finite and positive");

        const std::filesystem::path outputPath(outputPly);
        if (!outputPath.parent_path().empty()) {
            std::error_code directoryError;
            std::filesystem::create_directories(outputPath.parent_path(), directoryError);
            if (directoryError)
                throw std::runtime_error("cannot create model point-cloud directory: " +
                                         outputPath.parent_path().string());
        }
        const auto reconstructed = HeadSurfaceReconstructor::reconstructRegistrationCloud(
            stlPath, outputPly, reconstructionOptions);
        if (!reconstructed.success) throw std::runtime_error(reconstructed.message);

        result.success = true;
        result.pointCount = reconstructed.outputPointCount;
        result.stageTimingsMs = reconstructed.stageTimingsMs;
        result.message = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.message = exception.what();
    }
    return result;
}

RegistrationResult PointCloudRegistration::alignFiles(const std::string& sourcePly,
                                                       const std::string& targetPlyOrStl,
                                                       const std::string& outputPly,
                                                       const std::optional<Eigen::Matrix4d>& initialTransformation) const {
    Cloud source;
    Cloud target;
    if (!open3d::io::ReadPointCloud(sourcePly, source) || source.IsEmpty()) {
        RegistrationResult result;
        result.message = "cannot read source point cloud: " + sourcePly;
        return result;
    }
    if (!readTarget(targetPlyOrStl, target)) {
        RegistrationResult result;
        result.message = "cannot read target PLY/STL: " + targetPlyOrStl;
        return result;
    }
    return alignClouds(source, target, outputPly, initialTransformation);
}

RegistrationResult PointCloudRegistration::alignClouds(
    const open3d::geometry::PointCloud& sourceInput,
    const open3d::geometry::PointCloud& targetInput,
    const std::string& outputPly,
    const std::optional<Eigen::Matrix4d>& initialTransformation) const {
    RegistrationResult result;
    using TimingClock = std::chrono::steady_clock;
    const auto totalStart = TimingClock::now();
    const auto addTiming = [&result](const std::string& stage, const TimingClock::time_point& start) {
        result.stageTimingsMs.emplace_back(stage,
            std::chrono::duration<double, std::milli>(TimingClock::now() - start).count());
    };
    try {
        if (!outputPly.empty() && lowerExtension(outputPly) != ".ply")
            throw std::runtime_error(
                "aligned point-cloud output must use a .ply path; OBJ export is disabled: " +
                outputPly);
        if (options_.voxelSizeMm <= 0) throw std::runtime_error("voxel size must be positive");
        if (!(options_.initialSourceToTargetRmseToSkipGlobalMm > 0.0) ||
            !std::isfinite(options_.initialSourceToTargetRmseToSkipGlobalMm) ||
            !(options_.initialSourceToTargetP95ToSkipGlobalMm > 0.0) ||
            !std::isfinite(options_.initialSourceToTargetP95ToSkipGlobalMm))
            throw std::runtime_error("full-source registration thresholds must be finite and positive");
        Cloud source = sourceInput;
        Cloud target = targetInput;
        if (options_.reseedGlobalRandomEngine)
            open3d::utility::random::Seed(options_.randomSeed);
        auto stageStart = TimingClock::now();
        if (source.IsEmpty()) throw std::runtime_error("source point cloud is empty");
        if (target.IsEmpty()) throw std::runtime_error("target point cloud is empty");
        addTiming("input_cloud_copy", stageStart);
        if (!(options_.targetUnitScale > 0.0) || !std::isfinite(options_.targetUnitScale))
            throw std::runtime_error("target unit scale must be finite and positive");
        scalePointCloud(target, options_.targetUnitScale);
        result.sourcePointCount = source.points_.size();
        result.targetPointCount = target.points_.size();
        stageStart = TimingClock::now();
        source.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(options_.voxelSizeMm * 2.0, 30));
        target.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(options_.voxelSizeMm * 2.0, 30));
        addTiming("normal_estimation", stageStart);
        reg::RegistrationResult best;
        DistanceSummary bestSourceQuality;
        bool hasBest = false;
        std::string bestMethod = "none";
        const auto passesInitialGate = [this](
            const reg::RegistrationResult& candidate,
            const DistanceSummary& sourceQuality) {
            return candidate.fitness_ >= options_.initialFitnessToSkipGlobal &&
                   candidate.inlier_rmse_ <= options_.initialRmseToSkipGlobalMm &&
                   sourceQuality.rmse <= options_.initialSourceToTargetRmseToSkipGlobalMm &&
                   sourceQuality.p95 <= options_.initialSourceToTargetP95ToSkipGlobalMm;
        };
        const auto isBetter = [&best, &bestSourceQuality, &hasBest, &passesInitialGate](
            const reg::RegistrationResult& candidate,
            const DistanceSummary& sourceQuality) {
            if (!hasBest) return true;
            const bool candidatePasses = passesInitialGate(candidate, sourceQuality);
            const bool bestPasses = passesInitialGate(best, bestSourceQuality);
            if (candidatePasses != bestPasses) return candidatePasses;
            if (candidatePasses) {
                if (candidate.fitness_ != best.fitness_)
                    return candidate.fitness_ > best.fitness_;
                if (sourceQuality.p95 != bestSourceQuality.p95)
                    return sourceQuality.p95 < bestSourceQuality.p95;
            } else {
                if (sourceQuality.p95 != bestSourceQuality.p95)
                    return sourceQuality.p95 < bestSourceQuality.p95;
                if (sourceQuality.rmse != bestSourceQuality.rmse)
                    return sourceQuality.rmse < bestSourceQuality.rmse;
                if (candidate.fitness_ != best.fitness_)
                    return candidate.fitness_ > best.fitness_;
            }
            return candidate.inlier_rmse_ < best.inlier_rmse_;
        };

        if (initialTransformation && initialTransformation->array().isFinite().all()) {
            stageStart = TimingClock::now();
            const int stageIterations = std::max(20, options_.icpIterations / 2);
            auto initialized = reg::RegistrationICP(
                source, target, options_.voxelSizeMm * 4.0, *initialTransformation,
                reg::TransformationEstimationPointToPlane(),
                reg::ICPConvergenceCriteria(1e-6, 1e-6, stageIterations));
            initialized = reg::RegistrationICP(
                source, target, options_.voxelSizeMm * 2.0, initialized.transformation_,
                reg::TransformationEstimationPointToPlane(),
                reg::ICPConvergenceCriteria(1e-7, 1e-7, stageIterations));
            initialized = reg::RegistrationICP(
                source, target, options_.voxelSizeMm * 1.5, initialized.transformation_,
                reg::TransformationEstimationPointToPlane(),
                reg::ICPConvergenceCriteria(1e-7, 1e-7, options_.icpIterations));
            if (initialized.fitness_ > 0.0) {
                bestSourceQuality = sourceToTargetQuality(
                    source, target, initialized.transformation_);
                best = initialized;
                hasBest = true;
                bestMethod = "model_virtual_views_keypoints_icp";
            }
            addTiming("keypoint_initial_multiscale_icp", stageStart);
        }

        const bool runGlobal = !hasBest || !passesInitialGate(best, bestSourceQuality);
        if (runGlobal) {
            const auto globalStart = TimingClock::now();
            stageStart = TimingClock::now();
            const auto s = prepare(source, options_.voxelSizeMm);
            const auto t = prepare(target, options_.voxelSizeMm);
            addTiming("fpfh_feature_extraction", stageStart);
            reg::CorrespondenceCheckerBasedOnEdgeLength edge(0.9);
            reg::CorrespondenceCheckerBasedOnDistance distance(options_.voxelSizeMm * 1.5);
            std::vector<std::reference_wrapper<const reg::CorrespondenceChecker>> checkers{edge, distance};
            const int attempts = options_.globalAttempts > 1 ? options_.globalAttempts : 1;
            for (int attempt = 0; attempt < attempts; ++attempt) {
                stageStart = TimingClock::now();
                if (options_.reseedGlobalRandomEngine)
                    open3d::utility::random::Seed(options_.randomSeed + attempt);
                const auto ransacStart = TimingClock::now();
                const auto global = reg::RegistrationRANSACBasedOnFeatureMatching(
                    *s.cloud, *t.cloud, *s.feature, *t.feature, true, options_.voxelSizeMm * 1.5,
                    reg::TransformationEstimationPointToPoint(false), 3, checkers,
                    reg::RANSACConvergenceCriteria(options_.ransacIterations, 0.999));
                addTiming("ransac_attempt_" + std::to_string(attempt + 1), ransacStart);
                const auto icpStart = TimingClock::now();
                const auto refined = reg::RegistrationICP(
                    source, target, options_.voxelSizeMm * 1.5, global.transformation_,
                    reg::TransformationEstimationPointToPlane(),
                    reg::ICPConvergenceCriteria(1e-7, 1e-7, options_.icpIterations));
                addTiming("global_icp_attempt_" + std::to_string(attempt + 1), icpStart);
                const auto refinedSourceQuality = sourceToTargetQuality(
                    source, target, refined.transformation_);
                if (isBetter(refined, refinedSourceQuality)) {
                    best = refined;
                    bestSourceQuality = refinedSourceQuality;
                    hasBest = true;
                    bestMethod = "fpfh_ransac_icp";
                }
            }
            addTiming("global_fpfh_ransac_icp_total", globalStart);
        }
        result.fitness = best.fitness_;
        result.inlierRmseMm = best.inlier_rmse_;
        result.transformation = best.transformation_;
        result.initializationMethod = bestMethod;
        result.usedInitialTransformation = bestMethod == "model_virtual_views_keypoints_icp";
        result.success = hasBest && best.fitness_ > 0.0;
        result.message = result.success ? "ok" : "registration found no inliers";
        Cloud aligned = source;
        aligned.Transform(result.transformation);
        stageStart = TimingClock::now();
        calculateSymmetricMetrics(aligned, target, result);
        addTiming("quality_metrics", stageStart);
        if (!outputPly.empty()) {
            stageStart = TimingClock::now();
            const std::filesystem::path path(outputPly);
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            if (!open3d::io::WritePointCloud(outputPly, aligned))
                throw std::runtime_error("cannot write aligned point cloud: " + outputPly);
            std::ofstream matrix(path.parent_path() / (path.stem().string() + "_transformation.txt"));
            if (!matrix)
                throw std::runtime_error("cannot open aligned transformation output");
            matrix.precision(12);
            matrix << result.transformation << '\n';
            matrix.flush();
            if (!matrix)
                throw std::runtime_error("cannot write aligned transformation output");
            matrix.close();
            if (!matrix)
                throw std::runtime_error("cannot close aligned transformation output");
            std::ofstream metrics(path.parent_path() / (path.stem().string() + "_metrics.txt"));
            if (!metrics)
                throw std::runtime_error("cannot open aligned metrics output");
            metrics.precision(12);
            metrics << "fitness=" << result.fitness << '\n'
                    << "inlier_rmse_mm=" << result.inlierRmseMm << '\n'
                    << "source_to_target_mean_mm=" << result.sourceToTargetMeanMm << '\n'
                    << "source_to_target_rmse_mm=" << result.sourceToTargetRmseMm << '\n'
                    << "source_to_target_median_mm=" << result.sourceToTargetMedianMm << '\n'
                    << "source_to_target_p95_mm=" << result.sourceToTargetP95Mm << '\n'
                    << "source_to_target_max_mm=" << result.sourceToTargetMaxMm << '\n'
                    << "target_to_source_mean_mm=" << result.targetToSourceMeanMm << '\n'
                    << "target_to_source_rmse_mm=" << result.targetToSourceRmseMm << '\n'
                    << "target_to_source_p95_mm=" << result.targetToSourceP95Mm << '\n'
                    << "symmetric_mean_mm=" << result.symmetricMeanMm << '\n'
                    << "symmetric_rmse_mm=" << result.symmetricRmseMm << '\n'
                    << "symmetric_median_mm=" << result.symmetricMedianMm << '\n'
                    << "symmetric_p95_mm=" << result.symmetricP95Mm << '\n'
                    << "symmetric_max_mm=" << result.symmetricMaxMm << '\n'
                    << "source_points=" << result.sourcePointCount << '\n'
                    << "target_points=" << result.targetPointCount << '\n';
            metrics << "initialization_method=" << result.initializationMethod << '\n'
                    << "used_initial_transformation=" << (result.usedInitialTransformation ? 1 : 0) << '\n';
            addTiming("result_files_write", stageStart);
            metrics.flush();
            if (!metrics)
                throw std::runtime_error("cannot write aligned metrics output");
            metrics.close();
            if (!metrics)
                throw std::runtime_error("cannot close aligned metrics output");
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.message = e.what();
    }
    addTiming("align_clouds_total", totalStart);
    return result;
}
