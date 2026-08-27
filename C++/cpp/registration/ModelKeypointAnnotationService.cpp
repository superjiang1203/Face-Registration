#include "registration/ModelKeypointAnnotationService.hpp"

#include "detection/FaceKeypointService.hpp"
#include "registration/StlModelRenderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>

#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using Keypoint = ModelKeypointAnnotationService::Keypoint3D;

struct Candidate {
    double azimuthDeg{0.0};
    double elevationDeg{0.0};
    cv::Mat bgr;
    cv::Mat zBuffer;
    std::vector<Keypoint> keypoints;
    int validCount{0};
    int namedValidCount{0};
    double meanScore{0.0};
};

cv::Mat visualizeZBuffer(const cv::Mat& zBuffer) {
    cv::Mat valid = (zBuffer > 0.0f) & (zBuffer < 0.999999f);
    double minimum = 0.0, maximum = 1.0;
    cv::minMaxLoc(zBuffer, &minimum, &maximum, nullptr, nullptr, valid);
    cv::Mat gray(zBuffer.size(), CV_8U, cv::Scalar(0));
    if (cv::countNonZero(valid) > 0 && maximum > minimum) {
        cv::Mat normalized;
        zBuffer.convertTo(normalized, CV_32F, -255.0 / (maximum - minimum),
                          255.0 * maximum / (maximum - minimum));
        normalized.convertTo(gray, CV_8U);
        gray.setTo(0, ~valid);
    }
    cv::Mat colored;
    cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid);
    return colored;
}

std::array<unsigned char, 3> rgbForKeypoint(const std::string& name, int index) {
    if (name == "nose_root") return {255, 0, 255};
    if (name == "nose_tip") return {255, 0, 0};
    if (name == "right_eye_outer") return {0, 180, 255};
    if (name == "right_eye_inner") return {0, 255, 255};
    if (name == "left_eye_inner") return {100, 255, 50};
    if (name == "left_eye_outer") return {0, 255, 0};
    static constexpr std::array<unsigned char, 3> palette[] = {
        {230, 25, 75}, {60, 180, 75}, {0, 130, 200}, {245, 130, 48},
        {145, 30, 180}, {70, 240, 240}, {240, 50, 230}, {210, 245, 60}};
    return palette[static_cast<std::size_t>(std::max(0, index)) % std::size(palette)];
}

cv::Scalar bgrForKeypoint(const std::string& name, int index) {
    const auto rgb = rgbForKeypoint(name, index);
    return cv::Scalar(rgb[2], rgb[1], rgb[0]);
}

void drawKeypoints(cv::Mat& bgr, const std::vector<Keypoint>& keypoints) {
    for (const auto& keypoint : keypoints) {
        if (keypoint.imageX < 0 || keypoint.imageY < 0) continue;
        const bool named = ModelKeypointAnnotationService::isRegistrationKeypoint(keypoint.name);
        const cv::Scalar color = bgrForKeypoint(keypoint.name, keypoint.index);
        cv::circle(
            bgr,
            cv::Point(keypoint.imageX, keypoint.imageY),
            named ? 5 : 2,
            color,
            -1,
            cv::LINE_AA);
        if (named) {
            cv::putText(
                bgr,
                keypoint.name,
                cv::Point(keypoint.imageX + 7, std::max(14, keypoint.imageY - 7)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv::LINE_AA);
        }
    }
}

std::string jsonEscape(const std::string& input) {
    std::ostringstream output;
    for (const unsigned char ch : input) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

bool writeKeypointsPly(
    const std::filesystem::path& path,
    const std::vector<Keypoint>& keypoints,
    std::string* error) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        if (error) *error = "Failed to write model keypoint PLY: " + path.string();
        return false;
    }
    const auto validCount = std::count_if(
        keypoints.begin(), keypoints.end(), [](const Keypoint& keypoint) { return keypoint.hasPoint; });
    output << "ply\nformat ascii 1.0\n"
           << "element vertex " << validCount << '\n'
           << "property double x\nproperty double y\nproperty double z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    output << std::setprecision(12);
    for (const auto& keypoint : keypoints) {
        if (!keypoint.hasPoint) continue;
        const auto rgb = rgbForKeypoint(keypoint.name, keypoint.index);
        output << keypoint.point.x() << ' ' << keypoint.point.y() << ' ' << keypoint.point.z() << ' '
               << static_cast<int>(rgb[0]) << ' ' << static_cast<int>(rgb[1]) << ' '
               << static_cast<int>(rgb[2]) << '\n';
    }
    return static_cast<bool>(output);
}

bool writeKeypointsJson(
    const std::filesystem::path& path,
    const ModelKeypointAnnotationService::Result& result,
    std::string* error) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        if (error) *error = "Failed to write model keypoint JSON: " + path.string();
        return false;
    }
    output << std::setprecision(12)
           << "{\n  \"stlPath\": \"" << jsonEscape(result.stlPath.generic_string()) << "\",\n"
           << "  \"modelUnitScale\": " << result.modelUnitScale << ",\n"
           << "  \"selectedAzimuthDeg\": " << result.selectedAzimuthDeg << ",\n"
           << "  \"selectedElevationDeg\": " << result.selectedElevationDeg << ",\n"
           << "  \"renderedViewCount\": " << result.renderedViewCount << ",\n"
           << "  \"namedValidCount\": " << result.namedValidCount << ",\n"
           << "  \"keypoints\": [\n";
    for (std::size_t i = 0; i < result.keypoints.size(); ++i) {
        const auto& keypoint = result.keypoints[i];
        output << "    {\"index\": " << keypoint.index
               << ", \"name\": \"" << jsonEscape(keypoint.name) << "\""
               << ", \"score\": " << keypoint.score
               << ", \"imageX\": " << keypoint.imageX
               << ", \"imageY\": " << keypoint.imageY
               << ", \"hasPoint\": " << (keypoint.hasPoint ? "true" : "false");
        if (keypoint.hasPoint) {
            output << ", \"x\": " << keypoint.point.x()
                   << ", \"y\": " << keypoint.point.y()
                   << ", \"z\": " << keypoint.point.z();
        }
        output << '}' << (i + 1 == result.keypoints.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return static_cast<bool>(output);
}

bool finitePoint(const Eigen::Vector3d& point) {
    return point.array().isFinite().all();
}

std::optional<Eigen::Matrix4d> estimateRigid(
    const std::vector<Eigen::Vector3d>& source,
    const std::vector<Eigen::Vector3d>& target,
    const std::vector<std::size_t>& indices) {
    if (indices.size() < 3) return std::nullopt;
    Eigen::Vector3d sourceCenter = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetCenter = Eigen::Vector3d::Zero();
    for (const std::size_t index : indices) {
        sourceCenter += source[index];
        targetCenter += target[index];
    }
    sourceCenter /= static_cast<double>(indices.size());
    targetCenter /= static_cast<double>(indices.size());

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const std::size_t index : indices)
        covariance += (source[index] - sourceCenter) * (target[index] - targetCenter).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();
    if (rotation.determinant() < 0.0) {
        Eigen::Matrix3d correctedV = svd.matrixV();
        correctedV.col(2) *= -1.0;
        rotation = correctedV * svd.matrixU().transpose();
    }
    if (!rotation.array().isFinite().all() || rotation.determinant() < 0.0) return std::nullopt;

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = targetCenter - rotation * sourceCenter;
    return transform;
}

double pointError(
    const Eigen::Matrix4d& transform,
    const Eigen::Vector3d& source,
    const Eigen::Vector3d& target) {
    return (transform.block<3, 3>(0, 0) * source + transform.block<3, 1>(0, 3) - target).norm();
}

} // namespace

ModelKeypointAnnotationService::ModelKeypointAnnotationService(FaceKeypointService& keypointDetector)
    : keypointDetector_(keypointDetector) {}

bool ModelKeypointAnnotationService::isRegistrationKeypoint(const std::string& name) {
    return name == "nose_root" || name == "nose_tip" ||
           name == "right_eye_outer" || name == "right_eye_inner" ||
           name == "left_eye_inner" || name == "left_eye_outer";
}

ModelKeypointAnnotationService::Result ModelKeypointAnnotationService::annotateStlModelKeypoints(
    const std::filesystem::path& stlPath,
    const std::filesystem::path& outputDirectory,
    const Options& options,
    std::string* error) {
    Result result;
    result.stlPath = stlPath;
    result.modelUnitScale = options.modelUnitScale;
    if (!(options.modelUnitScale > 0.0) || !std::isfinite(options.modelUnitScale)) {
        result.message = "Model unit scale must be finite and positive";
        if (error) *error = result.message;
        return result;
    }
    if (options.azimuthCandidates.empty() || options.elevationCandidates.empty()) {
        result.message = "Model virtual-camera candidate list is empty";
        if (error) *error = result.message;
        return result;
    }

    std::string detectorError;
    if (!keypointDetector_.ensureLoaded(&detectorError)) {
        result.message = detectorError;
        if (error) *error = result.message;
        return result;
    }

    std::error_code directoryError;
    std::filesystem::create_directories(outputDirectory, directoryError);
    if (directoryError) {
        result.message = "Failed to create model keypoint output directory: " + outputDirectory.string();
        if (error) *error = result.message;
        return result;
    }

    StlModelRenderer renderer;
    std::string rendererError;
    if (!renderer.loadStl(stlPath, &rendererError)) {
        result.message = rendererError;
        if (error) *error = result.message;
        return result;
    }

    std::optional<Candidate> best;
    for (const double elevationDeg : options.elevationCandidates) {
        for (const double azimuthDeg : options.azimuthCandidates) {
            StlModelRenderer::RenderOptions renderOptions;
            renderOptions.width = options.renderWidth;
            renderOptions.height = options.renderHeight;
            renderOptions.azimuthDeg = azimuthDeg;
            renderOptions.elevationDeg = elevationDeg;
            std::string renderError;
            auto render = renderer.render(renderOptions, &renderError);
            if (render.bgr.empty() || render.zBuffer.empty()) {
                std::cerr << "STL keypoint render skipped az=" << azimuthDeg << " el=" << elevationDeg
                          << ": " << renderError << '\n';
                continue;
            }
            ++result.renderedViewCount;

            std::string detectError;
            const auto detection = keypointDetector_.detect(render.bgr, &detectError);
            if (detection.keypoints.empty()) {
                std::cerr << "STL keypoint detection skipped az=" << azimuthDeg << " el=" << elevationDeg
                          << ": " << detectError << '\n';
                continue;
            }

            Candidate candidate;
            candidate.azimuthDeg = azimuthDeg;
            candidate.elevationDeg = elevationDeg;
            candidate.bgr = render.bgr.clone();
            candidate.zBuffer = render.zBuffer.clone();
            double scoreSum = 0.0;
            int scoreCount = 0;
            for (const auto& detected : detection.keypoints) {
                if (!std::isfinite(detected.x) || !std::isfinite(detected.y) ||
                    !std::isfinite(detected.score) || detected.score < options.minScore)
                    continue;

                Keypoint keypoint;
                keypoint.index = detected.index;
                keypoint.name = detected.name;
                keypoint.score = detected.score;
                keypoint.imageX = static_cast<int>(std::lround(detected.x));
                keypoint.imageY = static_cast<int>(std::lround(detected.y));
                cv::Point sampledPixel;
                const auto world = renderer.worldPointFromPixel(
                    render, keypoint.imageX, keypoint.imageY, 4, &sampledPixel);
                keypoint.hasPoint = world.has_value();
                if (world) {
                    keypoint.point = Eigen::Vector3d((*world)[0], (*world)[1], (*world)[2]) * options.modelUnitScale;
                    keypoint.imageX = sampledPixel.x;
                    keypoint.imageY = sampledPixel.y;
                    ++candidate.validCount;
                    if (isRegistrationKeypoint(keypoint.name)) ++candidate.namedValidCount;
                }
                candidate.keypoints.push_back(std::move(keypoint));
                scoreSum += detected.score;
                ++scoreCount;
            }
            candidate.meanScore = scoreCount > 0 ? scoreSum / static_cast<double>(scoreCount) : 0.0;
            std::cout << "STL keypoint view az=" << azimuthDeg << " el=" << elevationDeg
                      << " valid=" << candidate.validCount
                      << " named=" << candidate.namedValidCount
                      << " mean_score=" << candidate.meanScore << '\n';

            if (!best || std::tie(candidate.namedValidCount, candidate.validCount, candidate.meanScore) >
                             std::tie(best->namedValidCount, best->validCount, best->meanScore))
                best = std::move(candidate);
        }
    }

    if (!best || best->namedValidCount < options.minNamedKeypointCount) {
        std::ostringstream message;
        message << "Not enough valid registration keypoints were recovered from "
                << result.renderedViewCount << " STL views: "
                << (best ? best->namedValidCount : 0) << '/' << options.minNamedKeypointCount;
        result.message = message.str();
        if (error) *error = result.message;
        return result;
    }

    result.selectedAzimuthDeg = best->azimuthDeg;
    result.selectedElevationDeg = best->elevationDeg;
    result.namedValidCount = best->namedValidCount;
    result.keypoints = std::move(best->keypoints);
    if (options.writeVisualizationArtifacts) {
        result.renderImagePath = outputDirectory / "model_keypoints_render.png";
        result.depthImagePath = outputDirectory / "model_keypoints_depth.png";
        cv::Mat annotated = best->bgr.clone();
        drawKeypoints(annotated, result.keypoints);
        if (!cv::imwrite(result.renderImagePath.string(), annotated) ||
            !cv::imwrite(result.depthImagePath.string(), visualizeZBuffer(best->zBuffer))) {
            result.message = "Failed to write selected STL RGB/depth render";
            if (error) *error = result.message;
            return result;
        }
    }
    if (options.write3DArtifacts) {
        result.keypointsJsonPath = outputDirectory / "model_keypoints.json";
        result.keypointsPlyPath = outputDirectory / "model_keypoints.ply";
        std::string writeError;
        if (!writeKeypointsPly(result.keypointsPlyPath, result.keypoints, &writeError) ||
            !writeKeypointsJson(result.keypointsJsonPath, result, &writeError)) {
            result.message = writeError;
            if (error) *error = result.message;
            return result;
        }
    }

    result.success = true;
    result.message = "ok";
    return result;
}

ModelKeypointAnnotationService::PoseEstimate ModelKeypointAnnotationService::estimateCameraToModel(
    const std::vector<Keypoint3D>& cameraKeypoints,
    const std::vector<Keypoint3D>& modelKeypoints,
    double inlierThresholdMm,
    PoseSolver solver) {
    PoseEstimate result;
    if (!(inlierThresholdMm > 0.0)) {
        result.message = "Keypoint inlier threshold must be positive";
        return result;
    }

    std::map<std::string, Eigen::Vector3d> modelByName;
    for (const auto& keypoint : modelKeypoints) {
        if (keypoint.hasPoint && isRegistrationKeypoint(keypoint.name) && finitePoint(keypoint.point))
            modelByName.emplace(keypoint.name, keypoint.point);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
    for (const auto& keypoint : cameraKeypoints) {
        if (!keypoint.hasPoint || !isRegistrationKeypoint(keypoint.name) || !finitePoint(keypoint.point)) continue;
        const auto model = modelByName.find(keypoint.name);
        if (model == modelByName.end()) continue;
        source.push_back(keypoint.point);
        target.push_back(model->second);
    }
    result.matchedCount = static_cast<int>(source.size());
    if (source.size() < 3) {
        result.message = "Need at least 3 shared camera/model 3D keypoints, got " +
                         std::to_string(source.size());
        return result;
    }

    if (solver == PoseSolver::OverdeterminedSvd) {
        std::vector<std::size_t> all(source.size());
        std::iota(all.begin(), all.end(), 0);
        const auto transform = estimateRigid(source, target, all);
        if (!transform || !transform->array().isFinite().all()) {
            result.message = "Overdetermined SVD failed to estimate a finite transform";
            return result;
        }
        double sumSquared = 0.0;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const double residual = pointError(*transform, source[i], target[i]);
            sumSquared += residual * residual;
        }
        result.inlierCount = static_cast<int>(source.size());
        result.rmseMm = std::sqrt(sumSquared / static_cast<double>(source.size()));
        result.cameraToModel = *transform;
        result.success = true;
        result.message = "ok_overdetermined_svd";
        return result;
    }

    std::vector<std::size_t> bestInliers;
    double bestRmse = std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a + 2 < source.size(); ++a) {
        for (std::size_t b = a + 1; b + 1 < source.size(); ++b) {
            for (std::size_t c = b + 1; c < source.size(); ++c) {
                if ((source[b] - source[a]).cross(source[c] - source[a]).norm() < 1e-3 ||
                    (target[b] - target[a]).cross(target[c] - target[a]).norm() < 1e-3)
                    continue;
                const std::vector<std::size_t> sample{a, b, c};
                const auto transform = estimateRigid(source, target, sample);
                if (!transform) continue;
                std::vector<std::size_t> inliers;
                double sumSquared = 0.0;
                for (std::size_t i = 0; i < source.size(); ++i) {
                    const double residual = pointError(*transform, source[i], target[i]);
                    if (residual <= inlierThresholdMm) {
                        inliers.push_back(i);
                        sumSquared += residual * residual;
                    }
                }
                const double rmse = inliers.empty()
                    ? std::numeric_limits<double>::infinity()
                    : std::sqrt(sumSquared / static_cast<double>(inliers.size()));
                if (inliers.size() > bestInliers.size() ||
                    (inliers.size() == bestInliers.size() && rmse < bestRmse)) {
                    bestInliers = std::move(inliers);
                    bestRmse = rmse;
                }
            }
        }
    }
    if (bestInliers.size() < 3) {
        result.message = "Shared face keypoints are geometrically degenerate or inconsistent";
        return result;
    }

    const auto refined = estimateRigid(source, target, bestInliers);
    if (!refined || !refined->array().isFinite().all()) {
        result.message = "Failed to estimate a finite camera-to-model keypoint transform";
        return result;
    }
    double sumSquared = 0.0;
    for (const std::size_t index : bestInliers) {
        const double residual = pointError(*refined, source[index], target[index]);
        sumSquared += residual * residual;
    }
    result.inlierCount = static_cast<int>(bestInliers.size());
    result.rmseMm = std::sqrt(sumSquared / static_cast<double>(bestInliers.size()));
    result.cameraToModel = *refined;
    result.success = true;
    result.message = "ok";
    return result;
}
