#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <open3d/Open3D.h>

#include "registration/HeadSurfaceReconstructor.hpp"

struct RegistrationOptions {
    double voxelSizeMm{5.0};
    int ransacIterations{100000};
    int globalAttempts{12};
    int icpIterations{100};
    int randomSeed{42};
    bool reseedGlobalRandomEngine{true};
    double targetUnitScale{1.0};
    double initialFitnessToSkipGlobal{0.90};
    double initialRmseToSkipGlobalMm{3.0};
    double initialSourceToTargetRmseToSkipGlobalMm{6.0};
    double initialSourceToTargetP95ToSkipGlobalMm{10.0};
};

struct RegistrationResult {
    bool success{false};
    double fitness{0.0};
    double inlierRmseMm{0.0};
    double sourceToTargetMeanMm{0.0};
    double sourceToTargetRmseMm{0.0};
    double sourceToTargetMedianMm{0.0};
    double sourceToTargetP95Mm{0.0};
    double sourceToTargetMaxMm{0.0};
    double targetToSourceMeanMm{0.0};
    double targetToSourceRmseMm{0.0};
    double targetToSourceP95Mm{0.0};
    double symmetricMeanMm{0.0};
    double symmetricRmseMm{0.0};
    double symmetricMedianMm{0.0};
    double symmetricP95Mm{0.0};
    double symmetricMaxMm{0.0};
    std::size_t sourcePointCount{0};
    std::size_t targetPointCount{0};
    bool usedInitialTransformation{false};
    std::string initializationMethod{"none"};
    Eigen::Matrix4d transformation{Eigen::Matrix4d::Identity()};
    std::string message;
    std::vector<std::pair<std::string, double>> stageTimingsMs;
};

struct StlPointCloudGenerationResult {
    bool success{false};
    std::string outputPly;
    std::size_t pointCount{0};
    std::string message;
    std::vector<std::pair<std::string, double>> stageTimingsMs;
};

class PointCloudRegistration {
public:
    explicit PointCloudRegistration(RegistrationOptions options = {});
    static StlPointCloudGenerationResult prepareRegistrationCloudFromStl(
        const std::string& stlPath,
        const std::string& outputPly,
        double modelUnitScale = 1.0);
    static StlPointCloudGenerationResult prepareRegistrationCloudFromStl(
        const std::string& stlPath,
        const std::string& outputPly,
        const HeadSurfaceReconstructor::Options& reconstructionOptions);
    RegistrationResult alignFiles(const std::string& sourcePly,
                                  const std::string& targetPlyOrStl,
                                  const std::string& outputPly = {},
                                  const std::optional<Eigen::Matrix4d>& initialTransformation = std::nullopt) const;
    RegistrationResult alignClouds(
        const open3d::geometry::PointCloud& source,
        const open3d::geometry::PointCloud& target,
        const std::string& outputPly = {},
        const std::optional<Eigen::Matrix4d>& initialTransformation = std::nullopt) const;
private:
    RegistrationOptions options_;
};
