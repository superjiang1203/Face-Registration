#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "camera/CameraTypes.hpp"
#include "registration/ModelKeypointAnnotationService.hpp"

class FaceKeypointService;

class CameraFaceKeypointExtractionService {
  public:
    using Keypoint3D = ModelKeypointAnnotationService::Keypoint3D;

    struct Snapshot {
        cv::Mat color;
        int pointCloudWidth{0};
        int pointCloudHeight{0};
        std::vector<PointXYZ> pointCloud;
    };

    struct Options {
        int searchRadiusPx{6};
        int minValidKeypointCount{3};
        double minKeypointScore{0.0};
        bool writeVisualizationArtifacts{true};
        bool write3DArtifacts{true};
    };

    struct CandidateMetrics {
        bool detectionSucceeded{false};
        bool geometryValid{false};
        int visibleCount{0};
        int validCount{0};
        double meanScore{0.0};
    };

    struct Result {
        bool success{false};
        std::string message;
        int visibleCount{0};
        int validCount{0};
        double meanScore{0.0};
        std::vector<Keypoint3D> keypoints;
        std::filesystem::path colorImagePath;
        std::filesystem::path keypointsJsonPath;
        std::filesystem::path keypointsPlyPath;
        std::filesystem::path renderImagePath;
    };

    explicit CameraFaceKeypointExtractionService(FaceKeypointService& keypointDetector);

    Result extractFromSnapshot(
        const Snapshot& snapshot,
        const std::filesystem::path& outputDirectory,
        const Options& options,
        std::string* error = nullptr);

  private:
    FaceKeypointService& keypointDetector_;
};
