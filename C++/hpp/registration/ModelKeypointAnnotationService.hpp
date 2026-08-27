#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

class FaceKeypointService;

class ModelKeypointAnnotationService {
  public:
    enum class PoseSolver {
        TripletVote,
        OverdeterminedSvd,
    };
    struct Keypoint3D {
        int index{-1};
        std::string name;
        double score{0.0};
        int imageX{-1};
        int imageY{-1};
        bool hasPoint{false};
        Eigen::Vector3d point{Eigen::Vector3d::Zero()};
    };

    struct Options {
        int renderWidth{1280};
        int renderHeight{960};
        double modelUnitScale{1.0};
        double minScore{0.0};
        int minNamedKeypointCount{3};
        bool writeVisualizationArtifacts{true};
        bool write3DArtifacts{true};
        std::vector<double> azimuthCandidates{0.0, -10.0, 10.0, -15.0, 15.0, -20.0, 20.0};
        std::vector<double> elevationCandidates{6.0, 0.0, -6.0, 10.0};
    };

    struct Result {
        bool success{false};
        std::string message;
        std::filesystem::path stlPath;
        double modelUnitScale{1.0};
        double selectedAzimuthDeg{0.0};
        double selectedElevationDeg{0.0};
        int renderedViewCount{0};
        int namedValidCount{0};
        std::vector<Keypoint3D> keypoints;
        std::filesystem::path keypointsJsonPath;
        std::filesystem::path keypointsPlyPath;
        std::filesystem::path renderImagePath;
        std::filesystem::path depthImagePath;
    };

    struct PoseEstimate {
        bool success{false};
        std::string message;
        int matchedCount{0};
        int inlierCount{0};
        double rmseMm{0.0};
        Eigen::Matrix4d cameraToModel{Eigen::Matrix4d::Identity()};
    };

    explicit ModelKeypointAnnotationService(FaceKeypointService& keypointDetector);

    Result annotateStlModelKeypoints(
        const std::filesystem::path& stlPath,
        const std::filesystem::path& outputDirectory,
        const Options& options,
        std::string* error = nullptr);

    static PoseEstimate estimateCameraToModel(
        const std::vector<Keypoint3D>& cameraKeypoints,
        const std::vector<Keypoint3D>& modelKeypoints,
        double inlierThresholdMm = 15.0,
        PoseSolver solver = PoseSolver::TripletVote);

    static bool isRegistrationKeypoint(const std::string& name);

  private:
    FaceKeypointService& keypointDetector_;
};
