#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "camera/CameraTypes.hpp"
#include "detection/FaceDetector.hpp"

class FaceDetectionService {
  public:
    struct Result {
        std::vector<FaceDetector::Detection> dets;
        double inferMs = 0.0;
        cv::Mat annotatedBgr;
    };

    struct Options {
        FaceDetector::Options detector;
        bool draw = true;
        bool drawInPlace = false;
        bool drawDepth = false;
        int depthSampleRadius = 2;
        float boxShiftUpRatio{0.0f};
        float boxExpandTopRatio{0.0f};
        float boxShrinkBottomRatio{0.0f};
        float boxExpandXRatio{0.0f};
    };

    FaceDetectionService();
    explicit FaceDetectionService(Options opts);

    bool loadOnnx(const std::string& onnxPath);
    bool isUsingCuda() const;

    std::optional<Result> process(cv::Mat& bgr, const cv::Mat& depthAligned16u = cv::Mat(),
        const std::vector<PointXYZ>* pointCloud = nullptr, int pointCloudWidth = 0, int pointCloudHeight = 0);

  private:
    void adjustAndDraw(cv::Mat& bgr, std::vector<FaceDetector::Detection>& dets, const cv::Mat& depthAligned16u,
        const std::vector<PointXYZ>* pointCloud, int pointCloudWidth, int pointCloudHeight) const;

    Options options_;
    FaceDetector detector_;
    bool modelLoaded_{false};
};
