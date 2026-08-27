#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

class Sapiens2PoseEstimator {
public:
    struct Options { bool preferCuda{true}; int deviceId{0}; };
    struct Keypoint { int index{}; cv::Point2f position; float score{}; };
    struct Result { std::vector<Keypoint> keypoints; cv::Rect2f personBox; cv::Rect2f cropBox; };

    explicit Sapiens2PoseEstimator(Options options = {});
    void load(const std::string& modelPath);
    Result infer(const cv::Mat& bgr, const cv::Rect2f& personBox) const;
    bool isUsingCuda() const { return usingCuda_; }

private:
    Options options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_, outputName_;
    bool usingCuda_{false};
    int inputWidth_{768}, inputHeight_{1024};
};
