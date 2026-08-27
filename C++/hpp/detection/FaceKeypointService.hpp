#pragma once

#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include <filesystem>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

class FaceKeypointService {
  public:
    struct Keypoint {
        int index = -1;
        std::string name;
        float x = 0.0f;
        float y = 0.0f;
        float score = 0.0f;
    };

    struct Detection {
        int imageWidth = 0;
        int imageHeight = 0;
        std::string modelPath;
        std::vector<Keypoint> keypoints;
    };

    struct TensorData {
        std::vector<int64_t> shape;
        std::vector<float> values;
    };

    explicit FaceKeypointService(std::string modelPath = {});
    ~FaceKeypointService() = default;

    bool ensureLoaded(std::string* error = nullptr);
    bool isLoaded() const;
    bool isUsingCuda() const;
    std::string resolvedModelPath() const;

    Detection detect(const cv::Mat& bgr, std::string* error = nullptr);

  private:
    bool load(std::string* error);
    std::vector<float> makeInputTensorData(const cv::Mat& bgr) const;
    std::vector<Keypoint> decodeOutputs(const std::vector<TensorData>& outputs, int imageWidth, int imageHeight) const;
    std::vector<Keypoint> decodeHeatmapOutput(const TensorData& output, int imageWidth, int imageHeight) const;
    std::vector<Keypoint> decodeCoordinateOutput(const TensorData& output, const TensorData* scoreOutput, int imageWidth, int imageHeight) const;
    TensorData tensorToData(const Ort::Value& value) const;

    static std::string defaultModelPath();
    static std::string keypointName(int index);

    mutable std::mutex mutex_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string modelPath_;
    bool usingCuda_ = false;
    bool inputIsNhwc_ = false;
    int inputWidth_ = 256;
    int inputHeight_ = 256;
    std::vector<int64_t> inputShape_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
};
