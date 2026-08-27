#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class FaceDetector {
  public:
    struct Options {
        int inputWidth = 640;
        int inputHeight = 640;
        // Low-confidence proposals are retained for synchronized 3D gates.
        float confThreshold = 0.1f;
        float nmsThreshold = 0.45f;
        bool preferCuda = true;
        bool verbose = false;
    };

    struct Detection {
        // Detection box in original camera-image coordinates.
        cv::Rect bbox;
        float score = 0.0f;
        float selectionScore = 0.0f;
        // Robust candidate depth in millimetres.  The live pipeline sets this
        // from the filtered face cloud's median before physical-face grouping.
        float depthMm = 0.0f;
    };

    FaceDetector();
    explicit FaceDetector(Options options);

    bool loadOnnx(const std::string& onnxPath);
    bool isLoaded() const;
    bool isUsingCuda() const;

    std::vector<Detection> detect(const cv::Mat& bgr) const;

  private:
    Options options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    bool usingCuda_ = false;
    bool inputIsNhwc_ = false;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
};
