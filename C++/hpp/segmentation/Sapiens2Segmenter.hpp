#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

class Sapiens2Segmenter {
public:
    struct Options {
        bool preferCuda{true};
        int deviceId{0};
    };

    struct Result {
        cv::Mat labels;       // CV_8U, class id, camera RGB resolution
        cv::Mat faceMask;     // CV_8U, 0/255; face/neck + lips/teeth/tongue
        cv::Mat coloredLabels;// CV_8UC3 visualization
    };

    explicit Sapiens2Segmenter(Options options = {});
    void load(const std::string& modelPath);
    Result infer(const cv::Mat& bgr) const;
    bool isUsingCuda() const { return usingCuda_; }

private:
    Options options_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
    bool usingCuda_{false};
    int inputWidth_{768};
    int inputHeight_{1024};
    int classCount_{29};
};
