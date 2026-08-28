#include "segmentation/Sapiens2Segmenter.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {
constexpr std::array<float, 3> kMean{123.675f, 116.28f, 103.53f};
constexpr std::array<float, 3> kStd{58.395f, 57.12f, 57.375f};

const std::array<cv::Vec3b, 29> kPalette{{
    {0,0,0}, {128,0,0}, {0,128,0}, {128,128,0}, {0,0,128}, {128,0,128},
    {0,128,128}, {128,128,128}, {64,0,0}, {192,0,0}, {64,128,0}, {192,128,0},
    {64,0,128}, {192,0,128}, {64,128,128}, {192,128,128}, {0,64,0}, {128,64,0},
    {0,192,0}, {128,192,0}, {0,64,128}, {128,64,128}, {0,192,128}, {128,192,128},
    {64,64,0}, {192,64,0}, {64,192,0}, {192,192,0}, {64,64,128}
}};
}

Sapiens2Segmenter::Sapiens2Segmenter(Options options)
    : options_(options), env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Sapiens2Seg")) {}

void Sapiens2Segmenter::load(const std::string& modelPath) {
    if (!std::filesystem::exists(modelPath))
        throw std::runtime_error("ONNX model not found: " + modelPath);
    if (!std::filesystem::exists(modelPath + ".data"))
        throw std::runtime_error("ONNX external data not found: " + modelPath + ".data");

    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    usingCuda_ = false;
    if (options_.preferCuda) {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = options_.deviceId;
            cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            so.AppendExecutionProvider_CUDA(cuda);
            usingCuda_ = true;
        } catch (const std::exception& exception) {
            std::cerr << "Sapiens Seg CUDA unavailable; falling back to CPU: "
                      << exception.what() << '\n';
        }
    }

#ifdef _WIN32
    const auto wide = std::filesystem::path(modelPath).wstring();
    session_ = std::make_unique<Ort::Session>(*env_, wide.c_str(), so);
#else
    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), so);
#endif
    Ort::AllocatorWithDefaultOptions allocator;
    inputName_ = session_->GetInputNameAllocated(0, allocator).get();
    outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

    const auto inShape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto outShape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inShape.size() != 4 || inShape[1] != 3 || outShape.size() != 4)
        throw std::runtime_error("unexpected Sapiens2 ONNX tensor layout");
    inputHeight_ = static_cast<int>(inShape[2]);
    inputWidth_ = static_cast<int>(inShape[3]);
    classCount_ = static_cast<int>(outShape[1]);
}

Sapiens2Segmenter::Result Sapiens2Segmenter::infer(const cv::Mat& bgr) const {
    if (!session_) throw std::runtime_error("Sapiens2 model is not loaded");
    if (bgr.empty() || bgr.type() != CV_8UC3) throw std::runtime_error("input must be a non-empty CV_8UC3 BGR image");

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputWidth_, inputHeight_), 0, 0, cv::INTER_LINEAR);
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0, {}, cv::Scalar(), true, false, CV_32F);
    const size_t plane = static_cast<size_t>(inputWidth_) * inputHeight_;
    float* input = blob.ptr<float>();
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < plane; ++i)
            input[static_cast<size_t>(c) * plane + i] =
                (input[static_cast<size_t>(c) * plane + i] - kMean[c]) / kStd[c];

    const std::array<int64_t, 4> shape{1, 3, inputHeight_, inputWidth_};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input, blob.total(), shape.data(), shape.size());
    const char* inputNames[]{inputName_.c_str()};
    const char* outputNames[]{outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &tensor, 1, outputNames, 1);
    const auto info = outputs.at(0).GetTensorTypeAndShapeInfo();
    const auto outputShape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || outputShape.size() != 4)
        throw std::runtime_error("unexpected Sapiens2 output type");

    const int outClasses = static_cast<int>(outputShape[1]);
    const int outH = static_cast<int>(outputShape[2]);
    const int outW = static_cast<int>(outputShape[3]);
    const size_t outPlane = static_cast<size_t>(outH) * outW;
    const float* logits = outputs[0].GetTensorData<float>();
    cv::Mat best(bgr.size(), CV_32F, cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat labels(bgr.size(), CV_8U, cv::Scalar(0));
    cv::Mat score;
    for (int c = 0; c < outClasses; ++c) {
        cv::Mat classLogits(outH, outW, CV_32F, const_cast<float*>(logits + static_cast<size_t>(c) * outPlane));
        cv::resize(classLogits, score, bgr.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat better = score > best;
        score.copyTo(best, better);
        labels.setTo(c, better);
    }

    Result result;
    result.labels = labels;
    result.faceMask = (labels == 3) | (labels == 24) | (labels == 25) |
                      (labels == 26) | (labels == 27) | (labels == 28);
    result.coloredLabels.create(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
        const auto* src = labels.ptr<unsigned char>(y);
        auto* dst = result.coloredLabels.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) dst[x] = kPalette[std::min<int>(src[x], 28)];
    }
    return result;
}
