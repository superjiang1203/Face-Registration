#include "pose/Sapiens2PoseEstimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {
constexpr std::array<float, 3> kMean{0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd{0.229f, 0.224f, 0.225f};

cv::Rect2f paddedCrop(const cv::Rect2f& box, int outW, int outH) {
    float w = box.width * 1.25f, h = box.height * 1.25f;
    const float aspect = static_cast<float>(outW) / outH;
    if (w > h * aspect) h = w / aspect; else w = h * aspect;
    const float cx = box.x + box.width * .5f, cy = box.y + box.height * .5f;
    return {cx - w * .5f, cy - h * .5f, w, h};
}
}

Sapiens2PoseEstimator::Sapiens2PoseEstimator(Options options)
    : options_(options), env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Sapiens2Pose")) {}

void Sapiens2PoseEstimator::load(const std::string& modelPath) {
    if (!std::filesystem::exists(modelPath)) throw std::runtime_error("Pose ONNX not found: " + modelPath);
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    if (options_.preferCuda) {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = options_.deviceId;
        cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        so.AppendExecutionProvider_CUDA(cuda);
        usingCuda_ = true;
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
    const auto in = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto out = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in.size() != 4 || in[1] != 3 || out.size() != 4 || out[1] != 308)
        throw std::runtime_error("unexpected Sapiens2 Pose tensor layout");
    inputHeight_ = static_cast<int>(in[2]); inputWidth_ = static_cast<int>(in[3]);
}

Sapiens2PoseEstimator::Result Sapiens2PoseEstimator::infer(const cv::Mat& bgr, const cv::Rect2f& personBox) const {
    if (!session_) throw std::runtime_error("Pose model is not loaded");
    if (bgr.empty() || bgr.type() != CV_8UC3 || personBox.width <= 0 || personBox.height <= 0)
        throw std::runtime_error("invalid pose input image or person box");
    const cv::Rect2f crop = paddedCrop(personBox, inputWidth_, inputHeight_);
    const double sx = (inputWidth_ - 1.0) / crop.width, sy = (inputHeight_ - 1.0) / crop.height;
    cv::Mat inverse = (cv::Mat_<double>(2, 3) << 1.0 / sx, 0, crop.x, 0, 1.0 / sy, crop.y);
    cv::Mat warped;
    cv::warpAffine(bgr, warped, inverse, {inputWidth_, inputHeight_},
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
    cv::Mat blob = cv::dnn::blobFromImage(warped, 1.0 / 255.0, {}, {}, true, false, CV_32F);
    const size_t plane = static_cast<size_t>(inputWidth_) * inputHeight_;
    float* values = blob.ptr<float>();
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < plane; ++i)
            values[c * plane + i] = (values[c * plane + i] - kMean[c]) / kStd[c];

    const std::array<int64_t, 4> shape{1, 3, inputHeight_, inputWidth_};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(mem, values, blob.total(), shape.data(), shape.size());
    const char* ins[]{inputName_.c_str()}; const char* outs[]{outputName_.c_str()};
    auto output = session_->Run({}, ins, &input, 1, outs, 1);
    const auto os = output[0].GetTensorTypeAndShapeInfo().GetShape();
    const int count = static_cast<int>(os[1]), hh = static_cast<int>(os[2]), hw = static_cast<int>(os[3]);
    const size_t heatPlane = static_cast<size_t>(hh) * hw;
    const float* heatmaps = output[0].GetTensorData<float>();

    Result result; result.personBox = personBox; result.cropBox = crop; result.keypoints.reserve(count);
    for (int k = 0; k < count; ++k) {
        cv::Mat raw(hh, hw, CV_32F, const_cast<float*>(heatmaps + k * heatPlane));
        double score; cv::Point peak;
        cv::minMaxLoc(raw, nullptr, &score, nullptr, &peak);
        cv::Mat blurred; cv::GaussianBlur(raw, blurred, {11, 11}, 0, 0, cv::BORDER_CONSTANT);
        double blurredMax; cv::minMaxLoc(blurred, nullptr, &blurredMax);
        if (blurredMax > 0) blurred *= score / blurredMax;
        cv::max(blurred, 1e-3, blurred); cv::min(blurred, 50.0, blurred); cv::log(blurred, blurred);
        float x = static_cast<float>(peak.x), y = static_cast<float>(peak.y);
        const int px = std::clamp(peak.x, 1, hw - 2), py = std::clamp(peak.y, 1, hh - 2);
        const float c = blurred.at<float>(py, px);
        const float gx = .5f * (blurred.at<float>(py, px + 1) - blurred.at<float>(py, px - 1));
        const float gy = .5f * (blurred.at<float>(py + 1, px) - blurred.at<float>(py - 1, px));
        const float hxx = blurred.at<float>(py, px + 1) - 2 * c + blurred.at<float>(py, px - 1);
        const float hyy = blurred.at<float>(py + 1, px) - 2 * c + blurred.at<float>(py - 1, px);
        const float hxy = .5f * (blurred.at<float>(py + 1, px + 1) - blurred.at<float>(py, px + 1)
            - blurred.at<float>(py + 1, px) + 2 * c - blurred.at<float>(py, px - 1)
            - blurred.at<float>(py - 1, px) + blurred.at<float>(py - 1, px - 1));
        const float det = hxx * hyy - hxy * hxy;
        if (std::abs(det) > 1e-8f) { x -= (hyy * gx - hxy * gy) / det; y -= (-hxy * gx + hxx * gy) / det; }
        const float imageX = x / (hw - 1.f) * crop.width + crop.x;
        const float imageY = y / (hh - 1.f) * crop.height + crop.y;
        result.keypoints.push_back({k, {imageX, imageY}, static_cast<float>(score)});
    }
    return result;
}
