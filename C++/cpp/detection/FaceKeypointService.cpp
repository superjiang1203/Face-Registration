#include "detection/FaceKeypointService.hpp"

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <onnxruntime_c_api.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

    constexpr const char* kOpenMmlabFaceKeypointOnnxPath =
        "models/face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx";
    constexpr int kDefaultInputWidth = 256;
    constexpr int kDefaultInputHeight = 256;

    constexpr float kMeanRgb[3] = {123.675f, 116.28f, 103.53f};
    constexpr float kStdRgb[3] = {58.395f, 57.12f, 57.375f};

    float halfToFloat(uint16_t v) {
        const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
        const uint32_t exp = static_cast<uint32_t>(v & 0x7C00u) >> 10;
        const uint32_t mant = static_cast<uint32_t>(v & 0x03FFu);
        uint32_t f = 0;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                uint32_t m = mant;
                uint32_t e = 127 - 15 + 1;
                while ((m & 0x0400u) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x03FFu;
                f = sign | (e << 23) | (m << 13);
            }
        } else if (exp == 0x1Fu) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float out = 0.0f;
        std::memcpy(&out, &f, sizeof(float));
        return out;
    }

    size_t elementCount(const std::vector<int64_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t count = 1;
        for (const int64_t d : shape) {
            if (d <= 0) {
                return 0;
            }
            count *= static_cast<size_t>(d);
        }
        return count;
    }

    float scoreAt(const FaceKeypointService::TensorData* scores, int index) {
        if (!scores || scores->values.empty() || index < 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        if (scores->shape.size() == 2 && scores->shape[0] == 1 && index < scores->shape[1]) {
            return scores->values[static_cast<size_t>(index)];
        }
        if (scores->shape.size() == 1 && index < scores->shape[0]) {
            return scores->values[static_cast<size_t>(index)];
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

} // namespace

FaceKeypointService::FaceKeypointService(std::string modelPath)
    // ONNX Runtime uses the logger of the first environment created in this
    // process for some CUDA kernel messages from later sessions.  In the
    // camera pipeline this service is created before the CUDA face detector,
    // so harmless detector ScatterND warnings were incorrectly labelled as
    // FaceKeypointService messages.  Keep actionable runtime errors while
    // suppressing those misleading provider warnings.
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "FaceRegistration")),
      modelPath_(modelPath.empty() ? defaultModelPath() : std::move(modelPath)),
      inputWidth_(kDefaultInputWidth), inputHeight_(kDefaultInputHeight) {
}

std::string FaceKeypointService::defaultModelPath() {
    return (std::filesystem::current_path() / kOpenMmlabFaceKeypointOnnxPath).lexically_normal().string();
}

bool FaceKeypointService::ensureLoaded(std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session_) {
        return true;
    }
    return load(error);
}

bool FaceKeypointService::isLoaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return session_ != nullptr;
}

bool FaceKeypointService::isUsingCuda() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return usingCuda_;
}

std::string FaceKeypointService::resolvedModelPath() const {
    return modelPath_;
}

bool FaceKeypointService::load(std::string* error) {
    if (!std::filesystem::exists(modelPath_)) {
        if (error) {
            *error = "HRNet-WFLW face keypoint ONNX model not found: " + modelPath_;
        }
        return false;
    }

    try {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // CPU is the portable default. It avoids a partially installed CUDA provider
        // preventing the otherwise valid model from loading.
        usingCuda_ = false;

#ifdef _WIN32
        const std::wstring wpath = std::filesystem::absolute(modelPath_).wstring();
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), so);
#else
        const std::string path = std::filesystem::absolute(modelPath_).string();
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), so);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        inputNames_.clear();
        outputNames_.clear();
        inputShape_.clear();
        inputIsNhwc_ = false;
        inputWidth_ = kDefaultInputWidth;
        inputHeight_ = kDefaultInputHeight;

        const size_t inputCount = session_->GetInputCount();
        inputNames_.reserve(inputCount);
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            inputNames_.push_back(name.get() ? std::string(name.get()) : std::string());
        }
        if (inputCount > 0) {
            auto typeInfo = session_->GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputShape_ = tensorInfo.GetShape();
            if (inputShape_.size() == 4) {
                const int64_t c1 = inputShape_[1];
                const int64_t h2 = inputShape_[2];
                const int64_t c3 = inputShape_[3];
                if (c3 == 3 && c1 != 3) {
                    inputIsNhwc_ = true;
                    if (inputShape_[1] > 0)
                        inputHeight_ = static_cast<int>(inputShape_[1]);
                    if (inputShape_[2] > 0)
                        inputWidth_ = static_cast<int>(inputShape_[2]);
                } else {
                    inputIsNhwc_ = false;
                    if (h2 > 0)
                        inputHeight_ = static_cast<int>(h2);
                    if (inputShape_[3] > 0)
                        inputWidth_ = static_cast<int>(inputShape_[3]);
                }
            }
        }

        const size_t outputCount = session_->GetOutputCount();
        outputNames_.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(name.get() ? std::string(name.get()) : std::string());
        }

        if (inputNames_.empty() || outputNames_.empty()) {
            if (error) {
                *error = "HRNet-WFLW ONNX model has no usable inputs or outputs: " + modelPath_;
            }
            session_.reset();
            return false;
        }

        return true;
    } catch (const std::exception& ex) {
        if (error) {
            *error = std::string("Failed to load HRNet-WFLW face keypoint ONNX model: ") + ex.what();
        }
    } catch (...) {
        if (error) {
            *error = "Failed to load HRNet-WFLW face keypoint ONNX model: unknown error";
        }
    }

    session_.reset();
    inputNames_.clear();
    outputNames_.clear();
    usingCuda_ = false;
    return false;
}

std::vector<float> FaceKeypointService::makeInputTensorData(const cv::Mat& bgr) const {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputWidth_, inputHeight_), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb32;
    rgb.convertTo(rgb32, CV_32F);

    if (inputIsNhwc_) {
        std::vector<float> data(static_cast<size_t>(inputWidth_) * static_cast<size_t>(inputHeight_) * 3u);
        size_t idx = 0;
        for (int y = 0; y < inputHeight_; ++y) {
            const cv::Vec3f* row = rgb32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < inputWidth_; ++x) {
                for (int c = 0; c < 3; ++c) {
                    data[idx++] = (row[x][c] - kMeanRgb[c]) / kStdRgb[c];
                }
            }
        }
        return data;
    }

    std::vector<float> data(static_cast<size_t>(inputWidth_) * static_cast<size_t>(inputHeight_) * 3u);
    const size_t plane = static_cast<size_t>(inputWidth_) * static_cast<size_t>(inputHeight_);
    for (int y = 0; y < inputHeight_; ++y) {
        const cv::Vec3f* row = rgb32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < inputWidth_; ++x) {
            const size_t pixel = static_cast<size_t>(y) * static_cast<size_t>(inputWidth_) + static_cast<size_t>(x);
            for (int c = 0; c < 3; ++c) {
                data[static_cast<size_t>(c) * plane + pixel] = (row[x][c] - kMeanRgb[c]) / kStdRgb[c];
            }
        }
    }
    return data;
}

FaceKeypointService::Detection FaceKeypointService::detect(const cv::Mat& bgr, std::string* error) {
    Detection result;
    result.modelPath = modelPath_;
    if (bgr.empty() || bgr.channels() != 3) {
        if (error) {
            *error = "Face keypoint input image must be a non-empty BGR image";
        }
        return result;
    }
    result.imageWidth = bgr.cols;
    result.imageHeight = bgr.rows;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!session_ && !load(error)) {
        return result;
    }

    try {
        std::vector<float> inputData = makeInputTensorData(bgr);
        std::vector<int64_t> inputShape = inputShape_;
        if (inputShape.size() != 4 || elementCount(inputShape) != inputData.size()) {
            inputShape = inputIsNhwc_ ? std::vector<int64_t>{1, inputHeight_, inputWidth_, 3}
                                      : std::vector<int64_t>{1, 3, inputHeight_, inputWidth_};
        }

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

        std::vector<const char*> inputNamesC;
        inputNamesC.reserve(inputNames_.size());
        for (const auto& name : inputNames_) {
            inputNamesC.push_back(name.c_str());
        }

        std::vector<const char*> outputNamesC;
        outputNamesC.reserve(outputNames_.size());
        for (const auto& name : outputNames_) {
            outputNamesC.push_back(name.c_str());
        }

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr}, inputNamesC.data(), &inputTensor, 1, outputNamesC.data(), outputNamesC.size());

        std::vector<TensorData> outputData;
        outputData.reserve(outputs.size());
        for (const auto& out : outputs) {
            outputData.push_back(tensorToData(out));
        }
        result.keypoints = decodeOutputs(outputData, bgr.cols, bgr.rows);
        if (result.keypoints.empty() && error) {
            *error = "HRNet-WFLW face keypoint ONNX output could not be decoded";
        }
        return result;
    } catch (const std::exception& ex) {
        if (error) {
            *error = std::string("HRNet-WFLW face keypoint inference failed: ") + ex.what();
        }
    } catch (...) {
        if (error) {
            *error = "HRNet-WFLW face keypoint inference failed: unknown error";
        }
    }

    return result;
}

FaceKeypointService::TensorData FaceKeypointService::tensorToData(const Ort::Value& value) const {
    TensorData out;
    if (!value.IsTensor()) {
        return out;
    }

    auto info = value.GetTensorTypeAndShapeInfo();
    out.shape = info.GetShape();
    const size_t count = info.GetElementCount();
    if (count == 0) {
        return out;
    }

    const auto elemType = info.GetElementType();
    out.values.resize(count);
    if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = value.GetTensorData<float>();
        if (data) {
            std::copy(data, data + count, out.values.begin());
        }
    } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* data = value.GetTensorData<Ort::Float16_t>();
        if (data) {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; ++i) {
                out.values[i] = halfToFloat(h[i]);
            }
        }
    } else {
        out.values.clear();
    }
    return out;
}

std::vector<FaceKeypointService::Keypoint> FaceKeypointService::decodeOutputs(
    const std::vector<TensorData>& outputs, int imageWidth, int imageHeight) const {
    for (const auto& out : outputs) {
        if (out.shape.size() == 4 && !out.values.empty()) {
            auto keypoints = decodeHeatmapOutput(out, imageWidth, imageHeight);
            if (!keypoints.empty()) {
                return keypoints;
            }
        }
    }

    const TensorData* scoreOutput = nullptr;
    for (const auto& out : outputs) {
        if ((out.shape.size() == 1 || out.shape.size() == 2) && !out.values.empty()) {
            scoreOutput = &out;
            break;
        }
    }
    for (const auto& out : outputs) {
        if (out.shape.size() == 3 && !out.values.empty()) {
            auto keypoints = decodeCoordinateOutput(out, scoreOutput, imageWidth, imageHeight);
            if (!keypoints.empty()) {
                return keypoints;
            }
        }
    }

    return {};
}

std::vector<FaceKeypointService::Keypoint> FaceKeypointService::decodeHeatmapOutput(
    const TensorData& output, int imageWidth, int imageHeight) const {
    if (output.shape.size() != 4 || output.values.empty()) {
        return {};
    }

    const int64_t d1 = output.shape[1];
    const int64_t d2 = output.shape[2];
    const int64_t d3 = output.shape[3];

    bool heatmapNhwc = false;
    int keypointCount = 0;
    int heatmapH = 0;
    int heatmapW = 0;
    if (d1 > 0 && d1 <= 256 && d2 > 2 && d3 > 2) {
        heatmapNhwc = false;
        keypointCount = static_cast<int>(d1);
        heatmapH = static_cast<int>(d2);
        heatmapW = static_cast<int>(d3);
    } else if (d3 > 0 && d3 <= 256 && d1 > 2 && d2 > 2) {
        heatmapNhwc = true;
        keypointCount = static_cast<int>(d3);
        heatmapH = static_cast<int>(d1);
        heatmapW = static_cast<int>(d2);
    } else {
        return {};
    }

    auto at = [&](int k, int y, int x) -> float {
        if (heatmapNhwc) {
            const size_t idx = ((static_cast<size_t>(y) * static_cast<size_t>(heatmapW) + static_cast<size_t>(x)) *
                                   static_cast<size_t>(keypointCount)) +
                               static_cast<size_t>(k);
            return output.values[idx];
        }
        const size_t idx = (static_cast<size_t>(k) * static_cast<size_t>(heatmapH) + static_cast<size_t>(y)) *
                               static_cast<size_t>(heatmapW) +
                           static_cast<size_t>(x);
        return output.values[idx];
    };

    std::vector<Keypoint> keypoints;
    keypoints.reserve(static_cast<size_t>(keypointCount));
    for (int k = 0; k < keypointCount; ++k) {
        float bestValue = -std::numeric_limits<float>::infinity();
        int bestX = 0;
        int bestY = 0;
        for (int y = 0; y < heatmapH; ++y) {
            for (int x = 0; x < heatmapW; ++x) {
                const float v = at(k, y, x);
                if (v > bestValue) {
                    bestValue = v;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        float refinedX = static_cast<float>(bestX);
        float refinedY = static_cast<float>(bestY);
        if (bestX > 0 && bestX < heatmapW - 1 && bestY > 0 && bestY < heatmapH - 1) {
            const float dx = at(k, bestY, bestX + 1) - at(k, bestY, bestX - 1);
            const float dy = at(k, bestY + 1, bestX) - at(k, bestY - 1, bestX);
            if (dx > 0.0f)
                refinedX += 0.25f;
            if (dx < 0.0f)
                refinedX -= 0.25f;
            if (dy > 0.0f)
                refinedY += 0.25f;
            if (dy < 0.0f)
                refinedY -= 0.25f;
        }

        const float inputX =
            (refinedX + 0.5f) * (static_cast<float>(inputWidth_) / static_cast<float>(heatmapW)) - 0.5f;
        const float inputY =
            (refinedY + 0.5f) * (static_cast<float>(inputHeight_) / static_cast<float>(heatmapH)) - 0.5f;
        Keypoint kp;
        kp.index = k;
        kp.name = keypointName(k);
        kp.x = std::clamp(inputX * static_cast<float>(imageWidth) / static_cast<float>(inputWidth_), 0.0f,
            static_cast<float>(std::max(0, imageWidth - 1)));
        kp.y = std::clamp(inputY * static_cast<float>(imageHeight) / static_cast<float>(inputHeight_), 0.0f,
            static_cast<float>(std::max(0, imageHeight - 1)));
        kp.score = bestValue;
        keypoints.push_back(kp);
    }
    return keypoints;
}

std::vector<FaceKeypointService::Keypoint> FaceKeypointService::decodeCoordinateOutput(
    const TensorData& output, const TensorData* scoreOutput, int imageWidth, int imageHeight) const {
    if (output.shape.size() != 3 || output.shape[0] != 1 || output.values.empty()) {
        return {};
    }

    int keypointCount = 0;
    int valueCount = 0;
    bool transposed = false;
    if (output.shape[2] >= 2 && output.shape[2] <= 4) {
        keypointCount = static_cast<int>(output.shape[1]);
        valueCount = static_cast<int>(output.shape[2]);
    } else if (output.shape[1] >= 2 && output.shape[1] <= 4) {
        transposed = true;
        keypointCount = static_cast<int>(output.shape[2]);
        valueCount = static_cast<int>(output.shape[1]);
    } else {
        return {};
    }

    auto valueAt = [&](int k, int c) -> float {
        if (transposed) {
            return output.values[static_cast<size_t>(c) * static_cast<size_t>(keypointCount) + static_cast<size_t>(k)];
        }
        return output.values[static_cast<size_t>(k) * static_cast<size_t>(valueCount) + static_cast<size_t>(c)];
    };

    std::vector<Keypoint> keypoints;
    keypoints.reserve(static_cast<size_t>(keypointCount));
    for (int k = 0; k < keypointCount; ++k) {
        float x = valueAt(k, 0);
        float y = valueAt(k, 1);
        float score = valueCount >= 3 ? valueAt(k, 2) : scoreAt(scoreOutput, k);
        if (!std::isfinite(score)) {
            score = 1.0f;
        }

        const float maxCoord = std::max(std::abs(x), std::abs(y));
        if (maxCoord <= 2.0f) {
            x *= static_cast<float>(imageWidth);
            y *= static_cast<float>(imageHeight);
        } else if (maxCoord <= static_cast<float>(std::max(inputWidth_, inputHeight_)) * 1.5f) {
            x = x * static_cast<float>(imageWidth) / static_cast<float>(inputWidth_);
            y = y * static_cast<float>(imageHeight) / static_cast<float>(inputHeight_);
        }

        Keypoint kp;
        kp.index = k;
        kp.name = keypointName(k);
        kp.x = std::clamp(x, 0.0f, static_cast<float>(std::max(0, imageWidth - 1)));
        kp.y = std::clamp(y, 0.0f, static_cast<float>(std::max(0, imageHeight - 1)));
        kp.score = score;
        keypoints.push_back(kp);
    }
    return keypoints;
}

std::string FaceKeypointService::keypointName(int index) {
    switch (index) {
        case 51:
            return "nose_root";
        case 54:
            return "nose_tip";
        case 60:
            return "right_eye_outer";
        case 64:
            return "right_eye_inner";
        case 68:
            return "left_eye_inner";
        case 72:
            return "left_eye_outer";
        default:
            char buf[16] = {};
            std::snprintf(buf, sizeof(buf), "kp_%03d", index);
            return std::string(buf);
    }
}
