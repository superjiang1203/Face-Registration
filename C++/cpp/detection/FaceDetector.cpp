#include "detection/FaceDetector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_c_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

struct LetterboxResult {
    cv::Mat image;
    float scale = 1.0f;
    int padX = 0;
    int padY = 0;
};

LetterboxResult letterboxBgr(const cv::Mat& bgr, int targetW, int targetH) {
    LetterboxResult r;
    if (bgr.empty()) {
        return r;
    }

    const int srcW = bgr.cols;
    const int srcH = bgr.rows;

    const float scale = std::min(static_cast<float>(targetW) / static_cast<float>(srcW),
        static_cast<float>(targetH) / static_cast<float>(srcH));
    const int newW = std::max(1, static_cast<int>(std::round(srcW * scale)));
    const int newH = std::max(1, static_cast<int>(std::round(srcH * scale)));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(newW, newH));

    const int padX = (targetW - newW) / 2;
    const int padY = (targetH - newH) / 2;

    cv::Mat out(targetH, targetW, bgr.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));

    r.image = out;
    r.scale = scale;
    r.padX = padX;
    r.padY = padY;
    return r;
}

std::vector<FaceDetector::Detection> decodeYoloOutput(const cv::Mat& output,
    const FaceDetector::Options& opts,
    int srcW,
    int srcH,
    float scale,
    int padX,
    int padY) {
    std::vector<FaceDetector::Detection> dets;
    if (output.empty() || output.dims < 2) {
        return dets;
    }

    auto clampRect = [&](float x1, float y1, float x2, float y2) -> cv::Rect {
        const int ix1 = std::max(0, std::min(static_cast<int>(std::floor(x1)), srcW - 1));
        const int iy1 = std::max(0, std::min(static_cast<int>(std::floor(y1)), srcH - 1));
        const int ix2 = std::max(0, std::min(static_cast<int>(std::ceil(x2)), srcW));
        const int iy2 = std::max(0, std::min(static_cast<int>(std::ceil(y2)), srcH));
        const int w = std::max(0, ix2 - ix1);
        const int h = std::max(0, iy2 - iy1);
        return cv::Rect(ix1, iy1, w, h);
    };

    auto toOriginal = [&](float cx, float cy, float w, float h) -> cv::Rect {
        const float x1 = (cx - w / 2.0f - static_cast<float>(padX)) / scale;
        const float y1 = (cy - h / 2.0f - static_cast<float>(padY)) / scale;
        const float x2 = (cx + w / 2.0f - static_cast<float>(padX)) / scale;
        const float y2 = (cy + h / 2.0f - static_cast<float>(padY)) / scale;
        return clampRect(x1, y1, x2, y2);
    };

    auto toOriginalXYXY = [&](float x1, float y1, float x2, float y2) -> cv::Rect {
        const float ox1 = (x1 - static_cast<float>(padX)) / scale;
        const float oy1 = (y1 - static_cast<float>(padY)) / scale;
        const float ox2 = (x2 - static_cast<float>(padX)) / scale;
        const float oy2 = (y2 - static_cast<float>(padY)) / scale;
        return clampRect(ox1, oy1, ox2, oy2);
    };

    cv::Mat mat = output;
    if (mat.dims == 3) {
        const int d0 = mat.size[0];
        const int d1 = mat.size[1];
        const int d2 = mat.size[2];
        if (d0 == 1 && d1 > 0 && d2 > 0) {
            mat = mat.reshape(1, d1);
        }
    }

    if (mat.rows == 0 || mat.cols == 0) {
        return dets;
    }

    bool rowIsCandidate = mat.cols >= 5;
    bool colIsCandidate = (mat.rows > 0 && mat.cols > mat.rows && mat.rows <= 16);
    if (colIsCandidate) {
        mat = mat.t();
        rowIsCandidate = mat.cols >= 5;
    }
    if (!rowIsCandidate) {
        return dets;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(static_cast<size_t>(mat.rows));
    scores.reserve(static_cast<size_t>(mat.rows));

    float maxScoreSeen = 0.0f;
    float maxRow[6] = {0, 0, 0, 0, 0, 0};

    auto maybeDenormalize = [&](float& x1, float& y1, float& x2, float& y2) {
        const float maxV = std::max(std::max(x1, y1), std::max(x2, y2));
        if (maxV <= 1.5f) {
            x1 *= static_cast<float>(opts.inputWidth);
            x2 *= static_cast<float>(opts.inputWidth);
            y1 *= static_cast<float>(opts.inputHeight);
            y2 *= static_cast<float>(opts.inputHeight);
        }
    };

    for (int i = 0; i < mat.rows; ++i) {
        const float* p = mat.ptr<float>(i);
        if (mat.cols >= 5) {
            const float s = p[4];
            if (s > maxScoreSeen) {
                maxScoreSeen = s;
                maxRow[0] = p[0];
                maxRow[1] = p[1];
                maxRow[2] = p[2];
                maxRow[3] = p[3];
                maxRow[4] = p[4];
                maxRow[5] = (mat.cols >= 6) ? p[5] : 0.0f;
            }
        }

        if (mat.cols == 6) {
            float a0 = p[0];
            float a1 = p[1];
            float a2 = p[2];
            float a3 = p[3];
            const float score = p[4];
            if (score < opts.confThreshold) {
                continue;
            }

            cv::Rect bbox;
            if (a2 > a0 && a3 > a1) {
                float x1 = a0;
                float y1 = a1;
                float x2 = a2;
                float y2 = a3;
                maybeDenormalize(x1, y1, x2, y2);
                bbox = toOriginalXYXY(x1, y1, x2, y2);
            } else {
                bbox = toOriginal(a0, a1, a2, a3);
            }
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            boxes.push_back(bbox);
            scores.push_back(score);
            continue;
        }

        if (mat.cols == 7) {
            float x1 = p[1];
            float y1 = p[2];
            float x2 = p[3];
            float y2 = p[4];
            const float score = p[5];
            if (score < opts.confThreshold) {
                continue;
            }
            maybeDenormalize(x1, y1, x2, y2);
            cv::Rect bbox = toOriginalXYXY(x1, y1, x2, y2);
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            boxes.push_back(bbox);
            scores.push_back(score);
            continue;
        }

        if (mat.cols > 7) {
            float x1 = p[0];
            float y1 = p[1];
            float x2 = p[2];
            float y2 = p[3];
            const float rawScore = p[4];

            const float maxV = std::max(std::max(std::abs(x1), std::abs(y1)), std::max(std::abs(x2), std::abs(y2)));
            const bool looksNormalized = maxV <= 1.5f;
            const bool looksPixel = maxV <= std::max(opts.inputWidth, opts.inputHeight) * 1.5f;
            const bool looksXYXY = (x2 > x1) && (y2 > y1) && (looksNormalized || looksPixel);

            if (looksXYXY) {
                if (rawScore < opts.confThreshold) {
                    continue;
                }
                maybeDenormalize(x1, y1, x2, y2);
                cv::Rect bbox = toOriginalXYXY(x1, y1, x2, y2);
                if (bbox.width <= 0 || bbox.height <= 0) {
                    continue;
                }
                boxes.push_back(bbox);
                scores.push_back(rawScore);
                continue;
            }
        }

        const float cx = p[0];
        const float cy = p[1];
        const float w = p[2];
        const float h = p[3];

        float score = 0.0f;
        if (mat.cols == 5) {
            score = p[4];
        } else {
            score = p[4];
            float bestClass = 0.0f;
            for (int c = 5; c < mat.cols; ++c) {
                bestClass = std::max(bestClass, p[c]);
            }
            score *= bestClass;
        }

        if (score < opts.confThreshold) {
            continue;
        }

        cv::Rect bbox = toOriginal(cx, cy, w, h);
        if (bbox.width <= 0 || bbox.height <= 0) {
            continue;
        }
        boxes.push_back(bbox);
        scores.push_back(score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, opts.confThreshold, opts.nmsThreshold, indices);

    dets.reserve(indices.size());
    for (int idx : indices) {
        FaceDetector::Detection d;
        d.bbox = boxes[idx];
        d.score = scores[idx];
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(), [](const FaceDetector::Detection& a, const FaceDetector::Detection& b) {
        return a.score > b.score;
    });

    static bool printed = false;
    if (opts.verbose && !printed && dets.empty()) {
        printed = true;
        std::cout << "FaceDetector decode: rows=" << mat.rows << " cols=" << mat.cols << " maxScore=" << maxScoreSeen
                  << " maxRow=[" << maxRow[0] << "," << maxRow[1] << "," << maxRow[2] << "," << maxRow[3] << ","
                  << maxRow[4] << "," << maxRow[5] << "]" << std::endl;
        if (mat.rows > 0 && mat.cols >= 6) {
            std::cout << "  mat[0]=[" << mat.at<float>(0, 0) << "," << mat.at<float>(0, 1) << "," << mat.at<float>(0, 2)
                      << "," << mat.at<float>(0, 3) << "," << mat.at<float>(0, 4) << "," << mat.at<float>(0, 5) << "]"
                      << std::endl;
        }
        if (mat.rows > 1 && mat.cols >= 6) {
            std::cout << "  mat[1]=[" << mat.at<float>(1, 0) << "," << mat.at<float>(1, 1) << "," << mat.at<float>(1, 2)
                      << "," << mat.at<float>(1, 3) << "," << mat.at<float>(1, 4) << "," << mat.at<float>(1, 5) << "]"
                      << std::endl;
        }
    }

    return dets;
}

} // namespace

FaceDetector::FaceDetector() : FaceDetector(Options{}) {
}

FaceDetector::FaceDetector(Options options)
    : options_(options),
      env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceDetector")) {}

bool FaceDetector::loadOnnx(const std::string& onnxPath) {
    try {
        Ort::SessionOptions so;
        // Frame-level parallelism is controlled by CameraPipeline. Keeping
        // each ONNX session single-threaded avoids N workers each creating an
        // additional full CPU thread pool.
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        usingCuda_ = false;
        if (options_.preferCuda) {
            try {
#ifdef _WIN32
                HMODULE cudnn = LoadLibraryA("cudnn64_9.dll");
                if (!cudnn) {
                    throw std::runtime_error("cudnn64_9.dll missing");
                }
                FreeLibrary(cudnn);
#endif
                OrtCUDAProviderOptions cudaOpts;
                std::memset(&cudaOpts, 0, sizeof(cudaOpts));
                cudaOpts.device_id = 0;
                so.AppendExecutionProvider_CUDA(cudaOpts);
                usingCuda_ = true;
            } catch (...) {
                usingCuda_ = false;
            }
        }

#ifdef _WIN32
        std::wstring wpath = std::filesystem::path(onnxPath).wstring();
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), so);
#else
        session_ = std::make_unique<Ort::Session>(*env_, onnxPath.c_str(), so);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        inputNames_.clear();
        outputNames_.clear();
        inputIsNhwc_ = false;

        const size_t inputCount = session_->GetInputCount();
        inputNames_.reserve(inputCount);
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            inputNames_.push_back(name.get() ? std::string(name.get()) : std::string());
        }
        if (inputCount > 0) {
            auto typeInfo = session_->GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();
            if (shape.size() == 4) {
                const int64_t c1 = shape[1];
                const int64_t c3 = shape[3];
                if (c3 == 3 && c1 != 3) {
                    inputIsNhwc_ = true;
                }
            }
        }

        const size_t outputCount = session_->GetOutputCount();
        outputNames_.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(name.get() ? std::string(name.get()) : std::string());
        }

        return session_ != nullptr;
    } catch (...) {
        session_.reset();
        inputNames_.clear();
        outputNames_.clear();
        usingCuda_ = false;
        return false;
    }
}

bool FaceDetector::isLoaded() const {
    return session_ != nullptr;
}

bool FaceDetector::isUsingCuda() const {
    return usingCuda_;
}

std::vector<FaceDetector::Detection> FaceDetector::detect(const cv::Mat& bgr) const {
    if (!session_ || bgr.empty()) {
        return {};
    }

    const int srcW = bgr.cols;
    const int srcH = bgr.rows;
    const auto lb = letterboxBgr(bgr, options_.inputWidth, options_.inputHeight);
    if (lb.image.empty()) {
        return {};
    }

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor{nullptr};
    std::vector<float> nhwcFloat;
    cv::Mat blob;

    if (inputIsNhwc_) {
        cv::Mat rgb;
        cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);
        cv::Mat rgbf;
        rgb.convertTo(rgbf, CV_32F, 1.0 / 255.0);
        const size_t count = rgbf.total() * rgbf.channels();
        nhwcFloat.resize(count);
        std::memcpy(nhwcFloat.data(), rgbf.ptr<float>(), count * sizeof(float));
        const int64_t inputShape[4] = {1, options_.inputHeight, options_.inputWidth, 3};
        inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, nhwcFloat.data(), nhwcFloat.size(), inputShape, 4);
    } else {
        blob = cv::dnn::blobFromImage(lb.image, 1.0 / 255.0, cv::Size(options_.inputWidth, options_.inputHeight),
            cv::Scalar(), true, false, CV_32F);
        if (blob.empty() || blob.total() == 0) {
            return {};
        }
        const int64_t inputShape[4] = {1, 3, options_.inputHeight, options_.inputWidth};
        inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, blob.ptr<float>(), static_cast<size_t>(blob.total()), inputShape, 4);
    }

    std::vector<const char*> inputNamesC;
    inputNamesC.reserve(inputNames_.size());
    for (const auto& n : inputNames_) {
        inputNamesC.push_back(n.c_str());
    }

    std::vector<const char*> outputNamesC;
    outputNamesC.reserve(outputNames_.size());
    for (const auto& n : outputNames_) {
        outputNamesC.push_back(n.c_str());
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNamesC.data(), &inputTensor, 1, outputNamesC.data(), outputNamesC.size());
    if (outputs.empty()) {
        return {};
    }

    static bool dumped = false;
    if (options_.verbose && !dumped) {
        dumped = true;
        std::cout << "FaceDetector input layout: " << (inputIsNhwc_ ? "NHWC" : "NCHW") << std::endl;
        std::cout << "FaceDetector outputs: " << outputs.size() << std::endl;
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i].IsTensor()) {
                std::cout << "  out[" << i << "] not tensor" << std::endl;
                continue;
            }
            auto si = outputs[i].GetTensorTypeAndShapeInfo();
            auto sh = si.GetShape();
            std::cout << "  out[" << i << "] elem=" << static_cast<int>(si.GetElementType()) << " shape=[";
            for (size_t j = 0; j < sh.size(); ++j) {
                std::cout << sh[j] << (j + 1 < sh.size() ? "," : "");
            }
            std::cout << "]" << std::endl;
        }
        if (!outputs.empty() && outputs[0].IsTensor()) {
            auto si = outputs[0].GetTensorTypeAndShapeInfo();
            auto sh = si.GetShape();
            if (si.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && si.GetElementCount() >= 12) {
                const float* d = outputs[0].GetTensorData<float>();
                if (d) {
                    std::cout << "  out[0] first12=";
                    for (int k = 0; k < 12; ++k) {
                        std::cout << d[k] << (k + 1 < 12 ? "," : "");
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    auto halfToFloat = [](uint16_t v) -> float {
            const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
            const uint32_t exp = static_cast<uint32_t>(v & 0x7C00u) >> 10;
            const uint32_t mant = static_cast<uint32_t>(v & 0x03FFu);
            uint32_t f;
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
            float out;
            std::memcpy(&out, &f, sizeof(float));
            return out;
    };

    auto tensorToMat = [&](const Ort::Value& v, std::vector<float>& scratch) -> cv::Mat {
        if (!v.IsTensor()) {
            return {};
        }
        auto si = v.GetTensorTypeAndShapeInfo();
        auto shape = si.GetShape();
        if (shape.empty()) {
            return {};
        }
        const auto elemType = si.GetElementType();
        const float* data = nullptr;
        scratch.clear();

        if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            data = v.GetTensorData<float>();
            if (!data) return {};
        } else if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const Ort::Float16_t* h = v.GetTensorData<Ort::Float16_t>();
            if (!h) return {};
            const size_t count = si.GetElementCount();
            scratch.resize(count);
            const uint16_t* hv = reinterpret_cast<const uint16_t*>(h);
            for (size_t i = 0; i < count; ++i) scratch[i] = halfToFloat(hv[i]);
            data = scratch.data();
        } else {
            return {};
        }

        if (shape.size() == 2) {
            return cv::Mat(static_cast<int>(shape[0]), static_cast<int>(shape[1]), CV_32F, const_cast<float*>(data));
        }
        if (shape.size() == 3) {
            int sizes[3] = {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])};
            return cv::Mat(3, sizes, CV_32F, const_cast<float*>(data));
        }
        if (shape.size() == 4 && shape[0] == 1) {
            if (shape[1] == 1) {
                return cv::Mat(static_cast<int>(shape[2]), static_cast<int>(shape[3]), CV_32F, const_cast<float*>(data));
            }
            int sizes[3] = {static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3])};
            return cv::Mat(3, sizes, CV_32F, const_cast<float*>(data));
        }
        return {};
    };

    std::vector<Detection> best;
    for (const auto& out : outputs) {
        std::vector<float> scratch;
        cv::Mat outMat = tensorToMat(out, scratch);
        if (outMat.empty()) {
            continue;
        }
        auto dets = decodeYoloOutput(outMat, options_, srcW, srcH, lb.scale, lb.padX, lb.padY);
        if (dets.size() > best.size()) {
            best = std::move(dets);
        }
    }
    return best;
}
