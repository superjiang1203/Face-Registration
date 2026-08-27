#include "detection/FaceDetectionService.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>

FaceDetectionService::FaceDetectionService() : FaceDetectionService(Options{}) {
}

FaceDetectionService::FaceDetectionService(Options opts) : options_(opts), detector_(opts.detector) {
}

bool FaceDetectionService::loadOnnx(const std::string& onnxPath) {
    modelLoaded_ = detector_.loadOnnx(onnxPath);
    return modelLoaded_;
}

bool FaceDetectionService::isUsingCuda() const {
    return detector_.isUsingCuda();
}

std::optional<FaceDetectionService::Result> FaceDetectionService::process(cv::Mat& bgr, const cv::Mat& depthAligned16u,
    const std::vector<PointXYZ>* pointCloud, int pointCloudWidth, int pointCloudHeight) {
    if (bgr.empty()) {
        return std::nullopt;
    }
    if (!modelLoaded_) {
        return std::nullopt;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto dets = detector_.detect(bgr);
    for (auto& d : dets) {
        d.selectionScore = d.score;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double inferMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    Result r;
    r.inferMs = inferMs;
    r.dets = std::move(dets);

    if (options_.draw) {
        if (options_.drawInPlace) {
            adjustAndDraw(bgr, r.dets, depthAligned16u, pointCloud, pointCloudWidth, pointCloudHeight);
        } else {
            r.annotatedBgr = bgr.clone();
            adjustAndDraw(r.annotatedBgr, r.dets, depthAligned16u, pointCloud, pointCloudWidth, pointCloudHeight);
        }
    }

    return r;
}

static uint16_t sampleDepthMmAt(const cv::Mat& depth16u, int x, int y, int radius) {
    if (depth16u.empty() || depth16u.type() != CV_16U) {
        return 0;
    }
    const int w = depth16u.cols;
    const int h = depth16u.rows;
    x = std::max(0, std::min(w - 1, x));
    y = std::max(0, std::min(h - 1, y));
    const int r = std::max(0, radius);

    std::vector<uint16_t> values;
    values.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
    for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy) {
        const uint16_t* row = depth16u.ptr<uint16_t>(yy);
        for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx) {
            const uint16_t v = row[xx];
            if (v != 0) {
                values.push_back(v);
            }
        }
    }
    if (values.empty()) {
        return 0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

static uint16_t sampleDepthMmAtPointCloud(
    const std::vector<PointXYZ>* pointCloud, int pcW, int pcH, int x, int y, int radius) {
    if (!pointCloud || pointCloud->empty() || pcW <= 0 || pcH <= 0) {
        return 0;
    }
    x = std::max(0, std::min(pcW - 1, x));
    y = std::max(0, std::min(pcH - 1, y));
    const int r = std::max(0, radius);

    std::vector<float> values;
    values.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
    for (int yy = std::max(0, y - r); yy <= std::min(pcH - 1, y + r); ++yy) {
        for (int xx = std::max(0, x - r); xx <= std::min(pcW - 1, x + r); ++xx) {
            const size_t idx = static_cast<size_t>(yy) * static_cast<size_t>(pcW) + static_cast<size_t>(xx);
            if (idx >= pointCloud->size()) {
                continue;
            }
            const float zMm = (*pointCloud)[idx].z;
            if (std::isfinite(zMm) && zMm > 0.0f) {
                values.push_back(zMm);
            }
        }
    }
    if (values.empty()) {
        return 0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const float mm = values[mid];
    if (!std::isfinite(mm) || mm <= 0.0f) {
        return 0;
    }
    const float clamped = std::min(65535.0f, std::max(0.0f, mm));
    return static_cast<uint16_t>(std::lround(clamped));
}

void FaceDetectionService::adjustAndDraw(
    cv::Mat& bgr, std::vector<FaceDetector::Detection>& dets, const cv::Mat& depthAligned16u,
    const std::vector<PointXYZ>* pointCloud, int pointCloudWidth, int pointCloudHeight) const {
    const int imgW = bgr.cols;
    const int imgH = bgr.rows;
    for (auto& d : dets) {
        cv::Rect r = d.bbox;
        const int w = r.width;
        const int h = r.height;
        const int dx = static_cast<int>(std::round(w * options_.boxExpandXRatio));
        const int dyUp = static_cast<int>(std::round(h * (options_.boxShiftUpRatio + options_.boxExpandTopRatio)));
        const int dyDown = static_cast<int>(std::round(h * (options_.boxShiftUpRatio - options_.boxShrinkBottomRatio)));

        r.x = std::max(0, r.x - dx);
        r.width = std::min(imgW - r.x, r.width + 2 * dx);
        r.y = std::max(0, r.y - dyUp);
        r.height = std::min(imgH - r.y, r.height + dyUp + dyDown);
        d.bbox = r;
    }

    for (const auto& d : dets) {
        cv::rectangle(bgr, d.bbox, cv::Scalar(0, 255, 0), 2);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << d.score;
        if (options_.drawDepth) {
            const int cx = d.bbox.x + d.bbox.width / 2;
            const int cy = d.bbox.y + d.bbox.height / 2;
            uint16_t depthMm = 0;
            if (!depthAligned16u.empty() && depthAligned16u.type() == CV_16U) {
                depthMm = sampleDepthMmAt(depthAligned16u, cx, cy, options_.depthSampleRadius);
            } else {
                depthMm = sampleDepthMmAtPointCloud(pointCloud, pointCloudWidth, pointCloudHeight, cx, cy, options_.depthSampleRadius);
            }
            if (depthMm != 0) {
                ss << " " << std::fixed << std::setprecision(2) << (depthMm / 1000.0) << "m";
            } else {
                ss << " N/A";
            }
        }
        const auto textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect bg(d.bbox.x, std::max(0, d.bbox.y - textSize.height - 6), textSize.width + 6, textSize.height + 6);
        cv::rectangle(bgr, bg, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(bgr, ss.str(), cv::Point(bg.x + 3, bg.y + bg.height - 4), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 2);
    }
}
