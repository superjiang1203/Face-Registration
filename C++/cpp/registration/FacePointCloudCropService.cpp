#include "registration/FacePointCloudCropService.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

FacePointCloudCropService::FacePointCloudCropService() : FacePointCloudCropService(Options{}) {
}

FacePointCloudCropService::FacePointCloudCropService(Options opts) : options_(opts) {
}

static int safeRoundToInt(double v) {
    return static_cast<int>(std::lround(v));
}

static float computeMedian(std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

static void filterOutliersInPlace(std::vector<PointXYZ>& pts, float maxZDevMm, float maxXYRadiusMm) {
    if (pts.empty()) {
        return;
    }
    std::vector<float> zs;
    zs.reserve(pts.size());
    for (const auto& p : pts) {
        zs.push_back(p.z);
    }
    const float medZ = computeMedian(zs);

    float meanX = 0.0f;
    float meanY = 0.0f;
    for (const auto& p : pts) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= static_cast<float>(pts.size());
    meanY /= static_cast<float>(pts.size());

    const float maxZ = std::max(0.0f, maxZDevMm);
    const float maxR = std::max(0.0f, maxXYRadiusMm);
    const float maxR2 = maxR * maxR;

    std::vector<PointXYZ> filtered;
    filtered.reserve(pts.size());
    for (const auto& p : pts) {
        if (maxZ > 0.0f && std::abs(p.z - medZ) > maxZ) {
            continue;
        }
        if (maxR > 0.0f) {
            const float dx = p.x - meanX;
            const float dy = p.y - meanY;
            if (dx * dx + dy * dy > maxR2) {
                continue;
            }
        }
        filtered.push_back(p);
    }
    pts.swap(filtered);
}

static void keepLargestConnectedComponentInPlace(std::vector<PointXYZ>& pts, const std::vector<uint16_t>& gx, const std::vector<uint16_t>& gy,
    const std::vector<int>& nodeIndex, int gridW, int gridH, float maxNeighborDzMm) {
    if (pts.empty()) {
        return;
    }
    const float maxDz = std::max(0.0f, maxNeighborDzMm);
    std::vector<uint8_t> visited(pts.size(), 0);
    std::vector<size_t> bestComp;
    bestComp.reserve(pts.size());

    std::queue<size_t> q;
    std::vector<size_t> comp;
    comp.reserve(1024);

    auto tryPush = [&](int nx, int ny, float zRef) {
        if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) return;
        const int ni = nodeIndex[static_cast<size_t>(ny) * static_cast<size_t>(gridW) + static_cast<size_t>(nx)];
        if (ni < 0) return;
        const size_t idx = static_cast<size_t>(ni);
        if (idx >= pts.size() || visited[idx]) return;
        if (maxDz > 0.0f && std::abs(pts[idx].z - zRef) > maxDz) return;
        visited[idx] = 1;
        q.push(idx);
        comp.push_back(idx);
    };

    for (size_t i = 0; i < pts.size(); ++i) {
        if (visited[i]) continue;
        visited[i] = 1;
        while (!q.empty()) q.pop();
        comp.clear();
        q.push(i);
        comp.push_back(i);

        while (!q.empty()) {
            const size_t cur = q.front();
            q.pop();
            const int x = static_cast<int>(gx[cur]);
            const int y = static_cast<int>(gy[cur]);
            const float z = pts[cur].z;
            tryPush(x - 1, y, z);
            tryPush(x + 1, y, z);
            tryPush(x, y - 1, z);
            tryPush(x, y + 1, z);
            tryPush(x - 1, y - 1, z);
            tryPush(x + 1, y - 1, z);
            tryPush(x - 1, y + 1, z);
            tryPush(x + 1, y + 1, z);
        }

        if (comp.size() > bestComp.size()) {
            bestComp = comp;
        }
    }

    if (bestComp.empty() || bestComp.size() == pts.size()) {
        return;
    }
    std::vector<PointXYZ> filtered;
    filtered.reserve(bestComp.size());
    for (const auto idx : bestComp) {
        filtered.push_back(pts[idx]);
    }
    pts.swap(filtered);
}

cv::Rect FacePointCloudCropService::mapRect(const cv::Rect& r, int srcW, int srcH, int dstW, int dstH) {
    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
        return clampRect(r, dstW, dstH);
    }
    const double sx = static_cast<double>(dstW) / static_cast<double>(srcW);
    const double sy = static_cast<double>(dstH) / static_cast<double>(srcH);
    const cv::Rect mapped(
        safeRoundToInt(static_cast<double>(r.x) * sx),
        safeRoundToInt(static_cast<double>(r.y) * sy),
        safeRoundToInt(static_cast<double>(r.width) * sx),
        safeRoundToInt(static_cast<double>(r.height) * sy));
    return clampRect(mapped, dstW, dstH);
}

cv::Rect FacePointCloudCropService::clampRect(const cv::Rect& r, int w, int h) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int x2 = std::min(w, r.x + r.width);
    int y2 = std::min(h, r.y + r.height);
    if (x2 <= x || y2 <= y) {
        return cv::Rect();
    }
    return cv::Rect(x, y, x2 - x, y2 - y);
}

std::optional<FacePointCloudCropService::CropRoi> FacePointCloudCropService::faceCropRoi(
    const cv::Size& colorSize, int pointCloudWidth, int pointCloudHeight, const cv::Rect& faceBbox) const {
    if (!options_.enabled) {
        return std::nullopt;
    }
    if (faceBbox.empty() || colorSize.width <= 0 || colorSize.height <= 0 || pointCloudWidth <= 0 || pointCloudHeight <= 0) {
        return std::nullopt;
    }

    cv::Rect pointCloudRoi = mapRect(faceBbox, colorSize.width, colorSize.height, pointCloudWidth, pointCloudHeight);
    if (pointCloudRoi.empty()) {
        return std::nullopt;
    }
    const int margin = std::max(0, options_.roiMargin);
    if (margin > 0) {
        pointCloudRoi.x += margin;
        pointCloudRoi.y += margin;
        pointCloudRoi.width -= 2 * margin;
        pointCloudRoi.height -= 2 * margin;
        pointCloudRoi = clampRect(pointCloudRoi, pointCloudWidth, pointCloudHeight);
        if (pointCloudRoi.empty()) {
            return std::nullopt;
        }
    }

    const cv::Rect colorRoi = mapRect(pointCloudRoi, pointCloudWidth, pointCloudHeight, colorSize.width, colorSize.height);
    if (colorRoi.empty()) {
        return std::nullopt;
    }
    return CropRoi{colorRoi, pointCloudRoi};
}

std::vector<PointXYZ> FacePointCloudCropService::cropFacePointCloud(
    const CameraFrame& frame, const cv::Size& colorSize, const cv::Rect& faceBbox) const {
    std::vector<PointXYZ> out;
    if (!options_.enabled) {
        return out;
    }
    if (frame.pointCloud.empty() || frame.pointCloudWidth <= 0 || frame.pointCloudHeight <= 0) {
        return out;
    }
    if (faceBbox.empty()) {
        return out;
    }

    const int w = frame.pointCloudWidth;
    const int h = frame.pointCloudHeight;
    const int colorW = colorSize.width;
    const int colorH = colorSize.height;
    if (colorW <= 0 || colorH <= 0) {
        return out;
    }
    const auto cropRoiOpt = faceCropRoi(colorSize, w, h, faceBbox);
    if (!cropRoiOpt.has_value()) {
        return out;
    }
    const cv::Rect roi = cropRoiOpt->pointCloudRoi;

    const size_t maxPoints = std::max<size_t>(1, options_.maxPoints);
    int stride = std::max(1, options_.stride);
    const double requestedSamples = static_cast<double>(roi.area()) /
                                    static_cast<double>(stride * stride);
    if (requestedSamples > static_cast<double>(maxPoints)) {
        stride = std::max(stride, static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(roi.area()) / static_cast<double>(maxPoints)))));
    }
    out.reserve(std::min<size_t>(maxPoints, static_cast<size_t>((roi.area() / (stride * stride)) + 1)));

    const int gridW = std::max(1, (roi.width + stride - 1) / stride);
    const int gridH = std::max(1, (roi.height + stride - 1) / stride);
    std::vector<int> nodeIndex(static_cast<size_t>(gridW) * static_cast<size_t>(gridH), -1);
    std::vector<uint16_t> gx;
    std::vector<uint16_t> gy;
    gx.reserve(out.capacity());
    gy.reserve(out.capacity());

    for (int y = roi.y; y < roi.y + roi.height; y += stride) {
        for (int x = roi.x; x < roi.x + roi.width; x += stride) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            if (idx >= frame.pointCloud.size()) {
                continue;
            }
            const auto& p = frame.pointCloud[idx];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= 0.0f) {
                continue;
            }
            const int cgx = (x - roi.x) / stride;
            const int cgy = (y - roi.y) / stride;
            if (cgx < 0 || cgy < 0 || cgx >= gridW || cgy >= gridH) {
                continue;
            }
            out.push_back(p);
            gx.push_back(static_cast<uint16_t>(cgx));
            gy.push_back(static_cast<uint16_t>(cgy));
            nodeIndex[static_cast<size_t>(cgy) * static_cast<size_t>(gridW) + static_cast<size_t>(cgx)] = static_cast<int>(out.size() - 1);
            if (out.size() >= maxPoints) break;
        }
        if (out.size() >= maxPoints) break;
    }

    if (options_.keepLargestComponent && static_cast<int>(out.size()) >= std::max(0, options_.componentMinPoints)) {
        keepLargestConnectedComponentInPlace(out, gx, gy, nodeIndex, gridW, gridH, options_.componentMaxNeighborDzMm);
    }

    if (options_.filterOutliers && static_cast<int>(out.size()) >= std::max(0, options_.outlierMinPoints)) {
        filterOutliersInPlace(out, options_.outlierMaxZDeviationMm, options_.outlierMaxXYRadiusMm);
    }

    return out;
}

std::optional<FacePointCloudCropService::OrganizedCrop> FacePointCloudCropService::cropFacePointCloudOrganized(
    const CameraFrame& frame, const cv::Size& colorSize, const cv::Rect& faceBbox) const {
    if (!options_.enabled) {
        return std::nullopt;
    }
    if (frame.pointCloud.empty() || frame.pointCloudWidth <= 0 || frame.pointCloudHeight <= 0 || faceBbox.empty()) {
        return std::nullopt;
    }

    const int w = frame.pointCloudWidth;
    const int h = frame.pointCloudHeight;
    if (frame.pointCloud.size() < static_cast<size_t>(w) * static_cast<size_t>(h)) {
        return std::nullopt;
    }

    const auto cropRoiOpt = faceCropRoi(colorSize, w, h, faceBbox);
    if (!cropRoiOpt.has_value()) {
        return std::nullopt;
    }

    const cv::Rect roi = cropRoiOpt->pointCloudRoi;
    OrganizedCrop crop;
    crop.roi = cropRoiOpt.value();
    crop.pointCloud.resize(static_cast<size_t>(roi.width) * static_cast<size_t>(roi.height));

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            const size_t srcIdx = static_cast<size_t>(roi.y + y) * static_cast<size_t>(w) + static_cast<size_t>(roi.x + x);
            const size_t dstIdx = static_cast<size_t>(y) * static_cast<size_t>(roi.width) + static_cast<size_t>(x);
            const auto& p = frame.pointCloud[srcIdx];
            crop.pointCloud[dstIdx] = p;
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && p.z > 0.0f) {
                ++crop.validPointCount;
            }
        }
    }

    if (crop.validPointCount == 0) {
        return std::nullopt;
    }
    return crop;
}
