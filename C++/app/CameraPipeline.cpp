// Live depth-camera face registration application.
#include "camera/OrbbecDepthAlignedCamera.hpp"
#if FACE_HAS_VCAMERA
#include "camera/VcameraDepthAlignedCamera.hpp"
#endif
#include "detection/FaceDetectionService.hpp"
#include "detection/FaceKeypointService.hpp"
#include "pose/Sapiens2PoseEstimator.hpp"
#include "registration/CameraFaceKeypointExtractionService.hpp"
#include "registration/FaceCandidateSelectionPolicy.hpp"
#include "registration/FacePointCloudCropService.hpp"
#include "registration/HeadSurfaceCache.hpp"
#include "registration/ModelKeypointAnnotationService.hpp"
#include "registration/PointCloudRegistration.hpp"
#include "registration/StlModelRenderer.hpp"
#include "segmentation/Sapiens2Segmenter.hpp"

#include <open3d/Open3D.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::optional<bool> parseYamlBool(std::string value) {
    value = trim(std::move(value));
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (value == "true" || value == "yes" || value == "on" || value == "1")
        return true;
    if (value == "false" || value == "no" || value == "off" || value == "0")
        return false;
    return std::nullopt;
}

std::optional<std::string> readYamlScalar(
    const std::filesystem::path& path,
    const std::string& section,
    const std::string& key) {
    std::ifstream input(path);
    if (!input) return std::nullopt;
    std::string line, activeSection;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        if (trim(line).empty()) continue;
        const bool indented = !line.empty() && std::isspace(static_cast<unsigned char>(line.front()));
        const std::string text = trim(line);
        const auto colon = text.find(':');
        if (colon == std::string::npos) continue;
        const std::string name = trim(text.substr(0, colon));
        const std::string value = trim(text.substr(colon + 1));
        if (!indented && value.empty()) {
            activeSection = name;
        } else if (indented && activeSection == section && name == key) {
            if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                      (value.front() == '\'' && value.back() == '\'')))
                return value.substr(1, value.size() - 2);
            return value;
        }
    }
    return std::nullopt;
}

double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void reportTiming(
    const std::filesystem::path& timingPath,
    const std::string& stage,
    double milliseconds,
    int frameIndex = -1) {
    // Detailed timings remain available internally for aggregation, but the
    // public pipeline output intentionally contains only the four business
    // stages written by writeFinalTimings().
    (void)timingPath;
    (void)stage;
    (void)milliseconds;
    (void)frameIndex;
}

void writeFinalTimings(const std::filesystem::path& timingPath,
                       const std::string& localizationStage,
                       double localizationMs,
                       const std::string& keypointStage,
                       double keypointMs,
                       double svdVotingMs,
                       double icpIterationsMs,
                       double totalMs,
                       const std::vector<std::pair<std::string, double>>& details) {
    std::ofstream output(timingPath, std::ios::trunc);
    const auto write = [&](const std::string& stage, double milliseconds) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(3)
             << "[timing] stage=" << stage
             << " elapsed_ms=" << std::max(0.0, milliseconds);
        std::cout << line.str() << '\n';
        if (output) output << line.str() << '\n';
    };
    if (!localizationStage.empty()) write(localizationStage, localizationMs);
    write(keypointStage, keypointMs);
    write("svd_voting_coarse_registration", svdVotingMs);
    write("icp_total_iterations", icpIterationsMs);
    for (const auto& detail : details) write(detail.first, detail.second);
    write("total_localization_to_registration", totalMs);
}

cv::Mat visualizeAlignedDepth(const cv::Mat& depthMm) {
    if (depthMm.empty()) return {};
    cv::Mat depthFloat;
    depthMm.convertTo(depthFloat, CV_32F);
    cv::Mat valid = depthFloat > 0.0f;
    double minimum = 0.0, maximum = 0.0;
    cv::minMaxLoc(depthFloat, &minimum, &maximum, nullptr, nullptr, valid);
    cv::Mat gray(depthMm.size(), CV_8U, cv::Scalar(0));
    if (cv::countNonZero(valid) > 0 && maximum > minimum) {
        depthFloat.convertTo(gray, CV_8U, -255.0 / (maximum - minimum),
                             255.0 * maximum / (maximum - minimum));
        gray.setTo(0, ~valid);
    }
    cv::Mat colored;
    cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid);
    return colored;
}

cv::Mat normalizeDepthCropForDisplay(const cv::Mat& depthCrop) {
    if (depthCrop.empty()) return {};
    cv::Mat converted = depthCrop;
    if (converted.type() == CV_32F || converted.type() == CV_64F) {
        cv::Mat valid = converted > 0.0f;
        if (cv::countNonZero(valid) == 0) {
            return cv::Mat(depthCrop.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        }
        double minValue = 0.0, maxValue = 0.0;
        cv::minMaxLoc(converted, &minValue, &maxValue, nullptr, nullptr, valid);
        cv::Mat normalized;
        if (maxValue > minValue) {
            converted.convertTo(normalized, CV_8U, 255.0 / (maxValue - minValue), -255.0 * minValue / (maxValue - minValue));
        } else {
            converted.convertTo(normalized, CV_8U, 1.0, 0.0);
        }
        cv::Mat color;
        cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
        cv::Mat validMask = valid;
        color.setTo(cv::Scalar(0, 0, 0), ~validMask);
        return color;
    }
    if (converted.type() != CV_16U && converted.type() != CV_32S && converted.type() != CV_16S) {
        return {};
    }
    cv::Mat validMask = converted > 0;
    if (cv::countNonZero(validMask) == 0) {
        return cv::Mat(depthCrop.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    }
    double minValue = 0.0, maxValue = 0.0;
    cv::minMaxLoc(converted, &minValue, &maxValue, nullptr, nullptr, validMask);
    cv::Mat normalized;
    if (maxValue > minValue) {
        cv::normalize(converted, normalized, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    } else {
        normalized = cv::Mat(depthCrop.size(), CV_8U, cv::Scalar(0));
    }
    cv::Mat color;
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), ~validMask);
    return color;
}

void saveRoiDebugArtifacts(
    const CameraFrame& frame,
    const cv::Rect& roi,
    const std::filesystem::path& outputDirectory,
    const std::string& stem) {
    if (frame.color.empty() || frame.depth.empty() || roi.empty()) {
        return;
    }
    const cv::Rect clipped = roi & cv::Rect(0, 0, frame.color.cols, frame.color.rows);
    if (clipped.empty()) {
        return;
    }
    const cv::Mat colorCrop = frame.color(clipped).clone();
    const cv::Mat depthCrop = frame.depth(clipped).clone();
    if (!colorCrop.empty()) {
        cv::imwrite((outputDirectory / (stem + "_color.png")).string(), colorCrop);
    }
    if (!depthCrop.empty()) {
        const cv::Mat depthDisplay = normalizeDepthCropForDisplay(depthCrop);
        if (!depthDisplay.empty()) {
            cv::imwrite((outputDirectory / (stem + "_depth.png")).string(), depthDisplay);
        }
        if (depthCrop.type() == CV_16U || depthCrop.type() == CV_16S || depthCrop.type() == CV_32S) {
            cv::imwrite((outputDirectory / (stem + "_depth_raw.png")).string(), depthCrop);
        }
    }
}

bool writeD2cDiagnostics(
    const CameraFrame& frame,
    const std::filesystem::path& outputDirectory) {
    cv::Mat depthColor = visualizeAlignedDepth(frame.depth);
    if (depthColor.empty() || frame.color.empty()) return false;
    if (depthColor.size() != frame.color.size())
        cv::resize(depthColor, depthColor, frame.color.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat overlay;
    cv::addWeighted(frame.color, 0.55, depthColor, 0.45, 0.0, overlay);
    return cv::imwrite((outputDirectory / "camera_depth_aligned.png").string(), depthColor) &&
           cv::imwrite((outputDirectory / "camera_d2c_overlay.png").string(), overlay);
}

void printDevices(const std::string& backend, const std::vector<CameraDeviceInfo>& devices) {
    std::cout << backend << " cameras found: " << devices.size() << '\n';
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        std::cout << "  [" << i << "] model=" << d.model
                  << ", SN=" << (d.serialNumber.empty() ? "-" : d.serialNumber)
                  << ", IP=" << (d.ipAddress.empty() ? "-" : d.ipAddress)
                  << ", connection=" << d.connectionType << '\n';
    }
}

std::filesystem::path createSessionDirectory(const std::filesystem::path& root) {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&local, "%Y-%m-%d_%H-%M-%S")
              << '-' << std::setw(3) << std::setfill('0') << milliseconds.count();
    auto session = root / timestamp.str();
    for (int suffix = 1; std::filesystem::exists(session); ++suffix) {
        session = root / (timestamp.str() + "_" + std::to_string(suffix));
    }
    std::filesystem::create_directories(session);
    return session;
}

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return extension;
}

bool isStlPath(const std::filesystem::path& path) {
    return lowerExtension(path) == ".stl";
}

bool isPlyPath(const std::filesystem::path& path) {
    return lowerExtension(path) == ".ply";
}

struct PointKey {
    std::uint32_t x{0};
    std::uint32_t y{0};
    std::uint32_t z{0};

    bool operator==(const PointKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PointKeyHash {
    std::size_t operator()(const PointKey& key) const {
        std::size_t hash = static_cast<std::size_t>(key.x);
        hash ^= static_cast<std::size_t>(key.y) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.z) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        return hash;
    }
};

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

PointKey makePointKey(const PointXYZ& point) {
    return PointKey{floatBits(point.x), floatBits(point.y), floatBits(point.z)};
}

bool isValidPoint(const PointXYZ& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && point.z > 0.0f;
}

void keepOnlyFilteredPoints(
    std::vector<PointXYZ>& organizedPointCloud,
    const std::vector<PointXYZ>& filteredPointCloud) {
    std::unordered_set<PointKey, PointKeyHash> kept;
    kept.reserve(filteredPointCloud.size());
    for (const auto& point : filteredPointCloud) {
        if (isValidPoint(point)) kept.insert(makePointKey(point));
    }
    for (auto& point : organizedPointCloud) {
        if (!isValidPoint(point) || kept.find(makePointKey(point)) == kept.end()) {
            point = PointXYZ{};
        }
    }
}

std::size_t intersectPointCloudWithMask(
    std::vector<PointXYZ>& filteredPointCloud,
    const CameraFrame& frame,
    const cv::Size& colorSize,
    const cv::Rect& cropBbox,
    const cv::Mat& fullColorMask,
    const FacePointCloudCropService& cropper) {
    if (fullColorMask.empty() || fullColorMask.size() != colorSize) return 0;
    const auto organized = cropper.cropFacePointCloudOrganized(
        frame, colorSize, cropBbox);
    if (!organized || organized->roi.colorRoi.empty() ||
        organized->roi.pointCloudRoi.empty()) {
        filteredPointCloud.clear();
        return 0;
    }

    cv::Mat mask = fullColorMask(organized->roi.colorRoi);
    cv::Mat pointMask;
    cv::resize(mask, pointMask, organized->roi.pointCloudRoi.size(),
               0.0, 0.0, cv::INTER_NEAREST);
    std::unordered_set<PointKey, PointKeyHash> allowed;
    allowed.reserve(organized->validPointCount);
    for (int y = 0; y < pointMask.rows; ++y) {
        const auto* maskRow = pointMask.ptr<std::uint8_t>(y);
        for (int x = 0; x < pointMask.cols; ++x) {
            if (maskRow[x] == 0) continue;
            const auto& point = organized->pointCloud[
                static_cast<std::size_t>(y) * pointMask.cols + x];
            if (isValidPoint(point)) allowed.insert(makePointKey(point));
        }
    }
    filteredPointCloud.erase(
        std::remove_if(filteredPointCloud.begin(), filteredPointCloud.end(),
            [&](const PointXYZ& point) {
                return !isValidPoint(point) ||
                       allowed.find(makePointKey(point)) == allowed.end();
            }),
        filteredPointCloud.end());
    return allowed.size();
}

// Shrinks a face detection box to the core face region used for registration.
// The detector box typically extends past the chin onto the neck and past the
// contour onto ears/hair, none of which exist on the STL head model.  Those
// unmatchable fringe points inflate the source->target tail distance and drag
// ICP fitness below the gate.  Trim more from the bottom (chin/neck) than the
// top or sides.
static cv::Rect shrinkFaceBoxForCrop(const cv::Rect& box,
                                     float shrinkX = 0.08f,
                                     float shrinkTop = 0.06f,
                                     float shrinkBottom = 0.22f) {
    if (box.width <= 4 || box.height <= 4) return box;
    const int dx = static_cast<int>(std::round(box.width * shrinkX));
    const int dyTop = static_cast<int>(std::round(box.height * shrinkTop));
    const int dyBottom = static_cast<int>(std::round(box.height * shrinkBottom));
    cv::Rect r = box;
    r.x += dx;
    r.y += dyTop;
    r.width = std::max(1, r.width - 2 * dx);
    r.height = std::max(1, r.height - dyTop - dyBottom);
    return r;
}

struct PreparedFaceCandidate {
    FaceDetector::Detection detection;
    cv::Rect cropBbox;
    cv::Mat semanticMask;
    std::size_t semanticPixelArea{0};
    std::vector<PointXYZ> points;
    FaceCandidateSelectionPolicy::Metrics metrics;
};

double validatedDetectionScore(const FaceDetector::Detection& detection) {
    // Keep the detector's intentional 0.10 recall floor for rotated faces.
    // Center/aspect heuristics and the synchronized 3D gates reject false
    // positives later; selectionScore must not silently raise this threshold.
    return static_cast<double>(detection.score);
}

std::optional<CameraFaceKeypointExtractionService::Snapshot> snapshotForCandidate(
    const CameraFrame& originalFrame,
    const PreparedFaceCandidate& candidate,
    const FacePointCloudCropService& cropper) {
    if (originalFrame.color.empty() || originalFrame.pointCloud.empty())
        return std::nullopt;
    // Keypoint inference needs the complete detection box.  cropBbox is
    // intentionally tightened for ICP and removes useful facial context.
    const cv::Rect keypointBbox = candidate.detection.bbox.empty()
        ? candidate.cropBbox : candidate.detection.bbox;
    const auto organizedCrop = cropper.cropFacePointCloudOrganized(
        originalFrame, originalFrame.color.size(), keypointBbox);
    if (!organizedCrop) return std::nullopt;

    CameraFaceKeypointExtractionService::Snapshot snapshot;
    snapshot.color = originalFrame.color(organizedCrop->roi.colorRoi).clone();
    snapshot.pointCloudWidth = organizedCrop->roi.pointCloudRoi.width;
    snapshot.pointCloudHeight = organizedCrop->roi.pointCloudRoi.height;
    snapshot.pointCloud = organizedCrop->pointCloud;
    keepOnlyFilteredPoints(snapshot.pointCloud, candidate.points);
    return snapshot;
}

std::optional<Eigen::Vector3d> sampleSnapshotPoint(
    const CameraFaceKeypointExtractionService::Snapshot& snapshot, int x, int y, int radius = 6) {
    if (snapshot.color.empty() || snapshot.pointCloudWidth <= 0 ||
        snapshot.pointCloudHeight <= 0) return std::nullopt;
    const int centerX = std::clamp(cvRound(
        (static_cast<double>(x) + 0.5) * snapshot.pointCloudWidth /
        snapshot.color.cols - 0.5), 0, snapshot.pointCloudWidth - 1);
    const int centerY = std::clamp(cvRound(
        (static_cast<double>(y) + 0.5) * snapshot.pointCloudHeight /
        snapshot.color.rows - 0.5), 0, snapshot.pointCloudHeight - 1);
    const int cloudRadius = cvRound(radius * std::max(
        static_cast<double>(snapshot.pointCloudWidth) / snapshot.color.cols,
        static_cast<double>(snapshot.pointCloudHeight) / snapshot.color.rows));
    for (int r = 0; r <= cloudRadius; ++r) {
        for (int dy = -r; dy <= r; ++dy) for (int dx = -r; dx <= r; ++dx) {
            if (r > 0 && std::max(std::abs(dx), std::abs(dy)) != r) continue;
            const int px = centerX + dx, py = centerY + dy;
            if (px < 0 || py < 0 || px >= snapshot.pointCloudWidth || py >= snapshot.pointCloudHeight) continue;
            const auto& point = snapshot.pointCloud[static_cast<std::size_t>(py) * snapshot.pointCloudWidth + px];
            if (isValidPoint(point)) return Eigen::Vector3d(point.x, point.y, point.z);
        }
    }
    return std::nullopt;
}

CameraFaceKeypointExtractionService::Result extractSapiensFaceKeypoints(
    Sapiens2PoseEstimator& estimator,
    const CameraFaceKeypointExtractionService::Snapshot& snapshot,
    const std::filesystem::path& outputDirectory,
    double minScore = 0.25,
    int pointSearchRadius = 6) {
    CameraFaceKeypointExtractionService::Result result;
    const cv::Rect2f box(0.0f, 0.0f, static_cast<float>(snapshot.color.cols),
                         static_cast<float>(snapshot.color.rows));
    const auto pose = estimator.infer(snapshot.color, box);
    cv::Mat rendered = snapshot.color.clone();
    double scoreSum = 0.0;
    for (const auto& detected : pose.keypoints) {
        if (detected.index < 70 || detected.score < minScore ||
            !std::isfinite(detected.position.x) || !std::isfinite(detected.position.y)) continue;
        ModelKeypointAnnotationService::Keypoint3D keypoint;
        keypoint.index = detected.index;
        keypoint.name = "sapiens_face_" + std::to_string(detected.index);
        keypoint.score = detected.score;
        keypoint.imageX = cvRound(detected.position.x);
        keypoint.imageY = cvRound(detected.position.y);
        const auto point = sampleSnapshotPoint(
            snapshot, keypoint.imageX, keypoint.imageY, pointSearchRadius);
        keypoint.hasPoint = point.has_value();
        if (point) keypoint.point = *point;
        result.visibleCount++;
        if (point) result.validCount++;
        scoreSum += detected.score;
        result.keypoints.push_back(keypoint);
        cv::circle(rendered, {keypoint.imageX, keypoint.imageY}, 2,
                   point ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
    result.meanScore = result.visibleCount ? scoreSum / result.visibleCount : 0.0;
    result.success = result.validCount >= 3;
    result.message = result.success ? "Sapiens face keypoints extracted" : "Not enough Sapiens face 3D points";
    std::filesystem::create_directories(outputDirectory);
    result.colorImagePath = outputDirectory / "camera_keypoints_color.png";
    result.renderImagePath = outputDirectory / "camera_keypoints_render.jpg";
    result.keypointsJsonPath = outputDirectory / "camera_keypoints.json";
    cv::imwrite(result.colorImagePath.string(), snapshot.color);
    cv::imwrite(result.renderImagePath.string(), rendered);
    std::ofstream json(result.keypointsJsonPath);
    json << "{\n  \"provider\": \"sapiens_pose\",\n  \"validCount\": " << result.validCount
         << ",\n  \"meanScore\": " << result.meanScore << ",\n  \"keypoints\": [\n";
    for (std::size_t i = 0; i < result.keypoints.size(); ++i) {
        const auto& p = result.keypoints[i];
        json << "    {\"index\":" << p.index << ",\"name\":\"" << p.name
             << "\",\"score\":" << p.score << ",\"imageX\":" << p.imageX
             << ",\"imageY\":" << p.imageY << ",\"hasPoint\":" << (p.hasPoint ? "true" : "false");
        if (p.hasPoint) json << ",\"x\":" << p.point.x() << ",\"y\":" << p.point.y() << ",\"z\":" << p.point.z();
        json << "}" << (i + 1 == result.keypoints.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    return result;
}

ModelKeypointAnnotationService::Result annotateSapiensModelKeypoints(
    Sapiens2PoseEstimator& estimator,
    const std::filesystem::path& stlPath,
    const std::filesystem::path& outputDirectory) {
    ModelKeypointAnnotationService::Result result;
    result.stlPath = stlPath;
    StlModelRenderer renderer;
    std::string error;
    if (!renderer.loadStl(stlPath, &error)) { result.message = error; return result; }
    StlModelRenderer::RenderOptions options;
    options.azimuthDeg = 0.0;
    options.elevationDeg = 0.0;
    const auto render = renderer.render(options, &error);
    if (render.bgr.empty()) { result.message = error; return result; }
    result.renderedViewCount = 1;
    const auto pose = estimator.infer(render.bgr,
        cv::Rect2f(0, 0, static_cast<float>(render.bgr.cols), static_cast<float>(render.bgr.rows)));
    cv::Mat visual = render.bgr.clone();
    for (const auto& detected : pose.keypoints) {
        if (detected.index < 70 || detected.score < 0.25f) continue;
        ModelKeypointAnnotationService::Keypoint3D keypoint;
        keypoint.index = detected.index;
        keypoint.name = "sapiens_face_" + std::to_string(detected.index);
        keypoint.score = detected.score;
        keypoint.imageX = cvRound(detected.position.x);
        keypoint.imageY = cvRound(detected.position.y);
        cv::Point sampled;
        const auto world = renderer.worldPointFromPixel(
            render, keypoint.imageX, keypoint.imageY, 6, &sampled);
        keypoint.hasPoint = world.has_value();
        if (world) {
            keypoint.point = Eigen::Vector3d((*world)[0], (*world)[1], (*world)[2]);
            keypoint.imageX = sampled.x; keypoint.imageY = sampled.y;
            result.namedValidCount++;
            cv::circle(visual, sampled, 2, {0, 255, 0}, -1, cv::LINE_AA);
        }
        result.keypoints.push_back(keypoint);
    }
    result.success = result.namedValidCount >= 3;
    result.message = result.success ? "ok" : "Not enough Sapiens model face points";
    std::filesystem::create_directories(outputDirectory);
    result.renderImagePath = outputDirectory / "model_keypoints_render.png";
    cv::imwrite(result.renderImagePath.string(), visual);
    return result;
}

bool writeMatrix(const std::filesystem::path& path, const Eigen::Matrix4d& matrix) {
    std::ofstream output(path, std::ios::trunc);
    if (output) output << std::setprecision(12) << matrix << '\n';
    return static_cast<bool>(output);
}

struct ManualRoiSelection {
    CameraFrame frame;
    cv::Rect roi;
};

std::optional<ManualRoiSelection> selectManualRoi(
    CameraBase& camera, const std::filesystem::path& roiDirectory) {
    std::optional<CameraFrame> selectedFrame;
    for (int attempt = 0; attempt < 60; ++attempt) {
        auto frame = camera.capture();
        if (frame && !frame->color.empty() && !frame->pointCloud.empty()) {
            selectedFrame = std::move(*frame);
            break;
        }
    }
    if (!selectedFrame) return std::nullopt;

    struct SelectionState {
        bool drawing{false};
        cv::Point start;
        cv::Rect roi;
    } state;
    const std::string windowName = "Manual ROI - ENTER/SPACE confirm, ESC cancel";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName,
        [](int event, int x, int y, int, void* data) {
            auto& selection = *static_cast<SelectionState*>(data);
            if (event == cv::EVENT_LBUTTONDOWN) {
                selection.drawing = true;
                selection.start = {x, y};
                selection.roi = {};
            } else if ((event == cv::EVENT_MOUSEMOVE && selection.drawing) ||
                       (event == cv::EVENT_LBUTTONUP && selection.drawing)) {
                const int left = std::min(selection.start.x, x);
                const int top = std::min(selection.start.y, y);
                selection.roi = {left, top, std::abs(x - selection.start.x),
                                 std::abs(y - selection.start.y)};
                if (event == cv::EVENT_LBUTTONUP) selection.drawing = false;
            }
        }, &state);
    bool confirmed = false;
    for (;;) {
        cv::Mat display = selectedFrame->color.clone();
        if (!state.roi.empty())
            cv::rectangle(display, state.roi, cv::Scalar(0, 255, 0), 2);
        cv::imshow(windowName, display);
        const int key = cv::waitKey(16) & 0xff;
        if (key == 27) break;
        if ((key == 13 || key == 32) && !state.roi.empty()) {
            confirmed = true;
            break;
        }
        if (key == 'c' || key == 'C') state.roi = {};
    }
    cv::destroyWindow(windowName);
    const cv::Rect roi = state.roi & cv::Rect(0, 0,
        selectedFrame->color.cols, selectedFrame->color.rows);
    if (!confirmed || roi.empty()) return std::nullopt;

    std::filesystem::create_directories(roiDirectory);
    std::ofstream output(roiDirectory / "manual_roi.txt", std::ios::trunc);
    if (!output) return std::nullopt;
    output << roi.x << ' ' << roi.y << ' ' << roi.width << ' ' << roi.height << '\n';
    cv::imwrite((roiDirectory / "manual_roi_color.png").string(),
                selectedFrame->color(roi).clone());
    if (!selectedFrame->depth.empty() && selectedFrame->depth.size() == selectedFrame->color.size())
        cv::imwrite((roiDirectory / "manual_roi_depth_raw.png").string(),
                    selectedFrame->depth(roi).clone());
    return ManualRoiSelection{std::move(*selectedFrame), roi};
}
}

int main(int argc, char** argv) {
    std::string cameraBackend = "orbbec";
    CameraDeviceSelector cameraSelector;
    bool laserAutoControl = false;
    int laserPower = 25;
    bool listCameras = false;
    bool manualRoiMode = false;
    unsigned workerCount = 0;
    bool workerCountSpecified = false;
    bool cameraBackendSpecified = false;
    bool cameraSnSpecified = false;
    bool cameraIpSpecified = false;
    bool laserAutoSpecified = false;
    bool laserPowerSpecified = false;
    std::filesystem::path runtimeConfig = "C++/config/runtime.yml";
    std::optional<std::string> targetLocatorOverride;
    std::optional<std::string> keypointProviderOverride;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--camera-backend" || arg == "--camera-sn" || arg == "--camera-ip" ||
             arg == "--laser-power" || arg == "--laser-auto" || arg == "--threads" ||
             arg == "--target-locator" || arg == "--keypoint-provider") && i + 1 >= argc) {
            std::cerr << "missing value after " << arg << '\n';
            return EXIT_FAILURE;
        }
        if (arg == "--camera-backend") { cameraBackend = argv[++i]; cameraBackendSpecified = true; }
        else if (arg == "--camera-sn") { cameraSelector.serialNumber = argv[++i]; cameraSnSpecified = true; }
        else if (arg == "--camera-ip") { cameraSelector.ipAddress = argv[++i]; cameraIpSpecified = true; }
        else if (arg == "--laser-power") { laserPower = std::stoi(argv[++i]); laserPowerSpecified = true; }
        else if (arg == "--laser-auto") {
            const std::string value = argv[++i];
            if (value != "on" && value != "off") {
                std::cerr << "--laser-auto must be on or off\n";
                return EXIT_FAILURE;
            }
            laserAutoControl = value == "on";
            laserAutoSpecified = true;
        }
        else if (arg == "--threads") {
            workerCount = static_cast<unsigned>(std::stoul(argv[++i]));
            workerCountSpecified = true;
        }
        else if (arg == "--target-locator") targetLocatorOverride = argv[++i];
        else if (arg == "--keypoint-provider") keypointProviderOverride = argv[++i];
        else if (arg == "--config") {
            if (i + 1 >= argc) { std::cerr << "missing value after --config\n"; return EXIT_FAILURE; }
            runtimeConfig = argv[++i];
        }
        else if (arg == "--list-cameras") listCameras = true;
        else if (arg == "--manual-roi") manualRoiMode = true;
        else if (arg.rfind("--", 0) == 0) {
            std::cerr << "unknown option: " << arg;
            if (arg == "--manaul-roi") std::cerr << "; did you mean --manual-roi?";
            std::cerr << '\n';
            return EXIT_FAILURE;
        }
        else positional.push_back(arg);
    }
    const auto cameraConfigPath = std::filesystem::path("C++") / "config" / "camera.yml";
    if (!cameraBackendSpecified)
        cameraBackend = readYamlScalar(cameraConfigPath, "camera", "backend").value_or("orbbec");
    if (!cameraSnSpecified)
        cameraSelector.serialNumber = readYamlScalar(
            cameraConfigPath, "camera", "serial_number").value_or("");
    if (!cameraIpSpecified)
        cameraSelector.ipAddress = readYamlScalar(
            cameraConfigPath, "camera", "ip_address").value_or("");
    if (!laserAutoSpecified)
        laserAutoControl = readYamlScalar(cameraConfigPath, "camera", "laser_auto_control")
            .value_or("false") == "true";
    if (!laserPowerSpecified)
        laserPower = std::stoi(readYamlScalar(
            cameraConfigPath, "camera", "laser_power").value_or("25"));
    if (cameraBackend != "orbbec" && cameraBackend != "vcamera") {
        std::cerr << "invalid camera backend: " << cameraBackend << '\n';
        return EXIT_FAILURE;
    }
    std::vector<CameraDeviceInfo> discoveredDevices;
    if (cameraBackend == "vcamera") {
#if FACE_HAS_VCAMERA
        discoveredDevices = VcameraDepthAlignedCamera::discoverDevices();
#else
        std::cerr << "vcamera backend is available only in Windows builds; use --camera-backend orbbec\n";
        return EXIT_FAILURE;
#endif
    } else {
        discoveredDevices = OrbbecDepthAlignedCamera::discoverDevices();
    }
    printDevices(cameraBackend, discoveredDevices);
    if (listCameras) return discoveredDevices.empty() ? 2 : EXIT_SUCCESS;

    if (positional.size() > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " [--camera-backend orbbec|vcamera] [--list-cameras]"
                     " [--camera-sn SN] [--camera-ip IP]"
                     " [--laser-auto on|off] [--laser-power VALUE] [--manual-roi] [--threads N] [--config FILE]"
                     " [--target-locator face_detection|sapiens_seg]"
                     " [--keypoint-provider hrnet|sapiens_pose]"
                      " [face_detector.onnx|-] [face_keypoints.onnx|-] [output_dir]"
                      " [min_fitness] [max_inlier_rmse_mm]"
                      " [max_source_rmse_mm] [max_source_p95_mm]\n"
                      "Target STL is fixed to data/head.stl relative to the repository root.\n";
        return EXIT_FAILURE;
    }
    const std::string detectorArgument = positional.size() >= 1 ? positional[0] : "-";
    const std::string keypointArgument = positional.size() >= 2 ? positional[1] : "-";
    const std::string outputArgument = positional.size() >= 3 ? positional[2] :
        readYamlScalar(runtimeConfig, "runtime", "output_directory").value_or("output");
    if (isStlPath(outputArgument) || isPlyPath(outputArgument)) {
        std::cerr << "legacy camera command detected: the third positional argument ('"
                  << outputArgument
                  << "') looks like the removed <target.ply|stl> argument. Remove it; "
                     "the live target is fixed to data/head.stl.\n";
        return EXIT_FAILURE;
    }
    const auto requestedOutput = std::filesystem::absolute(outputArgument).lexically_normal();
    const auto fixedOutput = std::filesystem::absolute(
        readYamlScalar(runtimeConfig, "runtime", "output_directory").value_or("output")).lexically_normal();
    if (requestedOutput != fixedOutput) {
        std::cerr << "camera pipeline output is fixed to " << fixedOutput.string()
                  << "; use the runtime.output_directory value\n";
        return EXIT_FAILURE;
    }

    std::vector<CameraDeviceInfo> matchingDevices;
    for (const auto& device : discoveredDevices) {
        const bool snMatches = cameraSelector.serialNumber.empty() ||
                               cameraSelector.serialNumber == device.serialNumber;
        const bool ipMatches = cameraSelector.ipAddress.empty() ||
                               cameraSelector.ipAddress == device.ipAddress;
        if (snMatches && ipMatches) matchingDevices.push_back(device);
    }
    if (matchingDevices.empty()) {
        std::cerr << "no " << cameraBackend << " camera matches the requested SN/IP\n";
        return 2;
    }
    if (matchingDevices.size() > 1) {
        std::cerr << "multiple " << cameraBackend
                  << " cameras match; use --camera-sn or --camera-ip shown above\n";
        return 2;
    }
    // Pin the only matching device so the session manifest records its real
    // identity even when the operator only supplied the backend type.
    cameraSelector.serialNumber = matchingDevices.front().serialNumber;
    cameraSelector.ipAddress = matchingDevices.front().ipAddress;

    const std::filesystem::path modelStlPath = std::filesystem::path("data") / "head.stl";
    const std::filesystem::path outputRoot = fixedOutput;
    const std::filesystem::path outputDir = createSessionDirectory(outputRoot);
    const auto faceDetectionDir = outputDir / "face_detection";
    const auto faceSegmentationDir = outputDir / "face_segmentation";
    const auto faceKeypointsDir = outputDir / "face_keypoints_detection";
    const auto roiDir = outputDir / "roi";
    const auto cameraDir = outputDir / "camera";
    const auto stlDir = outputDir / "STL";
    const auto logsDir = outputDir / "logs";
    for (const auto& directory : {faceKeypointsDir, roiDir, cameraDir, stlDir, logsDir})
        std::filesystem::create_directories(directory);
    const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    if (!workerCountSpecified) {
        if (manualRoiMode) {
            workerCount = static_cast<unsigned>(std::stoul(readYamlScalar(
                runtimeConfig, "pipeline", "manual_roi_frames").value_or("8")));
        } else {
            const auto configuredFrames = readYamlScalar(runtimeConfig, "pipeline", "detection_frames");
            workerCount = configuredFrames
                ? static_cast<unsigned>(std::stoul(*configuredFrames))
                : std::min(8u, std::max(1u, hardwareThreads > 2 ? hardwareThreads - 2 : 1u));
        }
    }
    if (workerCount == 0 || workerCount > 64) {
        std::cerr << "--threads must be between 1 and 64\n";
        return EXIT_FAILURE;
    }
    const int batchSize = static_cast<int>(workerCount);
    const double minFitness = positional.size() >= 4 ? std::stod(positional[3]) :
        std::stod(readYamlScalar(runtimeConfig, "registration", "min_fitness").value_or("0.90"));
    const double maxInlierRmse = positional.size() >= 5 ? std::stod(positional[4]) :
        std::stod(readYamlScalar(runtimeConfig, "registration", "max_inlier_rmse_mm").value_or("3.0"));
    const double maxSourceRmse = positional.size() >= 6 ? std::stod(positional[5]) :
        std::stod(readYamlScalar(runtimeConfig, "registration", "max_source_to_target_rmse_mm").value_or("6.0"));
    const double maxSourceP95 = positional.size() >= 7 ? std::stod(positional[6]) :
        std::stod(readYamlScalar(runtimeConfig, "registration", "max_source_to_target_p95_mm").value_or("10.0"));
    if (!std::isfinite(minFitness) || minFitness < 0.0 || minFitness > 1.0 ||
        !std::isfinite(maxInlierRmse) || !(maxInlierRmse > 0.0) ||
        !std::isfinite(maxSourceRmse) || !(maxSourceRmse > 0.0) ||
        !std::isfinite(maxSourceP95) || !(maxSourceP95 > 0.0)) {
        std::cerr << "registration quality thresholds are invalid\n";
        return EXIT_FAILURE;
    }
    const double modelUnitScale = 1.0;
    const auto timingPath = logsDir / "registration_timing.txt";
    double targetLocalizationMs = 0.0;
    double modelTransitionMs = 0.0;
    std::optional<Clock::time_point> endToEndStart;
    std::cout << "session output: " << outputDir.string() << '\n';
    {
        std::ofstream sessionInfo(outputDir / "session_info.txt");
        sessionInfo << "camera_backend=" << cameraBackend << '\n'
                    << "camera_serial_number=" << cameraSelector.serialNumber << '\n'
                    << "camera_ip_address=" << cameraSelector.ipAddress << '\n'
                    << "vcamera_laser_auto=" << (laserAutoControl ? "on" : "off") << '\n'
                    << "vcamera_laser_power=" << laserPower << '\n'
                    << "target=" << modelStlPath.string() << '\n'
                    << "min_fitness=" << minFitness << '\n'
                    << "max_inlier_rmse_mm=" << maxInlierRmse << '\n'
                    << "max_source_to_target_rmse_mm=" << maxSourceRmse << '\n'
                    << "max_source_to_target_p95_mm=" << maxSourceP95 << '\n'
                    << "input_mode=" << (manualRoiMode ? "manual_roi_geometry" : "onnx_detection") << '\n';
        sessionInfo << "worker_count=" << workerCount << '\n'
                    << "hardware_threads=" << hardwareThreads << '\n'
                    << "capture_batch_size=" << batchSize << '\n'
                    << "runtime_config=" << runtimeConfig.string() << '\n';
    }

    FaceDetectionService::Options detectorOptions;
    const auto configuredProvider = readYamlScalar(runtimeConfig, "runtime", "onnx_provider");
    detectorOptions.detector.preferCuda = configuredProvider && *configuredProvider == "cuda";
    detectorOptions.draw = false;
    const std::string targetLocator = targetLocatorOverride.value_or(
        readYamlScalar(runtimeConfig, "pipeline", "target_locator").value_or("face_detection"));
    const std::string keypointProvider = keypointProviderOverride.value_or(
        readYamlScalar(runtimeConfig, "pipeline", "keypoint_model").value_or("hrnet"));
    const auto residentSetting = parseYamlBool(readYamlScalar(
        runtimeConfig, "pipeline", "keep_sapiens_models_resident").value_or("false"));
    if (!residentSetting) {
        std::cerr << "pipeline.keep_sapiens_models_resident must be a boolean\n";
        return EXIT_FAILURE;
    }
    const bool keepSapiensModelsResident = *residentSetting;
    if (targetLocator == "sapiens_seg")
        std::filesystem::create_directories(faceSegmentationDir);
    else
        std::filesystem::create_directories(faceDetectionDir);
    if (targetLocator != "face_detection" && targetLocator != "sapiens_seg") {
        std::cerr << "pipeline.target_locator must be face_detection or sapiens_seg\n";
        return EXIT_FAILURE;
    }
    if (keypointProvider != "hrnet" && keypointProvider != "sapiens_pose") {
        std::cerr << "pipeline.keypoint_model must be hrnet or sapiens_pose\n";
        return EXIT_FAILURE;
    }
    const std::string sapiensSegModel = readYamlScalar(runtimeConfig, "runtime", "sapiens_seg_model")
        .value_or("models/face_segmentation/sapiens2_seg/sapiens2_seg_0.4b_fp32.onnx");
    const std::string sapiensPoseModel = readYamlScalar(runtimeConfig, "runtime", "sapiens_pose_model")
        .value_or("models/face_keypoints/sapiens2_pose/sapiens2_pose_0.4b_fp32.onnx");
    const std::string faceDetectorModel = detectorArgument != "-" ? detectorArgument :
        readYamlScalar(runtimeConfig, "runtime", "face_detector_model")
            .value_or("models/face_detection/yolo_face/yolov12n-face.onnx");
    const std::string faceKeypointModel = keypointArgument != "-" ? keypointArgument :
        readYamlScalar(runtimeConfig, "runtime", "face_keypoint_model")
            .value_or("models/face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx");
    auto poseSolver = ModelKeypointAnnotationService::PoseSolver::TripletVote;
    double keypointInlierThresholdMm = 15.0;
    if (const auto configuredThreshold = readYamlScalar(
            runtimeConfig, "camera_keypoints", "keypoint_inlier_threshold_mm"))
        keypointInlierThresholdMm = std::stod(*configuredThreshold);
    const auto configuredSolver = readYamlScalar(runtimeConfig, "camera_keypoints", "pose_solver");
    if (configuredSolver && *configuredSolver == "overdetermined_svd")
        poseSolver = ModelKeypointAnnotationService::PoseSolver::OverdeterminedSvd;
    else if (configuredSolver && *configuredSolver != "triplet_vote") {
        std::cerr << "camera_keypoints.pose_solver must be triplet_vote or overdetermined_svd\n";
        return EXIT_FAILURE;
    }
    {
        std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
        sessionInfo << "face_detection_provider="
                    << (detectorOptions.detector.preferCuda ? "cuda" : "cpu") << '\n'
                    << "target_locator=" << targetLocator << '\n'
                    << "keypoint_provider=" << keypointProvider << '\n'
                    << "keep_sapiens_models_resident="
                    << (keepSapiensModelsResident ? "true" : "false") << '\n'
                    << "parallel_detection_frames=" << batchSize << '\n'
                    << "keypoint_pose_solver="
                    << (poseSolver == ModelKeypointAnnotationService::PoseSolver::TripletVote
                            ? "triplet_vote" : "overdetermined_svd") << '\n';
    }
    if (manualRoiMode) {
        std::cout << "manual ROI mode: interactive ROI replaces target localization; "
                  << keypointProvider << " keypoints and registration remain enabled\n";
    }
    std::unique_ptr<FaceKeypointService> keypointDetector;
    std::unique_ptr<CameraFaceKeypointExtractionService> cameraKeypointExtractor;
    std::vector<std::unique_ptr<FaceDetectionService>> residentFaceDetectors;
    std::unique_ptr<Sapiens2PoseEstimator> sapiensPoseEstimator;
    std::unique_ptr<Sapiens2Segmenter> sapiensSegmenter;
    std::mutex sapiensSegMutex;
    std::optional<ModelKeypointAnnotationService::Result> modelKeypoints;
    std::optional<std::string> modelKeypointSourceSha256;
    const bool keepResidentSeg = !manualRoiMode &&
        targetLocator == "sapiens_seg" && keepSapiensModelsResident;
    const bool keepResidentPose = keypointProvider == "sapiens_pose" &&
        keepSapiensModelsResident;
    if (keepResidentSeg) {
        sapiensSegmenter = std::make_unique<Sapiens2Segmenter>(
            Sapiens2Segmenter::Options{detectorOptions.detector.preferCuda, 0});
        sapiensSegmenter->load(sapiensSegModel);
        std::cout << "resident Sapiens model: Seg loaded before STL preprocessing\n";
    }
    if (keypointProvider == "hrnet") {
        const auto keypointLoadStart = Clock::now();
        keypointDetector = std::make_unique<FaceKeypointService>(
            faceKeypointModel, detectorOptions.detector.preferCuda, 0);
        std::string error;
        if (!keypointDetector->ensureLoaded(&error)) {
            std::cerr << error << '\n';
            return EXIT_FAILURE;
        }
        reportTiming(timingPath, "face_keypoint_model_load", elapsedMs(keypointLoadStart));
        cameraKeypointExtractor =
            std::make_unique<CameraFaceKeypointExtractionService>(*keypointDetector);
        if (isStlPath(modelStlPath)) {
            const auto annotationSourceBefore = HeadSurfaceCache::digestFile(modelStlPath);
            if (!annotationSourceBefore.success) {
                std::cerr << "cannot hash STL before model-keypoint annotation: "
                          << annotationSourceBefore.message << '\n';
                return EXIT_FAILURE;
            }
            ModelKeypointAnnotationService modelAnnotation(*keypointDetector);
            ModelKeypointAnnotationService::Options annotationOptions;
            annotationOptions.modelUnitScale = modelUnitScale;
            annotationOptions.writeVisualizationArtifacts = true;
            annotationOptions.write3DArtifacts = false;
            std::cout << "annotating STL face keypoints from "
                      << annotationOptions.azimuthCandidates.size() * annotationOptions.elevationCandidates.size()
                      << " virtual-camera views\n";
            const auto modelKeypointStart = Clock::now();
            auto annotation = modelAnnotation.annotateStlModelKeypoints(
                modelStlPath, stlDir, annotationOptions, &error);
            reportTiming(timingPath, "stl_virtual_views_keypoints", elapsedMs(modelKeypointStart));
            if (annotation.success) {
                const auto annotationSourceAfter = HeadSurfaceCache::digestFile(modelStlPath);
                if (!annotationSourceAfter.success ||
                    annotationSourceAfter.sha256Hex != annotationSourceBefore.sha256Hex) {
                    std::cerr << "STL changed while model keypoints were being annotated; "
                                 "restart so keypoints and the cached registration target use "
                                 "the same model bytes\n";
                    return EXIT_FAILURE;
                }
                std::cout << "selected STL virtual view: azimuth=" << annotation.selectedAzimuthDeg
                          << " elevation=" << annotation.selectedElevationDeg
                          << " named 3D keypoints=" << annotation.namedValidCount << '\n';
                std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
                sessionInfo << "model_virtual_views=" << annotation.renderedViewCount << '\n'
                            << "model_selected_azimuth_deg=" << annotation.selectedAzimuthDeg << '\n'
                            << "model_selected_elevation_deg=" << annotation.selectedElevationDeg << '\n'
                            << "model_named_keypoints=" << annotation.namedValidCount << '\n';
                modelKeypointSourceSha256 = annotationSourceBefore.sha256Hex;
                modelKeypoints = std::move(annotation);
            } else {
                std::cerr << "STL multi-view keypoint annotation failed; FPFH/RANSAC fallback remains active: "
                          << error << '\n';
            }
        }
    } else if (keypointProvider == "sapiens_pose") {
        const auto loadStart = Clock::now();
        sapiensPoseEstimator = std::make_unique<Sapiens2PoseEstimator>(
            Sapiens2PoseEstimator::Options{detectorOptions.detector.preferCuda, 0});
        sapiensPoseEstimator->load(sapiensPoseModel);
        reportTiming(timingPath, "sapiens_pose_model_load", elapsedMs(loadStart));
        const auto annotationSourceBefore = HeadSurfaceCache::digestFile(modelStlPath);
        const auto modelStart = Clock::now();
        auto annotation = annotateSapiensModelKeypoints(
            *sapiensPoseEstimator, modelStlPath, stlDir);
        reportTiming(timingPath, "stl_sapiens_pose_keypoints", elapsedMs(modelStart));
        if (annotation.success) {
            modelKeypointSourceSha256 = annotationSourceBefore.sha256Hex;
            modelKeypoints = std::move(annotation);
            std::cout << "Sapiens model face keypoints=" << modelKeypoints->namedValidCount
                      << "; pose solver selected from runtime.yml\n";
        } else {
            std::cerr << "Sapiens STL keypoints failed; FPFH/RANSAC fallback remains active: "
                      << annotation.message << '\n';
        }
    }

    RegistrationOptions registrationOptions;
    const int manualRoiGlobalAttempts = std::stoi(readYamlScalar(
        runtimeConfig, "pipeline", "manual_roi_global_attempts").value_or("1"));
    if (manualRoiGlobalAttempts < 1) {
        std::cerr << "pipeline.manual_roi_global_attempts must be positive\n";
        return EXIT_FAILURE;
    }
    registrationOptions.initialFitnessToSkipGlobal = minFitness;
    registrationOptions.initialRmseToSkipGlobalMm = maxInlierRmse;
    registrationOptions.initialSourceToTargetRmseToSkipGlobalMm = maxSourceRmse;
    registrationOptions.initialSourceToTargetP95ToSkipGlobalMm = maxSourceP95;
    std::filesystem::path registrationTargetPath;
    HeadSurfaceCache::SessionLease modelCacheSessionLease;
    {
        const auto modelCloudStart = Clock::now();
        HeadSurfaceCache::Recipe cacheRecipe;
        cacheRecipe.reconstruction.modelUnitScale = modelUnitScale;
        const auto addressResult = HeadSurfaceCache::resolveAddress(
            modelStlPath, outputRoot / "model_cache", cacheRecipe);
        if (!addressResult.success) {
            std::cerr << "cannot resolve head-surface cache: "
                      << addressResult.message << '\n';
            return EXIT_FAILURE;
        }
        const auto& cacheAddress = addressResult.address;
        if (modelKeypointSourceSha256 &&
            *modelKeypointSourceSha256 != cacheAddress.sourceSha256Hex) {
            std::cerr << "STL changed between model-keypoint annotation and cache resolution; "
                         "restart so both stages use the same model bytes\n";
            return EXIT_FAILURE;
        }
        {
            std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
            sessionInfo << "model_stl_sha256=" << cacheAddress.sourceSha256Hex << '\n'
                        << "model_cache_recipe_sha256=" << cacheAddress.recipeSha256Hex << '\n'
                         << "model_cache_key=" << cacheAddress.cacheKeyHex << '\n';
        }

        auto initialLeaseResult = HeadSurfaceCache::acquireSessionLease(cacheAddress);
        reportTiming(timingPath, "head_surface_cloud_cache_lease_acquire",
                     elapsedMs(modelCloudStart));
        if (initialLeaseResult.success) {
            if (!initialLeaseResult.lease.valid() || !initialLeaseResult.lookup.hit() ||
                initialLeaseResult.lookup.surfacePlyPath != cacheAddress.surfacePlyPath) {
                std::cerr << "head-surface cache returned an inconsistent initial session lease\n";
                return EXIT_FAILURE;
            }
            registrationTargetPath = initialLeaseResult.lookup.surfacePlyPath;
            const std::size_t leasedPointCount = initialLeaseResult.lookup.pointCount;
            modelCacheSessionLease = std::move(initialLeaseResult.lease);
            std::cout << "reusing cached head registration surface: "
                      << registrationTargetPath.string() << '\n';
            std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
            sessionInfo << "model_ply_cache_status=reused\n"
                        << "model_ply_cache=" << registrationTargetPath.string() << '\n'
                        << "model_ply_cache_points=" << leasedPointCount << '\n'
                        << "model_ply_cache_lease=held\n";
        } else {
            if (initialLeaseResult.lease.valid() ||
                (initialLeaseResult.lookup.status != HeadSurfaceCache::LookupStatus::Miss &&
                 initialLeaseResult.lookup.status != HeadSurfaceCache::LookupStatus::Invalid)) {
                std::cerr << "cannot acquire initial head-surface cache lease: "
                          << initialLeaseResult.message << '\n';
                return EXIT_FAILURE;
            }
            const auto workspaceResult = HeadSurfaceCache::createWorkspace(
                cacheAddress, modelStlPath);
            if (!workspaceResult.ready()) {
                std::cerr << "cannot create a stable head-surface cache workspace: "
                          << workspaceResult.message << '\n';
                return EXIT_FAILURE;
            }
            const auto& workspace = workspaceResult.workspace;
            const auto reconstructionStart = Clock::now();
            const auto generated = PointCloudRegistration::prepareRegistrationCloudFromStl(
                workspace.sourceSnapshotStlPath.string(), workspace.surfacePlyPath.string(),
                cacheRecipe.reconstruction);
            const double modelReconstructionMs = elapsedMs(reconstructionStart);
            reportTiming(timingPath, "head_surface_cloud_generation_excluded_from_total",
                         modelReconstructionMs);
            for (const auto& timing : generated.stageTimingsMs)
                reportTiming(timingPath, "head_surface_" + timing.first, timing.second);
            if (!generated.success) {
                const auto cleanup = HeadSurfaceCache::cleanupWorkspace(cacheAddress, workspace);
                if (!cleanup.success)
                    std::cerr << "warning: cannot clean failed cache workspace: "
                              << cleanup.message << '\n';
                std::cerr << "cannot prepare head registration surface: "
                          << generated.message << '\n';
                return EXIT_FAILURE;
            }

            const auto published = HeadSurfaceCache::publish(
                cacheAddress, workspace, generated.pointCount);
            if (!published.success()) {
                const auto cleanup = HeadSurfaceCache::cleanupWorkspace(cacheAddress, workspace);
                if (!cleanup.success)
                    std::cerr << "warning: cannot clean unpublished cache workspace: "
                              << cleanup.message << '\n';
                std::cerr << "cannot publish head-surface cache: "
                          << published.message << '\n';
                return EXIT_FAILURE;
            }
            const bool generatedHere =
                published.status == HeadSurfaceCache::PublishStatus::Published;

            const auto finalLeaseStart = Clock::now();
            auto finalLeaseResult = HeadSurfaceCache::acquireSessionLease(cacheAddress);
            reportTiming(timingPath, "head_surface_cloud_cache_lease_reacquire",
                         elapsedMs(finalLeaseStart));
            if (!finalLeaseResult.success || !finalLeaseResult.lease.valid() ||
                !finalLeaseResult.lookup.hit() ||
                finalLeaseResult.lookup.surfacePlyPath != cacheAddress.surfacePlyPath) {
                std::cerr << "cannot acquire validated head-surface cache lease after publication: "
                          << finalLeaseResult.message << '\n';
                return EXIT_FAILURE;
            }
            registrationTargetPath = finalLeaseResult.lookup.surfacePlyPath;
            const std::size_t leasedPointCount = finalLeaseResult.lookup.pointCount;
            modelCacheSessionLease = std::move(finalLeaseResult.lease);
            std::cout << (generatedHere ? "generated and cached" : "reused concurrently generated")
                      << " head registration surface: " << registrationTargetPath.string()
                      << " (" << leasedPointCount << " points)\n";
            std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
            sessionInfo << "model_ply_cache_status="
                        << (generatedHere ? "generated" : "reused_after_concurrent_generation") << '\n'
                        << "model_ply_cache=" << registrationTargetPath.string() << '\n'
                        << "model_ply_cache_points=" << leasedPointCount << '\n'
                        << "model_ply_cache_lease=held\n"
                        << "generated_model_points=" << generated.pointCount << '\n'
                        << "generated_model_unit_scale=" << modelUnitScale << '\n';
        }
    }
    if (!modelCacheSessionLease.valid()) {
        std::cerr << "head-surface cache lease was lost before camera startup\n";
        return EXIT_FAILURE;
    }
    registrationOptions.targetUnitScale = 1.0;
    auto registrationTargetCloud = open3d::io::CreatePointCloudFromFile(
        registrationTargetPath.string());
    if (!registrationTargetCloud || registrationTargetCloud->IsEmpty()) {
        std::cerr << "cannot load cached STL surface into memory\n";
        return EXIT_FAILURE;
    }

    // STL surface reconstruction/cache loading and model-side keypoint
    // annotation are one startup-preprocessing phase. Only after every STL
    // artifact is ready do we release the keypoint provider. None of this is
    // included in localization-to-registration timing.
    if (!keepResidentPose)
        sapiensPoseEstimator.reset();
    if (!manualRoiMode && targetLocator == "sapiens_seg" && !keepResidentSeg) {
        sapiensSegmenter = std::make_unique<Sapiens2Segmenter>(
            Sapiens2Segmenter::Options{detectorOptions.detector.preferCuda, 0});
        sapiensSegmenter->load(sapiensSegModel);
    }
    std::cout << "STL artifacts complete: resident models="
              << (targetLocator == "face_detection" ? "face_detection" :
                  keepResidentSeg ? "sapiens_seg" : "none")
              << ','
              << (keypointProvider == "hrnet" ? "hrnet" :
                  keepResidentPose ? "sapiens_pose" : "none") << '\n';

    CameraBase::Options cameraOptions;
    cameraOptions.frameKind = FrameKind::ColorDepthPointCloud;
    if (const auto configuredWidth = readYamlScalar(cameraConfigPath, "camera", "color_width"))
        cameraOptions.width = std::stoi(*configuredWidth);
    if (const auto configuredHeight = readYamlScalar(cameraConfigPath, "camera", "color_height"))
        cameraOptions.height = std::stoi(*configuredHeight);
    if (const auto configuredFps = readYamlScalar(cameraConfigPath, "camera", "fps"))
        cameraOptions.fps = std::stod(*configuredFps);
    std::unique_ptr<CameraBase> camera;
    if (cameraBackend == "vcamera") {
#if FACE_HAS_VCAMERA
        cameraOptions.width = 1280;
        cameraOptions.height = 960;
        VcameraDepthAlignedCamera::CaptureSettings vcameraSettings;
        vcameraSettings.laserAutoControl = laserAutoControl;
        vcameraSettings.laserPower = laserPower;
        camera = std::make_unique<VcameraDepthAlignedCamera>("registration-vcamera", cameraOptions,
                                                             cameraSelector, vcameraSettings);
#endif
    } else {
        camera = std::make_unique<OrbbecDepthAlignedCamera>("registration-orbbec", cameraOptions, cameraSelector);
    }
    std::cout << "camera backend: " << cameraBackend << '\n';
    const auto cameraOpenStart = Clock::now();
    if (!camera->open()) {
        std::cerr << "cannot open " << cameraBackend << " depth camera\n";
        return EXIT_FAILURE;
    }
    reportTiming(timingPath, "camera_open", elapsedMs(cameraOpenStart));

    FacePointCloudCropService::Options cropOptions;
    cropOptions.enabled = true;
    cropOptions.roiMargin = 10;
    cropOptions.stride = 1;
    cropOptions.maxPoints = 100000;
    cropOptions.filterOutliers = true;
    cropOptions.keepLargestComponent = true;
    // Tighter outlier rejection: keep the facial shell and drop fringe
    // points (nose-side, eye-socket edges, contour) whose nearest distance
    // to the 5 mm-voxelized target never drops below the ICP threshold.
    cropOptions.outlierMaxZDeviationMm = 80.0f;
    cropOptions.outlierMaxXYRadiusMm = 120.0f;
    FacePointCloudCropService cropper(cropOptions);
    FaceCandidateSelectionPolicy::Options faceSelectionOptions;
    const auto configureDouble = [&](const std::string& section, const std::string& key, double& value) {
        if (const auto configured = readYamlScalar(runtimeConfig, section, key)) value = std::stod(*configured);
    };
    configureDouble("face_selection", "min_detection_score", faceSelectionOptions.minDetectionScore);
    if (const auto configured = readYamlScalar(runtimeConfig, "face_selection", "min_point_count"))
        faceSelectionOptions.minPointCount = static_cast<std::size_t>(std::stoull(*configured));
    configureDouble("face_selection", "min_point_density", faceSelectionOptions.minPointDensity);
    configureDouble("face_selection", "min_depth_mm", faceSelectionOptions.minDepthMm);
    configureDouble("face_selection", "max_depth_mm", faceSelectionOptions.maxDepthMm);
    configureDouble("face_selection", "min_planar_span_mm", faceSelectionOptions.minPlanarSpanMm);
    configureDouble("face_selection", "max_planar_span_mm", faceSelectionOptions.maxPlanarSpanMm);
    configureDouble("face_selection", "min_depth_relief_mm", faceSelectionOptions.minDepthReliefMm);
    configureDouble("face_selection", "max_depth_relief_mm", faceSelectionOptions.maxDepthReliefMm);
    configureDouble("face_selection", "min_depth_mad_mm", faceSelectionOptions.minDepthMadMm);
    configureDouble("face_selection", "max_depth_mad_mm", faceSelectionOptions.maxDepthMadMm);
    if (const auto configured = readYamlScalar(runtimeConfig, "camera_keypoints", "pose_min_inlier_keypoints"))
        faceSelectionOptions.minKeypointPoseInliers = std::stoi(*configured);
    configureDouble("camera_keypoints", "pose_max_rmse_mm", faceSelectionOptions.maxKeypointPoseRmseMm);
    configureDouble("camera_keypoints", "pose_min_mean_score", faceSelectionOptions.minKeypointMeanScore);
    {
        std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
        sessionInfo << "face_selection_policy=physical_gate_then_global_detection_score_top1\n"
                    << "face_selection_min_detection_score=" << faceSelectionOptions.minDetectionScore << '\n'
                    << "face_selection_min_points=" << faceSelectionOptions.minPointCount << '\n'
                    << "face_selection_min_density=" << faceSelectionOptions.minPointDensity << '\n'
                    << "face_selection_depth_min_mm=" << faceSelectionOptions.minDepthMm << '\n'
                    << "face_selection_depth_max_mm=" << faceSelectionOptions.maxDepthMm << '\n'
                    << "face_selection_preferred_depth_mm=" << faceSelectionOptions.preferredDepthMm << '\n'
                    << "face_selection_depth_preference_sigma_mm=" << faceSelectionOptions.depthPreferenceSigmaMm << '\n'
                    << "face_selection_planar_span_min_mm=" << faceSelectionOptions.minPlanarSpanMm << '\n'
                    << "face_selection_planar_span_max_mm=" << faceSelectionOptions.maxPlanarSpanMm << '\n'
                    << "face_selection_depth_relief_min_mm=" << faceSelectionOptions.minDepthReliefMm << '\n'
                    << "face_selection_depth_relief_max_mm=" << faceSelectionOptions.maxDepthReliefMm << '\n'
                    << "face_selection_depth_mad_min_mm=" << faceSelectionOptions.minDepthMadMm << '\n'
                    << "face_selection_depth_mad_max_mm=" << faceSelectionOptions.maxDepthMadMm << '\n'
                    << "keypoint_pose_min_inliers="
                    << faceSelectionOptions.minKeypointPoseInliers << '\n'
                    << "keypoint_pose_max_rmse_mm="
                    << faceSelectionOptions.maxKeypointPoseRmseMm << '\n'
                    << "keypoint_pose_min_mean_score="
                    << faceSelectionOptions.minKeypointMeanScore << '\n';
    }

    struct FrameResult {
        int frameIndex{-1};
        std::uint64_t sequence{0};
        bool candidateFound{false};
        bool passed{false};
        std::string message;
        cv::Rect bbox;
        cv::Mat semanticMask;
        double depthMm{0.0};
        std::vector<PointXYZ> sourcePoints;
        std::shared_ptr<open3d::geometry::PointCloud> sourceCloud;
        std::optional<CameraFaceKeypointExtractionService::Result> keypoints;
        double keypointDetectionMs{0.0};
        double svdVotingMs{0.0};
        double icpIterationsMs{0.0};
        double candidatePreparationMs{0.0};
        double pointCloudPreparationMs{0.0};
        bool keypointInitializationAccepted{false};
        RegistrationResult registration;
    };

    std::optional<cv::Rect> manualRoi;
    if (manualRoiMode) {
        const auto selection = selectManualRoi(*camera, roiDir);
        if (!selection) {
            std::cerr << "manual ROI selection was cancelled or no valid frame was captured\n";
            camera->close();
            return EXIT_FAILURE;
        }
        manualRoi = selection->roi;
        // User interaction is deliberately excluded. Timing starts only
        // after the operator confirms the ROI.
        targetLocalizationMs = 0.0;
        endToEndStart = Clock::now();
        std::cout << "manual ROI saved: " << (roiDir / "manual_roi.txt").string()
                  << " -> " << manualRoi->x << ',' << manualRoi->y << ','
                  << manualRoi->width << ',' << manualRoi->height << '\n';
    }

    std::vector<CameraFrame> frames;
    frames.reserve(batchSize);
    const auto captureBatchStart = Clock::now();
    for (int attempt = 0;
         attempt < std::max(30, batchSize * 10) &&
         frames.size() < static_cast<std::size_t>(batchSize);
         ++attempt) {
        auto frame = camera->capture();
        if (!frame || frame->color.empty() || frame->pointCloud.empty()) continue;
        frames.push_back(std::move(*frame));
    }
    const double captureBatchMs = manualRoiMode ? elapsedMs(captureBatchStart) : 0.0;
    camera->close();
    const std::size_t expectedFrames = static_cast<std::size_t>(batchSize);
    if (frames.size() != expectedFrames) {
        std::cerr << "captured " << frames.size() << " of " << expectedFrames << " frames\n";
        return 2;
    }

    // Load the frame-local YOLO CUDA Sessions immediately before the timed online
    // stage. No camera wait or branch-specific GPU work occurs between this
    // common deployment step and the first real detection; no synthetic
    // inference is executed.
    if (!manualRoiMode && targetLocator == "face_detection") {
        residentFaceDetectors.reserve(static_cast<std::size_t>(batchSize));
        for (int i = 0; i < batchSize; ++i) {
            auto detector = std::make_unique<FaceDetectionService>(detectorOptions);
            if (!detector->loadOnnx(faceDetectorModel)) {
                std::cerr << "cannot preload resident face detector " << i << '\n';
                return EXIT_FAILURE;
            }
            if (detectorOptions.detector.preferCuda && !detector->isUsingCuda()) {
                std::cerr << "face detector " << i
                          << " did not attach to the CUDA execution provider\n";
                return EXIT_FAILURE;
            }
            residentFaceDetectors.push_back(std::move(detector));
        }
        std::cout << "online model deployment complete: "
                  << residentFaceDetectors.size()
                  << " Face Detection CUDA session(s)\n";
    }

    struct GlobalDetectionCandidate {
        int frameIndex{-1};
        PreparedFaceCandidate candidate;
    };
    std::optional<GlobalDetectionCandidate> globalTopDetection;
    if (!manualRoiMode) {
        std::vector<double> faceDetectorInferenceMs(frames.size(), 0.0);
        std::promise<void> detectionStartPromise;
        const std::shared_future<void> detectionStartSignal =
            detectionStartPromise.get_future().share();
        std::vector<std::promise<void>> detectionWorkerReady(frames.size());
        std::vector<std::future<void>> detectionWorkerReadyFutures;
        detectionWorkerReadyFutures.reserve(frames.size());
        for (auto& ready : detectionWorkerReady)
            detectionWorkerReadyFutures.push_back(ready.get_future());
        std::vector<std::future<std::vector<GlobalDetectionCandidate>>> detectionFutures;
        detectionFutures.reserve(frames.size());
        for (int frameIndex = 0; frameIndex < static_cast<int>(frames.size()); ++frameIndex) {
            detectionFutures.push_back(std::async(std::launch::async, [&, frameIndex]() {
                detectionWorkerReady[static_cast<std::size_t>(frameIndex)].set_value();
                detectionStartSignal.wait();
                std::vector<GlobalDetectionCandidate> accepted;
                FacePointCloudCropService localCropper(cropOptions);
                struct LocatedDetection {
                    FaceDetector::Detection detection;
                    cv::Mat semanticMask;
                    std::size_t semanticPixelArea{0};
                };
                std::vector<LocatedDetection> detections;
                if (targetLocator == "face_detection") {
                    cv::Mat color = frames[static_cast<std::size_t>(frameIndex)].color.clone();
                    const auto detected = residentFaceDetectors[
                        static_cast<std::size_t>(frameIndex)]->process(
                        color, frames[static_cast<std::size_t>(frameIndex)].depth,
                        &frames[static_cast<std::size_t>(frameIndex)].pointCloud,
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudWidth,
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudHeight);
                    if (!detected) return accepted;
                    faceDetectorInferenceMs[static_cast<std::size_t>(frameIndex)] =
                        detected->inferMs;
                    detections.reserve(detected->dets.size());
                    for (const auto& detection : detected->dets)
                        detections.push_back({detection, {}, 0});
                } else {
                    Sapiens2Segmenter::Result segmentation;
                    {
                        // A single CUDA session is shared to avoid loading one
                        // multi-GB Sapiens model per frame.
                        std::lock_guard<std::mutex> lock(sapiensSegMutex);
                        segmentation = sapiensSegmenter->infer(
                            frames[static_cast<std::size_t>(frameIndex)].color);
                    }
                    // Persist segmentation diagnostics before candidate gates
                    // are evaluated.  This is intentionally done for every
                    // frame, including frames that produce no accepted
                    // candidate, so a gate failure remains debuggable.
                    {
                        const auto& segFrame = frames[static_cast<std::size_t>(frameIndex)];
                        std::filesystem::create_directories(faceSegmentationDir);
                        const std::string stem = "frame_" + std::to_string(frameIndex);
                        if (!segmentation.coloredLabels.empty() &&
                            segmentation.coloredLabels.size() == segFrame.color.size()) {
                            cv::Mat overlay;
                            cv::addWeighted(segFrame.color, 0.62,
                                            segmentation.coloredLabels, 0.38, 0.0, overlay);
                            cv::imwrite((faceSegmentationDir / (stem + "_overlay.png")).string(),
                                        overlay);
                        }
                    }
                    cv::Mat groupingMask;
                    const cv::Mat groupingKernel = cv::getStructuringElement(
                        cv::MORPH_ELLIPSE, cv::Size(25, 25));
                    cv::morphologyEx(segmentation.faceMask, groupingMask,
                                     cv::MORPH_CLOSE, groupingKernel);
                    cv::Mat labels, stats, centroids;
                    const int count = cv::connectedComponentsWithStats(
                        groupingMask, labels, stats, centroids, 8, CV_32S);
                    for (int component = 1; component < count; ++component) {
                        if (stats.at<int>(component, cv::CC_STAT_AREA) < 100) continue;
                        FaceDetector::Detection detection;
                        cv::Rect box(
                            stats.at<int>(component, cv::CC_STAT_LEFT),
                            stats.at<int>(component, cv::CC_STAT_TOP),
                            stats.at<int>(component, cv::CC_STAT_WIDTH),
                            stats.at<int>(component, cv::CC_STAT_HEIGHT));
                        const int expandX = std::max(8, box.width / 8);
                        const int expandY = std::max(8, box.height / 8);
                        detection.bbox = cv::Rect(
                            box.x - expandX, box.y - expandY,
                            box.width + 2 * expandX, box.height + 2 * expandY) &
                            cv::Rect(0, 0,
                                frames[static_cast<std::size_t>(frameIndex)].color.cols,
                                frames[static_cast<std::size_t>(frameIndex)].color.rows);
                        detection.score = 1.0f;
                        detection.selectionScore = 1.0f;
                        // Preserve semantic component area for the density
                        // gate; bbox area includes deliberately expanded
                        // context and must not be treated as mask area.
                        cv::Mat topology = labels == component;
                        cv::Mat visibleComponentMask;
                        cv::bitwise_and(segmentation.faceMask, topology,
                                        visibleComponentMask);
                        const auto visibleArea = static_cast<std::size_t>(
                            cv::countNonZero(visibleComponentMask));
                        if (visibleArea < 100) continue;
                        detections.push_back({detection,
                            std::move(visibleComponentMask), visibleArea});
                    }
                }
                for (const auto& located : detections) {
                    const auto& detection = located.detection;
                    const double score = validatedDetectionScore(detection);
                    if (score < faceSelectionOptions.minDetectionScore || detection.bbox.empty()) continue;
                    PreparedFaceCandidate candidate;
                    candidate.detection = detection;
                    candidate.semanticMask = located.semanticMask;
                    candidate.semanticPixelArea = located.semanticPixelArea;
                    candidate.cropBbox = targetLocator == "sapiens_seg"
                        ? detection.bbox : shrinkFaceBoxForCrop(detection.bbox);
                    candidate.points = localCropper.cropFacePointCloud(
                        frames[static_cast<std::size_t>(frameIndex)],
                        frames[static_cast<std::size_t>(frameIndex)].color.size(),
                        candidate.cropBbox);
                    if (!candidate.semanticMask.empty()) {
                        candidate.semanticPixelArea = intersectPointCloudWithMask(
                            candidate.points,
                            frames[static_cast<std::size_t>(frameIndex)],
                            frames[static_cast<std::size_t>(frameIndex)].color.size(),
                            candidate.cropBbox, candidate.semanticMask, localCropper);
                    }
                    const auto roi = localCropper.faceCropRoi(
                        frames[static_cast<std::size_t>(frameIndex)].color.size(),
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudWidth,
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudHeight,
                        candidate.cropBbox);
                    const std::size_t pixels = targetLocator == "sapiens_seg"
                        ? candidate.semanticPixelArea
                        : roi
                        ? static_cast<std::size_t>(std::max(0, roi->pointCloudRoi.area())) : 0;
                    auto evaluationOptions = faceSelectionOptions;
                    if (!candidate.semanticMask.empty()) {
                        // A semantic face mask is much smaller than a
                        // rectangular detection ROI, especially after depth
                        // stride/downsampling. Keep the physical geometry
                        // gates, but use a viable minimum sample count.
                        evaluationOptions.minPointCount = std::min<std::size_t>(
                            evaluationOptions.minPointCount, 150);
                        // The semantic mask intentionally excludes hair and
                        // separately labelled occluders, so its shorter axis is
                        // smaller than a detector rectangle.
                        evaluationOptions.minPlanarSpanMm = std::min(
                            evaluationOptions.minPlanarSpanMm, 20.0);
                    }
                    candidate.metrics = FaceCandidateSelectionPolicy::evaluate(
                        score, candidate.points, pixels, evaluationOptions);
                    if (targetLocator == "sapiens_seg") {
                        std::cout << "Sapiens Seg candidate: frame=" << frameIndex
                                  << " bbox=" << candidate.cropBbox.x << ',' << candidate.cropBbox.y
                                  << ',' << candidate.cropBbox.width << ',' << candidate.cropBbox.height
                                  << " points=" << candidate.points.size()
                                  << " depth_mm=" << candidate.metrics.medianDepthMm
                                  << " span_mm=" << candidate.metrics.xSpanMm << 'x'
                                  << candidate.metrics.ySpanMm
                                  << " relief_mm=" << candidate.metrics.depthReliefMm
                                  << " mad_mm=" << candidate.metrics.depthMadMm
                                  << " accepted=" << (candidate.metrics.accepted ? "yes" : "no")
                                  << " reason=" << candidate.metrics.rejectionReason << '\n';
                    }
                    if (candidate.metrics.accepted)
                        accepted.push_back({frameIndex, std::move(candidate)});
                }
                return accepted;
            }));
        }
        for (auto& ready : detectionWorkerReadyFutures) ready.wait();
        // All workers are blocked at the same start gate, so async
        // thread-launch jitter is excluded from localization timing.
        const auto detectionBatchStart = Clock::now();
        endToEndStart = detectionBatchStart;
        detectionStartPromise.set_value();
        std::vector<GlobalDetectionCandidate> allCandidates;
        for (auto& future : detectionFutures) {
            auto candidates = future.get();
            for (auto& candidate : candidates) allCandidates.push_back(std::move(candidate));
        }
        std::sort(allCandidates.begin(), allCandidates.end(), [](const auto& left, const auto& right) {
            if (left.candidate.metrics.detectionScore != right.candidate.metrics.detectionScore)
                return left.candidate.metrics.detectionScore > right.candidate.metrics.detectionScore;
            if (left.candidate.metrics.pointDensity != right.candidate.metrics.pointDensity)
                return left.candidate.metrics.pointDensity > right.candidate.metrics.pointDensity;
            if (left.candidate.metrics.pointCount != right.candidate.metrics.pointCount)
                return left.candidate.metrics.pointCount > right.candidate.metrics.pointCount;
            return left.frameIndex < right.frameIndex;
        });
        if (allCandidates.empty()) {
            std::cerr << targetLocator
                      << " produced no candidate passing physical gates\n";
            return 3;
        }
        globalTopDetection = std::move(allCandidates.front());
        targetLocalizationMs = targetLocator == "face_detection"
            ? *std::max_element(faceDetectorInferenceMs.begin(),
                                faceDetectorInferenceMs.end())
            : elapsedMs(detectionBatchStart);
        std::cout << "global target Top-1 (" << targetLocator << "): frame=" << globalTopDetection->frameIndex
                  << " score=" << globalTopDetection->candidate.metrics.detectionScore
                  << " depth_mm=" << globalTopDetection->candidate.metrics.medianDepthMm
                  << " accepted_candidates=" << allCandidates.size() << '\n';
    }

    if (!manualRoiMode && targetLocator == "sapiens_seg" && !keepResidentSeg) {
        // The selected candidate owns a cloned semantic mask, so the Seg
        // session is no longer needed. Free it before camera Pose inference.
        sapiensSegmenter.reset();
    }
    if (keypointProvider == "sapiens_pose" && !sapiensPoseEstimator) {
            const auto poseLoadStart = Clock::now();
            sapiensPoseEstimator = std::make_unique<Sapiens2PoseEstimator>(
                Sapiens2PoseEstimator::Options{detectorOptions.detector.preferCuda, 0});
            sapiensPoseEstimator->load(sapiensPoseModel);
            modelTransitionMs += elapsedMs(poseLoadStart);
            std::cout << "camera keypoint stage: Sapiens Pose loaded\n";
    } else if (keypointProvider == "hrnet" && !cameraKeypointExtractor) {
        const auto keypointLoadStart = Clock::now();
        keypointDetector = std::make_unique<FaceKeypointService>(
            faceKeypointModel, detectorOptions.detector.preferCuda, 0);
        std::string error;
        if (!keypointDetector->ensureLoaded(&error)) {
            std::cerr << error << '\n';
            return EXIT_FAILURE;
        }
        cameraKeypointExtractor =
            std::make_unique<CameraFaceKeypointExtractionService>(*keypointDetector);
        modelTransitionMs += elapsedMs(keypointLoadStart);
        std::cout << "camera keypoint stage: HRNet loaded\n";
    }

    const auto processFrame = [&](int frameIndex) -> FrameResult {
        FrameResult output;
        output.frameIndex = frameIndex;
        const CameraFrame& frame = frames[static_cast<std::size_t>(frameIndex)];
        output.sequence = frame.meta.sequence;
        try {
            const auto candidatePreparationStart = Clock::now();
            FacePointCloudCropService localCropper(cropOptions);
            std::vector<PreparedFaceCandidate> candidates;

            const auto addCandidate = [&](const FaceDetector::Detection& detection,
                                          const cv::Rect& cropBox,
                                          double score,
                                          bool manual) {
                PreparedFaceCandidate candidate;
                candidate.detection = detection;
                candidate.cropBbox = cropBox;
                candidate.points = localCropper.cropFacePointCloud(
                    frame, frame.color.size(), cropBox);
                const auto roi = localCropper.faceCropRoi(
                    frame.color.size(), frame.pointCloudWidth,
                    frame.pointCloudHeight, cropBox);
                const std::size_t pixels = roi
                    ? static_cast<std::size_t>(std::max(0, roi->pointCloudRoi.area())) : 0;
                candidate.metrics = FaceCandidateSelectionPolicy::evaluate(
                    score, candidate.points, pixels, faceSelectionOptions);
                if (manual && !candidate.metrics.accepted && !candidate.points.empty()) {
                    candidate.metrics.accepted = true;
                    candidate.metrics.rejectionReason = "manual_roi_override";
                }
                if (candidate.metrics.accepted) candidates.push_back(std::move(candidate));
            };

            if (manualRoiMode) {
                const cv::Rect bbox = *manualRoi &
                    cv::Rect(0, 0, frame.color.cols, frame.color.rows);
                if (!bbox.empty()) {
                    FaceDetector::Detection detection;
                    detection.bbox = bbox;
                    detection.score = 1.0f;
                    detection.selectionScore = 1.0f;
                    addCandidate(detection, bbox, 1.0, true);
                }
            } else if (globalTopDetection && globalTopDetection->frameIndex == frameIndex) {
                candidates.push_back(globalTopDetection->candidate);
            }

            if (candidates.empty()) {
                output.message = "no candidate passed candidate gates";
                return output;
            }
            output.candidateFound = true;
            std::vector<FaceCandidateSelectionPolicy::Metrics> metrics;
            for (const auto& candidate : candidates) metrics.push_back(candidate.metrics);
            const auto order = FaceCandidateSelectionPolicy::rankNearestFirst(
                metrics, faceSelectionOptions);
            output.candidatePreparationMs = elapsedMs(candidatePreparationStart);

            const bool useKeypoints = cameraKeypointExtractor || sapiensPoseEstimator;

            bool hasResult = false;
            for (const std::size_t index : order) {
                auto& candidate = candidates[index];
                const auto pointCloudPreparationStart = Clock::now();
                open3d::geometry::PointCloud raw;
                for (const auto& point : candidate.points)
                    raw.points_.emplace_back(point.x, point.y, point.z);
                auto source = raw.VoxelDownSample(2.5);
                if (!source || source->IsEmpty()) continue;
                output.pointCloudPreparationMs += elapsedMs(pointCloudPreparationStart);

                std::optional<Eigen::Matrix4d> initial;
                double candidateKeypointMs = 0.0;
                double candidateSvdVotingMs = 0.0;
                if (useKeypoints) {
                    const auto keypointStart = Clock::now();
                    const auto snapshot = snapshotForCandidate(
                        frame, candidate, localCropper);
                    if (snapshot) {
                        const auto keypointOutputDirectory = manualRoiMode
                            ? faceKeypointsDir / ("frame_" + std::to_string(frameIndex))
                            : faceKeypointsDir;
                        CameraFaceKeypointExtractionService::Result extracted;
                        if (keypointProvider == "hrnet") {
                            CameraFaceKeypointExtractionService::Options options;
                            // A semantic mask makes visibility a pixel-level
                            // decision. Do not let neighbourhood search cross
                            // from an occluder into an adjacent face pixel.
                            if (!candidate.semanticMask.empty())
                                options.searchRadiusPx = 0;
                            options.writeVisualizationArtifacts = true;
                            options.write3DArtifacts = true;
                            std::string error;
                            extracted = cameraKeypointExtractor->extractFromSnapshot(
                                *snapshot, keypointOutputDirectory, options, &error);
                        } else {
                            extracted = extractSapiensFaceKeypoints(
                                *sapiensPoseEstimator, *snapshot, keypointOutputDirectory,
                                0.25, candidate.semanticMask.empty() ? 6 : 0);
                        }
                        candidateKeypointMs = elapsedMs(keypointStart);
                        output.keypoints = extracted;
                        const auto validRegistrationKeypoints = std::count_if(
                            extracted.keypoints.begin(), extracted.keypoints.end(),
                            [](const auto& keypoint) {
                                return ModelKeypointAnnotationService::isRegistrationKeypoint(
                                           keypoint.name) &&
                                       keypoint.hasPoint;
                            });
                        if (extracted.keypoints.empty()) {
                            std::cout << "SVD keypoint warning: frame=" << frameIndex
                                      << " no keypoints detected\n";
                        } else if (validRegistrationKeypoints == 0) {
                            std::cout << "SVD keypoint warning: frame=" << frameIndex
                                      << " keypoints detected, but none has a valid 3D point "
                                         "inside the masked point cloud\n";
                        }
                        if (extracted.success && modelKeypoints) {
                            const auto svdVotingStart = Clock::now();
                            const auto pose =
                                ModelKeypointAnnotationService::estimateCameraToModel(
                                    extracted.keypoints, modelKeypoints->keypoints,
                                    keypointInlierThresholdMm,
                                    poseSolver);
                            if (!pose.success) {
                                std::cout << "SVD keypoint warning: frame=" << frameIndex
                                          << " pose estimation failed: " << pose.message << '\n';
                            }
                            candidateSvdVotingMs = elapsedMs(svdVotingStart);
                            if (FaceCandidateSelectionPolicy::isReliableKeypointPose(
                                    pose.success, pose.inlierCount, pose.rmseMm,
                                    extracted.meanScore, faceSelectionOptions))
                                initial = pose.cameraToModel;
                        }
                    }
                    if (candidateKeypointMs == 0.0)
                        candidateKeypointMs = elapsedMs(keypointStart);
                }

                RegistrationOptions options = registrationOptions;
                if (manualRoiMode)
                    options.globalAttempts = manualRoiGlobalAttempts;
                options.randomSeed += frameIndex * 101;
                options.reseedGlobalRandomEngine = false;
                PointCloudRegistration registrationForFrame(options);
                const auto registered = registrationForFrame.alignClouds(
                    *source, *registrationTargetCloud, {}, initial);
                double candidateIcpMs = 0.0;
                for (const auto& stage : registered.stageTimingsMs) {
                    if (stage.first == "keypoint_initial_multiscale_icp" ||
                        stage.first.rfind("global_icp_attempt_", 0) == 0)
                        candidateIcpMs += stage.second;
                }
                output.keypointDetectionMs += candidateKeypointMs;
                output.svdVotingMs += candidateSvdVotingMs;
                output.icpIterationsMs += candidateIcpMs;
                output.keypointInitializationAccepted =
                    output.keypointInitializationAccepted || initial.has_value();
                if (!registered.success) continue;
                const bool passed =
                    registered.fitness >= minFitness &&
                    registered.inlierRmseMm <= maxInlierRmse &&
                    registered.sourceToTargetRmseMm <= maxSourceRmse &&
                    registered.sourceToTargetP95Mm <= maxSourceP95;
                const bool better =
                    !hasResult || (passed && !output.passed) ||
                    (passed == output.passed &&
                     std::tie(registered.sourceToTargetRmseMm,
                              registered.sourceToTargetP95Mm) <
                     std::tie(output.registration.sourceToTargetRmseMm,
                              output.registration.sourceToTargetP95Mm));
                if (!better) continue;
                hasResult = true;
                output.passed = passed;
                output.bbox = candidate.detection.bbox;
                output.semanticMask = candidate.semanticMask.clone();
                output.depthMm = candidate.metrics.medianDepthMm;
                output.sourcePoints = candidate.points;
                output.sourceCloud = std::move(source);
                output.registration = registered;
            }
            output.message = hasResult ? "ok" : "registration failed";
        } catch (const std::exception& exception) {
            output.message = exception.what();
        }
        return output;
    };

    std::vector<FrameResult> results;
    open3d::utility::random::Seed(registrationOptions.randomSeed);
    const auto registrationBatchStart = Clock::now();
    if (manualRoiMode) {
        std::cout << "manual ROI: captured batch=" << frames.size()
                  << "; registering all frames concurrently; camera_fps="
                  << cameraOptions.fps
                  << " ransac_icp_attempts_per_frame="
                  << manualRoiGlobalAttempts << '\n';
        std::vector<std::future<FrameResult>> futures;
        futures.reserve(frames.size());
        for (int i = 0; i < static_cast<int>(frames.size()); ++i)
            futures.push_back(std::async(std::launch::async, processFrame, i));
        results.reserve(futures.size());
        for (auto& future : futures) results.push_back(future.get());
    } else {
        results.push_back(processFrame(globalTopDetection->frameIndex));
    }
    const double registrationBatchWallMs = elapsedMs(registrationBatchStart);
    const double endToEndTotalMs = endToEndStart ? elapsedMs(*endToEndStart) : 0.0;
    double totalKeypointDetectionMs = 0.0;
    double totalSvdVotingMs = 0.0;
    double totalIcpIterationsMs = 0.0;
    double candidatePreparationBatchMaxMs = 0.0;
    double pointCloudPreparationBatchMaxMs = 0.0;
    std::map<std::string, double> registrationStageBatchMaxMs;
    for (const auto& result : results) {
        // Frames run concurrently, so batch stage latency follows the slowest
        // frame rather than the arithmetic sum of all worker durations.
        totalKeypointDetectionMs = std::max(
            totalKeypointDetectionMs, result.keypointDetectionMs);
        totalSvdVotingMs = std::max(totalSvdVotingMs, result.svdVotingMs);
        totalIcpIterationsMs = std::max(
            totalIcpIterationsMs, result.icpIterationsMs);
        candidatePreparationBatchMaxMs = std::max(
            candidatePreparationBatchMaxMs, result.candidatePreparationMs);
        pointCloudPreparationBatchMaxMs = std::max(
            pointCloudPreparationBatchMaxMs, result.pointCloudPreparationMs);
        for (const auto& stage : result.registration.stageTimingsMs) {
            auto& maximum = registrationStageBatchMaxMs[stage.first];
            maximum = std::max(maximum, stage.second);
        }
    }

    const FrameResult* best = nullptr;
    for (const auto& result : results) {
        if (!result.passed || !result.sourceCloud) continue;
        const bool betterManualScore = manualRoiMode && best &&
            (result.registration.fitness > best->registration.fitness ||
             (result.registration.fitness == best->registration.fitness &&
              std::tie(result.registration.sourceToTargetRmseMm,
                       result.registration.sourceToTargetP95Mm) <
              std::tie(best->registration.sourceToTargetRmseMm,
                       best->registration.sourceToTargetP95Mm)));
        const bool betterAutomaticQuality = !manualRoiMode && best &&
            std::tie(result.registration.sourceToTargetRmseMm,
                     result.registration.sourceToTargetP95Mm) <
            std::tie(best->registration.sourceToTargetRmseMm,
                     best->registration.sourceToTargetP95Mm);
        if (!best || betterManualScore || betterAutomaticQuality)
            best = &result;
    }

    std::ofstream detectionLog(logsDir / "detection_debug.txt", std::ios::trunc);
    std::ofstream registrationLog(logsDir / "registration_attempts.txt", std::ios::trunc);
    for (const auto& result : results) {
        detectionLog << "frame=" << result.frameIndex
                     << " sequence=" << result.sequence
                     << " candidate=" << (result.candidateFound ? 1 : 0)
                     << " bbox=" << result.bbox.x << ',' << result.bbox.y << ','
                     << result.bbox.width << ',' << result.bbox.height
                     << " depth_mm=" << result.depthMm
                     << " points=" << result.sourcePoints.size()
                     << " message=" << result.message << '\n';
        registrationLog << "frame=" << result.frameIndex
                        << " gate=" << (result.passed ? "PASS" : "FAIL")
                        << " initialization=" << result.registration.initializationMethod
                        << " fitness=" << result.registration.fitness
                        << " inlier_rmse_mm=" << result.registration.inlierRmseMm
                        << " source_rmse_mm=" << result.registration.sourceToTargetRmseMm
                        << " source_p95_mm=" << result.registration.sourceToTargetP95Mm
                        << '\n';
    }
    detectionLog.close();
    registrationLog.close();
    if (!best) {
        std::cerr << "no parallel result passed the quality gate\n";
        return 2;
    }

    std::cout << "registration path: " << best->registration.initializationMethod << '\n';
    if (best->registration.initializationMethod == "fpfh_ransac_icp") {
        if (!best->keypointInitializationAccepted) {
            std::cout << "keypoints detection failed or produced no reliable 3D pose; "
                         "fallback to FPFH/RANSAC/ICP\n";
        } else {
            std::cout << "keypoint initialization failed the registration quality gate; "
                         "fallback to FPFH/RANSAC/ICP\n";
        }
    } else {
        std::cout << "keypoint initialization accepted; FPFH not used\n";
    }

    const CameraFrame& bestFrame = frames[static_cast<std::size_t>(best->frameIndex)];
    const Eigen::Matrix4d cameraToStl = best->registration.transformation;
    const Eigen::Matrix4d stlToCamera = cameraToStl.inverse();
    if (!cameraToStl.array().isFinite().all() ||
        !stlToCamera.array().isFinite().all()) return EXIT_FAILURE;

    if (!open3d::io::WritePointCloud(
            (cameraDir / "camera_face_cloud.ply").string(), *best->sourceCloud))
        return EXIT_FAILURE;
    if (!best->semanticMask.empty()) {
        cv::imwrite((cameraDir / "camera_face_mask.png").string(),
                    best->semanticMask);
        cv::Mat overlay = bestFrame.color.clone();
        cv::Mat tint(overlay.size(), overlay.type(), cv::Scalar(0, 255, 0));
        cv::Mat blended;
        cv::addWeighted(overlay, 0.65, tint, 0.35, 0.0, blended);
        blended.copyTo(overlay, best->semanticMask);
        cv::imwrite((cameraDir / "camera_face_mask_overlay.png").string(), overlay);
    }
    auto aligned = std::make_shared<open3d::geometry::PointCloud>(*best->sourceCloud);
    aligned->Transform(cameraToStl);
    if (!open3d::io::WritePointCloud(
            (cameraDir / "aligned_camera_face.ply").string(), *aligned))
        return EXIT_FAILURE;
    if (!writeMatrix(stlDir / "camera_to_stl_transformation.txt", cameraToStl) ||
        !writeMatrix(stlDir / "pose_stl_to_camera.txt", stlToCamera))
        return EXIT_FAILURE;
    auto stlInCamera =
        std::make_shared<open3d::geometry::PointCloud>(*registrationTargetCloud);
    stlInCamera->Transform(stlToCamera);
    if (!open3d::io::WritePointCloud(
            (stlDir / "stl_surface_in_camera.ply").string(), *stlInCamera))
        return EXIT_FAILURE;

    cv::Mat annotated = bestFrame.color.clone();
    cv::rectangle(annotated, best->bbox, cv::Scalar(0, 255, 0), 2);
    if (manualRoiMode) {
        cv::imwrite((roiDir / "selected_roi.png").string(), annotated);
        saveRoiDebugArtifacts(bestFrame, best->bbox, roiDir, "manual_roi");
    } else if (targetLocator == "sapiens_seg") {
        cv::imwrite((faceSegmentationDir / "segmented_face.png").string(), annotated);
    } else {
        cv::imwrite((faceDetectionDir / "detected_face.png").string(), annotated);
    }
    writeD2cDiagnostics(bestFrame, cameraDir);

    std::ofstream metrics(logsDir / "aligned_camera_face_metrics.txt", std::ios::trunc);
    metrics << std::setprecision(12)
            << "selected_frame=" << best->frameIndex << '\n'
            << "selection_policy=min_source_rmse_then_p95\n"
            << "fitness=" << best->registration.fitness << '\n'
            << "inlier_rmse_mm=" << best->registration.inlierRmseMm << '\n'
            << "source_to_target_rmse_mm="
            << best->registration.sourceToTargetRmseMm << '\n'
            << "source_to_target_median_mm="
            << best->registration.sourceToTargetMedianMm << '\n'
            << "source_to_target_p95_mm="
            << best->registration.sourceToTargetP95Mm << '\n'
            << "target_to_source_rmse_mm="
            << best->registration.targetToSourceRmseMm << '\n'
            << "symmetric_rmse_mm=" << best->registration.symmetricRmseMm << '\n'
            << "initialization_method="
            << best->registration.initializationMethod << '\n';
    metrics.close();
    if (!metrics) {
        std::cerr << "cannot write final registration metrics\n";
        return EXIT_FAILURE;
    }

    const auto temporaryReady = outputDir / ".aligned_camera_face.ready.tmp";
    bool readyWritten = false;
    {
        std::ofstream ready(temporaryReady, std::ios::trunc);
        ready << "schema=2\nquality_gate=PASS\n"
              << "batch_size=" << frames.size() << '\n'
              << "worker_count=" << workerCount << '\n'
              << "selected_frame=" << best->frameIndex << '\n'
              << "camera_cloud=camera/camera_face_cloud.ply\n"
              << "aligned_cloud=camera/aligned_camera_face.ply\n"
              << "camera_to_stl=STL/camera_to_stl_transformation.txt\n"
              << "stl_to_camera=STL/pose_stl_to_camera.txt\n"
              << "stl_surface_in_camera=STL/stl_surface_in_camera.ply\n"
              << "metrics=logs/aligned_camera_face_metrics.txt\n";
        ready.flush();
        readyWritten = static_cast<bool>(ready);
    }
    if (!readyWritten) return EXIT_FAILURE;
    std::error_code readyError;
    std::filesystem::rename(
        temporaryReady, outputDir / "aligned_camera_face.ready", readyError);
    if (readyError) return EXIT_FAILURE;

    std::ofstream sessionInfo(outputDir / "session_info.txt", std::ios::app);
    sessionInfo << "processing="
                << (manualRoiMode ? "batch_concurrent_in_memory"
                                  : "global_top1_in_memory") << '\n'
                << "selected_frame=" << best->frameIndex << '\n'
                << "selected_source_rmse_mm="
                << best->registration.sourceToTargetRmseMm << '\n'
                << "selected_source_p95_mm="
                << best->registration.sourceToTargetP95Mm << '\n';
    const std::string localizationStage = manualRoiMode
        ? ""
        : targetLocator == "sapiens_seg" ? "face_segmentation" : "face_detection";
    const std::string keypointStage = keypointProvider == "sapiens_pose"
        ? "sapiens_pose_keypoint_detection" : "hrnet_keypoint_detection";
    // Pose must be loaded after Seg releases GPU memory. Count that required
    // runtime transition as part of the keypoint stage, rather than exposing
    // internal diagnostic/test timings.
    totalKeypointDetectionMs += modelTransitionMs;
    std::vector<std::pair<std::string, double>> detailedTimings;
    double fpfhRansacMs = 0.0;
    if (best->registration.initializationMethod == "fpfh_ransac_icp") {
        for (const auto& stage : best->registration.stageTimingsMs) {
            if (stage.first == "fpfh_feature_extraction" ||
                stage.first.rfind("ransac_attempt_", 0) == 0)
                fpfhRansacMs += stage.second;
        }
        detailedTimings.emplace_back("fpfh_ransac_coarse_registration",
                                     fpfhRansacMs);
    }
    const double pointCloudProcessingMs = std::max(0.0,
        endToEndTotalMs - targetLocalizationMs - totalKeypointDetectionMs -
        totalSvdVotingMs - totalIcpIterationsMs - fpfhRansacMs);
    detailedTimings.emplace_back("point_cloud_processing",
                                 pointCloudProcessingMs);
    writeFinalTimings(timingPath, localizationStage, targetLocalizationMs,
                      keypointStage, totalKeypointDetectionMs,
                      totalSvdVotingMs, totalIcpIterationsMs,
                      endToEndTotalMs, detailedTimings);
    return EXIT_SUCCESS;
}
