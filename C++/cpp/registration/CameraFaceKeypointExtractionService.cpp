#include "registration/CameraFaceKeypointExtractionService.hpp"

#include "detection/FaceKeypointService.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using Service = CameraFaceKeypointExtractionService;
using Keypoint = Service::Keypoint3D;

struct SnapshotData {
    cv::Mat color;
    int pointCloudWidth{0};
    int pointCloudHeight{0};
    std::vector<PointXYZ> pointCloud;
};

struct Candidate {
    Service::CandidateMetrics metrics;
    SnapshotData snapshot;
    std::vector<Keypoint> keypoints;
};

bool hasPlausibleFaceGeometry(
    const std::vector<Keypoint>& keypoints, const cv::Size& imageSize) {
    std::unordered_map<std::string, cv::Point2d> points;
    for (const auto& keypoint : keypoints) {
        if (keypoint.imageX >= 0 && keypoint.imageY >= 0)
            points[keypoint.name] = cv::Point2d(keypoint.imageX, keypoint.imageY);
    }
    if (points.size() < 3 || imageSize.width <= 0 || imageSize.height <= 0) return false;
    double minX = INFINITY, minY = INFINITY, maxX = -INFINITY, maxY = -INFINITY;
    for (const auto& entry : points) {
        minX = std::min(minX, entry.second.x); minY = std::min(minY, entry.second.y);
        maxX = std::max(maxX, entry.second.x); maxY = std::max(maxY, entry.second.y);
    }
    if (maxX - minX < 0.10 * imageSize.width ||
        maxY - minY < 0.05 * imageSize.height) return false;

    const char* required[] = {"right_eye_outer", "right_eye_inner", "left_eye_inner",
                              "left_eye_outer", "nose_root", "nose_tip"};
    for (const char* name : required) if (points.find(name) == points.end()) return true;
    const auto distance = [](const cv::Point2d& a, const cv::Point2d& b) { return cv::norm(a - b); };
    const auto rightCenter = 0.5 * (points["right_eye_outer"] + points["right_eye_inner"]);
    const auto leftCenter = 0.5 * (points["left_eye_outer"] + points["left_eye_inner"]);
    const double interocular = distance(rightCenter, leftCenter);
    const double rightEyeWidth = distance(points["right_eye_outer"], points["right_eye_inner"]);
    const double leftEyeWidth = distance(points["left_eye_outer"], points["left_eye_inner"]);
    const double noseLength = distance(points["nose_root"], points["nose_tip"]);
    const double eyeY = 0.5 * (rightCenter.y + leftCenter.y);
    return interocular >= 0.12 * imageSize.width &&
           rightEyeWidth >= 0.02 * imageSize.width &&
           leftEyeWidth >= 0.02 * imageSize.width &&
           noseLength >= 0.03 * imageSize.height &&
           points["nose_tip"].y > points["nose_root"].y &&
           points["nose_tip"].y > eyeY - 0.05 * imageSize.height;
}

bool isValidPoint(const PointXYZ& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && point.z > 0.0f;
}

double median(std::vector<double>& values) {
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    return values[middle];
}

std::optional<Eigen::Vector3d> samplePointCloudAtPixel(
    const SnapshotData& snapshot,
    int imageX,
    int imageY,
    int searchRadius) {
    const int cloudWidth = snapshot.pointCloudWidth;
    const int cloudHeight = snapshot.pointCloudHeight;
    const int imageWidth = snapshot.color.cols;
    const int imageHeight = snapshot.color.rows;
    if (cloudWidth <= 0 || cloudHeight <= 0 || imageWidth <= 0 || imageHeight <= 0 ||
        snapshot.pointCloud.size() < static_cast<std::size_t>(cloudWidth) * static_cast<std::size_t>(cloudHeight)) {
        return std::nullopt;
    }

    const double scaleX = imageWidth > 1
        ? static_cast<double>(cloudWidth - 1) / static_cast<double>(imageWidth - 1)
        : 1.0;
    const double scaleY = imageHeight > 1
        ? static_cast<double>(cloudHeight - 1) / static_cast<double>(imageHeight - 1)
        : 1.0;
    const int cloudX = std::clamp(
        static_cast<int>(std::lround(static_cast<double>(imageX) * scaleX)), 0, cloudWidth - 1);
    const int cloudY = std::clamp(
        static_cast<int>(std::lround(static_cast<double>(imageY) * scaleY)), 0, cloudHeight - 1);

    const int imageRadius = std::max(0, searchRadius);
    const int radiusX = static_cast<int>(std::ceil(static_cast<double>(imageRadius) * scaleX));
    const int radiusY = static_cast<int>(std::ceil(static_cast<double>(imageRadius) * scaleY));
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    xs.reserve(static_cast<std::size_t>((2 * radiusX + 1) * (2 * radiusY + 1)));
    ys.reserve(xs.capacity());
    zs.reserve(xs.capacity());
    for (int y = std::max(0, cloudY - radiusY); y <= std::min(cloudHeight - 1, cloudY + radiusY); ++y) {
        for (int x = std::max(0, cloudX - radiusX); x <= std::min(cloudWidth - 1, cloudX + radiusX); ++x) {
            const auto& point = snapshot.pointCloud[
                static_cast<std::size_t>(y) * static_cast<std::size_t>(cloudWidth) + static_cast<std::size_t>(x)];
            if (!isValidPoint(point)) continue;
            xs.push_back(point.x);
            ys.push_back(point.y);
            zs.push_back(point.z);
        }
    }
    if (zs.empty()) return std::nullopt;
    return Eigen::Vector3d(median(xs), median(ys), median(zs));
}

std::array<unsigned char, 3> rgbForKeypoint(const std::string& name, int index) {
    if (name == "nose_root") return {255, 0, 255};
    if (name == "nose_tip") return {255, 0, 0};
    if (name == "right_eye_outer") return {0, 180, 255};
    if (name == "right_eye_inner") return {0, 255, 255};
    if (name == "left_eye_inner") return {100, 255, 50};
    if (name == "left_eye_outer") return {0, 255, 0};
    static constexpr std::array<unsigned char, 3> palette[] = {
        {230, 25, 75}, {60, 180, 75}, {0, 130, 200}, {245, 130, 48},
        {145, 30, 180}, {70, 240, 240}, {240, 50, 230}, {210, 245, 60}};
    return palette[static_cast<std::size_t>(std::max(0, index)) % std::size(palette)];
}

void drawKeypoints(cv::Mat& image, const std::vector<Keypoint>& keypoints) {
    for (const auto& keypoint : keypoints) {
        if (keypoint.imageX < 0 || keypoint.imageY < 0) continue;
        const auto rgb = rgbForKeypoint(keypoint.name, keypoint.index);
        const cv::Scalar bgr(rgb[2], rgb[1], rgb[0]);
        cv::circle(image, cv::Point(keypoint.imageX, keypoint.imageY), 5, bgr, -1, cv::LINE_AA);
        cv::putText(image, keypoint.name,
            cv::Point(keypoint.imageX + 7, std::max(14, keypoint.imageY - 7)),
            cv::FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv::LINE_AA);
    }
}

std::string jsonEscape(const std::string& input) {
    std::ostringstream output;
    for (const unsigned char ch : input) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(ch) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

bool writeKeypointsPly(
    const std::filesystem::path& path,
    const std::vector<Keypoint>& keypoints,
    std::string* error) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        if (error) *error = "Failed to write camera keypoint PLY: " + path.string();
        return false;
    }
    const auto validCount = std::count_if(
        keypoints.begin(), keypoints.end(), [](const Keypoint& keypoint) { return keypoint.hasPoint; });
    output << "ply\nformat ascii 1.0\nelement vertex " << validCount
           << "\nproperty double x\nproperty double y\nproperty double z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
           << std::setprecision(12);
    for (const auto& keypoint : keypoints) {
        if (!keypoint.hasPoint) continue;
        const auto rgb = rgbForKeypoint(keypoint.name, keypoint.index);
        output << keypoint.point.x() << ' ' << keypoint.point.y() << ' ' << keypoint.point.z() << ' '
               << static_cast<int>(rgb[0]) << ' ' << static_cast<int>(rgb[1]) << ' '
               << static_cast<int>(rgb[2]) << '\n';
    }
    return static_cast<bool>(output);
}

bool writeKeypointsJson(
    const std::filesystem::path& path,
    const Service::Result& result,
    const SnapshotData& selectedSnapshot,
    std::string* error) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        if (error) *error = "Failed to write camera keypoint JSON: " + path.string();
        return false;
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"colorImagePath\": \"" << jsonEscape(result.colorImagePath.generic_string()) << "\",\n"
           << "  \"keypointsJsonPath\": \"" << jsonEscape(result.keypointsJsonPath.generic_string()) << "\",\n"
           << "  \"keypointsPlyPath\": \"" << jsonEscape(result.keypointsPlyPath.generic_string()) << "\",\n"
           << "  \"renderImagePath\": \"" << jsonEscape(result.renderImagePath.generic_string()) << "\",\n"
           << "  \"visibleCount\": " << result.visibleCount << ",\n"
           << "  \"validCount\": " << result.validCount << ",\n"
           << "  \"meanScore\": " << result.meanScore << ",\n"
           << "  \"imageWidth\": " << selectedSnapshot.color.cols << ",\n"
           << "  \"imageHeight\": " << selectedSnapshot.color.rows << ",\n"
           << "  \"pointCloudWidth\": " << selectedSnapshot.pointCloudWidth << ",\n"
           << "  \"pointCloudHeight\": " << selectedSnapshot.pointCloudHeight << ",\n"
           << "  \"keypoints\": [\n";
    for (std::size_t i = 0; i < result.keypoints.size(); ++i) {
        const auto& keypoint = result.keypoints[i];
        output << "    {\"index\": " << keypoint.index
               << ", \"name\": \"" << jsonEscape(keypoint.name) << "\""
               << ", \"score\": " << keypoint.score
               << ", \"imageX\": " << keypoint.imageX
               << ", \"imageY\": " << keypoint.imageY
               << ", \"hasPoint\": " << (keypoint.hasPoint ? "true" : "false");
        if (keypoint.hasPoint) {
            output << ", \"x\": " << keypoint.point.x()
                   << ", \"y\": " << keypoint.point.y()
                   << ", \"z\": " << keypoint.point.z();
        }
        output << '}' << (i + 1 == result.keypoints.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return static_cast<bool>(output);
}

Candidate evaluateCandidate(
    FaceKeypointService& detector,
    const Service::Snapshot& source,
    const Service::Options& options) {
    Candidate candidate;
    candidate.snapshot.color = source.color.clone();
    candidate.snapshot.pointCloud = source.pointCloud;
    candidate.snapshot.pointCloudWidth = source.pointCloudWidth;
    candidate.snapshot.pointCloudHeight = source.pointCloudHeight;
    if (candidate.snapshot.color.empty() || candidate.snapshot.pointCloud.empty()) return candidate;

    std::string detectionError;
    auto detection = detector.detect(candidate.snapshot.color, &detectionError);
    if (detection.keypoints.empty()) {
        if (!detectionError.empty()) {
            std::cerr << "camera keypoint detection failed: " << detectionError << '\n';
        }
        // Fallback: try a centered crop of the rotated snapshot when detector fails
        const float centerCropRatio = 0.6f; // fraction of min(width,height)
        const int w = candidate.snapshot.color.cols;
        const int h = candidate.snapshot.color.rows;
        const int side = static_cast<int>(std::round(std::min(w, h) * centerCropRatio));
        const int x0 = std::max(0, (w - side) / 2);
        const int y0 = std::max(0, (h - side) / 2);
        const cv::Rect centerBox(x0, y0, std::min(side, w - x0), std::min(side, h - y0));
        if (centerBox.area() > 0) {
            try {
                const cv::Mat crop = candidate.snapshot.color(centerBox).clone();
                std::string cropError;
                auto cropDetection = detector.detect(crop, &cropError);
                if (!cropDetection.keypoints.empty()) {
                    // Remap crop coordinates back to full image coordinates
                    for (auto& kp : cropDetection.keypoints) {
                        kp.x += static_cast<float>(centerBox.x);
                        kp.y += static_cast<float>(centerBox.y);
                    }
                    detection = std::move(cropDetection);
                } else if (!cropError.empty()) {
                    std::cerr << "camera keypoint crop failed: " << cropError << '\n';
                }
            } catch (const cv::Exception& ex) {
                std::cerr << "camera keypoint crop exception: " << ex.what() << '\n';
            }
        }
    }
    if (detection.keypoints.empty()) {
        return candidate;
    }
    candidate.metrics.detectionSucceeded = true;

    double scoreSum = 0.0;
    int scoreCount = 0;
    for (const auto& detected : detection.keypoints) {
        if (!ModelKeypointAnnotationService::isRegistrationKeypoint(detected.name) ||
            !std::isfinite(detected.x) || !std::isfinite(detected.y) ||
            !std::isfinite(detected.score) || detected.score < options.minKeypointScore) {
            continue;
        }
        ++candidate.metrics.visibleCount;
        scoreSum += detected.score;
        ++scoreCount;

        Keypoint keypoint;
        keypoint.index = detected.index;
        keypoint.name = detected.name;
        keypoint.score = detected.score;
        keypoint.imageX = std::clamp(
            static_cast<int>(std::lround(detected.x)), 0, std::max(0, candidate.snapshot.color.cols - 1));
        keypoint.imageY = std::clamp(
            static_cast<int>(std::lround(detected.y)), 0, std::max(0, candidate.snapshot.color.rows - 1));
        const auto point = samplePointCloudAtPixel(
            candidate.snapshot, keypoint.imageX, keypoint.imageY, options.searchRadiusPx);
        keypoint.hasPoint = point.has_value();
        if (point) {
            keypoint.point = *point;
            ++candidate.metrics.validCount;
        }
        candidate.keypoints.push_back(std::move(keypoint));
    }
    candidate.metrics.meanScore = scoreCount > 0 ? scoreSum / static_cast<double>(scoreCount) : 0.0;
    candidate.metrics.geometryValid = hasPlausibleFaceGeometry(
        candidate.keypoints, candidate.snapshot.color.size());
    std::cout << "camera keypoint candidate: visible=" << candidate.metrics.visibleCount
              << " valid3d=" << candidate.metrics.validCount
              << " mean_score=" << candidate.metrics.meanScore
              << " geometry=" << (candidate.metrics.geometryValid ? "valid" : "invalid") << '\n';
    return candidate;
}

void setFailure(Service::Result& result, const std::string& message, std::string* error) {
    result.success = false;
    result.message = message;
    if (error) *error = message;
}

} // namespace

CameraFaceKeypointExtractionService::CameraFaceKeypointExtractionService(
    FaceKeypointService& keypointDetector)
    : keypointDetector_(keypointDetector) {
}

CameraFaceKeypointExtractionService::Result
CameraFaceKeypointExtractionService::extractFromSnapshot(
    const Snapshot& snapshot,
    const std::filesystem::path& outputDirectory,
    const Options& options,
    std::string* error) {
    Result result;
    if (error) error->clear();
    if (snapshot.color.empty() || snapshot.color.type() != CV_8UC3) {
        setFailure(result, "Camera keypoint snapshot must contain a non-empty CV_8UC3 image", error);
        return result;
    }
    const std::size_t expectedPointCount =
        static_cast<std::size_t>(std::max(0, snapshot.pointCloudWidth)) *
        static_cast<std::size_t>(std::max(0, snapshot.pointCloudHeight));
    if (snapshot.pointCloudWidth <= 0 || snapshot.pointCloudHeight <= 0 ||
        snapshot.pointCloud.size() < expectedPointCount) {
        setFailure(result, "Camera keypoint snapshot must contain a valid organized point cloud", error);
        return result;
    }
    if (options.minValidKeypointCount < 3) {
        setFailure(result, "Camera keypoint minValidKeypointCount must be at least 3", error);
        return result;
    }
    std::string loadError;
    if (!keypointDetector_.ensureLoaded(&loadError)) {
        setFailure(result, loadError, error);
        return result;
    }
    if (options.writeVisualizationArtifacts || options.write3DArtifacts) {
        try {
            std::filesystem::create_directories(outputDirectory);
        } catch (const std::exception& exception) {
            setFailure(result, std::string("Failed to create camera keypoint output directory: ") + exception.what(), error);
            return result;
        }
    }

    Candidate best = evaluateCandidate(keypointDetector_, snapshot, options);
    bool hasBest = best.metrics.detectionSucceeded;
    result.visibleCount = best.metrics.visibleCount;
    result.validCount = best.metrics.validCount;
    result.meanScore = best.metrics.meanScore;
    result.keypoints = best.keypoints;
    if (options.writeVisualizationArtifacts) {
        result.colorImagePath = outputDirectory / "camera_keypoints_color.png";
        result.renderImagePath = outputDirectory / "camera_keypoints_render.jpg";

        cv::Mat annotated = snapshot.color.clone();
        drawKeypoints(annotated, result.keypoints);
        try {
            if (!cv::imwrite(result.colorImagePath.string(), snapshot.color) ||
                !cv::imwrite(result.renderImagePath.string(), annotated)) {
                setFailure(result, "Failed to write camera keypoint images", error);
                return result;
            }
        } catch (const cv::Exception& exception) {
            setFailure(result, std::string("Failed to write camera keypoint images: ") + exception.what(), error);
            return result;
        }
    }

    if (!hasBest || best.metrics.validCount < options.minValidKeypointCount ||
        !best.metrics.geometryValid) {
        setFailure(result,
            "Face keypoint detector failed to produce at least " +
                std::to_string(options.minValidKeypointCount) + " camera 3D keypoints",
            error);
        return result;
    }

    if (options.write3DArtifacts) {
        result.keypointsJsonPath = outputDirectory / "camera_keypoints.json";
        result.keypointsPlyPath = outputDirectory / "camera_keypoints.ply";
        if (!writeKeypointsPly(result.keypointsPlyPath, result.keypoints, error)) {
            result.message = error ? *error : "Failed to write camera keypoint PLY";
            return result;
        }
        if (!writeKeypointsJson(result.keypointsJsonPath, result, best.snapshot, error)) {
            result.message = error ? *error : "Failed to write camera keypoint JSON";
            return result;
        }
    }

    result.success = true;
    result.message = "Camera face keypoints extracted";
    return result;
}
