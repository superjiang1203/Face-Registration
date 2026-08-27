// Live depth-camera face registration application.
#include "camera/OrbbecDepthAlignedCamera.hpp"
#if FACE_HAS_VCAMERA
#include "camera/VcameraDepthAlignedCamera.hpp"
#endif
#include "detection/FaceDetectionService.hpp"
#include "detection/FaceKeypointService.hpp"
#include "registration/CameraFaceKeypointExtractionService.hpp"
#include "registration/FaceCandidateSelectionPolicy.hpp"
#include "registration/FacePointCloudCropService.hpp"
#include "registration/HeadSurfaceCache.hpp"
#include "registration/ModelKeypointAnnotationService.hpp"
#include "registration/PointCloudRegistration.hpp"

#include <open3d/Open3D.h>
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
    std::ostringstream line;
    line << std::fixed << std::setprecision(3) << "[timing] ";
    if (frameIndex >= 0) line << "frame=" << frameIndex << ' ';
    line << "stage=" << stage << " elapsed_ms=" << milliseconds;
    std::cout << line.str() << '\n';
    std::ofstream output(timingPath, std::ios::app);
    if (output) output << line.str() << '\n';
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

bool writeMatrix(const std::filesystem::path& path, const Eigen::Matrix4d& matrix) {
    std::ofstream output(path, std::ios::trunc);
    if (output) output << std::setprecision(12) << matrix << '\n';
    return static_cast<bool>(output);
}

std::filesystem::path resolveManualRoiPath() {
    return std::filesystem::path("output") / "manual_roi.txt";
}
}

int main(int argc, char** argv) {
    const auto registrationTotalStart = Clock::now();
    double excludedInitialModelReconstructionMs = 0.0;
    std::string cameraBackend = "orbbec";
    CameraDeviceSelector cameraSelector;
    bool laserAutoControl = false;
    int laserPower = 25;
    bool listCameras = false;
    bool manualRoiMode = false;
    unsigned workerCount = 0;
    bool workerCountSpecified = false;
    std::filesystem::path runtimeConfig = "C++/config/runtime.yml";
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--camera-backend" || arg == "--camera-sn" || arg == "--camera-ip" ||
             arg == "--laser-power" || arg == "--laser-auto" || arg == "--threads") && i + 1 >= argc) {
            std::cerr << "missing value after " << arg << '\n';
            return EXIT_FAILURE;
        }
        if (arg == "--camera-backend") cameraBackend = argv[++i];
        else if (arg == "--camera-sn") cameraSelector.serialNumber = argv[++i];
        else if (arg == "--camera-ip") cameraSelector.ipAddress = argv[++i];
        else if (arg == "--laser-power") laserPower = std::stoi(argv[++i]);
        else if (arg == "--laser-auto") {
            const std::string value = argv[++i];
            if (value != "on" && value != "off") {
                std::cerr << "--laser-auto must be on or off\n";
                return EXIT_FAILURE;
            }
            laserAutoControl = value == "on";
        }
        else if (arg == "--threads") {
            workerCount = static_cast<unsigned>(std::stoul(argv[++i]));
            workerCountSpecified = true;
        }
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

    if (positional.size() < 3 || positional.size() > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " [--camera-backend orbbec|vcamera] [--list-cameras]"
                     " [--camera-sn SN] [--camera-ip IP]"
                     " [--laser-auto on|off] [--laser-power VALUE] [--manual-roi] [--threads N] [--config FILE]"
                      " <face_detector.onnx|manual:-> <face_keypoints.onnx|manual:-> <output_dir>"
                      " [min_fitness] [max_inlier_rmse_mm]"
                      " [max_source_rmse_mm] [max_source_p95_mm]\n"
                      "Target STL is fixed to data/head.stl relative to the repository root.\n";
        return EXIT_FAILURE;
    }
    if (isStlPath(positional[2]) || isPlyPath(positional[2])) {
        std::cerr << "legacy camera command detected: the third positional argument ('"
                  << positional[2]
                  << "') looks like the removed <target.ply|stl> argument. Remove it; "
                     "the live target is fixed to data/head.stl.\n";
        return EXIT_FAILURE;
    }
    const auto requestedOutput = std::filesystem::absolute(positional[2]).lexically_normal();
    const auto fixedOutput = std::filesystem::absolute("output").lexically_normal();
    if (requestedOutput != fixedOutput) {
        std::cerr << "camera pipeline output is fixed to " << fixedOutput.string()
                  << "; replace output argument '" << positional[2] << "' with .\\output\n";
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
    const auto faceKeypointsDir = outputDir / "face_keypoints_detection";
    const auto roiDir = outputDir / "roi";
    const auto cameraDir = outputDir / "camera";
    const auto stlDir = outputDir / "STL";
    const auto logsDir = outputDir / "logs";
    for (const auto& directory : {faceDetectionDir, faceKeypointsDir, roiDir,
                                  cameraDir, stlDir, logsDir})
        std::filesystem::create_directories(directory);
    const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    if (!workerCountSpecified) {
        const auto configuredFrames = readYamlScalar(runtimeConfig, "pipeline", "detection_frames");
        workerCount = configuredFrames
            ? static_cast<unsigned>(std::stoul(*configuredFrames))
            : std::min(8u, std::max(1u, hardwareThreads > 2 ? hardwareThreads - 2 : 1u));
    }
    if (workerCount == 0 || workerCount > 64) {
        std::cerr << "--threads must be between 1 and 64\n";
        return EXIT_FAILURE;
    }
    const int batchSize = static_cast<int>(workerCount);
    const double minFitness = positional.size() >= 4 ? std::stod(positional[3]) : 0.90;
    const double maxInlierRmse = positional.size() >= 5 ? std::stod(positional[4]) : 3.0;
    const double maxSourceRmse = positional.size() >= 6 ? std::stod(positional[5]) : 6.0;
    const double maxSourceP95 = positional.size() >= 7 ? std::stod(positional[6]) : 10.0;
    if (!std::isfinite(minFitness) || minFitness < 0.0 || minFitness > 1.0 ||
        !std::isfinite(maxInlierRmse) || !(maxInlierRmse > 0.0) ||
        !std::isfinite(maxSourceRmse) || !(maxSourceRmse > 0.0) ||
        !std::isfinite(maxSourceP95) || !(maxSourceP95 > 0.0)) {
        std::cerr << "registration quality thresholds are invalid\n";
        return EXIT_FAILURE;
    }
    const double modelUnitScale = 1.0;
    const auto timingPath = logsDir / "pipeline_timing.txt";
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
                    << "parallel_detection_frames=" << batchSize << '\n'
                    << "keypoint_pose_solver="
                    << (poseSolver == ModelKeypointAnnotationService::PoseSolver::TripletVote
                            ? "triplet_vote" : "overdetermined_svd") << '\n';
    }
    if (manualRoiMode) {
        std::cout << "manual ROI mode: ONNX detection and RGB keypoint initialization disabled\n";
    }
    std::unique_ptr<FaceKeypointService> keypointDetector;
    std::unique_ptr<CameraFaceKeypointExtractionService> cameraKeypointExtractor;
    std::optional<ModelKeypointAnnotationService::Result> modelKeypoints;
    std::optional<std::string> modelKeypointSourceSha256;
    if (!manualRoiMode && positional[1] != "-") {
        const auto keypointLoadStart = Clock::now();
        keypointDetector = std::make_unique<FaceKeypointService>(positional[1]);
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
    }

    RegistrationOptions registrationOptions;
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
            excludedInitialModelReconstructionMs = modelReconstructionMs;
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


    CameraBase::Options cameraOptions;
    cameraOptions.frameKind = FrameKind::ColorDepthPointCloud;
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
        double depthMm{0.0};
        std::vector<PointXYZ> sourcePoints;
        std::shared_ptr<open3d::geometry::PointCloud> sourceCloud;
        std::optional<CameraFaceKeypointExtractionService::Result> keypoints;
        RegistrationResult registration;
    };

    if (!manualRoiMode && positional[1] == "-") {
        std::cerr << "face-detection mode requires a keypoint model\n";
        camera->close();
        return EXIT_FAILURE;
    }

    std::optional<cv::Rect> manualRoi;
    const auto manualRoiPath = resolveManualRoiPath();
    if (manualRoiMode) {
        std::ifstream input(manualRoiPath);
        int x = 0, y = 0, width = 0, height = 0;
        if (!(input >> x >> y >> width >> height) || width <= 0 || height <= 0) {
            std::cerr << "invalid manual ROI file: " << manualRoiPath.string() << '\n';
            camera->close();
            return EXIT_FAILURE;
        }
        manualRoi = cv::Rect(x, y, width, height);
    }

    std::vector<CameraFrame> frames;
    frames.reserve(batchSize);
    for (int attempt = 0;
         attempt < std::max(30, batchSize * 10) &&
         frames.size() < static_cast<std::size_t>(batchSize);
         ++attempt) {
        auto frame = camera->capture();
        if (!frame || frame->color.empty() || frame->pointCloud.empty()) continue;
        frames.push_back(std::move(*frame));
    }
    camera->close();
    if (frames.size() != static_cast<std::size_t>(batchSize)) {
        std::cerr << "captured " << frames.size() << " of " << batchSize << " frames\n";
        return 2;
    }

    struct GlobalDetectionCandidate {
        int frameIndex{-1};
        PreparedFaceCandidate candidate;
    };
    std::optional<GlobalDetectionCandidate> globalTopDetection;
    if (!manualRoiMode) {
        const auto detectionBatchStart = Clock::now();
        std::vector<std::future<std::vector<GlobalDetectionCandidate>>> detectionFutures;
        detectionFutures.reserve(frames.size());
        for (int frameIndex = 0; frameIndex < static_cast<int>(frames.size()); ++frameIndex) {
            detectionFutures.push_back(std::async(std::launch::async, [&, frameIndex]() {
                std::vector<GlobalDetectionCandidate> accepted;
                FacePointCloudCropService localCropper(cropOptions);
                FaceDetectionService detector(detectorOptions);
                if (!detector.loadOnnx(positional[0])) return accepted;
                cv::Mat color = frames[static_cast<std::size_t>(frameIndex)].color.clone();
                const auto detected = detector.process(
                    color,
                    frames[static_cast<std::size_t>(frameIndex)].depth,
                    &frames[static_cast<std::size_t>(frameIndex)].pointCloud,
                    frames[static_cast<std::size_t>(frameIndex)].pointCloudWidth,
                    frames[static_cast<std::size_t>(frameIndex)].pointCloudHeight);
                if (!detected) return accepted;
                for (const auto& detection : detected->dets) {
                    const double score = validatedDetectionScore(detection);
                    if (score < faceSelectionOptions.minDetectionScore || detection.bbox.empty()) continue;
                    PreparedFaceCandidate candidate;
                    candidate.detection = detection;
                    candidate.cropBbox = shrinkFaceBoxForCrop(detection.bbox);
                    candidate.points = localCropper.cropFacePointCloud(
                        frames[static_cast<std::size_t>(frameIndex)],
                        frames[static_cast<std::size_t>(frameIndex)].color.size(),
                        candidate.cropBbox);
                    const auto roi = localCropper.faceCropRoi(
                        frames[static_cast<std::size_t>(frameIndex)].color.size(),
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudWidth,
                        frames[static_cast<std::size_t>(frameIndex)].pointCloudHeight,
                        candidate.cropBbox);
                    const std::size_t pixels = roi
                        ? static_cast<std::size_t>(std::max(0, roi->pointCloudRoi.area())) : 0;
                    candidate.metrics = FaceCandidateSelectionPolicy::evaluate(
                        score, candidate.points, pixels, faceSelectionOptions);
                    if (candidate.metrics.accepted)
                        accepted.push_back({frameIndex, std::move(candidate)});
                }
                return accepted;
            }));
        }
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
            std::cerr << "parallel face detection produced no candidate passing physical gates\n";
            return 3;
        }
        globalTopDetection = std::move(allCandidates.front());
        reportTiming(timingPath, "parallel_face_detection_global_top1", elapsedMs(detectionBatchStart));
        std::cout << "global detection Top-1: frame=" << globalTopDetection->frameIndex
                  << " score=" << globalTopDetection->candidate.metrics.detectionScore
                  << " depth_mm=" << globalTopDetection->candidate.metrics.medianDepthMm
                  << " accepted_candidates=" << allCandidates.size() << '\n';
    }

    const auto processFrame = [&](int frameIndex) -> FrameResult {
        FrameResult output;
        output.frameIndex = frameIndex;
        const CameraFrame& frame = frames[static_cast<std::size_t>(frameIndex)];
        output.sequence = frame.meta.sequence;
        try {
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

            CameraFaceKeypointExtractionService* keypointExtractorForFrame =
                manualRoiMode ? nullptr : cameraKeypointExtractor.get();

            bool hasResult = false;
            for (const std::size_t index : order) {
                auto& candidate = candidates[index];
                open3d::geometry::PointCloud raw;
                for (const auto& point : candidate.points)
                    raw.points_.emplace_back(point.x, point.y, point.z);
                auto source = raw.VoxelDownSample(2.5);
                if (!source || source->IsEmpty()) continue;

                std::optional<Eigen::Matrix4d> initial;
                if (keypointExtractorForFrame) {
                    const auto keypointStart = Clock::now();
                    const auto snapshot = snapshotForCandidate(
                        frame, candidate, localCropper);
                    if (snapshot) {
                        CameraFaceKeypointExtractionService::Options options;
                        options.writeVisualizationArtifacts = true;
                        options.write3DArtifacts = true;
                        std::string error;
                        const auto extracted =
                            keypointExtractorForFrame->extractFromSnapshot(
                                *snapshot, faceKeypointsDir, options, &error);
                        output.keypoints = extracted;
                        if (extracted.success && modelKeypoints) {
                            const auto pose =
                                ModelKeypointAnnotationService::estimateCameraToModel(
                                    extracted.keypoints, modelKeypoints->keypoints,
                                    keypointInlierThresholdMm,
                                    poseSolver);
                            if (FaceCandidateSelectionPolicy::isReliableKeypointPose(
                                    pose.success, pose.inlierCount, pose.rmseMm,
                                    extracted.meanScore, faceSelectionOptions))
                                initial = pose.cameraToModel;
                        }
                    }
                    reportTiming(timingPath, "face_keypoints_detection_top1",
                                 elapsedMs(keypointStart));
                }

                RegistrationOptions options = registrationOptions;
                options.randomSeed += frameIndex * 101;
                options.reseedGlobalRandomEngine = false;
                PointCloudRegistration registrationForFrame(options);
                const auto registered = registrationForFrame.alignClouds(
                    *source, *registrationTargetCloud, {}, initial);
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

    std::vector<std::future<FrameResult>> futures;
    open3d::utility::random::Seed(registrationOptions.randomSeed);
    if (manualRoiMode) {
        for (int i = 0; i < static_cast<int>(frames.size()); ++i)
            futures.push_back(std::async(std::launch::async, processFrame, i));
    } else {
        futures.push_back(std::async(
            std::launch::async, processFrame, globalTopDetection->frameIndex));
    }
    std::vector<FrameResult> results;
    for (auto& future : futures) results.push_back(future.get());

    const FrameResult* best = nullptr;
    for (const auto& result : results) {
        if (!result.passed || !result.sourceCloud) continue;
        if (!best ||
            std::tie(result.registration.sourceToTargetRmseMm,
                     result.registration.sourceToTargetP95Mm) <
            std::tie(best->registration.sourceToTargetRmseMm,
                     best->registration.sourceToTargetP95Mm))
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
    for (const auto& stage : best->registration.stageTimingsMs)
        reportTiming(timingPath, "registration_" + stage.first, stage.second);

    const CameraFrame& bestFrame = frames[static_cast<std::size_t>(best->frameIndex)];
    const Eigen::Matrix4d cameraToStl = best->registration.transformation;
    const Eigen::Matrix4d stlToCamera = cameraToStl.inverse();
    if (!cameraToStl.array().isFinite().all() ||
        !stlToCamera.array().isFinite().all()) return EXIT_FAILURE;

    if (!open3d::io::WritePointCloud(
            (cameraDir / "camera_face_cloud.ply").string(), *best->sourceCloud))
        return EXIT_FAILURE;
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
    sessionInfo << "processing=batch_parallel_in_memory\n"
                << "selected_frame=" << best->frameIndex << '\n'
                << "selected_source_rmse_mm="
                << best->registration.sourceToTargetRmseMm << '\n'
                << "selected_source_p95_mm="
                << best->registration.sourceToTargetP95Mm << '\n';
    reportTiming(
        timingPath, "registration_total_excluding_initial_model_reconstruction",
        std::max(0.0, elapsedMs(registrationTotalStart) -
                      excludedInitialModelReconstructionMs));
    return EXIT_SUCCESS;
}
