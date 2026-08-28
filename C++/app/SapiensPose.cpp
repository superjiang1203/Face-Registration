#include "pose/Sapiens2PoseEstimator.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
float maskSupportRatio(const cv::Mat& mask, const cv::Point2f& point, int radius) {
    const int cx = cvRound(point.x), cy = cvRound(point.y);
    int supported = 0, sampled = 0;
    for (int dy = -radius; dy <= radius; ++dy) for (int dx = -radius; dx <= radius; ++dx) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const int x = cx + dx, y = cy + dy;
        ++sampled;
        if (x >= 0 && x < mask.cols && y >= 0 && y < mask.rows && mask.at<unsigned char>(y, x) != 0)
            ++supported;
    }
    return sampled > 0 ? static_cast<float>(supported) / sampled : 0.0f;
}
}

int main(int argc, char** argv) try {
    if (argc < 5) {
        std::cerr << "Usage: sapiens_pose <model.onnx> <color.png> <face_mask.png> <output_dir>"
                     " [pose_score_threshold=0.25] [mask_support_threshold=0.20] [mask_radius_px=5]\n";
        return 2;
    }
    const std::filesystem::path model = argv[1], imagePath = argv[2], maskPath = argv[3], outDir = argv[4];
    const float scoreThreshold = argc > 5 ? std::stof(argv[5]) : 0.25f;
    const float maskSupportThreshold = argc > 6 ? std::stof(argv[6]) : 0.20f;
    const int maskRadius = argc > 7 ? std::stoi(argv[7]) : 5;
    if (scoreThreshold < 0 || maskSupportThreshold < 0 || maskSupportThreshold > 1 || maskRadius < 0)
        throw std::runtime_error("invalid visibility thresholds");
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(maskPath.string(), cv::IMREAD_GRAYSCALE);
    if (image.empty() || mask.empty()) throw std::runtime_error("failed to read image or mask");
    if (mask.size() != image.size()) throw std::runtime_error("face mask resolution must match the RGB image");
    std::vector<cv::Point> pixels; cv::findNonZero(mask, pixels);
    if (pixels.empty()) throw std::runtime_error("target mask is empty");
    const cv::Rect face = cv::boundingRect(pixels);
    const int x = std::max(0, face.x - face.width), y = std::max(0, face.y - face.height / 2);
    cv::Rect2f body(static_cast<float>(x), static_cast<float>(y),
        static_cast<float>(std::min(image.cols - x, face.width * 3)),
        static_cast<float>(std::min(image.rows - y, face.height * 4)));

    Sapiens2PoseEstimator estimator; estimator.load(model.string());
    const auto result = estimator.infer(image, body);
    std::filesystem::create_directories(outDir);
    cv::Mat visibleVis = image.clone(), classifiedVis = image.clone();
    cv::rectangle(classifiedVis, result.personBox, {0, 255, 255}, 2);
    struct Classified { Sapiens2PoseEstimator::Keypoint point; float maskSupport; bool visible; };
    std::vector<Classified> facePoints;
    int visibleFace = 0, occludedFace = 0;
    for (const auto& p : result.keypoints) if (p.index >= 70) {
        const float support = maskSupportRatio(mask, p.position, maskRadius);
        const bool visible = p.score >= scoreThreshold && support >= maskSupportThreshold;
        facePoints.push_back({p, support, visible});
        if (visible) {
            cv::circle(visibleVis, p.position, 2, {0, 255, 0}, -1, cv::LINE_AA);
            cv::circle(classifiedVis, p.position, 2, {0, 255, 0}, -1, cv::LINE_AA);
            ++visibleFace;
        } else {
            const cv::Point q(cvRound(p.position.x), cvRound(p.position.y));
            cv::drawMarker(classifiedVis, q, {0, 0, 255}, cv::MARKER_TILTED_CROSS, 5, 1, cv::LINE_AA);
            ++occludedFace;
        }
    }
    cv::imwrite((outDir / "sapiens2_visible_face_keypoints.png").string(), visibleVis);
    cv::imwrite((outDir / "sapiens2_face_keypoints_visibility.png").string(), classifiedVis);
    cv::imwrite((outDir / "sapiens2_face_keypoints_onnx.png").string(), visibleVis);
    std::ofstream json(outDir / "sapiens2_pose_keypoints_onnx.json");
    json << std::setprecision(8);
    json << "{\n  \"person_box_xywh\": [" << body.x << ',' << body.y << ',' << body.width << ',' << body.height
         << "],\n  \"visibility\": {\"pose_score_threshold\":" << scoreThreshold
         << ",\"mask_support_threshold\":" << maskSupportThreshold << ",\"mask_radius_px\":" << maskRadius
         << "},\n  \"keypoints\": [\n";
    for (size_t i = 0; i < result.keypoints.size(); ++i) {
        const auto& p = result.keypoints[i];
        const float support = p.index >= 70 ? maskSupportRatio(mask, p.position, maskRadius) : 0.0f;
        const bool visible = p.index >= 70 && p.score >= scoreThreshold && support >= maskSupportThreshold;
        json << "    {\"index\":" << p.index << ",\"x\":" << p.position.x << ",\"y\":" << p.position.y
             << ",\"score\":" << p.score << ",\"mask_support\":" << support
             << ",\"visible\":" << (visible ? "true" : "false") << "}"
             << (i + 1 == result.keypoints.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";

    std::ofstream visibleJson(outDir / "sapiens2_visible_face_keypoints.json");
    visibleJson << std::setprecision(8)
                << "{\n  \"pose_score_threshold\":" << scoreThreshold
                << ",\n  \"mask_support_threshold\":" << maskSupportThreshold
                << ",\n  \"mask_radius_px\":" << maskRadius
                << ",\n  \"visible_keypoint_count\":" << visibleFace << ",\n  \"keypoints\": [\n";
    bool first = true;
    for (const auto& item : facePoints) if (item.visible) {
        if (!first) visibleJson << ",\n";
        first = false;
        const auto& p = item.point;
        visibleJson << "    {\"index\":" << p.index << ",\"x\":" << p.position.x << ",\"y\":"
                    << p.position.y << ",\"score\":" << p.score << ",\"mask_support\":" << item.maskSupport << "}";
    }
    visibleJson << "\n  ]\n}\n";
    std::cout << "Sapiens2 Pose ONNX completed; CUDA: " << (estimator.isUsingCuda() ? "yes" : "no")
              << "; visible face keypoints: " << visibleFace << "; rejected/occluded: " << occludedFace
              << "; output: " << outDir.string() << '\n';
    return 0;
} catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
