#include "camera/OrbbecDepthAlignedCamera.hpp"
#include "detection/FaceKeypointService.hpp"
#include "segmentation/Sapiens2Segmenter.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    fs::path model = "models/face_segmentation/sapiens2_seg/sapiens2_seg_0.4b_fp32.onnx";
    fs::path keypointModel;
    fs::path output = "output/sapiens2_capture";
    std::string cameraSn, cameraIp;
    int width = 1280, height = 800, fps = 30, warmup = 15;
    int minDepthMm = 600, maxDepthMm = 800;
    bool useCpu = false, useHrnet = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + a);
            return argv[i];
        };
        if (a == "--model") model = value();
        else if (a == "--keypoint-model") keypointModel = value();
        else if (a == "--output") output = value();
        else if (a == "--camera-sn") cameraSn = value();
        else if (a == "--camera-ip") cameraIp = value();
        else if (a == "--width") width = std::stoi(value());
        else if (a == "--height") height = std::stoi(value());
        else if (a == "--fps") fps = std::stoi(value());
        else if (a == "--warmup") warmup = std::stoi(value());
        else if (a == "--min-depth-mm") minDepthMm = std::stoi(value());
        else if (a == "--max-depth-mm") maxDepthMm = std::stoi(value());
        else if (a == "--cpu") useCpu = true;
        else if (a == "--no-hrnet") useHrnet = false;
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [--model FILE] [--output DIR] [--camera-sn SN]"
                         " [--camera-ip IP] [--width 1280] [--height 800] [--fps 30] [--warmup 15]"
                         " [--min-depth-mm N] [--max-depth-mm N] [--keypoint-model FILE] [--no-hrnet] [--cpu]\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + a);
    }

    try {
        if (minDepthMm < 0 || maxDepthMm <= minDepthMm || maxDepthMm > 65535)
            throw std::runtime_error("invalid depth range; require 0 <= min < max <= 65535 mm");
        if (keypointModel.empty())
            keypointModel = "models/face_keypoints/hrnet/hrnetv2_w18_wflw_256x256_heatmap.onnx";
        std::cout << "Loading Sapiens2: " << fs::absolute(model).string() << '\n';
        Sapiens2Segmenter segmenter({!useCpu, 0});
        segmenter.load(model.string());
        std::cout << "Provider: " << (segmenter.isUsingCuda() ? "CUDA" : "CPU") << '\n';

        CameraBase::Options cameraOptions;
        cameraOptions.width = width; cameraOptions.height = height; cameraOptions.fps = fps;
        cameraOptions.timeoutMs = 1000; cameraOptions.frameKind = FrameKind::ColorDepth;
        CameraDeviceSelector selector{cameraSn, cameraIp};
        OrbbecDepthAlignedCamera camera("orbbec_sapiens2", cameraOptions, selector);
        if (!camera.start()) throw std::runtime_error("failed to open/start Orbbec camera");

        std::optional<CameraFrame> frame;
        for (int n = 0; n <= warmup; ++n) {
            frame = camera.capture();
            if (!frame || frame->color.empty() || frame->depth.empty()) --n;
        }
        camera.stop();
        std::cout << "Captured aligned frame: " << frame->color.cols << 'x' << frame->color.rows << '\n';

        const auto begin = std::chrono::steady_clock::now();
        auto seg = segmenter.infer(frame->color);

        fs::create_directories(output);
        cv::Mat overlay;
        cv::addWeighted(frame->color, 0.62, seg.coloredLabels, 0.38, 0.0, overlay);
        cv::Mat faceOnly(frame->color.size(), frame->color.type(), cv::Scalar());
        frame->color.copyTo(faceOnly, seg.faceMask);
        cv::Mat faceDepth(frame->depth.size(), frame->depth.type(), cv::Scalar());
        frame->depth.copyTo(faceDepth, seg.faceMask);
        // The Orbbec camera class has already aligned depth to RGB and converted
        // it to millimetres. Exclude zero/invalid depth and keep [min, max].
        cv::Mat depthRangeMask = (frame->depth >= minDepthMm) & (frame->depth <= maxDepthMm);
        cv::Mat depthSeedMask;
        cv::bitwise_and(seg.faceMask, depthRangeMask, depthSeedMask);

        // Depth is used to select one complete face component, not to cut
        // individual pixels from a curved face. The nose/front surface may be
        // inside the requested range while cheeks and ears are several cm
        // farther away. Pick the face component containing the most in-range
        // seed pixels, then retain that entire semantic face component.
        // Eyeglass (class 2) can split one face into disconnected forehead,
        // cheek and ear islands. Close only the topology used for grouping;
        // the final pixels still come from the original face mask, so glasses
        // are not added back into the result.
        cv::Mat selectionTopology;
        const cv::Mat groupingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
        cv::morphologyEx(seg.faceMask, selectionTopology, cv::MORPH_CLOSE, groupingKernel);
        cv::Mat componentLabels, componentStats, componentCentroids;
        const int componentCount = cv::connectedComponentsWithStats(
            selectionTopology, componentLabels, componentStats, componentCentroids, 8, CV_32S);
        int selectedComponent = 0;
        int selectedSeedPixels = 0;
        for (int component = 1; component < componentCount; ++component) {
            cv::Mat componentMask = componentLabels == component;
            cv::Mat componentSeeds;
            cv::bitwise_and(componentMask, depthSeedMask, componentSeeds);
            const int seedPixels = cv::countNonZero(componentSeeds);
            if (seedPixels > selectedSeedPixels) {
                selectedSeedPixels = seedPixels;
                selectedComponent = component;
            }
        }
        // A few isolated depth samples may be sensor noise. Require a small
        // cluster before accepting the complete target face.
        constexpr int kMinimumDepthSeedPixels = 25;
        cv::Mat filteredFaceMask(seg.faceMask.size(), CV_8U, cv::Scalar(0));
        if (selectedComponent != 0 && selectedSeedPixels >= kMinimumDepthSeedPixels) {
            cv::Mat selectedTopology = componentLabels == selectedComponent;
            cv::bitwise_and(seg.faceMask, selectedTopology, filteredFaceMask);
        }

        // First pass above is only for locating the unique depth-selected
        // target. Run a second, aspect-preserving pass on an expanded target
        // ROI so the face occupies much more of the 768x1024 model input.
        // All allowed face fragments inside this one target ROI are retained;
        // they do not need to form one connected component (glasses/masks may
        // split forehead, cheeks and ears).
        if (cv::countNonZero(filteredFaceMask) > 0) {
            cv::Rect targetRoi = cv::boundingRect(filteredFaceMask);
            const int expandX = std::max(24, targetRoi.width / 3);
            const int expandY = std::max(24, targetRoi.height / 3);
            targetRoi = cv::Rect(targetRoi.x - expandX, targetRoi.y - expandY,
                targetRoi.width + 2 * expandX, targetRoi.height + 2 * expandY) &
                cv::Rect(0, 0, frame->color.cols, frame->color.rows);

            constexpr int kModelW = 768;
            constexpr int kModelH = 1024;
            const double roiScale = std::min(
                static_cast<double>(kModelW) / targetRoi.width,
                static_cast<double>(kModelH) / targetRoi.height);
            const int resizedW = std::max(1, cvRound(targetRoi.width * roiScale));
            const int resizedH = std::max(1, cvRound(targetRoi.height * roiScale));
            const int padLeft = (kModelW - resizedW) / 2;
            const int padTop = (kModelH - resizedH) / 2;
            cv::Mat resizedRoi;
            cv::resize(frame->color(targetRoi), resizedRoi, cv::Size(resizedW, resizedH), 0, 0, cv::INTER_LINEAR);
            // BGR values corresponding to the RGB normalization mean make the
            // padded input approximately zero after normalization.
            cv::Mat letterboxed(kModelH, kModelW, CV_8UC3, cv::Scalar(104, 116, 124));
            resizedRoi.copyTo(letterboxed(cv::Rect(padLeft, padTop, resizedW, resizedH)));
            const auto refined = segmenter.infer(letterboxed);
            const cv::Rect validModelRect(padLeft, padTop, resizedW, resizedH);

            cv::Mat mappedMask, mappedLabels, mappedColors;
            cv::resize(refined.faceMask(validModelRect), mappedMask, targetRoi.size(), 0, 0, cv::INTER_NEAREST);
            cv::resize(refined.labels(validModelRect), mappedLabels, targetRoi.size(), 0, 0, cv::INTER_NEAREST);
            cv::resize(refined.coloredLabels(validModelRect), mappedColors, targetRoi.size(), 0, 0, cv::INTER_NEAREST);

            seg.faceMask = cv::Mat(frame->color.size(), CV_8U, cv::Scalar(0));
            seg.labels = cv::Mat(frame->color.size(), CV_8U, cv::Scalar(0));
            seg.coloredLabels = cv::Mat(frame->color.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            mappedMask.copyTo(seg.faceMask(targetRoi));
            mappedLabels.copyTo(seg.labels(targetRoi));
            mappedColors.copyTo(seg.coloredLabels(targetRoi));
            filteredFaceMask = seg.faceMask.clone();
        }

        cv::Mat finalFaceMask = filteredFaceMask.clone();
        cv::Mat landmarkFaceContour(frame->color.size(), CV_8U, cv::Scalar(0));
        cv::Mat faceOcclusionMask(frame->color.size(), CV_8U, cv::Scalar(0));
        cv::Mat keypointDebug = frame->color.clone();
        bool neckTrimmed = false;
        if (useHrnet && selectedComponent != 0 && !filteredFaceMask.empty() && cv::countNonZero(filteredFaceMask) > 0) {
            const cv::Rect refinedBounds = cv::boundingRect(filteredFaceMask);
            const int left = refinedBounds.x;
            const int top = refinedBounds.y;
            const int compW = refinedBounds.width;
            // Face_Neck may extend far below the chin. HRNet-WFLW expects a
            // top-down face crop, so cap the unpadded crop height relative to
            // face width instead of feeding the entire long neck.
            const int compH = std::min(refinedBounds.height,
                std::max(compW, cvRound(1.35 * compW)));
            const int padX = std::max(12, compW / 8);
            const int padY = std::max(12, compH / 8);
            const cv::Rect imageBounds(0, 0, frame->color.cols, frame->color.rows);
            const cv::Rect cropRect = cv::Rect(left - padX, top - padY,
                compW + 2 * padX, compH + 2 * padY) & imageBounds;

            FaceKeypointService keypoints(keypointModel.string());
            std::string keypointError;
            const auto detection = keypoints.detect(frame->color(cropRect), &keypointError);
            if (detection.keypoints.size() >= 33) {
                std::vector<cv::Point> jaw;
                jaw.reserve(33);
                for (int i = 0; i <= 32; ++i) {
                    const auto& kp = detection.keypoints[static_cast<size_t>(i)];
                    jaw.emplace_back(cvRound(kp.x) + cropRect.x, cvRound(kp.y) + cropRect.y);
                    cv::circle(keypointDebug, jaw.back(), 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
                cv::polylines(keypointDebug, jaw, false, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

                // Remove the area below the jaw only between its two end
                // points. Keeping pixels outside that horizontal interval is
                // important because visible ears lie outside the jaw curve.
                std::vector<cv::Point> belowJaw = jaw;
                belowJaw.emplace_back(jaw.back().x, frame->color.rows - 1);
                belowJaw.emplace_back(jaw.front().x, frame->color.rows - 1);
                cv::Mat aboveJaw(frame->color.size(), CV_8U, cv::Scalar(0));
                aboveJaw.setTo(255);
                cv::fillPoly(aboveJaw, std::vector<std::vector<cv::Point>>{belowJaw}, cv::Scalar(0));
                const auto chin = std::max_element(jaw.begin(), jaw.end(),
                    [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
                const auto leftJaw = std::min_element(jaw.begin(), jaw.end(),
                    [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
                const auto rightJaw = std::max_element(jaw.begin(), jaw.end(),
                    [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
                const int chinY = chin != jaw.end() ? chin->y : std::max(jaw.front().y, jaw.back().y);
                const int earAllowance = std::max(8,
                    static_cast<int>(0.45 * std::max(0, chinY - std::min(leftJaw->y, rightJaw->y))));
                const int leftCutY = std::min(frame->color.rows, leftJaw->y + earAllowance);
                const int rightCutY = std::min(frame->color.rows, rightJaw->y + earAllowance);
                if (leftJaw->x > 0 && leftCutY < frame->color.rows)
                    aboveJaw(cv::Rect(0, leftCutY, leftJaw->x, frame->color.rows - leftCutY)).setTo(0);
                if (rightJaw->x + 1 < frame->color.cols && rightCutY < frame->color.rows)
                    aboveJaw(cv::Rect(rightJaw->x + 1, rightCutY,
                        frame->color.cols - rightJaw->x - 1, frame->color.rows - rightCutY)).setTo(0);

                // Build a complete face oval. WFLW 0..32 form the lower
                // contour; an oriented forehead arc is extrapolated using the
                // eyebrow band (33..50) and chin. This contour is independent
                // of segmentation connectivity, so a mask or glasses cannot
                // split away the forehead or cheeks.
                const cv::Point2f leftTemple = jaw.front();
                const cv::Point2f rightTemple = jaw.back();
                cv::Point2f faceCenter = (leftTemple + rightTemple) * 0.5f;
                cv::Point2f horizontal = rightTemple - leftTemple;
                const float templeDistance = std::max(1.0f, std::sqrt(horizontal.dot(horizontal)));
                horizontal *= 1.0f / templeDistance;
                cv::Point2f upward(-horizontal.y, horizontal.x);
                cv::Point2f browCenter(0.0f, 0.0f);
                for (int i = 33; i <= 50; ++i) {
                    const auto& kp = detection.keypoints[static_cast<size_t>(i)];
                    browCenter += cv::Point2f(kp.x + cropRect.x, kp.y + cropRect.y);
                }
                browCenter *= 1.0f / 18.0f;
                if ((browCenter - faceCenter).dot(upward) < 0.0f) upward *= -1.0f;
                const cv::Point2f chinPoint = jaw[16];
                const float browProjection = (browCenter - faceCenter).dot(upward);
                const float chinProjection = (chinPoint - faceCenter).dot(upward);
                const float foreheadHeight = std::max(0.35f * templeDistance,
                    browProjection + 0.45f * std::max(0.0f, browProjection - chinProjection));

                std::vector<cv::Point> completeFacePolygon = jaw;
                constexpr int kForeheadArcSamples = 32;
                for (int i = 0; i <= kForeheadArcSamples; ++i) {
                    const float theta = static_cast<float>(CV_PI) * i / kForeheadArcSamples;
                    const cv::Point2f p = faceCenter +
                        horizontal * (0.5f * templeDistance * std::cos(theta)) +
                        upward * (foreheadHeight * std::sin(theta));
                    completeFacePolygon.emplace_back(cvRound(p.x), cvRound(p.y));
                }
                cv::fillPoly(landmarkFaceContour,
                    std::vector<std::vector<cv::Point>>{completeFacePolygon}, cv::Scalar(255));

                // Subtract every non-face semantic pixel inside the theoretical
                // contour. This removes hair, eyeglasses, masks, hands and
                // other occluders rather than hallucinating hidden skin.
                cv::Mat notVisibleFace;
                cv::bitwise_not(filteredFaceMask, notVisibleFace);
                cv::bitwise_and(landmarkFaceContour, notVisibleFace, faceOcclusionMask);
                cv::bitwise_and(landmarkFaceContour, filteredFaceMask, finalFaceMask);

                // Ear boundaries from Sapiens2 are more reliable than an
                // extrapolated landmark oval. Add visible semantic pixels just
                // outside the left/right temple sides, while aboveJaw prevents
                // the neck from returning.
                cv::Mat outsideContour, semanticOutside, visibleEarCandidates;
                cv::bitwise_not(landmarkFaceContour, outsideContour);
                cv::bitwise_and(filteredFaceMask, outsideContour, semanticOutside);
                cv::bitwise_and(semanticOutside, aboveJaw, visibleEarCandidates);
                const int lowerTempleY = std::max(leftJaw->y, rightJaw->y);
                const int earBottomY = std::clamp(
                    chinY - cvRound(0.35 * std::max(0, chinY - lowerTempleY)), 0, frame->color.rows);
                if (earBottomY < frame->color.rows)
                    visibleEarCandidates.rowRange(earBottomY, frame->color.rows).setTo(0);
                cv::bitwise_or(finalFaceMask, visibleEarCandidates, finalFaceMask);

                cv::polylines(keypointDebug,
                    std::vector<std::vector<cv::Point>>{completeFacePolygon}, true,
                    cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
                neckTrimmed = true;
            } else {
                std::cerr << "Face keypoint neck trim skipped: "
                          << (keypointError.empty() ? "fewer than 33 WFLW keypoints" : keypointError) << '\n';
            }
        }
        cv::Mat filteredFace(frame->color.size(), frame->color.type(), cv::Scalar());
        frame->color.copyTo(filteredFace, finalFaceMask);
        cv::Mat filteredFaceDepth(frame->depth.size(), frame->depth.type(), cv::Scalar());
        frame->depth.copyTo(filteredFaceDepth, finalFaceMask);
        // Overlay only the final target. Pixels outside the selected face and
        // depth range remain identical to the original RGB image. Eyeglass
        // (class 2) is not part of faceMask and is therefore excluded.
        cv::Mat targetTint(frame->color.size(), frame->color.type(), cv::Scalar(0, 255, 0));
        cv::Mat blendedTarget;
        cv::addWeighted(frame->color, 0.55, targetTint, 0.45, 0.0, blendedTarget);
        cv::Mat targetOverlay = frame->color.clone();
        blendedTarget.copyTo(targetOverlay, finalFaceMask);
        cv::imwrite((output / "color.png").string(), frame->color);
        cv::imwrite((output / "depth_aligned_raw.png").string(), frame->depth);
        cv::imwrite((output / "labels.png").string(), seg.labels);
        cv::imwrite((output / "labels_color.png").string(), seg.coloredLabels);
        cv::imwrite((output / "face_mask.png").string(), seg.faceMask);
        cv::imwrite((output / "face_only.png").string(), faceOnly);
        cv::imwrite((output / "face_depth_raw.png").string(), faceDepth);
        cv::imwrite((output / "face_mask_depth_filtered.png").string(), finalFaceMask);
        cv::imwrite((output / "depth_gate_seed_mask.png").string(), depthSeedMask);
        cv::imwrite((output / "target_face_mask.png").string(), finalFaceMask);
        cv::imwrite((output / "landmark_face_contour.png").string(), landmarkFaceContour);
        cv::imwrite((output / "face_occlusion_mask.png").string(), faceOcclusionMask);
        cv::imwrite((output / "face_keypoints_jaw.png").string(), keypointDebug);
        cv::imwrite((output / "face_only_depth_filtered.png").string(), filteredFace);
        cv::imwrite((output / "face_depth_filtered_raw.png").string(), filteredFaceDepth);
        cv::imwrite((output / "target_overlay_rgb.png").string(), targetOverlay);
        // overlay.png is the user-facing final result. Keep the unfiltered
        // 29-class visualization under an explicit diagnostic filename.
        cv::imwrite((output / "overlay.png").string(), targetOverlay);
        cv::imwrite((output / "all_classes_overlay.png").string(), overlay);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - begin).count();
        std::cout << "Two-pass segmentation completed in " << ms << " ms; depth gate: ["
                  << minDepthMm << ", " << maxDepthMm << "] mm; seed pixels: "
                  << selectedSeedPixels << "; complete target pixels: "
                  << cv::countNonZero(finalFaceMask) << "; contour source: "
                  << (neckTrimmed ? "HRNet-WFLW" : "Sapiens2 Seg only") << "; outputs: "
                  << fs::absolute(output).string() << '\n';
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n'; return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
