#include "camera/OrbbecDepthAlignedCamera.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--camera-sn SN] [--camera-ip IP] [--width W] [--height H]"
              << " [--fps F] [--output output/manual_roi.txt]\n";
}

cv::Mat normalizeDepthForDisplay(const cv::Mat& depth) {
    if (depth.empty() || depth.type() != CV_16U) {
        return cv::Mat();
    }
    cv::Mat depth8u;
    cv::normalize(depth, depth8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat color;
    cv::applyColorMap(depth8u, color, cv::COLORMAP_JET);
    return color;
}

cv::Mat normalizeDepthForImage(const cv::Mat& depth) {
    if (depth.empty()) {
        return cv::Mat();
    }
    cv::Mat validMask = depth > 0;
    if (cv::countNonZero(validMask) == 0) {
        return cv::Mat(depth.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    }
    cv::Mat normalized;
    cv::normalize(depth, normalized, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::Mat color;
    cv::applyColorMap(normalized, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), ~validMask);
    return color;
}

void saveRoi(const std::filesystem::path& path, const cv::Rect& roi, const cv::Mat& colorFrame, const cv::Mat& depthFrame) {
    if (roi.width <= 0 || roi.height <= 0) {
        throw std::runtime_error("invalid ROI: width and height must be positive");
    }
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open ROI output path: " + path.string());
    }
    out << roi.x << ' ' << roi.y << ' ' << roi.width << ' ' << roi.height << '\n';

    const cv::Rect safeRoi = roi & cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
    if (!safeRoi.empty()) {
        cv::imwrite((path.parent_path() / "manual_roi_color.png").string(), colorFrame(safeRoi).clone());
        if (!depthFrame.empty()) {
            const cv::Mat roiDepth = depthFrame(safeRoi).clone();
            if (!roiDepth.empty()) {
                cv::imwrite((path.parent_path() / "manual_roi_depth_raw.png").string(), roiDepth);
                const cv::Mat depthDisplay = normalizeDepthForImage(roiDepth);
                if (!depthDisplay.empty()) {
                    cv::imwrite((path.parent_path() / "manual_roi_depth.png").string(), depthDisplay);
                }
            }
        }
    }

    std::cout << "saved ROI to " << path.string() << " -> x=" << roi.x
              << " y=" << roi.y << " width=" << roi.width
              << " height=" << roi.height << '\n';
}

} // namespace

int main(int argc, char** argv) {
    std::string cameraSn;
    std::string cameraIp;
    int width = 1280;
    int height = 800;
    int fps = 30;
    std::filesystem::path outputPath = "output/manual_roi.txt";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--camera-sn" && i + 1 < argc) cameraSn = argv[++i];
        else if (arg == "--camera-ip" && i + 1 < argc) cameraIp = argv[++i];
        else if (arg == "--width" && i + 1 < argc) width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) height = std::stoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) fps = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) outputPath = argv[++i];
        else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            printUsage(argv[0]);
            return 1;
        }
    }

    CameraBase::Options options;
    options.width = width;
    options.height = height;
    options.fps = fps;
    options.frameKind = FrameKind::ColorDepthPointCloud;

    CameraDeviceSelector selector;
    selector.serialNumber = cameraSn;
    selector.ipAddress = cameraIp;

    OrbbecDepthAlignedCamera camera("orbbec_preview", options, selector);
    if (!camera.open()) {
        std::cerr << "failed to open Orbbec camera; check --camera-sn/--camera-ip\n";
        return 2;
    }
    if (!camera.start()) {
        std::cerr << "failed to start Orbbec camera stream\n";
        return 3;
    }

    cv::namedWindow("Orbbec Color", cv::WINDOW_NORMAL);
    cv::namedWindow("Orbbec Depth", cv::WINDOW_NORMAL);
    std::cout << "drag a rectangle in the Color window, then press Space/Enter to confirm; press q to quit\n";

    struct SelectionState {
        bool drawing{false};
        cv::Point start{-1, -1};
        cv::Point current{-1, -1};
        cv::Rect roi;
    } state;
    auto onMouse = [](int event, int x, int y, int, void* userData) {
        auto* s = static_cast<SelectionState*>(userData);
        if (event == cv::EVENT_LBUTTONDOWN) {
            s->drawing = true;
            s->start = cv::Point(x, y);
            s->current = cv::Point(x, y);
            s->roi = cv::Rect();
        } else if (event == cv::EVENT_MOUSEMOVE && s->drawing) {
            s->current = cv::Point(x, y);
            const int x0 = std::min(s->start.x, s->current.x);
            const int y0 = std::min(s->start.y, s->current.y);
            const int x1 = std::max(s->start.x, s->current.x);
            const int y1 = std::max(s->start.y, s->current.y);
            s->roi = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        } else if (event == cv::EVENT_LBUTTONUP && s->drawing) {
            s->drawing = false;
            const int x0 = std::min(s->start.x, x);
            const int y0 = std::min(s->start.y, y);
            const int x1 = std::max(s->start.x, x);
            const int y1 = std::max(s->start.y, y);
            s->roi = cv::Rect(x0, y0, x1 - x0, y1 - y0);
            s->current = cv::Point(x, y);
        }
    };
    cv::setMouseCallback("Orbbec Color", onMouse, &state);

    try {
        while (true) {
            const auto frame = camera.capture();
            if (!frame || frame->color.empty() || frame->depth.empty()) {
                cv::waitKey(5);
                continue;
            }
            cv::Mat color = frame->color.clone();
            cv::Mat depth = normalizeDepthForDisplay(frame->depth);
            if (!depth.empty()) {
                cv::imshow("Orbbec Depth", depth);
            }
            if (state.roi.area() > 0) {
                cv::rectangle(color, state.roi, cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Orbbec Color", color);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
            if (key == 'c' || key == 'C') {
                state.roi = cv::Rect();
                state.start = cv::Point(-1, -1);
                state.current = cv::Point(-1, -1);
                state.drawing = false;
                continue;
            }
            if ((key == 13 || key == 32) && state.roi.area() > 0) {
                saveRoi(outputPath, state.roi, color, frame->depth);
                break;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "preview error: " << ex.what() << '\n';
        camera.stop();
        return 4;
    }

    camera.stop();
    cv::destroyAllWindows();
    return 0;
}
