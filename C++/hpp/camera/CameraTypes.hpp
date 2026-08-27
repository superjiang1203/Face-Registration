#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct PointXYZ { float x{0}; float y{0}; float z{0}; };

struct CameraDeviceInfo {
    std::string backend;
    std::string model;
    std::string serialNumber;
    std::string ipAddress;
    std::string connectionType;
};

struct CameraDeviceSelector {
    std::string serialNumber;
    std::string ipAddress;
};

enum class FrameKind { ColorOnly, ColorDepth, ColorDepthPointCloud };

inline bool wantsDepthStream(FrameKind kind) { return kind != FrameKind::ColorOnly; }
inline bool wantsDepth(FrameKind kind) { return kind != FrameKind::ColorOnly; }
inline bool wantsPointCloud(FrameKind kind) { return kind == FrameKind::ColorDepthPointCloud; }

struct FrameMeta {
    std::uint64_t sequence{0};
    double fps{0.0};
    std::chrono::steady_clock::time_point captureTime{};
};

struct CameraFrame {
    FrameKind kind{FrameKind::ColorOnly};
    FrameMeta meta;
    cv::Mat color;
    cv::Mat depth;
    int pointCloudWidth{0};
    int pointCloudHeight{0};
    std::vector<PointXYZ> pointCloud;
};
