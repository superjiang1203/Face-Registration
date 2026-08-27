#pragma once

#include "camera/CameraTypes.hpp"

#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>

class CameraBase {
public:
    struct Options {
        int width{1280};
        int height{800};
        double fps{30.0};
        int timeoutMs{100};
        int pointCloudEveryNFrames{1};
        FrameKind frameKind{FrameKind::ColorDepthPointCloud};
    };

    explicit CameraBase(std::string name, Options options = {});
    virtual ~CameraBase();

    bool open();
    void close() noexcept;
    bool start();
    void stop() noexcept;
    std::optional<CameraFrame> capture();

    virtual void setFrameKind(FrameKind kind);
    FrameKind getFrameKind() const;
    double getFrameRate() const;
    std::chrono::milliseconds captureTimeout() const;

protected:
    virtual bool openImpl() = 0;
    virtual void closeImpl() noexcept = 0;
    virtual std::optional<CameraFrame> captureOnce() = 0;

    std::string name_;
    std::pair<int, int> resolution_;
    double frameRate_;
    int pointCloudEveryNFrames_;
    cv::Mat cameraMatrix_;
    cv::Mat distortionCoeffs_;

private:
    Options options_;
    FrameKind frameKind_;
    bool opened_{false};
    std::uint64_t sequence_{0};
};
