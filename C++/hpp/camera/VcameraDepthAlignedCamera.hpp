#pragma once

#include "camera/CameraBase.hpp"

#include <memory>
#include <string>

class VcameraDepthAlignedCamera final : public CameraBase {
public:
    using DeviceSelector = CameraDeviceSelector;
    struct CaptureSettings {
        bool laserAutoControl{false};
        int laserPower{25};
    };

    explicit VcameraDepthAlignedCamera(const std::string& name = "VcameraDepthAligned");
    VcameraDepthAlignedCamera(const std::string& name, const CameraBase::Options& options,
                             DeviceSelector selector = {}, CaptureSettings settings = {});
    ~VcameraDepthAlignedCamera() override;

    static std::vector<CameraDeviceInfo> discoverDevices();

    void setFrameKind(FrameKind kind) override;

protected:
    bool openImpl() override;
    void closeImpl() noexcept override;
    std::optional<CameraFrame> captureOnce() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DeviceSelector selector_;
    CaptureSettings settings_;
};
