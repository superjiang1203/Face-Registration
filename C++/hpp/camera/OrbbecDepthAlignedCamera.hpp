#pragma once

#include "camera/CameraBase.hpp"
#include <cstdint>
#include <memory>

namespace ob {
class Pipeline;
class Align;
class PointCloudFilter;
class Context;
}

class OrbbecDepthAlignedCamera : public CameraBase {
public:
    explicit OrbbecDepthAlignedCamera(const std::string& name = "OrbbecDepthAligned");
    OrbbecDepthAlignedCamera(const std::string& name, const CameraBase::Options& options);
    OrbbecDepthAlignedCamera(const std::string& name, const CameraBase::Options& options,
                             CameraDeviceSelector selector);
    ~OrbbecDepthAlignedCamera() override;
    static std::vector<CameraDeviceInfo> discoverDevices();
    void setFrameKind(FrameKind kind) override;

protected:
    bool openImpl() override;
    void closeImpl() noexcept override;
    std::optional<CameraFrame> captureOnce() override;

private:
    std::unique_ptr<ob::Pipeline> pipeline_;
    std::unique_ptr<ob::Context> context_;
    std::shared_ptr<ob::Align> alignFilter_;
    std::shared_ptr<ob::PointCloudFilter> pointCloudFilter_;
    int colorFormat_{0};
    bool usingHwD2CAlign_{false};
    std::uint64_t pointCloudFrameCounter_{0};
    CameraDeviceSelector selector_;
};
