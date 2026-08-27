#pragma once

#include "camera/CameraBase.hpp"
#include <memory>
#include <opencv2/core.hpp>

namespace ob { class Pipeline; }

class OrbbecCamera : public CameraBase {
public:
    explicit OrbbecCamera(const std::string& name = "Orbbec");
    OrbbecCamera(const std::string& name, const CameraBase::Options& options);
    ~OrbbecCamera() override;
    void setFrameKind(FrameKind kind) override;

protected:
    bool openImpl() override;
    void closeImpl() noexcept override;
    std::optional<CameraFrame> captureOnce() override;

private:
    std::unique_ptr<ob::Pipeline> pipeline_;
    cv::Mat map1_;
    cv::Mat map2_;
};
