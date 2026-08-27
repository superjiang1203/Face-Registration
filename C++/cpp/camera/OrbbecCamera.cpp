#include "camera/OrbbecCamera.hpp"

#include <libobsensor/ObSensor.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

OrbbecCamera::OrbbecCamera(const std::string& name) : OrbbecCamera(name, CameraBase::Options{}) {}
OrbbecCamera::OrbbecCamera(const std::string& name, const CameraBase::Options& options)
    : CameraBase(name, options) { setFrameKind(FrameKind::ColorOnly); }
OrbbecCamera::~OrbbecCamera() { stop(); }
void OrbbecCamera::setFrameKind(FrameKind) { CameraBase::setFrameKind(FrameKind::ColorOnly); }

bool OrbbecCamera::openImpl() {
    try {
        pipeline_ = std::make_unique<ob::Pipeline>();
        auto config = std::make_shared<ob::Config>();
        const auto width = static_cast<uint32_t>(resolution_.first > 0 ? resolution_.first : 1280);
        const auto height = static_cast<uint32_t>(resolution_.second > 0 ? resolution_.second : 720);
        const int fps = frameRate_ > 0 ? static_cast<int>(frameRate_) : 30;
        config->enableVideoStream(OB_STREAM_COLOR, width, height, fps, OB_FORMAT_BGR);
        pipeline_->start(config);
        resolution_ = {static_cast<int>(width), static_cast<int>(height)};
        frameRate_ = fps;
        const auto params = pipeline_->getCameraParam();
        const auto& k = params.rgbIntrinsic;
        const auto& d = params.rgbDistortion;
        cameraMatrix_ = (cv::Mat_<double>(3, 3) << k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1);
        distortionCoeffs_ = (cv::Mat_<double>(8, 1) << d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6);
        cv::initUndistortRectifyMap(cameraMatrix_, distortionCoeffs_, {}, cameraMatrix_,
                                    cv::Size(width, height), CV_32FC1, map1_, map2_);
        return true;
    } catch (const ob::Error&) { pipeline_.reset(); return false; }
}

void OrbbecCamera::closeImpl() noexcept {
    try { if (pipeline_) pipeline_->stop(); } catch (...) {}
    pipeline_.reset();
}

std::optional<CameraFrame> OrbbecCamera::captureOnce() {
    try {
        auto frames = pipeline_->waitForFrameset(static_cast<uint32_t>(captureTimeout().count()));
        auto color = frames ? frames->colorFrame() : nullptr;
        if (!color) return std::nullopt;
        cv::Mat raw(color->height(), color->width(), CV_8UC3, color->getData());
        CameraFrame frame;
        frame.kind = FrameKind::ColorOnly;
        if (map1_.empty()) frame.color = raw.clone();
        else cv::remap(raw, frame.color, map1_, map2_, cv::INTER_LINEAR);
        frame.meta.fps = frameRate_;
        frame.meta.captureTime = std::chrono::steady_clock::now();
        return frame;
    } catch (const ob::Error&) { return std::nullopt; }
}
