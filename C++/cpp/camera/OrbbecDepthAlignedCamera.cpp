#include "camera/OrbbecDepthAlignedCamera.hpp"

#include <libobsensor/ObSensor.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <utility>

OrbbecDepthAlignedCamera::OrbbecDepthAlignedCamera(const std::string& name)
    : OrbbecDepthAlignedCamera(name, CameraBase::Options{}) {}
OrbbecDepthAlignedCamera::OrbbecDepthAlignedCamera(const std::string& name, const CameraBase::Options& options)
    : OrbbecDepthAlignedCamera(name, options, {}) {}
OrbbecDepthAlignedCamera::OrbbecDepthAlignedCamera(const std::string& name, const CameraBase::Options& options,
                                                   CameraDeviceSelector selector)
    : CameraBase(name, options), selector_(std::move(selector)) {}
OrbbecDepthAlignedCamera::~OrbbecDepthAlignedCamera() { stop(); }
void OrbbecDepthAlignedCamera::setFrameKind(FrameKind kind) { CameraBase::setFrameKind(kind); }

std::vector<CameraDeviceInfo> OrbbecDepthAlignedCamera::discoverDevices() {
    std::vector<CameraDeviceInfo> result;
    try {
        ob::Context context;
        context.enableNetDeviceEnumeration(true);
        const auto devices = context.queryDeviceList();
        for (uint32_t i = 0; i < devices->getCount(); ++i) {
            result.push_back({"orbbec", devices->getName(i), devices->getSerialNumber(i),
                              devices->getIpAddress(i), devices->getConnectionType(i)});
        }
    } catch (const ob::Error&) {}
    return result;
}

bool OrbbecDepthAlignedCamera::openImpl() {
    try {
        context_ = std::make_unique<ob::Context>();
        context_->enableNetDeviceEnumeration(true);
        const auto devices = context_->queryDeviceList();
        if (devices->getCount() == 0) return false;
        int selectedIndex = -1;
        for (uint32_t i = 0; i < devices->getCount(); ++i) {
            const bool snMatches = selector_.serialNumber.empty() || selector_.serialNumber == devices->getSerialNumber(i);
            const bool ipMatches = selector_.ipAddress.empty() || selector_.ipAddress == devices->getIpAddress(i);
            if (snMatches && ipMatches) {
                if (selectedIndex >= 0 && selector_.serialNumber.empty() && selector_.ipAddress.empty()) {
                    std::cerr << "multiple Orbbec cameras found; specify --camera-sn or --camera-ip\n";
                    closeImpl();
                    return false;
                }
                selectedIndex = static_cast<int>(i);
            }
        }
        if (selectedIndex < 0) return false;
        std::cout << "selected Orbbec: model=" << devices->getName(selectedIndex)
                  << ", SN=" << devices->getSerialNumber(selectedIndex)
                  << ", IP=" << devices->getIpAddress(selectedIndex) << '\n';
        pipeline_ = std::make_unique<ob::Pipeline>(devices->getDevice(selectedIndex));
        auto config = std::make_shared<ob::Config>();
        const auto width = static_cast<uint32_t>(resolution_.first > 0 ? resolution_.first : 1280);
        const auto height = static_cast<uint32_t>(resolution_.second > 0 ? resolution_.second : 800);
        const int fps = frameRate_ > 0 ? static_cast<int>(frameRate_) : 30;
        config->enableVideoStream(OB_STREAM_COLOR, width, height, fps, OB_FORMAT_BGR);
        config->enableVideoStream(OB_STREAM_DEPTH, width, height, fps, OB_FORMAT_Y16);
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
        pipeline_->enableFrameSync();
        pipeline_->start(config);
        alignFilter_ = std::make_shared<ob::Align>(OB_STREAM_COLOR);
        alignFilter_->setMatchTargetResolution(true);
        const auto params = pipeline_->getCameraParam();
        const auto& k = params.rgbIntrinsic;
        cameraMatrix_ = (cv::Mat_<double>(3, 3) << k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1);
        resolution_ = {static_cast<int>(width), static_cast<int>(height)};
        frameRate_ = fps;
        return true;
    } catch (const ob::Error& e) { std::cerr << "Orbbec open failed: " << e.what() << '\n'; closeImpl(); return false; }
}

void OrbbecDepthAlignedCamera::closeImpl() noexcept {
    alignFilter_.reset(); pointCloudFilter_.reset();
    try { if (pipeline_) pipeline_->stop(); } catch (...) {}
    pipeline_.reset();
    context_.reset();
}

std::optional<CameraFrame> OrbbecDepthAlignedCamera::captureOnce() {
    try {
        auto frames = pipeline_->waitForFrameset(static_cast<uint32_t>(captureTimeout().count()));
        if (!frames || !alignFilter_) return std::nullopt;
        auto aligned = alignFilter_->process(frames);
        auto set = aligned ? aligned->as<ob::FrameSet>() : nullptr;
        auto color = set ? set->colorFrame() : nullptr;
        auto depth = set ? set->depthFrame() : nullptr;
        if (!color || !depth) return std::nullopt;
        CameraFrame frame;
        frame.kind = getFrameKind();
        frame.color = cv::Mat(color->height(), color->width(), CV_8UC3, color->getData()).clone();
        const cv::Mat rawDepth(depth->height(), depth->width(), CV_16U, depth->getData());
        rawDepth.convertTo(frame.depth, CV_16U, static_cast<double>(depth->getValueScale()));
        if (wantsPointCloud(frame.kind)) {
            frame.pointCloudWidth = frame.depth.cols;
            frame.pointCloudHeight = frame.depth.rows;
            frame.pointCloud.resize(static_cast<size_t>(frame.depth.total()));
            const double fx = cameraMatrix_.at<double>(0, 0), fy = cameraMatrix_.at<double>(1, 1);
            const double cx = cameraMatrix_.at<double>(0, 2), cy = cameraMatrix_.at<double>(1, 2);
            for (int y = 0; y < frame.depth.rows; ++y) {
                const auto* row = frame.depth.ptr<uint16_t>(y);
                for (int x = 0; x < frame.depth.cols; ++x) {
                    const float z = static_cast<float>(row[x]);
                    auto& p = frame.pointCloud[static_cast<size_t>(y * frame.depth.cols + x)];
                    if (z > 0) p = {static_cast<float>((x - cx) * z / fx), static_cast<float>((y - cy) * z / fy), z};
                }
            }
        }
        frame.meta.fps = frameRate_;
        frame.meta.captureTime = std::chrono::steady_clock::now();
        return frame;
    } catch (const ob::Error&) { return std::nullopt; }
}
