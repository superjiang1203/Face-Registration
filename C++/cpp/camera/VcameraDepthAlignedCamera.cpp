#include "camera/VcameraDepthAlignedCamera.hpp"

#include <vcamera/camera.h>
#include <vcamera/commonUtils.h>
#include <vcamera/imageProc.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

struct VcameraDepthAlignedCamera::Impl {
    vcamera::Camera camera;
    std::mutex mutex;
    std::condition_variable ready;
    std::optional<CameraFrame> latest;
    bool connected{false};
    bool capturing{false};
    bool reportedFrameFormat{false};
};

namespace {
bool ok(vcamera::CameraApiStatus status, const char* operation) {
    if (status.IsSuccess()) return true;
    std::cerr << "Vcamera " << operation << " failed: " << status.message() << '\n';
    return false;
}

bool configureImageMode(vcamera::Camera& camera, const std::string& sensor,
                        int requestedWidth, int requestedHeight) {
    std::vector<vcamera::ImageMode> modes;
    auto status = camera.GetImageModes(sensor, modes);
    if (!status.IsSuccess()) {
        std::cerr << "Vcamera cannot query " << sensor << " modes: " << status.message() << '\n';
        return false;
    }
    std::cout << "Vcamera " << sensor << " modes:";
    const vcamera::ImageMode* selected = nullptr;
    for (const auto& mode : modes) {
        std::cout << ' ' << mode.width << 'x' << mode.height;
        if (!selected && mode.width == requestedWidth && mode.height == requestedHeight) selected = &mode;
    }
    std::cout << '\n';
    if (!selected) {
        std::cerr << "Vcamera " << sensor << " does not support requested "
                  << requestedWidth << 'x' << requestedHeight << '\n';
        return false;
    }
    return ok(camera.SetImageMode(sensor, *selected), "set image mode");
}

bool configureLaser(vcamera::Camera& camera, bool autoControl, int requestedPower) {
    vcamera::Feature autoFeature;
    auto status = camera.GetFeature("LaserPowerAutoCtrl", autoFeature);
    if (!status.IsSuccess() || !ok(autoFeature.SetValue(autoControl), "set LaserPowerAutoCtrl")) return false;

    vcamera::Feature powerFeature;
    status = camera.GetFeature("LightController0/LightBrightness", powerFeature);
    if (!status.IsSuccess()) {
        std::cerr << "Vcamera get laser power feature failed: " << status.message() << '\n';
        return false;
    }
    vcamera::Int64Range range;
    if (!ok(powerFeature.GetRange(range), "get laser power range")) return false;
    if (requestedPower < range.minValue || requestedPower > range.maxValue ||
        ((requestedPower - range.minValue) % std::max<int64_t>(1, range.step)) != 0) {
        std::cerr << "Vcamera laser power " << requestedPower << " is outside valid range ["
                  << range.minValue << ", " << range.maxValue << "], step=" << range.step << '\n';
        return false;
    }
    if (!ok(powerFeature.SetValue(requestedPower), "set laser power")) return false;

    vcamera::Value autoValue, powerValue;
    if (!ok(autoFeature.GetValue(autoValue), "read LaserPowerAutoCtrl") ||
        !ok(powerFeature.GetValue(powerValue), "read laser power")) return false;
    std::cout << "Vcamera laser: auto_control=" << vcamera::GetValueString(autoValue)
              << ", power=" << vcamera::GetValueString(powerValue)
              << ", valid_range=[" << range.minValue << ',' << range.maxValue << "]\n";
    return true;
}
}

VcameraDepthAlignedCamera::VcameraDepthAlignedCamera(const std::string& name)
    : VcameraDepthAlignedCamera(name, CameraBase::Options{}, {}) {}

VcameraDepthAlignedCamera::VcameraDepthAlignedCamera(const std::string& name,
                                                     const CameraBase::Options& options,
                                                     DeviceSelector selector, CaptureSettings settings)
    : CameraBase(name, options), impl_(std::make_unique<Impl>()), selector_(std::move(selector)),
      settings_(settings) {}

VcameraDepthAlignedCamera::~VcameraDepthAlignedCamera() { stop(); }

std::vector<CameraDeviceInfo> VcameraDepthAlignedCamera::discoverDevices() {
    std::vector<CameraDeviceInfo> result;
    if (!ok(vcamera::CameraUtils::Init(true), "initialization")) return result;
    for (const auto& device : vcamera::CameraUtils::DiscoverCameras()) {
        result.push_back({"vcamera", device.model, device.serial_number, device.network_info.ip,
                          device.interface_info.interface_type == vcamera::InterfaceType::Network ? "Network" : "USB"});
    }
    return result;
}

void VcameraDepthAlignedCamera::setFrameKind(FrameKind kind) { CameraBase::setFrameKind(kind); }

bool VcameraDepthAlignedCamera::openImpl() {
    try {
        if (!ok(vcamera::CameraUtils::Init(true), "initialization")) return false;
        const auto devices = vcamera::CameraUtils::DiscoverCameras();
        if (devices.empty()) {
            std::cerr << "Vcamera found no cameras\n";
            return false;
        }

        vcamera::CameraInfo selectedDevice;
        if (!selector_.serialNumber.empty() || !selector_.ipAddress.empty()) {
            bool found = false;
            for (const auto& device : devices) {
                const bool snMatches = selector_.serialNumber.empty() || device.serial_number == selector_.serialNumber;
                const bool ipMatches = selector_.ipAddress.empty() || device.network_info.ip == selector_.ipAddress;
                if (snMatches && ipMatches) { selectedDevice = device; found = true; break; }
            }
            if (!found) {
                std::cerr << "Requested Vcamera was not discovered (SN=" << selector_.serialNumber
                          << ", IP=" << selector_.ipAddress << ")\n";
                return false;
            }
            impl_->camera = vcamera::CameraFactory::GetCamera(selector_.serialNumber, selector_.ipAddress);
        } else {
            if (devices.size() != 1) {
                std::cerr << "Vcamera discovered " << devices.size()
                          << " cameras; specify --camera-sn or --camera-ip\n";
                return false;
            }
            selectedDevice = devices.front();
            impl_->camera = vcamera::CameraFactory::GetCameraByCameraInfo(selectedDevice);
        }

        std::cout << "selected Vcamera: model=" << selectedDevice.model
                  << ", SN=" << selectedDevice.serial_number
                  << ", IP=" << selectedDevice.network_info.ip << '\n';

        if (!ok(impl_->camera.Connect(), "connect")) return false;
        impl_->connected = true;
        impl_->camera.StopCapture();

        if (!configureLaser(impl_->camera, settings_.laserAutoControl, settings_.laserPower)) {
            closeImpl();
            return false;
        }

        if (!configureImageMode(impl_->camera, vcamera::SensorType::Depth,
                                resolution_.first, resolution_.second) ||
            !configureImageMode(impl_->camera, vcamera::SensorType::Texture,
                                resolution_.first, resolution_.second)) {
            closeImpl();
            return false;
        }

        vcamera::Feature acquisitionMode;
        if (!ok(impl_->camera.GetFeature("AcquisitionMode", acquisitionMode), "get AcquisitionMode") ||
            !ok(acquisitionMode.SetValue(2), "set continuous acquisition")) {
            closeImpl();
            return false;
        }
        if (!ok(impl_->camera.SetSensorEnabled(vcamera::SensorType::Depth, true), "enable depth") ||
            !ok(impl_->camera.SetSensorEnabled(vcamera::SensorType::Texture, true), "enable texture") ||
            !ok(impl_->camera.SetMapDepthToTextureEnabled(true), "enable depth-to-texture alignment")) {
            closeImpl();
            return false;
        }

        impl_->camera.RegisterFrameSetCallback([this](const vcamera::FrameSet& frameset) {
            const auto colorImage = frameset.GetImage(vcamera::SensorType::Texture);
            const auto depthImage = frameset.GetImage(vcamera::SensorType::Depth);
            if (!colorImage.IsValid() || !depthImage.IsValid() ||
                colorImage.pixel_format() != vcamera::PixelFormat::BGR24 ||
                depthImage.pixel_format() != vcamera::PixelFormat::DEPTH16) return;
            if (colorImage.width() != depthImage.width() || colorImage.height() != depthImage.height()) return;

            CameraFrame frame;
            frame.kind = getFrameKind();
            frame.color = cv::Mat(colorImage.height(), colorImage.width(), CV_8UC3,
                                  const_cast<uint8_t*>(colorImage.data())).clone();

            const cv::Mat rawDepth(depthImage.height(), depthImage.width(), CV_16U,
                                   const_cast<uint8_t*>(depthImage.data()));
            rawDepth.convertTo(frame.depth, CV_16U, depthImage.scale_unit());
            frame.depth.setTo(0, rawDepth == 65535);

            if (wantsPointCloud(frame.kind)) {
                const auto cloud = vcamera::ImageProc::DepthImageToPointCloud(depthImage, nullptr);
                const auto vendorPoints = cloud.GetPoints();
                frame.pointCloudWidth = depthImage.width();
                frame.pointCloudHeight = depthImage.height();
                frame.pointCloud.resize(vendorPoints.size());
                for (std::size_t i = 0; i < vendorPoints.size(); ++i) {
                    frame.pointCloud[i] = {vendorPoints[i].x, vendorPoints[i].y, vendorPoints[i].z};
                }
                if (!impl_->reportedFrameFormat) {
                    std::cout << "Vcamera aligned frame: color=" << colorImage.width() << 'x' << colorImage.height()
                              << ", depth=" << depthImage.width() << 'x' << depthImage.height()
                              << ", depth_scale=" << depthImage.scale_unit()
                              << ", points=" << vendorPoints.size() << '\n';
                    impl_->reportedFrameFormat = true;
                }
            }
            frame.meta.fps = frameRate_;
            frame.meta.captureTime = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->latest = std::move(frame);
            }
            impl_->ready.notify_one();
        });

        if (!ok(impl_->camera.StartCapture(), "start capture")) {
            closeImpl();
            return false;
        }
        impl_->capturing = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Vcamera open failed: " << e.what() << '\n';
        closeImpl();
        return false;
    }
}

void VcameraDepthAlignedCamera::closeImpl() noexcept {
    try {
        if (impl_->capturing) impl_->camera.StopCapture();
        if (impl_->connected) impl_->camera.Disconnect();
    } catch (...) {}
    impl_->capturing = false;
    impl_->connected = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->latest.reset();
    }
    impl_->ready.notify_all();
}

std::optional<CameraFrame> VcameraDepthAlignedCamera::captureOnce() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (!impl_->ready.wait_for(lock, captureTimeout(), [this] { return impl_->latest.has_value(); })) {
        return std::nullopt;
    }
    auto frame = std::move(impl_->latest);
    impl_->latest.reset();
    return frame;
}
