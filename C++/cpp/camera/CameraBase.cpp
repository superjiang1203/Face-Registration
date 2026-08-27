#include "camera/CameraBase.hpp"

#include <utility>

CameraBase::CameraBase(std::string name, Options options)
    : name_(std::move(name)), resolution_{options.width, options.height}, frameRate_(options.fps),
      pointCloudEveryNFrames_(options.pointCloudEveryNFrames), options_(options), frameKind_(options.frameKind) {}

CameraBase::~CameraBase() = default;
bool CameraBase::open() { if (opened_) return true; opened_ = openImpl(); return opened_; }
void CameraBase::close() noexcept { if (opened_) { closeImpl(); opened_ = false; } }
bool CameraBase::start() { return open(); }
void CameraBase::stop() noexcept { close(); }
std::optional<CameraFrame> CameraBase::capture() {
    if (!opened_ && !open()) return std::nullopt;
    auto frame = captureOnce();
    if (frame) frame->meta.sequence = ++sequence_;
    return frame;
}
void CameraBase::setFrameKind(FrameKind kind) { frameKind_ = kind; }
FrameKind CameraBase::getFrameKind() const { return frameKind_; }
double CameraBase::getFrameRate() const { return frameRate_; }
std::chrono::milliseconds CameraBase::captureTimeout() const { return std::chrono::milliseconds(options_.timeoutMs); }
