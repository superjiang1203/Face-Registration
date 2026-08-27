#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera/CameraTypes.hpp"

class FacePointCloudCropService {
  public:
    struct CropRoi {
        cv::Rect colorRoi;
        cv::Rect pointCloudRoi;
    };

    struct OrganizedCrop {
        CropRoi roi;
        std::vector<PointXYZ> pointCloud;
        size_t validPointCount{0};
    };

    struct Options {
        bool enabled{false};
        int roiMargin{0};
        int stride{2};
        size_t maxPoints{50000};

        bool filterOutliers{false};
        int outlierMinPoints{300};
        float outlierMaxZDeviationMm{250.0f};
        float outlierMaxXYRadiusMm{200.0f};

        bool keepLargestComponent{false};
        int componentMinPoints{300};
        float componentMaxNeighborDzMm{80.0f};
    };

    FacePointCloudCropService();
    explicit FacePointCloudCropService(Options opts);

    std::optional<CropRoi> faceCropRoi(
        const cv::Size& colorSize, int pointCloudWidth, int pointCloudHeight, const cv::Rect& faceBbox) const;
    std::vector<PointXYZ> cropFacePointCloud(const CameraFrame& frame, const cv::Size& colorSize, const cv::Rect& faceBbox) const;
    std::optional<OrganizedCrop> cropFacePointCloudOrganized(
        const CameraFrame& frame, const cv::Size& colorSize, const cv::Rect& faceBbox) const;

  private:
    static cv::Rect clampRect(const cv::Rect& r, int w, int h);
    static cv::Rect mapRect(const cv::Rect& r, int srcW, int srcH, int dstW, int dstH);

    Options options_;
};
