#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

class vtkPolyData;

class StlModelRenderer {
  public:
    struct RenderOptions {
        int width{1280};
        int height{960};
        double azimuthDeg{0.0};
        double elevationDeg{8.0};
        double zoomFactor{1.8};
    };

    struct RenderResult {
        cv::Mat bgr;
        cv::Mat zBuffer;
        int width{0};
        int height{0};
        double azimuthDeg{0.0};
        double elevationDeg{0.0};
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkRenderWindow> renderWindow;
    };

    bool loadStl(const std::filesystem::path& stlPath, std::string* error = nullptr);
    bool isLoaded() const;
    cv::Vec3d center() const;
    double diagonal() const;

    RenderResult render(const RenderOptions& options, std::string* error = nullptr) const;
    std::optional<cv::Vec3d> worldPointFromPixel(
        const RenderResult& result,
        int u,
        int v,
        int searchRadius,
        cv::Point* sampledPixel = nullptr) const;

  private:
    vtkSmartPointer<vtkPolyData> poly_;
    cv::Vec3d center_{0.0, 0.0, 0.0};
    double diagonal_{1.0};
};
