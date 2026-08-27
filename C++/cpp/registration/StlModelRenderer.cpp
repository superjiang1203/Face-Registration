#include "registration/StlModelRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCleanPolyData.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkLight.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataNormals.h>
#include <vtkProperty.h>
#include <vtkSTLReader.h>
#include <vtkWindowToImageFilter.h>

namespace {

cv::Vec3d directionFromAzEl(double azimuthDeg, double elevationDeg) {
    const double azimuth = azimuthDeg * CV_PI / 180.0;
    const double elevation = elevationDeg * CV_PI / 180.0;
    cv::Vec3d direction(
        std::sin(azimuth) * std::cos(elevation),
        -std::cos(azimuth) * std::cos(elevation),
        std::sin(elevation));
    const double norm = cv::norm(direction);
    return norm < 1e-8 ? cv::Vec3d(0.0, -1.0, 0.0) : direction / norm;
}

cv::Mat vtkRgbImageToBgrMat(vtkImageData* image) {
    if (!image || !image->GetPointData() || !image->GetPointData()->GetScalars()) return {};
    int dimensions[3]{};
    image->GetDimensions(dimensions);
    const int width = dimensions[0];
    const int height = dimensions[1];
    vtkDataArray* scalars = image->GetPointData()->GetScalars();
    if (width <= 0 || height <= 0 || scalars->GetNumberOfComponents() < 3) return {};

    cv::Mat bgr(height, width, CV_8UC3);
    double tuple[4]{};
    for (int y = 0; y < height; ++y) {
        cv::Vec3b* row = bgr.ptr<cv::Vec3b>(height - 1 - y);
        for (int x = 0; x < width; ++x) {
            const vtkIdType index =
                static_cast<vtkIdType>(y) * static_cast<vtkIdType>(width) + static_cast<vtkIdType>(x);
            scalars->GetTuple(index, tuple);
            row[x] = cv::Vec3b(
                static_cast<unsigned char>(std::clamp(tuple[2], 0.0, 255.0)),
                static_cast<unsigned char>(std::clamp(tuple[1], 0.0, 255.0)),
                static_cast<unsigned char>(std::clamp(tuple[0], 0.0, 255.0)));
        }
    }
    return bgr;
}

cv::Mat vtkZImageToMat(vtkImageData* image) {
    if (!image || !image->GetPointData() || !image->GetPointData()->GetScalars()) return {};
    int dimensions[3]{};
    image->GetDimensions(dimensions);
    const int width = dimensions[0];
    const int height = dimensions[1];
    vtkDataArray* scalars = image->GetPointData()->GetScalars();
    if (width <= 0 || height <= 0 || scalars->GetNumberOfComponents() < 1) return {};

    cv::Mat zBuffer(height, width, CV_32FC1);
    for (int y = 0; y < height; ++y) {
        float* row = zBuffer.ptr<float>(height - 1 - y);
        for (int x = 0; x < width; ++x) {
            const vtkIdType index =
                static_cast<vtkIdType>(y) * static_cast<vtkIdType>(width) + static_cast<vtkIdType>(x);
            row[x] = static_cast<float>(scalars->GetComponent(index, 0));
        }
    }
    return zBuffer;
}

} // namespace

bool StlModelRenderer::loadStl(const std::filesystem::path& stlPath, std::string* error) {
    if (stlPath.empty() || !std::filesystem::is_regular_file(stlPath)) {
        if (error) *error = "STL file not found: " + stlPath.string();
        return false;
    }

    vtkNew<vtkSTLReader> reader;
    reader->SetFileName(stlPath.string().c_str());
    reader->Update();

    vtkNew<vtkCleanPolyData> clean;
    clean->SetInputConnection(reader->GetOutputPort());
    clean->Update();

    vtkNew<vtkPolyDataNormals> normals;
    normals->SetInputConnection(clean->GetOutputPort());
    normals->SplittingOff();
    normals->ConsistencyOn();
    normals->AutoOrientNormalsOn();
    normals->Update();

    vtkPolyData* output = normals->GetOutput();
    if (!output || output->GetNumberOfPoints() == 0) {
        if (error) *error = "Failed to load STL polydata points: " + stlPath.string();
        return false;
    }

    poly_ = vtkSmartPointer<vtkPolyData>::New();
    poly_->ShallowCopy(output);

    double bounds[6]{};
    poly_->GetBounds(bounds);
    center_ = cv::Vec3d(
        (bounds[0] + bounds[1]) * 0.5,
        (bounds[2] + bounds[3]) * 0.5,
        (bounds[4] + bounds[5]) * 0.5);
    const double dx = bounds[1] - bounds[0];
    const double dy = bounds[3] - bounds[2];
    const double dz = bounds[5] - bounds[4];
    diagonal_ = std::max(1.0, std::sqrt(dx * dx + dy * dy + dz * dz));
    return true;
}

bool StlModelRenderer::isLoaded() const {
    return poly_ != nullptr && poly_->GetNumberOfPoints() > 0;
}

cv::Vec3d StlModelRenderer::center() const {
    return center_;
}

double StlModelRenderer::diagonal() const {
    return diagonal_;
}

StlModelRenderer::RenderResult StlModelRenderer::render(
    const RenderOptions& options,
    std::string* error) const {
    RenderResult result;
    result.width = std::max(16, options.width);
    result.height = std::max(16, options.height);
    result.azimuthDeg = options.azimuthDeg;
    result.elevationDeg = options.elevationDeg;

    if (!isLoaded()) {
        if (error) *error = "STL renderer has no loaded model";
        return result;
    }

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(poly_);

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(0.86, 0.79, 0.74);
    actor->GetProperty()->SetAmbient(0.25);
    actor->GetProperty()->SetDiffuse(0.70);
    actor->GetProperty()->SetSpecular(0.18);
    actor->GetProperty()->SetSpecularPower(18.0);

    result.renderer = vtkSmartPointer<vtkRenderer>::New();
    result.renderer->SetBackground(0.08, 0.08, 0.08);
    result.renderer->AddActor(actor);

    vtkNew<vtkLight> keyLight;
    keyLight->SetLightTypeToSceneLight();
    keyLight->SetPosition(-300.0, -700.0, 500.0);
    keyLight->SetFocalPoint(center_[0], center_[1], center_[2]);
    keyLight->SetIntensity(1.0);
    result.renderer->AddLight(keyLight);

    vtkNew<vtkLight> fillLight;
    fillLight->SetLightTypeToSceneLight();
    fillLight->SetPosition(250.0, -450.0, 220.0);
    fillLight->SetFocalPoint(center_[0], center_[1], center_[2]);
    fillLight->SetIntensity(0.45);
    result.renderer->AddLight(fillLight);

    result.renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    result.renderWindow->SetOffScreenRendering(1);
    result.renderWindow->SetShowWindow(false);
    result.renderWindow->SetMultiSamples(0);
    result.renderWindow->AddRenderer(result.renderer);
    result.renderWindow->SetSize(result.width, result.height);

    vtkNew<vtkCamera> camera;
    camera->SetViewAngle(28.0);
    const cv::Vec3d direction = directionFromAzEl(options.azimuthDeg, options.elevationDeg);
    const double distance = 2.4 * diagonal_;
    const cv::Vec3d cameraPosition = center_ + direction * distance;
    camera->SetPosition(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
    camera->SetFocalPoint(center_[0], center_[1], center_[2]);
    camera->SetViewUp(0.0, 0.0, 1.0);
    if (std::abs(direction.dot(cv::Vec3d(0.0, 0.0, 1.0))) > 0.95) camera->SetViewUp(1.0, 0.0, 0.0);
    camera->SetClippingRange(0.1, std::max(10000.0, distance * 4.0));
    result.renderer->SetActiveCamera(camera);
    if (options.zoomFactor > 1.0) {
        camera->Dolly(options.zoomFactor);
        result.renderer->ResetCameraClippingRange();
    }

    result.renderWindow->Render();

    vtkNew<vtkWindowToImageFilter> rgbFilter;
    rgbFilter->SetInput(result.renderWindow);
    rgbFilter->SetInputBufferTypeToRGB();
    rgbFilter->ReadFrontBufferOff();
    rgbFilter->Update();
    result.bgr = vtkRgbImageToBgrMat(rgbFilter->GetOutput());

    vtkNew<vtkWindowToImageFilter> zFilter;
    zFilter->SetInput(result.renderWindow);
    zFilter->SetInputBufferTypeToZBuffer();
    zFilter->ReadFrontBufferOff();
    zFilter->Update();
    result.zBuffer = vtkZImageToMat(zFilter->GetOutput());

    if ((result.bgr.empty() || result.zBuffer.empty()) && error)
        *error = "Failed to capture off-screen STL render buffers";
    return result;
}

std::optional<cv::Vec3d> StlModelRenderer::worldPointFromPixel(
    const RenderResult& result,
    int u,
    int v,
    int searchRadius,
    cv::Point* sampledPixel) const {
    if (!result.renderer || result.zBuffer.empty()) return std::nullopt;
    const int width = result.zBuffer.cols;
    const int height = result.zBuffer.rows;
    if (width <= 0 || height <= 0) return std::nullopt;

    u = std::clamp(u, 0, width - 1);
    v = std::clamp(v, 0, height - 1);
    bool found = false;
    int bestU = u;
    int bestV = v;
    float zValue = 1.0f;
    int bestDistanceSquared = std::numeric_limits<int>::max();
    const int radius = std::max(0, searchRadius);
    for (int y = std::max(0, v - radius); y <= std::min(height - 1, v + radius); ++y) {
        const float* row = result.zBuffer.ptr<float>(y);
        for (int x = std::max(0, u - radius); x <= std::min(width - 1, u + radius); ++x) {
            const float z = row[x];
            if (!std::isfinite(z) || z <= 0.0f || z >= 0.999999f) continue;
            const int distanceSquared = (x - u) * (x - u) + (y - v) * (y - v);
            if (distanceSquared < bestDistanceSquared ||
                (distanceSquared == bestDistanceSquared && z < zValue)) {
                bestDistanceSquared = distanceSquared;
                bestU = x;
                bestV = y;
                zValue = z;
                found = true;
            }
        }
    }
    if (!found) return std::nullopt;
    if (sampledPixel) *sampledPixel = cv::Point(bestU, bestV);

    result.renderer->SetDisplayPoint(
        static_cast<double>(bestU),
        static_cast<double>(height - 1 - bestV),
        static_cast<double>(zValue));
    result.renderer->DisplayToWorld();
    const double* world = result.renderer->GetWorldPoint();
    if (!world || std::abs(world[3]) < 1e-8) return std::nullopt;
    return cv::Vec3d(world[0] / world[3], world[1] / world[3], world[2] / world[3]);
}
