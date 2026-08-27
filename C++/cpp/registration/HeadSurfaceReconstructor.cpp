#include "registration/HeadSurfaceReconstructor.hpp"

#include <open3d/Open3D.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Point = Eigen::Vector3d;

struct VoxelKey {
    long long x{0}, y{0}, z{0};
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const noexcept {
        auto hash = static_cast<std::uint64_t>(key.x);
        hash ^= static_cast<std::uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::uint64_t>(key.z) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        return static_cast<std::size_t>(hash);
    }
};

VoxelKey voxelKey(const Point& point, double voxel) {
    return {static_cast<long long>(std::floor(point.x() / voxel)),
            static_cast<long long>(std::floor(point.y() / voxel)),
            static_cast<long long>(std::floor(point.z() / voxel))};
}

void voxelAverage(std::vector<Point>& points, double voxel) {
    if (!(voxel > 0.0) || points.empty()) return;
    struct Accumulator { Point sum{Point::Zero()}; std::size_t count{0}; };
    std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> grid;
    grid.reserve(points.size());
    for (const auto& point : points) {
        auto& value = grid[voxelKey(point, voxel)];
        value.sum += point;
        ++value.count;
    }
    std::vector<Point> reduced;
    reduced.reserve(grid.size());
    for (const auto& entry : grid) {
        if (entry.second.count > 0)
            reduced.push_back(entry.second.sum / static_cast<double>(entry.second.count));
    }
    points = std::move(reduced);
}

void filterVisibleFromMultipleViews(
    std::vector<Point>& points, int axisIndex, double voxelSizeMm, int viewCount) {
    if (points.empty() || viewCount <= 0) return;
    Point minimum = points.front(), maximum = points.front();
    for (const auto& point : points) {
        minimum = minimum.cwiseMin(point);
        maximum = maximum.cwiseMax(point);
    }
    const Point center = 0.5 * (minimum + maximum);
    const double baseMm = voxelSizeMm > 1e-6 ? voxelSizeMm : 1.0;
    const double binMm = std::clamp(2.0 * baseMm, 1.5, 3.0);
    const double bandMm = std::clamp(3.0 * baseMm, 2.0, 6.0);
    std::vector<std::uint8_t> keep(points.size(), 0);
    const double golden = 3.14159265358979323846 * (3.0 - std::sqrt(5.0));
    int usedViews = 0;
    const int maxAttempts = std::max(128, viewCount * 12);
    for (int attempt = 0; attempt < maxAttempts && usedViews < viewCount; ++attempt) {
        const double fraction = (static_cast<double>(attempt) + 0.5) / maxAttempts;
        const double wz = 1.0 - 2.0 * fraction;
        const double radius = std::sqrt(std::max(0.0, 1.0 - wz * wz));
        const double phi = golden * attempt;
        const Point direction(radius * std::cos(phi), radius * std::sin(phi), wz);
        if (direction[axisIndex] < 0.25 || direction.z() < -0.15) continue;
        ++usedViews;
        Point up = std::abs(wz) > 0.9 ? Point(0, 1, 0) : Point(0, 0, 1);
        Point horizontal = up.cross(direction).normalized();
        Point vertical = direction.cross(horizontal);
        struct DepthBin { double first{INFINITY}, second{INFINITY}; int firstCount{0}; };
        std::unordered_map<std::uint64_t, DepthBin> bins;
        bins.reserve(std::max<std::size_t>(1024, points.size() / 16));
        const auto keyFor = [&](const Point& point) {
            const Point local = point - center;
            const auto x = static_cast<std::uint32_t>(static_cast<long long>(std::floor(local.dot(horizontal) / binMm)));
            const auto y = static_cast<std::uint32_t>(static_cast<long long>(std::floor(local.dot(vertical) / binMm)));
            return (static_cast<std::uint64_t>(x) << 32) | y;
        };
        for (const auto& point : points) {
            const double depth = (point - center).dot(direction);
            auto& bin = bins[keyFor(point)];
            if (depth + 1e-6 < bin.first) {
                bin.second = bin.first; bin.first = depth; bin.firstCount = 1;
            } else if (std::abs(depth - bin.first) <= 1e-6) {
                ++bin.firstCount;
            } else if (depth < bin.second) {
                bin.second = depth;
            }
        }
        for (std::size_t index = 0; index < points.size(); ++index) {
            const double depth = (points[index] - center).dot(direction);
            const auto found = bins.find(keyFor(points[index]));
            if (found == bins.end()) continue;
            const auto& bin = found->second;
            const bool firstIsOutlier = bin.firstCount <= 1 && std::isfinite(bin.second) &&
                                        bin.second - bin.first > bandMm;
            const double visibleDepth = firstIsOutlier ? bin.second : bin.first;
            if (depth <= visibleDepth + bandMm) keep[index] = 1;
        }
    }
    std::vector<Point> visible;
    visible.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
        if (keep[index]) visible.push_back(points[index]);
    if (!visible.empty()) points = std::move(visible);
}

void filterIslands(std::vector<Point>& points, double voxel, bool multiView) {
    if (!(voxel > 0.0) || points.size() < 2000) return;
    std::unordered_map<VoxelKey, int, VoxelKeyHash> grid;
    std::vector<VoxelKey> keys;
    keys.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const auto key = voxelKey(points[index], voxel);
        keys.push_back(key);
        grid.emplace(key, static_cast<int>(index));
    }
    std::vector<int> component(points.size(), -1), sizes;
    std::vector<int> queue;
    for (std::size_t start = 0; start < points.size(); ++start) {
        if (component[start] >= 0) continue;
        const int id = static_cast<int>(sizes.size());
        int size = 0;
        queue.assign(1, static_cast<int>(start));
        component[start] = id;
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const int index = queue[cursor]; ++size;
            const auto key = keys[static_cast<std::size_t>(index)];
            for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const auto found = grid.find({key.x + dx, key.y + dy, key.z + dz});
                    if (found == grid.end() || component[static_cast<std::size_t>(found->second)] >= 0) continue;
                    component[static_cast<std::size_t>(found->second)] = id;
                    queue.push_back(found->second);
                }
        }
        sizes.push_back(size);
    }
    const int largest = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
    const int threshold = std::max(200, static_cast<int>(std::llround(
        (multiView ? 0.01 : 0.05) * static_cast<double>(largest))));
    std::vector<Point> filtered;
    filtered.reserve(points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
        if (sizes[static_cast<std::size_t>(component[index])] >= threshold) filtered.push_back(points[index]);
    if (!filtered.empty()) points = std::move(filtered);
}
} // namespace

HeadSurfaceReconstructor::Result HeadSurfaceReconstructor::reconstructRegistrationCloud(
    const std::string& stlPath, const std::string& outputPly, const Options& requested) {
    Result result;
    result.outputPly = outputPly;
    const auto totalStart = Clock::now();
    const auto timed = [&result](const std::string& name, const Clock::time_point& start) {
        result.stageTimingsMs.emplace_back(name,
            std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    };
    try {
        Options options = requested;
        options.cropThresholdRatio = std::clamp(options.cropThresholdRatio, 0.05, 0.95);
        options.voxelSizeMm = std::max(0.0, options.voxelSizeMm);
        if (!(options.modelUnitScale > 0.0) || !std::isfinite(options.modelUnitScale))
            throw std::runtime_error("model unit scale must be finite and positive");
        auto stage = Clock::now();
        auto mesh = open3d::io::CreateMeshFromFile(stlPath);
        if (!mesh || mesh->IsEmpty() || mesh->triangles_.empty())
            throw std::runtime_error("cannot read STL triangle mesh: " + stlPath);
        if (options.modelUnitScale != 1.0) mesh->Scale(options.modelUnitScale, Eigen::Vector3d::Zero());
        result.triangleCount = mesh->triangles_.size();
        timed("read_stl", stage);

        stage = Clock::now();
        std::vector<double> cdf;
        cdf.reserve(mesh->triangles_.size());
        double totalArea = 0.0;
        for (const auto& triangle : mesh->triangles_) {
            const auto& a = mesh->vertices_[triangle(0)];
            const auto& b = mesh->vertices_[triangle(1)];
            const auto& c = mesh->vertices_[triangle(2)];
            totalArea += 0.5 * (b - a).cross(c - a).norm();
            cdf.push_back(totalArea);
        }
        if (!(totalArea > 0.0)) throw std::runtime_error("STL has no positive-area triangles");
        const std::size_t initialSamples = options.targetPoints == 0 ? 200000 :
            static_cast<std::size_t>(std::clamp<long long>(
                static_cast<long long>(options.targetPoints) * 4LL, 20000LL, 1200000LL));
        result.initialSampleCount = initialSamples;
        std::mt19937 random(options.randomSeed);
        std::uniform_real_distribution<double> unit(0.0, 1.0), area(0.0, totalArea);
        std::vector<Point> points;
        points.reserve(initialSamples);
        for (std::size_t index = 0; index < initialSamples; ++index) {
            const auto iterator = std::lower_bound(cdf.begin(), cdf.end(), area(random));
            const std::size_t triangleIndex = std::min<std::size_t>(
                static_cast<std::size_t>(iterator - cdf.begin()), mesh->triangles_.size() - 1);
            const auto& triangle = mesh->triangles_[triangleIndex];
            const auto& p0 = mesh->vertices_[triangle(0)];
            const auto& p1 = mesh->vertices_[triangle(1)];
            const auto& p2 = mesh->vertices_[triangle(2)];
            const double root = std::sqrt(unit(random));
            const double second = unit(random);
            points.push_back((1.0 - root) * p0 + root * (1.0 - second) * p1 + root * second * p2);
        }
        timed("area_weighted_surface_sampling", stage);

        stage = Clock::now();
        const char axis = static_cast<char>(std::toupper(static_cast<unsigned char>(options.cropAxis)));
        const int axisIndex = axis == 'X' ? 0 : (axis == 'Z' ? 2 : 1);
        double minimum = points.front()[axisIndex], maximum = minimum;
        for (const auto& point : points) {
            minimum = std::min(minimum, point[axisIndex]);
            maximum = std::max(maximum, point[axisIndex]);
        }
        const double threshold = minimum + options.cropThresholdRatio * (maximum - minimum);
        points.erase(std::remove_if(points.begin(), points.end(), [&](const Point& point) {
            return point[axisIndex] > threshold;
        }), points.end());
        if (points.empty()) throw std::runtime_error("face-side ROI removed every sampled point");
        timed("face_roi_crop", stage);

        stage = Clock::now();
        voxelAverage(points, options.voxelSizeMm);
        timed("voxel_average", stage);
        if (options.enableMultiViewVisibility) {
            stage = Clock::now();
            filterVisibleFromMultipleViews(points, axisIndex, options.voxelSizeMm, options.multiViewCount);
            timed("multi_view_visibility", stage);
        }
        stage = Clock::now();
        filterIslands(points, options.voxelSizeMm, options.enableMultiViewVisibility);
        timed("island_filter", stage);
        if (options.targetPoints > 0 && points.size() > options.targetPoints) {
            const std::size_t step = std::max<std::size_t>(1, points.size() / options.targetPoints);
            std::vector<Point> bounded;
            bounded.reserve(options.targetPoints);
            for (std::size_t index = 0; index < points.size() && bounded.size() < options.targetPoints; index += step)
                bounded.push_back(points[index]);
            points = std::move(bounded);
        }
        if (points.empty()) throw std::runtime_error("surface preparation produced no points");

        stage = Clock::now();
        const std::filesystem::path outputPath(outputPly);
        if (!outputPath.parent_path().empty()) std::filesystem::create_directories(outputPath.parent_path());
        open3d::geometry::PointCloud cloud;
        cloud.points_ = std::move(points);
        if (!open3d::io::WritePointCloud(outputPly, cloud))
            throw std::runtime_error("cannot write reconstructed PLY: " + outputPly);
        result.outputPointCount = cloud.points_.size();
        timed("write_ply", stage);
        result.success = true;
        result.message = "ok";
    } catch (const std::exception& exception) {
        result.message = exception.what();
    }
    timed("total", totalStart);
    return result;
}
