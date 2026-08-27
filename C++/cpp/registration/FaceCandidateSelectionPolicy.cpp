#include "registration/FaceCandidateSelectionPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdlib>

namespace {

double parseDoubleEnv(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) return fallback;
    return parsed;
}

bool finitePoint(const PointXYZ& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && point.z > 0.0f;
}

double percentileSorted(const std::vector<double>& values, double percentile) {
    if (values.empty()) return 0.0;
    const double position = std::clamp(percentile, 0.0, 1.0) *
                            static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(low);
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

} // namespace

FaceCandidateSelectionPolicy::Metrics FaceCandidateSelectionPolicy::evaluate(
    double detectionScore,
    const std::vector<PointXYZ>& points,
    std::size_t roiPixelArea,
    const Options& options) {
    Metrics metrics;
    metrics.detectionScore = detectionScore;
    metrics.roiPixelArea = roiPixelArea;

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> depths;
    xs.reserve(points.size());
    ys.reserve(points.size());
    depths.reserve(points.size());
    for (const auto& point : points) {
        if (!finitePoint(point)) continue;
        xs.push_back(static_cast<double>(point.x));
        ys.push_back(static_cast<double>(point.y));
        depths.push_back(static_cast<double>(point.z));
    }
    metrics.pointCount = depths.size();
    metrics.pointDensity = roiPixelArea > 0
        ? static_cast<double>(metrics.pointCount) / static_cast<double>(roiPixelArea)
        : 0.0;

    const auto reject = [&metrics](const std::string& reason) {
        metrics.accepted = false;
        metrics.rejectionReason = reason;
        return metrics;
    };
    if (!std::isfinite(detectionScore) || detectionScore < options.minDetectionScore)
        return reject("detection_score");
    if (metrics.pointCount < options.minPointCount)
        return reject("point_count");
    if (roiPixelArea == 0 || metrics.pointDensity < options.minPointDensity)
        return reject("point_density");

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(depths.begin(), depths.end());
    metrics.medianDepthMm = percentileSorted(depths, 0.50);
    metrics.depthP10Mm = percentileSorted(depths, 0.10);
    metrics.depthP90Mm = percentileSorted(depths, 0.90);
    metrics.depthReliefMm = metrics.depthP90Mm - metrics.depthP10Mm;
    metrics.xSpanMm = percentileSorted(xs, 0.95) - percentileSorted(xs, 0.05);
    metrics.ySpanMm = percentileSorted(ys, 0.95) - percentileSorted(ys, 0.05);
    metrics.planarMinSpanMm = std::min(metrics.xSpanMm, metrics.ySpanMm);
    metrics.planarMaxSpanMm = std::max(metrics.xSpanMm, metrics.ySpanMm);

    std::vector<double> deviations;
    deviations.reserve(depths.size());
    for (const double depth : depths)
        deviations.push_back(std::abs(depth - metrics.medianDepthMm));
    std::sort(deviations.begin(), deviations.end());
    metrics.depthMadMm = percentileSorted(deviations, 0.50);

    if (metrics.medianDepthMm < options.minDepthMm ||
        metrics.medianDepthMm > options.maxDepthMm)
        return reject("depth_range");
    if (metrics.planarMinSpanMm < options.minPlanarSpanMm ||
        metrics.planarMaxSpanMm > options.maxPlanarSpanMm)
        return reject("physical_span");
    if (metrics.depthReliefMm < options.minDepthReliefMm ||
        metrics.depthReliefMm > options.maxDepthReliefMm)
        return reject("depth_relief");
    if (metrics.depthMadMm < options.minDepthMadMm ||
        metrics.depthMadMm > options.maxDepthMadMm)
        return reject("depth_mad");

    metrics.accepted = true;
    metrics.rejectionReason = "ok";
    return metrics;
}

std::vector<std::size_t> FaceCandidateSelectionPolicy::rankNearestFirst(
    const std::vector<Metrics>& candidates,
    const Options& options) {
    const double preferredDepthMm = std::max(
        1.0, parseDoubleEnv("FACE_SELECT_PREFERRED_DEPTH_MM", options.preferredDepthMm));
    const double sigmaMm = std::max(
        1.0, parseDoubleEnv("FACE_SELECT_DEPTH_SIGMA_MM", options.depthPreferenceSigmaMm));
    const double twoSigmaSquared = 2.0 * sigmaMm * sigmaMm;

    std::vector<std::size_t> remaining;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (candidates[index].accepted &&
            std::isfinite(candidates[index].medianDepthMm) &&
            candidates[index].medianDepthMm > 0.0) {
            remaining.push_back(index);
        }
    }

    std::sort(remaining.begin(), remaining.end(), [&](std::size_t left, std::size_t right) {
        // Physical/depth constraints are hard gates in evaluate(). Among all
        // accepted faces, detector confidence is the primary identity score.
        if (candidates[left].detectionScore != candidates[right].detectionScore)
            return candidates[left].detectionScore > candidates[right].detectionScore;
        const double leftDelta = candidates[left].medianDepthMm - preferredDepthMm;
        const double rightDelta = candidates[right].medianDepthMm - preferredDepthMm;
        // Soft Gaussian preference around the expected capture distance.  A
        // nearer candidate still ranks first when the depth difference is
        // large, but surface quality and confidence now resolve near-ties.
        const double leftDepthScore =
            std::exp(-(leftDelta * leftDelta) / twoSigmaSquared);
        const double rightDepthScore =
            std::exp(-(rightDelta * rightDelta) / twoSigmaSquared);
        const double epsilon = 1e-9;
        if (std::abs(leftDepthScore - rightDepthScore) > epsilon)
            return leftDepthScore > rightDepthScore;
        if (candidates[left].pointDensity != candidates[right].pointDensity)
            return candidates[left].pointDensity > candidates[right].pointDensity;
        if (candidates[left].pointCount != candidates[right].pointCount)
            return candidates[left].pointCount > candidates[right].pointCount;
        if (candidates[left].planarMinSpanMm != candidates[right].planarMinSpanMm)
            return candidates[left].planarMinSpanMm > candidates[right].planarMinSpanMm;
        return left < right;
    });
    return remaining;
}

bool FaceCandidateSelectionPolicy::isReliableKeypointPose(
    bool poseSuccess,
    int inlierCount,
    double poseRmseMm,
    double meanKeypointScore,
    const Options& options) {
    return poseSuccess &&
           inlierCount >= std::max(3, options.minKeypointPoseInliers) &&
           std::isfinite(poseRmseMm) && poseRmseMm >= 0.0 &&
           poseRmseMm <= options.maxKeypointPoseRmseMm &&
           std::isfinite(meanKeypointScore) &&
           meanKeypointScore >= options.minKeypointMeanScore;
}
