#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "camera/CameraTypes.hpp"

// Texture-independent validation and ordering for face point-cloud candidates.
// Landmark confidence is intentionally absent: landmarks are evaluated only
// after a physical face candidate has been selected.
class FaceCandidateSelectionPolicy {
  public:
    struct Options {
        // Low-confidence proposals are retained for the physical 3D gates.
        double minDetectionScore{0.10};
        std::size_t minPointCount{1000};
        double minPointDensity{0.15};
        double minDepthMm{200.0};
        double maxDepthMm{2000.0};
        // The head model is captured around 70 cm.  Ranking is a soft Gaussian
        // preference centred on this distance rather than a strict
        // "closest wins" rule, so a nearer torso/limb false positive cannot
        // steal the lock from the intended head.  Surface quality and detector
        // confidence only break near-ties.
        double preferredDepthMm{700.0};
        double depthPreferenceSigmaMm{250.0};
        double minPlanarSpanMm{45.0};
        double maxPlanarSpanMm{220.0};
        double minDepthReliefMm{12.0};
        double maxDepthReliefMm{100.0};
        double minDepthMadMm{2.0};
        double maxDepthMadMm{30.0};
        int minKeypointPoseInliers{5};
        double maxKeypointPoseRmseMm{6.0};
        double minKeypointMeanScore{0.50};
    };

    struct Metrics {
        bool accepted{false};
        std::string rejectionReason;
        double detectionScore{0.0};
        std::size_t pointCount{0};
        std::size_t roiPixelArea{0};
        double pointDensity{0.0};
        double medianDepthMm{0.0};
        double depthMadMm{0.0};
        double depthP10Mm{0.0};
        double depthP90Mm{0.0};
        double depthReliefMm{0.0};
        double xSpanMm{0.0};
        double ySpanMm{0.0};
        double planarMinSpanMm{0.0};
        double planarMaxSpanMm{0.0};
    };

    static Metrics evaluate(
        double detectionScore,
        const std::vector<PointXYZ>& points,
        std::size_t roiPixelArea,
        const Options& options = {});

    // Returns accepted physical-face indices ordered primarily by detector
    // confidence. Depth preference and surface quality only break ties after
    // evaluate() has applied the hard physical/depth gates.
    static std::vector<std::size_t> rankNearestFirst(
        const std::vector<Metrics>& candidates,
        const Options& options = {});

    // Landmarks may provide an initial pose only after a candidate passes the
    // physical gates. Low-texture detections must pass both network
    // confidence and 3D geometric-consistency gates before they are trusted.
    static bool isReliableKeypointPose(
        bool poseSuccess,
        int inlierCount,
        double poseRmseMm,
        double meanKeypointScore,
        const Options& options = {});
};
