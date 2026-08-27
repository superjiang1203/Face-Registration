#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class HeadSurfaceReconstructor {
public:
    struct Options {
        char cropAxis{'Y'};
        double cropThresholdRatio{0.40};
        std::size_t targetPoints{100000};
        double voxelSizeMm{1.0};
        bool enableMultiViewVisibility{false};
        int multiViewCount{48};
        unsigned int randomSeed{1337};
        double modelUnitScale{1.0};
    };

    struct Result {
        bool success{false};
        std::string message;
        std::string outputPly;
        std::size_t triangleCount{0};
        std::size_t initialSampleCount{0};
        std::size_t outputPointCount{0};
        std::vector<std::pair<std::string, double>> stageTimingsMs;
    };

    // Preserves the original registration-surface algorithm:
    // area-weighted sampling -> face-side ROI -> voxel averaging -> optional
    // multi-view visibility -> island filtering -> bounded PLY output.
    static Result reconstructRegistrationCloud(
        const std::string& stlPath,
        const std::string& outputPly,
        const Options& options);
};
