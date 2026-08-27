// Offline point-cloud registration command-line application.
#include "registration/PointCloudRegistration.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <source.ply> <prepared_model.ply> <output.ply>"
                     " [voxel_mm] [min_fitness] [max_inlier_rmse_mm]"
                     " [expected_4x4.txt] [initial_4x4.txt]\n";
        return EXIT_FAILURE;
    }
    RegistrationOptions options;
    if (std::filesystem::path(argv[2]).extension() == ".stl") {
        std::cerr << "direct STL targets are no longer accepted; use the cached PLY produced by "
                     "HeadSurfaceReconstructor\n";
        return EXIT_FAILURE;
    }
    if (argc >= 5) options.voxelSizeMm = std::stod(argv[4]);
    const double minFitness = argc >= 6 ? std::stod(argv[5]) : 0.90;
    const double maxInlierRmse = argc >= 7 ? std::stod(argv[6]) : 3.0;
    constexpr double maxSourceRmse = 6.0;
    constexpr double maxSourceP95 = 10.0;
    options.initialFitnessToSkipGlobal = minFitness;
    options.initialRmseToSkipGlobalMm = maxInlierRmse;
    options.initialSourceToTargetRmseToSkipGlobalMm = maxSourceRmse;
    options.initialSourceToTargetP95ToSkipGlobalMm = maxSourceP95;
    const auto readMatrix = [](const std::string& path) -> std::optional<Eigen::Matrix4d> {
        Eigen::Matrix4d matrix;
        std::ifstream input(path);
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (!(input >> matrix(row, col))) return std::nullopt;
        return matrix;
    };
    std::optional<Eigen::Matrix4d> initial;
    if (argc == 9) {
        initial = readMatrix(argv[8]);
        if (!initial) {
            std::cerr << "cannot parse initial transform: " << argv[8] << '\n';
            return EXIT_FAILURE;
        }
    }
    const auto result = PointCloudRegistration(options).alignFiles(argv[1], argv[2], argv[3], initial);
    if (!result.success) {
        std::cerr << "registration failed: " << result.message << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "initialization: " << result.initializationMethod << '\n'
              << "fitness: " << result.fitness << '\n'
              << "inlier RMSE: " << result.inlierRmseMm << " mm\n"
              << "source->target mean/RMSE/median/P95/max: "
              << result.sourceToTargetMeanMm << " / " << result.sourceToTargetRmseMm << " / "
              << result.sourceToTargetMedianMm << " / " << result.sourceToTargetP95Mm << " / "
              << result.sourceToTargetMaxMm << " mm\n"
              << "symmetric mean/RMSE/P95: " << result.symmetricMeanMm << " / "
              << result.symmetricRmseMm << " / " << result.symmetricP95Mm << " mm\n"
              << "transformation:\n" << result.transformation << '\n';
    std::optional<double> translationError;
    std::optional<double> rotationErrorDeg;
    if (argc >= 8) {
        const auto expectedValue = readMatrix(argv[7]);
        if (!expectedValue) {
            std::cerr << "cannot parse expected transform: " << argv[7] << '\n';
            return EXIT_FAILURE;
        }
        const Eigen::Matrix4d& expected = *expectedValue;
        translationError =
            (result.transformation.block<3, 1>(0, 3) - expected.block<3, 1>(0, 3)).norm();
        const Eigen::Matrix3d rotationDelta =
            result.transformation.block<3, 3>(0, 0) * expected.block<3, 3>(0, 0).transpose();
        const double cosine = std::clamp((rotationDelta.trace() - 1.0) * 0.5, -1.0, 1.0);
        rotationErrorDeg = std::acos(cosine) * 180.0 / 3.14159265358979323846;
        std::cout << "ground-truth translation error: " << *translationError << " mm\n"
                  << "ground-truth rotation error: " << *rotationErrorDeg << " deg\n";
    }
    const bool passed = result.fitness >= minFitness &&
                        result.inlierRmseMm <= maxInlierRmse &&
                        result.sourceToTargetRmseMm <= maxSourceRmse &&
                        result.sourceToTargetP95Mm <= maxSourceP95;
    std::cout << "quality gate: " << (passed ? "PASS" : "FAIL")
              << " (fitness >= " << minFitness
              << ", inlier RMSE <= " << maxInlierRmse
              << " mm, source RMSE <= " << maxSourceRmse
              << " mm, source P95 <= " << maxSourceP95 << " mm)\n";
    const std::filesystem::path outputPath(argv[3]);
    std::ofstream metrics(outputPath.parent_path() /
        (outputPath.stem().string() + "_metrics.txt"), std::ios::app);
    metrics << "min_fitness=" << minFitness << '\n'
            << "max_inlier_rmse_mm=" << maxInlierRmse << '\n'
            << "max_source_to_target_rmse_mm=" << maxSourceRmse << '\n'
            << "max_source_to_target_p95_mm=" << maxSourceP95 << '\n'
            << "quality_gate=" << (passed ? "PASS" : "FAIL") << '\n';
    if (translationError) metrics << "ground_truth_translation_error_mm=" << *translationError << '\n';
    if (rotationErrorDeg) metrics << "ground_truth_rotation_error_deg=" << *rotationErrorDeg << '\n';
    return passed ? EXIT_SUCCESS : 2;
}
