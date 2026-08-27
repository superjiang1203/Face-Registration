#include "registration/HeadSurfaceCache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {
using Digest = std::array<std::uint8_t, 32>;

constexpr const char* kSurfaceFileName = "head_surface.ply";
constexpr const char* kManifestFileName = "head_surface_cache.txt";
constexpr const char* kSourceSnapshotFileName = "source.stl";
constexpr const char* kWorkspacePrefix = ".head-surface-work-";
constexpr const char* kLockFileName = ".head_surface_cache.lock";
constexpr const char* kLegacyV2DirectoryName = "head_surface_v2";
constexpr const char* kLegacySurfaceFileName = "surface.ply";
constexpr const char* kLegacyManifestFileName = "manifest.txt";
constexpr const char* kLegacyV1FileName = "head_head_surface_v1_scale_1.ply";
constexpr std::size_t kIoBufferBytes = 1024 * 1024;
constexpr std::uintmax_t kMaximumManifestBytes = 64 * 1024;
constexpr std::size_t kMaximumPlyHeaderBytes = 1024 * 1024;

std::uint32_t rotateRight(std::uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
}

class Sha256 final {
public:
    Sha256()
        : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const std::uint8_t* data, std::size_t size) {
        if (size == 0) return;
        totalBytes_ += static_cast<std::uint64_t>(size);
        while (size > 0) {
            const std::size_t count = std::min(size, block_.size() - bufferedBytes_);
            std::memcpy(block_.data() + bufferedBytes_, data, count);
            data += count;
            size -= count;
            bufferedBytes_ += count;
            if (bufferedBytes_ == block_.size()) {
                transform(block_.data());
                bufferedBytes_ = 0;
            }
        }
    }

    void update(const std::vector<std::uint8_t>& bytes) {
        update(bytes.data(), bytes.size());
    }

    Digest finish() {
        const std::uint64_t bitCount = totalBytes_ * 8U;
        block_[bufferedBytes_++] = 0x80U;
        if (bufferedBytes_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(bufferedBytes_),
                      block_.end(), std::uint8_t{0});
            transform(block_.data());
            bufferedBytes_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(bufferedBytes_),
                  block_.begin() + 56, std::uint8_t{0});
        for (int index = 0; index < 8; ++index)
            block_[56 + index] = static_cast<std::uint8_t>(bitCount >> (56 - 8 * index));
        transform(block_.data());

        Digest digest{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            digest[word * 4] = static_cast<std::uint8_t>(state_[word] >> 24);
            digest[word * 4 + 1] = static_cast<std::uint8_t>(state_[word] >> 16);
            digest[word * 4 + 2] = static_cast<std::uint8_t>(state_[word] >> 8);
            digest[word * 4 + 3] = static_cast<std::uint8_t>(state_[word]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16; ++index) {
            const std::size_t offset = index * 4;
            schedule[index] = (static_cast<std::uint32_t>(block[offset]) << 24) |
                              (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                              (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                              static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < schedule.size(); ++index) {
            const std::uint32_t s0 = rotateRight(schedule[index - 15], 7) ^
                                     rotateRight(schedule[index - 15], 18) ^
                                     (schedule[index - 15] >> 3);
            const std::uint32_t s1 = rotateRight(schedule[index - 2], 17) ^
                                     rotateRight(schedule[index - 2], 19) ^
                                     (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }

        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t upperSigma1 =
                rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 =
                h + upperSigma1 + choice + constants[index] + schedule[index];
            const std::uint32_t upperSigma0 =
                rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = upperSigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::uint64_t totalBytes_{0};
    std::size_t bufferedBytes_{0};
};

std::string digestHex(const Digest& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.resize(digest.size() * 2);
    for (std::size_t index = 0; index < digest.size(); ++index) {
        output[index * 2] = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 0x0fU];
    }
    return output;
}

bool decodeDigest(const std::string& text, Digest& digest) {
    if (text.size() != digest.size() * 2) return false;
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const int high = nibble(text[index * 2]);
        const int low = nibble(text[index * 2 + 1]);
        if (high < 0 || low < 0) return false;
        digest[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

struct DigestWorkResult {
    bool success{false};
    Digest digest{};
    std::uintmax_t byteCount{0};
    std::string message;
};

DigestWorkResult digestFileInternal(
    const std::filesystem::path& inputPath,
    const std::filesystem::path* copyPath = nullptr) {
    DigestWorkResult result;
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        result.message = "cannot open file for SHA-256: " + inputPath.string();
        return result;
    }
    std::ofstream copy;
    if (copyPath) {
        copy.open(*copyPath, std::ios::binary | std::ios::trunc);
        if (!copy) {
            result.message = "cannot create source snapshot: " + copyPath->string();
            return result;
        }
    }

    Sha256 sha;
    std::vector<char> buffer(kIoBufferBytes);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            sha.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                       static_cast<std::size_t>(count));
            result.byteCount += static_cast<std::uintmax_t>(count);
            if (copyPath) {
                copy.write(buffer.data(), count);
                if (!copy) {
                    result.message = "cannot write source snapshot: " + copyPath->string();
                    return result;
                }
            }
        }
    }
    if (input.bad() || (input.fail() && !input.eof())) {
        result.message = "cannot finish reading file for SHA-256: " + inputPath.string();
        return result;
    }
    if (copyPath) {
        copy.flush();
        if (!copy) {
            result.message = "cannot flush source snapshot: " + copyPath->string();
            return result;
        }
        copy.close();
        if (!copy) {
            result.message = "cannot close source snapshot: " + copyPath->string();
            return result;
        }
    }
    result.digest = sha.finish();
    result.success = true;
    result.message = "ok";
    return result;
}

void appendU8(std::vector<std::uint8_t>& output, std::uint8_t value) {
    output.push_back(value);
}

void appendU32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
        output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendString(std::vector<std::uint8_t>& output, const std::string& value) {
    appendU64(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

std::uint64_t doubleBits(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "64-bit double is required");
    static_assert(std::numeric_limits<double>::is_iec559, "IEEE-754 double is required");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool validateRecipe(const HeadSurfaceCache::Recipe& recipe, std::string& message) {
    if (recipe.algorithmVersion.empty() || recipe.algorithmVersion.size() > 128) {
        message = "algorithm version must contain 1 to 128 characters";
        return false;
    }
    for (unsigned char character : recipe.algorithmVersion) {
        const bool allowed = (character >= 'a' && character <= 'z') ||
                             (character >= 'A' && character <= 'Z') ||
                             (character >= '0' && character <= '9') ||
                             character == '.' || character == '_' || character == '-';
        if (!allowed) {
            message = "algorithm version may contain only ASCII letters, digits, '.', '_' and '-'";
            return false;
        }
    }
    const auto& options = recipe.reconstruction;
    if (!std::isfinite(options.cropThresholdRatio)) {
        message = "crop threshold ratio must be finite";
        return false;
    }
    if (!std::isfinite(options.voxelSizeMm)) {
        message = "voxel size must be finite";
        return false;
    }
    if (!std::isfinite(options.modelUnitScale) || !(options.modelUnitScale > 0.0)) {
        message = "model unit scale must be finite and positive";
        return false;
    }
    message = "ok";
    return true;
}

std::vector<std::uint8_t> recipeBytes(const HeadSurfaceCache::Recipe& recipe) {
    const auto& options = recipe.reconstruction;
    std::vector<std::uint8_t> bytes;
    bytes.reserve(256);
    appendString(bytes, "face-registration/head-surface-recipe");
    appendU32(bytes, HeadSurfaceCache::kCacheSchemaVersion);
    appendString(bytes, recipe.algorithmVersion);
    appendU8(bytes, static_cast<std::uint8_t>(
                        static_cast<unsigned char>(options.cropAxis)));
    appendU64(bytes, doubleBits(options.cropThresholdRatio));
    appendU64(bytes, static_cast<std::uint64_t>(options.targetPoints));
    appendU64(bytes, doubleBits(options.voxelSizeMm));
    appendU8(bytes, options.enableMultiViewVisibility ? 1U : 0U);
    appendU64(bytes, static_cast<std::uint64_t>(
                        static_cast<std::int64_t>(options.multiViewCount)));
    appendU64(bytes, static_cast<std::uint64_t>(options.randomSeed));
    appendU64(bytes, doubleBits(options.modelUnitScale));
    return bytes;
}

Digest sha256Bytes(const std::vector<std::uint8_t>& bytes) {
    Sha256 sha;
    sha.update(bytes);
    return sha.finish();
}

Digest cacheKeyDigest(const Digest& sourceDigest, const Digest& recipeDigest) {
    std::vector<std::uint8_t> bytes;
    appendString(bytes, "face-registration/head-surface-cache-key");
    appendU32(bytes, HeadSurfaceCache::kCacheSchemaVersion);
    bytes.insert(bytes.end(), sourceDigest.begin(), sourceDigest.end());
    bytes.insert(bytes.end(), recipeDigest.begin(), recipeDigest.end());
    return sha256Bytes(bytes);
}

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) {
                       if (character >= 'A' && character <= 'Z')
                           return static_cast<char>(character - 'A' + 'a');
                       return static_cast<char>(character);
                   });
    return extension;
}

std::filesystem::path normalized(const std::filesystem::path& path) {
    return path.lexically_normal();
}

std::filesystem::path workspaceRootFor(const std::filesystem::path& cacheRoot) {
    const auto normalizedRoot = normalized(cacheRoot);
    std::string cacheName = normalizedRoot.filename().string();
    if (cacheName.empty()) cacheName = "model_cache";
    return normalized(normalizedRoot.parent_path() /
        ("." + cacheName + "-head-surface-work"));
}

bool containedPath(
    const std::filesystem::path& root,
    const std::filesystem::path& candidate,
    bool allowEqual,
    std::string& message) {
    std::error_code error;
    auto canonicalRoot = std::filesystem::weakly_canonical(root, error);
    if (error) {
        error.clear();
        canonicalRoot = std::filesystem::absolute(root, error).lexically_normal();
    }
    if (error) {
        message = "cannot resolve containment root: " + error.message();
        return false;
    }
    error.clear();
    auto canonicalCandidate = std::filesystem::weakly_canonical(candidate, error);
    if (error) {
        error.clear();
        canonicalCandidate = std::filesystem::absolute(candidate, error).lexically_normal();
    }
    if (error) {
        message = "cannot resolve contained cache path: " + error.message();
        return false;
    }

    auto rootIterator = canonicalRoot.begin();
    auto candidateIterator = canonicalCandidate.begin();
    for (; rootIterator != canonicalRoot.end(); ++rootIterator, ++candidateIterator) {
        if (candidateIterator == canonicalCandidate.end() ||
            *candidateIterator != *rootIterator) {
            message = "cache path escapes its expected root";
            return false;
        }
    }
    if (!allowEqual && candidateIterator == canonicalCandidate.end()) {
        message = "contained cache path unexpectedly equals its root";
        return false;
    }
    message = "ok";
    return true;
}

bool validateAddress(const HeadSurfaceCache::Address& address, std::string& message) {
    if (address.cacheRoot.empty()) {
        message = "cache root is empty";
        return false;
    }
    if (address.sourceByteCount == 0) {
        message = "source STL byte count is zero";
        return false;
    }
    Digest sourceDigest{};
    Digest storedRecipeDigest{};
    Digest storedKeyDigest{};
    if (!decodeDigest(address.sourceSha256Hex, sourceDigest) ||
        !decodeDigest(address.recipeSha256Hex, storedRecipeDigest) ||
        !decodeDigest(address.cacheKeyHex, storedKeyDigest)) {
        message = "cache address contains a malformed SHA-256 value";
        return false;
    }
    if (!validateRecipe(address.recipe, message)) return false;
    const Digest expectedRecipeDigest = sha256Bytes(recipeBytes(address.recipe));
    if (expectedRecipeDigest != storedRecipeDigest) {
        message = "cache address recipe digest does not match its recipe";
        return false;
    }
    if (cacheKeyDigest(sourceDigest, expectedRecipeDigest) != storedKeyDigest) {
        message = "cache address key does not match its STL and recipe digests";
        return false;
    }
    if (normalized(address.surfacePlyPath) !=
            normalized(address.cacheRoot / kSurfaceFileName) ||
        normalized(address.manifestPath) !=
            normalized(address.cacheRoot / kManifestFileName)) {
        message = "cache address paths are inconsistent";
        return false;
    }
    message = "ok";
    return true;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

class PublicationLock final {
public:
    PublicationLock() = default;
    PublicationLock(const PublicationLock&) = delete;
    PublicationLock& operator=(const PublicationLock&) = delete;

    ~PublicationLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            if (locked_)
                UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &overlapped_);
            CloseHandle(handle_);
        }
#else
        if (descriptor_ >= 0) {
            flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
#endif
    }

    bool acquire(const std::filesystem::path& cacheRoot, std::string& message) {
        std::error_code directoryError;
        std::filesystem::create_directories(cacheRoot, directoryError);
        if (directoryError) {
            message = "cannot create cache root for publication lock: " +
                directoryError.message();
            return false;
        }
        const auto lockPath = normalized(cacheRoot / kLockFileName);
        if (!containedPath(cacheRoot, lockPath, false, message)) return false;
#ifdef _WIN32
        handle_ = CreateFileW(
            lockPath.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_HIDDEN, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            message = "cannot open head-surface publication lock file: " +
                std::system_category().message(static_cast<int>(GetLastError()));
            return false;
        }
        if (!LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0,
                        MAXDWORD, MAXDWORD, &overlapped_)) {
            message = "cannot acquire head-surface publication lock file: " +
                std::system_category().message(static_cast<int>(GetLastError()));
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        locked_ = true;
#else
        descriptor_ = open(lockPath.c_str(), O_CREAT | O_RDWR, 0600);
        if (descriptor_ < 0) {
            message = "cannot open head-surface publication lock: " +
                std::system_category().message(errno);
            return false;
        }
        if (flock(descriptor_, LOCK_EX) != 0) {
            message = "cannot acquire head-surface publication lock: " +
                std::system_category().message(errno);
            close(descriptor_);
            descriptor_ = -1;
            return false;
        }
#endif
        message = "ok";
        return true;
    }

private:
#ifdef _WIN32
    HANDLE handle_{INVALID_HANDLE_VALUE};
    OVERLAPPED overlapped_{};
    bool locked_{false};
#else
    int descriptor_{-1};
#endif
};

bool atomicReplaceFile(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string& message) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        message = "cannot atomically replace " + destination.string() + ": " +
            std::system_category().message(static_cast<int>(GetLastError()));
        return false;
    }
#else
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error) {
        message = "cannot atomically replace " + destination.string() + ": " +
            error.message();
        return false;
    }
#endif
    message = "ok";
    return true;
}

std::map<std::string, std::string> fixedManifestFields(
    const HeadSurfaceCache::Address& address) {
    const auto& options = address.recipe.reconstruction;
    return {
        {"format", "face-registration-head-surface-cache"},
        {"schema", std::to_string(HeadSurfaceCache::kCacheSchemaVersion)},
        {"cache_key", address.cacheKeyHex},
        {"source_sha256", address.sourceSha256Hex},
        {"source_size", std::to_string(address.sourceByteCount)},
        {"recipe_sha256", address.recipeSha256Hex},
        {"algorithm_version", address.recipe.algorithmVersion},
        {"crop_axis_byte", std::to_string(static_cast<unsigned int>(
                               static_cast<unsigned char>(options.cropAxis)))},
        {"crop_threshold_ratio_bits", hex64(doubleBits(options.cropThresholdRatio))},
        {"target_points", std::to_string(options.targetPoints)},
        {"voxel_size_mm_bits", hex64(doubleBits(options.voxelSizeMm))},
        {"enable_multi_view_visibility", options.enableMultiViewVisibility ? "1" : "0"},
        {"multi_view_count", std::to_string(options.multiViewCount)},
        {"random_seed", std::to_string(options.randomSeed)},
        {"model_unit_scale_bits", hex64(doubleBits(options.modelUnitScale))},
    };
}

struct ManifestReadResult {
    bool success{false};
    std::map<std::string, std::string> fields;
    std::string message;
};

ManifestReadResult readManifest(const std::filesystem::path& path) {
    ManifestReadResult result;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) {
        result.message = "cannot query cache manifest size: " + error.message();
        return result;
    }
    if (bytes == 0 || bytes > kMaximumManifestBytes) {
        result.message = "cache manifest has an invalid size";
        return result;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        result.message = "cannot open cache manifest: " + path.string();
        return result;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0) {
            result.message = "cache manifest contains a malformed line";
            return result;
        }
        const std::string name = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (!result.fields.emplace(name, value).second) {
            result.message = "cache manifest contains duplicate field: " + name;
            return result;
        }
    }
    if (input.bad()) {
        result.message = "cannot finish reading cache manifest";
        return result;
    }
    result.success = true;
    result.message = "ok";
    return result;
}

bool parseUnsigned(const std::string& text, std::uint64_t& value) {
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value, 10);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

struct PlyHeaderResult {
    bool success{false};
    std::uint64_t vertexCount{0};
    std::string message;
};

PlyHeaderResult readPlyHeader(const std::filesystem::path& path) {
    PlyHeaderResult result;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        result.message = "cannot open cached PLY: " + path.string();
        return result;
    }
    std::string line;
    std::size_t headerBytes = 0;
    bool firstLine = true;
    bool formatSeen = false;
    bool vertexSeen = false;
    while (std::getline(input, line)) {
        headerBytes += line.size() + 1;
        if (headerBytes > kMaximumPlyHeaderBytes) {
            result.message = "cached PLY header is unreasonably large";
            return result;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (firstLine) {
            firstLine = false;
            if (line != "ply") {
                result.message = "cached surface is not a PLY file";
                return result;
            }
            continue;
        }
        std::istringstream tokens(line);
        tokens.imbue(std::locale::classic());
        std::string first;
        tokens >> first;
        if (first == "format") {
            std::string representation, version;
            tokens >> representation >> version;
            formatSeen = !representation.empty() && version == "1.0";
        } else if (first == "element") {
            std::string elementName, countText;
            tokens >> elementName >> countText;
            if (elementName == "vertex") {
                std::uint64_t count = 0;
                if (!parseUnsigned(countText, count) || count == 0) {
                    result.message = "cached PLY has an invalid vertex count";
                    return result;
                }
                result.vertexCount = count;
                vertexSeen = true;
            }
        } else if (first == "end_header") {
            if (!formatSeen || !vertexSeen) {
                result.message = "cached PLY header is missing format or vertices";
                return result;
            }
            result.success = true;
            result.message = "ok";
            return result;
        }
    }
    result.message = "cached PLY header is incomplete";
    return result;
}

std::string uniqueToken() {
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t sequence = counter.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::uint64_t thread = static_cast<std::uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uint64_t entropy = 0;
    try {
        std::random_device random;
        entropy = (static_cast<std::uint64_t>(random()) << 32) ^ random();
    } catch (...) {
        entropy = ticks ^ (sequence * 0x9e3779b97f4a7c15ULL);
    }
    return hex64(ticks) + "-" + hex64(thread) + "-" + hex64(sequence ^ entropy);
}

bool validWorkspace(
    const HeadSurfaceCache::Address& address,
    const HeadSurfaceCache::Workspace& workspace,
    std::string& message) {
    if (workspace.cacheKeyHex != address.cacheKeyHex ||
        workspace.sourceSha256Hex != address.sourceSha256Hex) {
        message = "workspace does not belong to this cache address";
        return false;
    }
    const auto directory = normalized(workspace.directory);
    const auto workRoot = workspaceRootFor(address.cacheRoot);
    if (directory.parent_path() != workRoot) {
        message = "workspace is outside the cache work root";
        return false;
    }
    if (!containedPath(workRoot, directory, false, message)) return false;
    const std::string filename = directory.filename().string();
    if (filename.rfind(kWorkspacePrefix, 0) != 0) {
        message = "workspace name is not owned by this cache address";
        return false;
    }
    if (normalized(workspace.sourceSnapshotStlPath) != normalized(directory / kSourceSnapshotFileName) ||
        normalized(workspace.surfacePlyPath) != normalized(directory / kSurfaceFileName) ||
        normalized(workspace.manifestPath) != normalized(directory / kManifestFileName)) {
        message = "workspace paths are inconsistent";
        return false;
    }
    message = "ok";
    return true;
}

bool writeManifest(
    const HeadSurfaceCache::Address& address,
    const std::filesystem::path& path,
    const std::string& surfaceSha256,
    std::uintmax_t surfaceBytes,
    std::size_t pointCount,
    std::string& message) {
    auto fields = fixedManifestFields(address);
    fields.emplace("surface_sha256", surfaceSha256);
    fields.emplace("surface_size", std::to_string(surfaceBytes));
    fields.emplace("point_count", std::to_string(pointCount));
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        message = "cannot create cache manifest: " + path.string();
        return false;
    }
    for (const auto& field : fields) output << field.first << '=' << field.second << '\n';
    output.flush();
    if (!output) {
        message = "cannot flush cache manifest: " + path.string();
        return false;
    }
    output.close();
    if (!output) {
        message = "cannot close cache manifest: " + path.string();
        return false;
    }
    message = "ok";
    return true;
}

bool onlyPublishableFilesPresent(
    const HeadSurfaceCache::Workspace& workspace,
    std::string& message) {
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(workspace.directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        const auto path = normalized(iterator->path());
        if (path != normalized(workspace.surfacePlyPath) &&
            path != normalized(workspace.manifestPath)) {
            message = "workspace contains an unexpected file: " + path.string();
            return false;
        }
    }
    if (error) {
        message = "cannot enumerate cache workspace: " + error.message();
        return false;
    }
    message = "ok";
    return true;
}

HeadSurfaceCache::LookupResult lookupCachePair(
    const HeadSurfaceCache::Address& address,
    const std::filesystem::path& surfacePath,
    const std::filesystem::path& manifestPath) {
    HeadSurfaceCache::LookupResult result;
    std::error_code error;
    const bool surfaceExists = std::filesystem::exists(surfacePath, error);
    if (error) {
        result.status = HeadSurfaceCache::LookupStatus::Error;
        result.message = "cannot inspect cached surface: " + error.message();
        return result;
    }
    error.clear();
    const bool manifestExists = std::filesystem::exists(manifestPath, error);
    if (error) {
        result.status = HeadSurfaceCache::LookupStatus::Error;
        result.message = "cannot inspect cache manifest: " + error.message();
        return result;
    }
    if (!surfaceExists && !manifestExists) {
        result.status = HeadSurfaceCache::LookupStatus::Miss;
        result.message = "cache files do not exist";
        return result;
    }
    if (!surfaceExists || !manifestExists) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = "cache pair is incomplete";
        return result;
    }
    error.clear();
    if (!std::filesystem::is_regular_file(manifestPath, error) || error) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = error ? "cannot inspect cache manifest: " + error.message()
                               : "cache manifest path is not a regular file";
        return result;
    }
    const auto manifest = readManifest(manifestPath);
    if (!manifest.success) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = manifest.message;
        return result;
    }
    const auto expectedFields = fixedManifestFields(address);
    for (const auto& expected : expectedFields) {
        const auto found = manifest.fields.find(expected.first);
        if (found == manifest.fields.end() || found->second != expected.second) {
            result.status = HeadSurfaceCache::LookupStatus::Invalid;
            result.message = "cache manifest field mismatch: " + expected.first;
            return result;
        }
    }
    const auto surfaceHash = manifest.fields.find("surface_sha256");
    const auto surfaceSize = manifest.fields.find("surface_size");
    const auto pointCount = manifest.fields.find("point_count");
    Digest ignored{};
    if (surfaceHash == manifest.fields.end() || !decodeDigest(surfaceHash->second, ignored) ||
        surfaceSize == manifest.fields.end() || pointCount == manifest.fields.end()) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = "cache manifest is missing valid surface fields";
        return result;
    }
    std::uint64_t expectedSurfaceBytes = 0;
    std::uint64_t expectedPointCount = 0;
    if (!parseUnsigned(surfaceSize->second, expectedSurfaceBytes) || expectedSurfaceBytes == 0 ||
        !parseUnsigned(pointCount->second, expectedPointCount) || expectedPointCount == 0 ||
        expectedPointCount > std::numeric_limits<std::size_t>::max()) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = "cache manifest surface size or point count is invalid";
        return result;
    }
    error.clear();
    if (!std::filesystem::is_regular_file(surfacePath, error) || error) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = error ? "cannot inspect cached surface: " + error.message()
                               : "cached surface PLY is missing";
        return result;
    }
    const auto actualSurfaceBytes = std::filesystem::file_size(surfacePath, error);
    if (error || actualSurfaceBytes != expectedSurfaceBytes) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = error ? "cannot query cached surface size: " + error.message()
                               : "cached surface size does not match its manifest";
        return result;
    }
    const auto surface = digestFileInternal(surfacePath);
    if (!surface.success || digestHex(surface.digest) != surfaceHash->second) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = surface.success ? "cached surface SHA-256 does not match its manifest"
                                         : surface.message;
        return result;
    }
    const auto ply = readPlyHeader(surfacePath);
    if (!ply.success || ply.vertexCount != expectedPointCount) {
        result.status = HeadSurfaceCache::LookupStatus::Invalid;
        result.message = ply.success ? "cached PLY vertex count does not match its manifest"
                                     : ply.message;
        return result;
    }
    result.status = HeadSurfaceCache::LookupStatus::Hit;
    result.surfacePlyPath = surfacePath;
    result.pointCount = static_cast<std::size_t>(expectedPointCount);
    result.message = "cache hit";
    return result;
}

bool removeEmptyWorkRoot(
    const std::filesystem::path& workRoot,
    std::string& warning) {
    std::error_code error;
    const bool exists = std::filesystem::exists(workRoot, error);
    if (error) {
        warning = "cannot inspect cache work root: " + error.message();
        return false;
    }
    if (!exists) return true;
    const bool empty = std::filesystem::is_empty(workRoot, error);
    if (error) {
        warning = "cannot inspect whether cache work root is empty: " + error.message();
        return false;
    }
    if (!empty) return true;
    std::filesystem::remove(workRoot, error);
    if (error && error != std::errc::directory_not_empty) {
        warning = "cannot remove empty cache work root: " + error.message();
        return false;
    }
    return true;
}

bool createTemporaryWorkDirectory(
    const HeadSurfaceCache::Address& address,
    const std::string& label,
    std::filesystem::path& directory,
    std::string& message) {
    const auto workRoot = workspaceRootFor(address.cacheRoot);
    const auto siblingContainer = normalized(address.cacheRoot).parent_path();
    const auto containmentRoot = siblingContainer.empty()
        ? std::filesystem::path(".") : siblingContainer;
    if (!containedPath(containmentRoot, workRoot, false, message)) return false;
    std::error_code error;
    std::filesystem::create_directories(workRoot, error);
    if (error) {
        message = "cannot create cache work root: " + error.message();
        return false;
    }
    if (!std::filesystem::is_directory(workRoot, error) || error) {
        message = error ? "cannot inspect cache work root: " + error.message()
                        : "cache work root is not a directory";
        return false;
    }
    const std::string prefix = std::string(kWorkspacePrefix) + label + "-";
    for (int attempt = 0; attempt < 64; ++attempt) {
        directory = normalized(workRoot / (prefix + uniqueToken()));
        if (!containedPath(workRoot, directory, false, message)) return false;
        error.clear();
        if (std::filesystem::create_directory(directory, error)) {
            message = "ok";
            return true;
        }
        // Another process may have removed the empty sibling work root after
        // our initial check. Recreate it and retry with a fresh child name.
        if (error == std::errc::no_such_file_or_directory) {
            error.clear();
            std::filesystem::create_directories(workRoot, error);
            if (error) {
                message = "cannot recreate cache work root: " + error.message();
                return false;
            }
            continue;
        }
        if (error && error != std::errc::file_exists) {
            message = "cannot create cache work directory: " + error.message();
            return false;
        }
    }
    message = "cannot allocate a unique cache work directory after 64 attempts";
    return false;
}

bool normalizeLegacyLayoutLocked(
    const HeadSurfaceCache::Address& address,
    std::string& message) {
    const auto cacheRoot = normalized(address.cacheRoot);
    const auto legacyV2Root = normalized(cacheRoot / kLegacyV2DirectoryName);
    const auto legacyEntry = normalized(legacyV2Root / address.cacheKeyHex);
    const auto legacySurface = normalized(legacyEntry / kLegacySurfaceFileName);
    const auto legacyManifest = normalized(legacyEntry / kLegacyManifestFileName);
    const auto legacyV1 = normalized(cacheRoot / kLegacyV1FileName);

    if (legacyV2Root.parent_path() != cacheRoot ||
        legacyEntry.parent_path() != legacyV2Root ||
        legacySurface.parent_path() != legacyEntry ||
        legacyManifest.parent_path() != legacyEntry ||
        legacyV1.parent_path() != cacheRoot) {
        message = "legacy cache paths are inconsistent";
        return false;
    }

    std::error_code error;
    const bool legacyV2Exists = std::filesystem::exists(legacyV2Root, error);
    if (error) {
        message = "cannot inspect legacy v2 cache root: " + error.message();
        return false;
    }
    if (legacyV2Exists && !containedPath(cacheRoot, legacyV2Root, false, message))
        return false;

    auto fixed = lookupCachePair(address, address.surfacePlyPath, address.manifestPath);
    if (fixed.status == HeadSurfaceCache::LookupStatus::Error) {
        message = fixed.message;
        return false;
    }

    if (fixed.status != HeadSurfaceCache::LookupStatus::Hit && legacyV2Exists) {
        if (!containedPath(legacyV2Root, legacyEntry, false, message) ||
            !containedPath(legacyEntry, legacySurface, false, message) ||
            !containedPath(legacyEntry, legacyManifest, false, message)) {
            return false;
        }
        const auto legacy = lookupCachePair(address, legacySurface, legacyManifest);
        if (legacy.status == HeadSurfaceCache::LookupStatus::Error) {
            message = legacy.message;
            return false;
        }
        if (legacy.status == HeadSurfaceCache::LookupStatus::Hit) {
            std::filesystem::path migrationDirectory;
            if (!createTemporaryWorkDirectory(
                    address, "migration", migrationDirectory, message)) {
                return false;
            }
            const auto stagedSurface = migrationDirectory / "head_surface.tmp";
            const auto stagedManifest = migrationDirectory / "head_surface_cache.tmp";
            error.clear();
            std::filesystem::copy_file(
                legacySurface, stagedSurface,
                std::filesystem::copy_options::overwrite_existing, error);
            if (!error) {
                std::filesystem::copy_file(
                    legacyManifest, stagedManifest,
                    std::filesystem::copy_options::overwrite_existing, error);
            }
            if (error) {
                std::error_code ignored;
                std::filesystem::remove_all(migrationDirectory, ignored);
                message = "cannot stage legacy cache migration: " + error.message();
                return false;
            }
            const auto staged = lookupCachePair(address, stagedSurface, stagedManifest);
            if (staged.status != HeadSurfaceCache::LookupStatus::Hit) {
                std::error_code ignored;
                std::filesystem::remove_all(migrationDirectory, ignored);
                message = "staged legacy cache failed validation: " + staged.message;
                return false;
            }
            if (!atomicReplaceFile(stagedSurface, address.surfacePlyPath, message) ||
                !atomicReplaceFile(stagedManifest, address.manifestPath, message)) {
                std::error_code ignored;
                std::filesystem::remove(address.manifestPath, ignored);
                std::filesystem::remove_all(migrationDirectory, ignored);
                return false;
            }
            std::filesystem::remove_all(migrationDirectory, error);
            if (error) {
                message = "cannot clean legacy migration workspace: " + error.message();
                return false;
            }
            if (!removeEmptyWorkRoot(workspaceRootFor(address.cacheRoot), message))
                return false;
            fixed = lookupCachePair(address, address.surfacePlyPath, address.manifestPath);
            if (fixed.status != HeadSurfaceCache::LookupStatus::Hit) {
                message = "migrated fixed cache failed validation: " + fixed.message;
                return false;
            }
        }
    }

    const bool legacyV1Exists = std::filesystem::exists(legacyV1, error);
    if (error) {
        message = "cannot inspect legacy v1 cache file: " + error.message();
        return false;
    }
    if (legacyV1Exists) {
        if (!containedPath(cacheRoot, legacyV1, false, message)) return false;
        error.clear();
        if (!std::filesystem::is_regular_file(legacyV1, error) || error) {
            message = error ? "cannot inspect legacy v1 cache file: " + error.message()
                            : "legacy v1 cache path is not a regular file";
            return false;
        }
        std::filesystem::remove(legacyV1, error);
        if (error) {
            message = "cannot remove legacy v1 cache file: " + error.message();
            return false;
        }
    }
    if (legacyV2Exists) {
        error.clear();
        std::filesystem::remove_all(legacyV2Root, error);
        if (error) {
            message = "cannot remove legacy v2 cache tree: " + error.message();
            return false;
        }
    }
    message = "ok";
    return true;
}
} // namespace

struct HeadSurfaceCache::SessionLease::Impl final {
    PublicationLock lock;
};

HeadSurfaceCache::SessionLease::SessionLease() noexcept = default;
HeadSurfaceCache::SessionLease::~SessionLease() = default;
HeadSurfaceCache::SessionLease::SessionLease(SessionLease&&) noexcept = default;
HeadSurfaceCache::SessionLease& HeadSurfaceCache::SessionLease::operator=(
    SessionLease&&) noexcept = default;

HeadSurfaceCache::SessionLease::SessionLease(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

bool HeadSurfaceCache::SessionLease::valid() const noexcept {
    return static_cast<bool>(impl_);
}

HeadSurfaceCache::FileDigestResult HeadSurfaceCache::digestFile(
    const std::filesystem::path& path) {
    FileDigestResult result;
    const auto digest = digestFileInternal(path);
    result.success = digest.success;
    result.byteCount = digest.byteCount;
    result.message = digest.message;
    if (digest.success) result.sha256Hex = digestHex(digest.digest);
    return result;
}

HeadSurfaceCache::AddressResult HeadSurfaceCache::resolveAddress(
    const std::filesystem::path& sourceStlPath,
    const std::filesystem::path& cacheRoot,
    const Recipe& recipe) {
    AddressResult result;
    if (sourceStlPath.empty()) {
        result.message = "source STL path is empty";
        return result;
    }
    if (cacheRoot.empty()) {
        result.message = "cache root is empty";
        return result;
    }
    if (lowerExtension(sourceStlPath) != ".stl") {
        result.message = "model cache input is not an STL file: " + sourceStlPath.string();
        return result;
    }
    if (!validateRecipe(recipe, result.message)) return result;
    std::error_code fileError;
    if (!std::filesystem::is_regular_file(sourceStlPath, fileError)) {
        result.message = fileError
            ? "cannot inspect source STL: " + fileError.message()
            : "source STL is not a regular file: " + sourceStlPath.string();
        return result;
    }
    const auto source = digestFileInternal(sourceStlPath);
    if (!source.success) {
        result.message = source.message;
        return result;
    }
    if (source.byteCount == 0) {
        result.message = "source STL is empty: " + sourceStlPath.string();
        return result;
    }
    const Digest recipeDigest = sha256Bytes(recipeBytes(recipe));
    const Digest keyDigest = cacheKeyDigest(source.digest, recipeDigest);

    Address address;
    address.sourceStlPath = normalized(sourceStlPath);
    address.cacheRoot = normalized(cacheRoot);
    address.sourceSha256Hex = digestHex(source.digest);
    address.recipeSha256Hex = digestHex(recipeDigest);
    address.cacheKeyHex = digestHex(keyDigest);
    address.surfacePlyPath = normalized(address.cacheRoot / kSurfaceFileName);
    address.manifestPath = normalized(address.cacheRoot / kManifestFileName);
    address.sourceByteCount = source.byteCount;
    address.recipe = recipe;
    result.address = std::move(address);
    result.success = true;
    result.message = "ok";
    return result;
}

HeadSurfaceCache::LookupResult HeadSurfaceCache::lookup(const Address& address) {
    LookupResult result;
    std::string addressMessage;
    if (!validateAddress(address, addressMessage)) {
        result.status = LookupStatus::Error;
        result.message = addressMessage;
        return result;
    }
    return lookupCachePair(address, address.surfacePlyPath, address.manifestPath);
}

HeadSurfaceCache::SessionLeaseResult HeadSurfaceCache::acquireSessionLease(
    const Address& address) {
    SessionLeaseResult result;
    std::string message;
    if (!validateAddress(address, message)) {
        result.message = message;
        return result;
    }

    auto implementation = std::make_unique<SessionLease::Impl>();
    if (!implementation->lock.acquire(address.cacheRoot, result.message)) return result;

    const auto currentSource = digestFileInternal(address.sourceStlPath);
    if (!currentSource.success || currentSource.byteCount != address.sourceByteCount ||
        digestHex(currentSource.digest) != address.sourceSha256Hex) {
        result.message = currentSource.success
            ? "source STL changed before session cache lease acquisition"
            : currentSource.message;
        return result;
    }
    if (!normalizeLegacyLayoutLocked(address, result.message)) return result;

    result.lookup = lookupCachePair(
        address, address.surfacePlyPath, address.manifestPath);
    if (result.lookup.status != LookupStatus::Hit) {
        result.message = "session cache lease requires a validated hit: " +
            result.lookup.message;
        return result;
    }
    result.lease = SessionLease(std::move(implementation));
    result.success = true;
    result.message = "validated session cache lease acquired";
    return result;
}

HeadSurfaceCache::WorkspaceResult HeadSurfaceCache::createWorkspace(
    const Address& address,
    const std::filesystem::path& currentSourceStlPath) {
    WorkspaceResult result;
    std::string addressMessage;
    if (!validateAddress(address, addressMessage)) {
        result.message = addressMessage;
        return result;
    }
    if (currentSourceStlPath.empty()) {
        result.message = "current source STL path is empty";
        return result;
    }
    Workspace workspace;
    workspace.cacheKeyHex = address.cacheKeyHex;
    workspace.sourceSha256Hex = address.sourceSha256Hex;
    if (!createTemporaryWorkDirectory(
            address, "generation", workspace.directory, result.message)) {
        return result;
    }
    workspace.sourceSnapshotStlPath = normalized(workspace.directory / kSourceSnapshotFileName);
    workspace.surfacePlyPath = normalized(workspace.directory / kSurfaceFileName);
    workspace.manifestPath = normalized(workspace.directory / kManifestFileName);

    const auto snapshot = digestFileInternal(currentSourceStlPath, &workspace.sourceSnapshotStlPath);
    if (!snapshot.success) {
        result.workspace = workspace;
        result.message = snapshot.message;
        cleanupWorkspace(address, workspace);
        return result;
    }
    if (snapshot.byteCount != address.sourceByteCount ||
        digestHex(snapshot.digest) != address.sourceSha256Hex) {
        result.status = WorkspaceStatus::SourceChanged;
        result.workspace = workspace;
        result.message = "source STL changed after its cache address was resolved";
        cleanupWorkspace(address, workspace);
        return result;
    }
    result.status = WorkspaceStatus::Ready;
    result.workspace = std::move(workspace);
    result.message = "workspace ready";
    return result;
}

HeadSurfaceCache::PathOperationResult HeadSurfaceCache::cleanupWorkspace(
    const Address& address,
    const Workspace& workspace) {
    PathOperationResult result;
    std::string addressMessage;
    if (!validateAddress(address, addressMessage)) {
        result.message = addressMessage;
        return result;
    }
    if (!validWorkspace(address, workspace, result.message)) return result;
    std::error_code error;
    const bool exists = std::filesystem::exists(workspace.directory, error);
    if (error) {
        result.message = "cannot inspect cache workspace during cleanup: " + error.message();
        return result;
    }
    if (!exists) {
        std::string rootCleanupMessage;
        if (!removeEmptyWorkRoot(workspaceRootFor(address.cacheRoot), rootCleanupMessage)) {
            result.message = rootCleanupMessage;
            return result;
        }
        result.success = true;
        result.path = workspace.directory;
        result.message = "workspace is already absent";
        return result;
    }
    if (!std::filesystem::is_directory(workspace.directory, error) || error) {
        result.message = error ? "cannot inspect cache workspace during cleanup: " + error.message()
                               : "workspace path is not a directory";
        return result;
    }
    std::filesystem::remove_all(workspace.directory, error);
    if (error) {
        result.message = "cannot remove cache workspace: " + error.message();
        return result;
    }
    std::string rootCleanupMessage;
    if (!removeEmptyWorkRoot(workspaceRootFor(address.cacheRoot), rootCleanupMessage)) {
        result.message = rootCleanupMessage;
        return result;
    }
    result.success = true;
    result.path = workspace.directory;
    result.message = "workspace removed";
    return result;
}

HeadSurfaceCache::PathOperationResult HeadSurfaceCache::discardInvalidEntry(
    const Address& address) {
    PathOperationResult result;
    std::string message;
    if (!validateAddress(address, message)) {
        result.message = message;
        return result;
    }

    const auto initial = lookup(address);
    if (initial.status == LookupStatus::Hit) {
        result.success = true;
        result.path = address.surfacePlyPath;
        result.message = "cache entry is valid and was not discarded";
        return result;
    }
    if (initial.status == LookupStatus::Miss) {
        result.success = true;
        result.path = address.cacheRoot;
        result.message = "cache entry is already absent";
        return result;
    }
    if (initial.status == LookupStatus::Error) {
        result.message = initial.message;
        return result;
    }

    PublicationLock lock;
    if (!lock.acquire(address.cacheRoot, result.message)) return result;
    if (!normalizeLegacyLayoutLocked(address, result.message)) return result;

    const auto current = lookup(address);
    if (current.status == LookupStatus::Hit) {
        result.success = true;
        result.path = address.surfacePlyPath;
        result.message = "another process replaced the invalid cache entry";
        return result;
    }
    if (current.status == LookupStatus::Miss) {
        result.success = true;
        result.path = address.cacheRoot;
        result.message = "another process discarded the invalid cache entry";
        return result;
    }
    if (current.status == LookupStatus::Error) {
        result.message = current.message;
        return result;
    }

    // Remove the commit record first. An interruption can therefore leave only
    // an uncommitted PLY, never a pair that lookup could accept as a hit.
    std::error_code error;
    std::filesystem::remove(address.manifestPath, error);
    if (error) {
        result.message = "cannot discard invalid cache manifest: " + error.message();
        return result;
    }
    error.clear();
    std::filesystem::remove(address.surfacePlyPath, error);
    if (error) {
        result.message = "cannot discard invalid cached surface: " + error.message();
        return result;
    }
    result.success = true;
    result.path = address.cacheRoot;
    result.message = "invalid fixed cache files discarded";
    return result;
}

HeadSurfaceCache::PublishResult HeadSurfaceCache::publish(
    const Address& address,
    const Workspace& workspace,
    std::size_t generatedPointCount) {
    PublishResult result;
    std::string message;
    if (!validateAddress(address, message)) {
        result.message = message;
        return result;
    }
    if (!validWorkspace(address, workspace, message)) {
        result.message = message;
        return result;
    }
    if (generatedPointCount == 0) {
        result.message = "cannot publish a surface with zero points";
        return result;
    }
    std::error_code error;
    if (!std::filesystem::is_directory(workspace.directory, error) || error) {
        result.message = error ? "cannot inspect cache workspace: " + error.message()
                               : "cache workspace is missing";
        return result;
    }
    if (!std::filesystem::is_regular_file(workspace.sourceSnapshotStlPath, error) || error) {
        result.message = error ? "cannot inspect workspace source snapshot: " + error.message()
                               : "workspace source snapshot is missing";
        return result;
    }
    const auto snapshot = digestFileInternal(workspace.sourceSnapshotStlPath);
    if (!snapshot.success || snapshot.byteCount != address.sourceByteCount ||
        digestHex(snapshot.digest) != address.sourceSha256Hex) {
        result.message = snapshot.success ? "workspace source snapshot does not match the cache key"
                                          : snapshot.message;
        return result;
    }
    if (!std::filesystem::is_regular_file(workspace.surfacePlyPath, error) || error) {
        result.message = error ? "cannot inspect generated surface PLY: " + error.message()
                               : "generated surface PLY is missing";
        return result;
    }
    const auto ply = readPlyHeader(workspace.surfacePlyPath);
    if (!ply.success || ply.vertexCount != generatedPointCount) {
        result.message = ply.success ? "generated PLY vertex count does not match generator result"
                                     : ply.message;
        return result;
    }
    const auto surface = digestFileInternal(workspace.surfacePlyPath);
    if (!surface.success || surface.byteCount == 0) {
        result.message = surface.success ? "generated surface PLY is empty" : surface.message;
        return result;
    }

    std::filesystem::remove(workspace.sourceSnapshotStlPath, error);
    if (error) {
        result.message = "cannot remove source snapshot before publication: " + error.message();
        return result;
    }
    if (!writeManifest(address, workspace.manifestPath, digestHex(surface.digest),
                       surface.byteCount, generatedPointCount, message)) {
        result.message = message;
        return result;
    }
    if (!onlyPublishableFilesPresent(workspace, message)) {
        result.message = message;
        return result;
    }

    PublicationLock lock;
    if (!lock.acquire(address.cacheRoot, result.message)) return result;

    // A fixed cache slot makes a late writer dangerous: it could otherwise
    // overwrite a surface generated for newer STL bytes. Re-hash the canonical
    // source immediately before touching the public files.
    const auto currentSource = digestFileInternal(address.sourceStlPath);
    if (!currentSource.success || currentSource.byteCount != address.sourceByteCount ||
        digestHex(currentSource.digest) != address.sourceSha256Hex) {
        result.message = currentSource.success
            ? "source STL changed before fixed-cache publication"
            : currentSource.message;
        return result;
    }
    if (!normalizeLegacyLayoutLocked(address, result.message)) return result;

    // Recheck under the publication lock. Another process may have completed
    // the same recipe while this process generated its workspace.
    const auto existing = lookup(address);
    if (existing.status == LookupStatus::Hit) {
        const auto cleanup = cleanupWorkspace(address, workspace);
        if (!cleanup.success) {
            result.message = "cache entry exists but workspace cleanup failed: " + cleanup.message;
            return result;
        }
        result.status = PublishStatus::ExistingEntryReused;
        result.surfacePlyPath = existing.surfacePlyPath;
        result.pointCount = existing.pointCount;
        result.message = "concurrent cache entry reused";
        return result;
    }
    if (existing.status == LookupStatus::Error) {
        result.message = existing.message;
        return result;
    }

    // The PLY is replaced first and the manifest (the commit record) last. If
    // the process stops between them, lookup observes a hash mismatch and
    // rejects the pair instead of consuming partially published state.
    if (!atomicReplaceFile(workspace.surfacePlyPath, address.surfacePlyPath, message)) {
        result.message = message;
        return result;
    }
    if (!atomicReplaceFile(workspace.manifestPath, address.manifestPath, message)) {
        // Best effort: remove any old commit record so the newly replaced PLY
        // cannot accidentally be paired with stale metadata on the next run.
        std::error_code ignored;
        std::filesystem::remove(address.manifestPath, ignored);
        result.message = message;
        return result;
    }

    const auto published = lookup(address);
    if (published.status != LookupStatus::Hit) {
        result.message = "published cache entry failed validation: " + published.message;
        return result;
    }
    const auto cleanup = cleanupWorkspace(address, workspace);
    if (!cleanup.success) {
        result.message = "cache published but workspace cleanup failed: " + cleanup.message;
        return result;
    }
    result.status = PublishStatus::Published;
    result.surfacePlyPath = published.surfacePlyPath;
    result.pointCount = published.pointCount;
    result.message = "cache entry published";
    return result;
}
