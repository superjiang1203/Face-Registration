#pragma once

#include "registration/HeadSurfaceReconstructor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

// Single-slot, process-safe cache plumbing for the STL registration surface.
//
// The public cache names are deliberately stable and readable. SHA-256 is used
// only to validate that head_surface.ply was generated from the current STL
// bytes and reconstruction recipe; it is never exposed as a directory name.
class HeadSurfaceCache final {
public:
    // Keep the reconstruction schema at 2 because changing only the public
    // storage path does not change the generated point-cloud bytes. This lets
    // an already validated head.stl surface migrate to the readable fixed path
    // without an unnecessary reconstruction.
    static constexpr std::uint32_t kCacheSchemaVersion{2};
    static constexpr const char* kDefaultAlgorithmVersion{
        "head-surface-reconstructor-v1"};

    struct Recipe {
        // Increment/change this whenever reconstruction behavior changes without
        // a corresponding Options-field change.
        std::string algorithmVersion{kDefaultAlgorithmVersion};
        HeadSurfaceReconstructor::Options reconstruction{};
    };

    struct FileDigestResult {
        bool success{false};
        std::string sha256Hex;
        std::uintmax_t byteCount{0};
        std::string message;
    };

    struct Address {
        std::filesystem::path sourceStlPath;
        std::filesystem::path cacheRoot;
        std::filesystem::path surfacePlyPath;
        std::filesystem::path manifestPath;
        std::string sourceSha256Hex;
        std::string recipeSha256Hex;
        std::string cacheKeyHex;
        std::uintmax_t sourceByteCount{0};
        Recipe recipe;
    };

    struct AddressResult {
        bool success{false};
        Address address;
        std::string message;
    };

    enum class LookupStatus {
        Hit,
        Miss,
        Invalid,
        Error,
    };

    struct LookupResult {
        LookupStatus status{LookupStatus::Error};
        std::filesystem::path surfacePlyPath;
        std::size_t pointCount{0};
        std::string message;

        bool hit() const noexcept { return status == LookupStatus::Hit; }
    };

    // Move-only lease that holds the same cross-process file lock used by cache
    // publication. Keep it alive for as long as a consumer may reopen the fixed
    // head_surface.ply path. Do not call publish/discard/acquireSessionLease for
    // the same cache root while the current process still owns a lease.
    class SessionLease final {
    public:
        SessionLease() noexcept;
        ~SessionLease();
        SessionLease(SessionLease&&) noexcept;
        SessionLease& operator=(SessionLease&&) noexcept;

        SessionLease(const SessionLease&) = delete;
        SessionLease& operator=(const SessionLease&) = delete;

        bool valid() const noexcept;

    private:
        friend class HeadSurfaceCache;
        struct Impl;
        explicit SessionLease(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> impl_;
    };

    struct SessionLeaseResult {
        bool success{false};
        SessionLease lease;
        LookupResult lookup;
        std::string message;
    };

    struct Workspace {
        std::filesystem::path directory;
        std::filesystem::path sourceSnapshotStlPath;
        std::filesystem::path surfacePlyPath;
        std::filesystem::path manifestPath;
        std::string cacheKeyHex;
        std::string sourceSha256Hex;
    };

    enum class WorkspaceStatus {
        Ready,
        SourceChanged,
        Error,
    };

    struct WorkspaceResult {
        WorkspaceStatus status{WorkspaceStatus::Error};
        Workspace workspace;
        std::string message;

        bool ready() const noexcept { return status == WorkspaceStatus::Ready; }
    };

    enum class PublishStatus {
        Published,
        ExistingEntryReused,
        Error,
    };

    struct PublishResult {
        PublishStatus status{PublishStatus::Error};
        std::filesystem::path surfacePlyPath;
        std::size_t pointCount{0};
        std::string message;

        bool success() const noexcept { return status != PublishStatus::Error; }
    };

    struct PathOperationResult {
        bool success{false};
        std::filesystem::path path;
        std::string message;
    };

    // Computes SHA-256 without loading the complete file into memory.
    static FileDigestResult digestFile(const std::filesystem::path& path);

    // cacheRoot should normally be <output_root>/model_cache. This function is
    // read-only: it hashes the STL and resolves the two fixed public paths
    // <cacheRoot>/head_surface.ply and <cacheRoot>/head_surface_cache.txt.
    static AddressResult resolveAddress(
        const std::filesystem::path& sourceStlPath,
        const std::filesystem::path& cacheRoot,
        const Recipe& recipe);

    // A hit requires a matching manifest, byte count, SHA-256, and non-empty PLY
    // vertex declaration. Missing pairs are misses; incomplete, stale, or
    // corrupt pairs are invalid and must never be consumed.
    static LookupResult lookup(const Address& address);

    // Creates a collision-resistant temporary workspace in a hidden sibling of
    // the cache root (on the same volume) and copies the STL into it while
    // hashing. SourceChanged means the STL no longer matches the bytes used by
    // resolveAddress.
    static WorkspaceResult createWorkspace(
        const Address& address,
        const std::filesystem::path& currentSourceStlPath);

    // Validates the generated workspace and atomically replaces each fixed
    // cache file under a process-wide publication lock. The manifest is the
    // commit record and is replaced last, so an interrupted publication can be
    // detected but never mistaken for a cache hit.
    static PublishResult publish(
        const Address& address,
        const Workspace& workspace,
        std::size_t generatedPointCount);

    // Acquires the publication lock, normalizes any known legacy layout, then
    // validates the fixed pair while still holding the lock. Success requires a
    // cache hit. The returned move-only lease keeps that exact fixed path from
    // being replaced by another cooperating process until the lease is destroyed.
    static SessionLeaseResult acquireSessionLease(const Address& address);

    // Deletes only the two fixed cache files, and only when the current pair is
    // invalid. No backup or quarantine copy is retained.
    static PathOperationResult discardInvalidEntry(const Address& address);

    // Removes only a workspace created for this address. The containment and
    // name checks deliberately reject arbitrary paths.
    static PathOperationResult cleanupWorkspace(
        const Address& address,
        const Workspace& workspace);
};
