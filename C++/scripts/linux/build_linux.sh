#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: $0 [--config Release|Debug] [--jobs N] [--clean] [--force-unpack]"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
config="Release"
jobs="$(nproc)"
clean=false
force_unpack=false

while (($#)); do
  case "$1" in
    --config) config="${2:?missing value for --config}"; shift 2 ;;
    --jobs) jobs="${2:?missing value for --jobs}"; shift 2 ;;
    --clean) clean=true; shift ;;
    --force-unpack) force_unpack=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$config" == "Release" || "$config" == "Debug" ]] || { echo "--config must be Release or Debug" >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }

build_dir="$project_root/build/linux-$config"
required=(
  "$project_root/thirdparty/opencv/lib/cmake"
  "$project_root/thirdparty/open3d/lib/cmake"
  "$project_root/thirdparty/OrbbecSDK/lib/OrbbecSDKConfig.cmake"
  "$project_root/thirdparty/onnxruntime/include/onnxruntime_cxx_api.h"
)
dependencies_ready=true
for path in "${required[@]}"; do [[ -e "$path" ]] || dependencies_ready=false; done
# A full Windows workspace can contain thirdparty/opencv/x64 and a root-level
# Windows OpenCVConfig.cmake. Remove only these generated unpack directories
# before installing the Linux archives; source packages under thirdparty/src
# are left untouched.
if [[ -d "$project_root/thirdparty/opencv/x64" ]]; then
  echo "Detected unpacked Windows OpenCV; replacing it with the Linux package."
  rm -rf -- "$project_root/thirdparty/opencv"
  dependencies_ready=false
fi
if [[ -f "$project_root/thirdparty/vtk/lib/cmake/vtk-9.4/VTK-targets-debug.cmake" ]]; then
  echo "Detected stale Windows/Debug VTK targets; replacing VTK with the Linux package."
  rm -rf -- "$project_root/thirdparty/vtk"
  dependencies_ready=false
fi
if [[ "$force_unpack" == true || "$dependencies_ready" == false ]]; then
  # Invoke through bash instead of relying on the executable bit. This also
  # works when the project is copied from Windows or stored on a noexec mount.
  bash "$project_root/thirdparty/scripts/linux/packages.sh" unpack
fi

if [[ "$clean" == true ]]; then rm -rf -- "$build_dir"; fi
cmake -S "$project_root" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE="$config"
cmake --build "$build_dir" --parallel "$jobs"
echo "Build completed: $build_dir/C++"
