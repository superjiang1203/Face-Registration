#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_itk.sh

Environment:
  VTK_DIR  (optional) Override VTK CMake directory; must contain vtk-config.cmake
          Default: <project_root>/thirdparty/vtk/lib/cmake/vtk-9.4
  JOBS     (optional) Parallel build jobs
          Example: export JOBS=16
EOF
}

require_ninja() {
  if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required. Install: sudo apt update && sudo apt install -y ninja-build" >&2
    exit 1
  fi
}

resolve_project_root() {
  local dir="$1"
  while true; do
    if [[ -f "$dir/CMakeLists.txt" && -d "$dir/thirdparty" ]]; then
      echo "$dir"
      return 0
    fi
    local parent
    parent="$(cd "$dir/.." && pwd)"
    if [[ "$parent" == "$dir" ]]; then
      break
    fi
    dir="$parent"
  done
  echo "Cannot locate project root from: $1" >&2
  return 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(resolve_project_root "$script_dir")"

install_dir="$project_root/thirdparty/itk"
source_dir="$project_root/thirdparty/src/InsightToolkit-5.4.5"
build_dir="$project_root/thirdparty/build/itk-5.4.5-linux"
vtk_dir="${VTK_DIR:-$project_root/thirdparty/vtk/lib/cmake/vtk-9.4}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$source_dir" ]]; then
  echo "ITK source directory not found: $source_dir" >&2
  exit 1
fi
if [[ ! -f "$vtk_dir/vtk-config.cmake" ]]; then
  echo "VTK_DIR invalid or not found: $vtk_dir" >&2
  echo "Build/install VTK first, or export VTK_DIR to override." >&2
  usage >&2
  exit 1
fi

require_ninja
generator="Ninja"

jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

rm -rf "$build_dir"
mkdir -p "$build_dir"

cmake -S "$source_dir" -B "$build_dir" -G "$generator" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_SHARED_LIBS=ON \
  -DITK_BUILD_DEFAULT_MODULES=ON \
  -DModule_ITKVtkGlue=ON \
  -DVTK_DIR="$vtk_dir" \
  -DITK_SKIP_PATH_LENGTH_CHECKS=ON \
  -DBUILD_TESTING=OFF

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! ITK installed to: $install_dir"
echo "Set ITK_DIR to: $install_dir/lib/cmake/ITK-5.4"
