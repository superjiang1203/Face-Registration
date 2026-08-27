#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_vtk.sh

Environment:
  Qt6_DIR  (required) Qt6 CMake directory, must contain Qt6Config.cmake
          Example: export Qt6_DIR=/opt/Qt/6.8.3/gcc_64/lib/cmake/Qt6
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

install_dir="$project_root/thirdparty/vtk"
source_dir="$project_root/thirdparty/src/VTK-9.4.2"
build_dir="$project_root/thirdparty/build/vtk-9.4.2-linux"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

qt6_dir="${Qt6_DIR:-}"
if [[ -z "$qt6_dir" ]]; then
  echo "Qt6_DIR is required for VTK Qt build." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$qt6_dir/Qt6Config.cmake" ]]; then
  echo "Invalid Qt6_DIR: missing Qt6Config.cmake under: $qt6_dir" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$source_dir" ]]; then
  echo "VTK source directory not found: $source_dir" >&2
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
  -DVTK_GROUP_ENABLE_Qt=YES \
  -DVTK_QT_VERSION=6 \
  -DQt6_DIR="$qt6_dir" \
  -DVTK_MODULE_ENABLE_VTK_FiltersHybrid=YES \
  -DVTK_BUILD_TESTING=OFF \
  -DVTK_BUILD_EXAMPLES=OFF

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! VTK installed to: $install_dir"
echo "Set VTK_DIR to: $install_dir/lib/cmake/vtk-9.4"
