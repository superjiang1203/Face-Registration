#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_opencv.sh

Environment:
  Qt6_DIR  (optional) Enable Qt backend for HighGUI if provided; must contain Qt6Config.cmake
          Example: export Qt6_DIR=/opt/Qt/6.8.3/gcc_64/lib/cmake/Qt6
  WITH_OPENGL (optional) Force OpenGL support ON/OFF (default: auto)
          Example: export WITH_OPENGL=ON
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

install_dir="$project_root/thirdparty/opencv"
source_dir="$project_root/thirdparty/src/opencv-4.12.0"
contrib_dir="$project_root/thirdparty/src/opencv_contrib-4.12.0/modules"
build_dir="$project_root/thirdparty/build/opencv-4.12.0-linux"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$source_dir" ]]; then
  echo "OpenCV source directory not found: $source_dir" >&2
  exit 1
fi
if [[ ! -d "$contrib_dir" ]]; then
  echo "OpenCV contrib modules not found: $contrib_dir" >&2
  exit 1
fi

require_ninja
generator="Ninja"

jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

with_qt="OFF"
qt6_dir="${Qt6_DIR:-}"
qt_args=()
if [[ -n "$qt6_dir" ]]; then
  if [[ ! -f "$qt6_dir/Qt6Config.cmake" ]]; then
    echo "Invalid Qt6_DIR: missing Qt6Config.cmake under: $qt6_dir" >&2
    usage >&2
    exit 1
  fi
  with_qt="ON"
  qt_args+=(-DQt6_DIR="$qt6_dir")
else
  echo "Qt disabled for OpenCV (set Qt6_DIR to enable)." >&2
fi

want_opengl="${WITH_OPENGL:-auto}"
with_opengl="OFF"
if [[ "$want_opengl" == "ON" || "$want_opengl" == "OFF" ]]; then
  with_opengl="$want_opengl"
else
  if pkg-config --exists glew >/dev/null 2>&1; then
    with_opengl="ON"
  else
    with_opengl="OFF"
    echo "OpenGL disabled for OpenCV (install libglew-dev to enable)." >&2
  fi
fi

rm -rf "$build_dir"
mkdir -p "$build_dir"

cmake -S "$source_dir" -B "$build_dir" -G "$generator" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_opencv_world=OFF \
  -DOPENCV_EXTRA_MODULES_PATH="$contrib_dir" \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DWITH_QT="$with_qt" \
  -DWITH_OPENGL="$with_opengl" \
  -DBUILD_opencv_rgbd=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python2=OFF \
  "${qt_args[@]}"

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! OpenCV installed to: $install_dir"
echo "Set OpenCV_DIR to: $install_dir"
