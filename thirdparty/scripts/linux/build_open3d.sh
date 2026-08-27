#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_open3d.sh

Environment:
  JOBS  (optional) Parallel build jobs
       Example: export JOBS=16
  BUILD_ISPC_MODULE (optional) Enable ISPC module (default: OFF)
       Note: If enabled, Open3D may fetch ISPC during configure. For offline/slow networks, install system ISPC.
             On Ubuntu 22.04, apt may not provide ispc; use snap:
             sudo snap install ispc
  USE_SYSTEM_OPENSSL (optional) Use system OpenSSL ON/OFF (default: auto)
       Recommended on Ubuntu:
             sudo apt install -y libssl-dev
  USE_SYSTEM_CURL (optional) Use system curl ON/OFF (default: auto)
       Recommended on Ubuntu:
             sudo apt install -y libcurl4-openssl-dev
  USE_SYSTEM_VTK (optional) Use system VTK ON/OFF (default: auto)
       If auto, will use project thirdparty/vtk if detected; otherwise Open3D may download/build VTK.
  USE_SYSTEM_ZEROMQ (optional) Force ZeroMQ usage ON/OFF (default: auto)
       Note: If OFF, Open3D may fetch ZeroMQ during build. Recommended on Ubuntu:
             sudo apt install -y libzmq3-dev
  USE_BLAS (optional) Use BLAS/LAPACK instead of MKL ON/OFF (default: ON)
  USE_SYSTEM_BLAS (optional) Use system BLAS/LAPACK ON/OFF (default: ON; only applies when USE_BLAS=ON)
       Note: If USE_BLAS=OFF, Open3D will use MKL on x86_64 and may fetch MKL packages during build.
             Recommended on Ubuntu:
             sudo apt install -y libopenblas-dev liblapack-dev liblapacke-dev
  USE_SYSTEM_EIGEN3 (optional) Force system Eigen3 ON/OFF (default: auto)
       Note: If OFF, Open3D may fetch Eigen3 from GitLab.
             Recommended on Ubuntu:
             sudo apt install -y libeigen3-dev
  OPEN3D_THIRD_PARTY_DOWNLOAD_DIR (optional) Shared download cache directory for Open3D third-party archives
       Default: <project_root>/thirdparty/downloads/open3d
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 1
  fi
}

require_ninja() {
  if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required. Install: sudo apt update && sudo apt install -y ninja-build" >&2
    exit 1
  fi
}

require_cmake_min_version() {
  local required="$1"
  local current
  current="$(cmake --version | head -n 1 | awk '{print $3}')"
  if [[ -z "$current" ]]; then
    echo "Cannot detect CMake version." >&2
    exit 1
  fi
  if [[ "$(printf '%s\n%s\n' "$required" "$current" | sort -V | head -n 1)" != "$required" ]]; then
    echo "CMake $required or higher is required. You are running version $current" >&2
    echo "Recommended (user install): pip3 install --user \"cmake>=3.24\"" >&2
    echo "Then ensure PATH contains ~/.local/bin before running this script." >&2
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

install_dir="$project_root/thirdparty/open3d"
source_dir="$project_root/thirdparty/src/Open3D-main"
build_dir="$project_root/thirdparty/build/open3d-linux"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$source_dir" ]]; then
  echo "Open3D source directory not found: $source_dir" >&2
  exit 1
fi

require_cmd cmake
require_cmake_min_version "3.24.0"

require_ninja
generator="Ninja"

jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
build_ispc="${BUILD_ISPC_MODULE:-OFF}"
ispc_args=()
if [[ "$build_ispc" == "ON" ]]; then
  if command -v ispc >/dev/null 2>&1; then
    ispc_args+=(-DISPC_EXECUTABLE="$(command -v ispc)")
  else
    echo "ISPC not found. Install via snap: sudo snap install ispc (recommended), or ensure network access for Open3D to fetch ISPC." >&2
  fi
fi

want_system_zeromq="${USE_SYSTEM_ZEROMQ:-auto}"
use_system_zeromq="OFF"
if [[ "$want_system_zeromq" == "ON" || "$want_system_zeromq" == "OFF" ]]; then
  use_system_zeromq="$want_system_zeromq"
else
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libzmq >/dev/null 2>&1; then
    use_system_zeromq="ON"
  else
    use_system_zeromq="OFF"
  fi
fi

want_system_vtk="${USE_SYSTEM_VTK:-auto}"
use_system_vtk="OFF"
vtk_args=()
if [[ "$want_system_vtk" == "ON" || "$want_system_vtk" == "OFF" ]]; then
  use_system_vtk="$want_system_vtk"
else
  vtk_cfg="$(ls -1 "$project_root/thirdparty/vtk/lib/cmake/"vtk-*/vtk-config.cmake 2>/dev/null | head -n 1 || true)"
  if [[ -n "$vtk_cfg" ]]; then
    use_system_vtk="ON"
  else
    use_system_vtk="OFF"
  fi
fi
if [[ "$use_system_vtk" == "ON" ]]; then
  vtk_cfg="${vtk_cfg:-$(ls -1 "$project_root/thirdparty/vtk/lib/cmake/"vtk-*/vtk-config.cmake 2>/dev/null | head -n 1 || true)}"
  if [[ -n "$vtk_cfg" ]]; then
    vtk_dir="$(cd "$(dirname "$vtk_cfg")" && pwd)"
    vtk_args+=(-DUSE_SYSTEM_VTK=ON -DVTK_DIR="$vtk_dir")
  else
    use_system_vtk="OFF"
  fi
fi

use_system_blas="${USE_SYSTEM_BLAS:-ON}"
if [[ "$use_system_blas" != "ON" && "$use_system_blas" != "OFF" ]]; then
  use_system_blas="ON"
fi

use_blas="${USE_BLAS:-$use_system_blas}"
if [[ "$use_blas" != "ON" && "$use_blas" != "OFF" ]]; then
  use_blas="ON"
fi

want_system_eigen3="${USE_SYSTEM_EIGEN3:-auto}"
use_system_eigen3="OFF"
if [[ "$want_system_eigen3" == "ON" || "$want_system_eigen3" == "OFF" ]]; then
  use_system_eigen3="$want_system_eigen3"
else
  if [[ -d "/usr/include/eigen3/Eigen" || -d "/usr/local/include/eigen3/Eigen" ]]; then
    use_system_eigen3="ON"
  else
    use_system_eigen3="OFF"
  fi
fi

want_system_openssl="${USE_SYSTEM_OPENSSL:-auto}"
use_system_openssl="OFF"
if [[ "$want_system_openssl" == "ON" || "$want_system_openssl" == "OFF" ]]; then
  use_system_openssl="$want_system_openssl"
else
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openssl >/dev/null 2>&1; then
    use_system_openssl="ON"
  elif [[ -f "/usr/include/openssl/ssl.h" ]]; then
    use_system_openssl="ON"
  else
    use_system_openssl="OFF"
  fi
fi

want_system_curl="${USE_SYSTEM_CURL:-auto}"
use_system_curl="OFF"
if [[ "$want_system_curl" == "ON" || "$want_system_curl" == "OFF" ]]; then
  use_system_curl="$want_system_curl"
else
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libcurl >/dev/null 2>&1; then
    use_system_curl="ON"
  else
    use_system_curl="OFF"
  fi
fi

toolchain_args=()
if command -v nm >/dev/null 2>&1; then
  toolchain_args+=(-DCMAKE_NM="$(command -v nm)")
fi
if command -v ar >/dev/null 2>&1; then
  toolchain_args+=(-DCMAKE_AR="$(command -v ar)")
fi
if command -v ranlib >/dev/null 2>&1; then
  toolchain_args+=(-DCMAKE_RANLIB="$(command -v ranlib)")
fi

echo "Config:"
echo "  BUILD_ISPC_MODULE=$build_ispc"
echo "  USE_SYSTEM_VTK=$use_system_vtk"
echo "  USE_SYSTEM_ZEROMQ=$use_system_zeromq"
echo "  USE_BLAS=$use_blas"
echo "  USE_SYSTEM_BLAS=$use_system_blas"
echo "  USE_SYSTEM_EIGEN3=$use_system_eigen3"
echo "  USE_SYSTEM_OPENSSL=$use_system_openssl"
echo "  USE_SYSTEM_CURL=$use_system_curl"
echo ""

rm -rf "$build_dir"
mkdir -p "$build_dir"

thirdparty_download_dir="${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR:-$project_root/thirdparty/downloads/open3d}"
mkdir -p "$thirdparty_download_dir"

cmake -S "$source_dir" -B "$build_dir" -G "$generator" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_CUDA_MODULE=OFF \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_PYTORCH_OPS=OFF \
  -DWITH_OPENMP=ON \
  -DWITH_IPP=OFF \
  -DBUILD_ISPC_MODULE="$build_ispc" \
  -DOPEN3D_THIRD_PARTY_DOWNLOAD_DIR="$thirdparty_download_dir" \
  -DUSE_SYSTEM_ZEROMQ="$use_system_zeromq" \
  -DUSE_BLAS="$use_blas" \
  -DUSE_SYSTEM_BLAS="$use_system_blas" \
  -DUSE_SYSTEM_EIGEN3="$use_system_eigen3" \
  -DUSE_SYSTEM_OPENSSL="$use_system_openssl" \
  -DUSE_SYSTEM_CURL="$use_system_curl" \
  "${vtk_args[@]}" \
  "${toolchain_args[@]}" \
  "${ispc_args[@]}"

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! Open3D installed to: $install_dir"
echo "Set Open3D_DIR to: $install_dir/lib/cmake/Open3D"
