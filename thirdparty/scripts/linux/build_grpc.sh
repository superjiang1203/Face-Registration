#!/usr/bin/env bash
# Usage:
#   ./thirdparty/scripts/linux/build_grpc.sh
#
# Options (env):
#   JOBS=16
#
# Notes:
#   - 默认使用源码目录：thirdparty/src/grpc
#   - 若源码目录不存在，会按 TAG 自动 git clone（包含 submodules）
#   - 安装到：thirdparty/grpc
#   - 下载源码：git clone --depth 1 -b v1.78.1 --recurse-submodules --shallow-submodules https://github.com/grpc/grpc.git 
#   - 默认安装 Debug + Release 两个版本
#   - 与 Windows 脚本保持一致，直接使用 thirdparty/src/grpc 内置的 protobuf 子模块
set -euo pipefail

grpc_tag="v1.78.1"

jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
configs=("Debug" "Release")

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
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

require_cmd git
require_cmd cmake
require_cmd ninja

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(resolve_project_root "$script_dir")"

src_dir="$project_root/thirdparty/src/grpc"
build_dir="$project_root/thirdparty/build/grpc-linux"
install_dir="$project_root/thirdparty/grpc"

if [[ ! -d "$src_dir" ]]; then
  mkdir -p "$(dirname "$src_dir")"
  echo "Cloning gRPC $grpc_tag -> $src_dir"
  git clone --depth 1 --branch "$grpc_tag" --recurse-submodules --shallow-submodules https://github.com/grpc/grpc.git "$src_dir"
else
  if [[ -d "$src_dir/.git" ]]; then
    (cd "$src_dir" && git submodule update --init --recursive)
  fi
fi

rm -rf "$build_dir"
mkdir -p "$build_dir"

cmake -S "$src_dir" -B "$build_dir" -G "Ninja Multi-Config" \
  -DCMAKE_CONFIGURATION_TYPES="Debug;Release" \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=ON \
  -DgRPC_PROTOBUF_PROVIDER=module \
  -DgRPC_ZLIB_PROVIDER=module \
  -DgRPC_SSL_PROVIDER=module \
  -DgRPC_ABSL_PROVIDER=module \
  -DgRPC_RE2_PROVIDER=module \
  -DgRPC_CARES_PROVIDER=module

for config in "${configs[@]}"; do
  cmake --build "$build_dir" --config "$config" --target install --parallel "$jobs"
done

echo "Success! gRPC (Debug, Release) installed to: $install_dir"
echo "Set gRPC_DIR to: $install_dir/lib/cmake/grpc"
echo "Set Protobuf_DIR to: $install_dir/lib/cmake/protobuf"
