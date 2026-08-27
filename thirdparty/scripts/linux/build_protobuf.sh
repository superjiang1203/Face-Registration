#!/usr/bin/env bash
# Usage:
#   ./thirdparty/scripts/linux/build_protobuf.sh
#
# Options (env):
#   JOBS=16
#
# Notes:
#   - 默认使用源码目录：thirdparty/src/protobuf
#   - Protobuf CMake 导出可能依赖 absl/utf8_range，本脚本会先编译安装：
#       thirdparty/absl
#       thirdparty/utf8_range
#   - 安装到：thirdparty/protobuf
set -euo pipefail

protobuf_tag="v34.2"

jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
config="${CONFIG:-Release}"
build_shared="${BUILD_SHARED_LIBS:-OFF}"
with_zlib="${WITH_ZLIB:-OFF}"
reclone_if_incomplete="${RECLONE_IF_INCOMPLETE:-1}"

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

src_dir="$project_root/thirdparty/src/protobuf"
build_dir="$project_root/thirdparty/build/protobuf-linux"
install_dir="$project_root/thirdparty/protobuf"

utf8_src="$src_dir/third_party/utf8_range"

ensure_protobuf_sources_ready() {
  local src="$1"
  local reclone="$2"
  local expected_ver="${protobuf_tag#v}"
  local expected_protoc_ver="$expected_ver"
  if [[ "$expected_ver" =~ ^3\.([0-9]+)\.([0-9]+)$ ]]; then
    expected_protoc_ver="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  fi

  if [[ -d "$src" && -f "$src/version.json" ]]; then
    local actual_protoc_ver=""
    actual_protoc_ver="$(sed -n 's/.*"protoc_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$src/version.json" | head -n 1 || true)"
    if [[ -n "$actual_protoc_ver" && -n "$expected_protoc_ver" && "$actual_protoc_ver" != "$expected_protoc_ver" ]]; then
      if [[ "$reclone" != "1" ]]; then
        echo "Protobuf source version mismatch: expected $expected_protoc_ver but found $actual_protoc_ver under $src" >&2
        exit 1
      fi
      local backup="$src.mismatch"
      local i="1"
      while [[ -e "$backup" ]]; do
        backup="$src.mismatch.$i"
        i="$((i + 1))"
      done
      echo "Protobuf source version mismatch, moving aside: $src -> $backup" >&2
      mv "$src" "$backup"
    fi
  fi

  if [[ -d "$src/third_party/utf8_range" ]]; then
    echo "$src"
    return 0
  fi

  if [[ ! -d "$src" ]]; then
    mkdir -p "$(dirname "$src")"
    echo "Cloning protobuf $protobuf_tag -> $src"
    git clone --depth 1 --branch "$protobuf_tag" --recurse-submodules --shallow-submodules https://github.com/protocolbuffers/protobuf.git "$src"
  fi

  if [[ -d "$src/.git" ]]; then
    (cd "$src" && git submodule update --init --recursive --depth 1)
  fi

  local utf8_src="$src/third_party/utf8_range"
  local missing="0"
  [[ -d "$utf8_src" ]] || missing="1"

  if [[ "$missing" == "1" && -d "$src/.git" ]]; then
    (cd "$src" && git submodule update --init --recursive --depth 1)
    missing="0"
    [[ -d "$utf8_src" ]] || missing="1"
  fi

  if [[ "$missing" == "1" && "$reclone" == "1" ]]; then
    local backup="$src.incomplete"
    local i="1"
    while [[ -e "$backup" ]]; do
      backup="$src.incomplete.$i"
      i="$((i + 1))"
    done
    echo "Protobuf source tree seems incomplete, moving aside: $src -> $backup" >&2
    mv "$src" "$backup"

    mkdir -p "$(dirname "$src")"
    echo "Cloning protobuf $protobuf_tag -> $src"
    git clone --depth 1 --branch "$protobuf_tag" --recurse-submodules --shallow-submodules https://github.com/protocolbuffers/protobuf.git "$src"
    (cd "$src" && git submodule update --init --recursive --depth 1)

    missing="0"
    [[ -d "$absl_src" ]] || missing="1"
    [[ -d "$utf8_src" ]] || missing="1"
  fi

  if [[ "$missing" == "1" ]]; then
    echo "Protobuf third_party dependencies are missing under: $src/third_party. Please ensure submodules are present, or delete $src and rerun." >&2
    exit 1
  fi

  echo "$src"
}

src_dir="$(ensure_protobuf_sources_ready "$src_dir" "$reclone_if_incomplete")"

rm -rf "$build_dir"
mkdir -p "$build_dir"

cmake -S "$src_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE="$config" \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -Dprotobuf_BUILD_SHARED_LIBS="$build_shared" \
  -DBUILD_SHARED_LIBS="$build_shared" \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dprotobuf_BUILD_EXAMPLES=OFF \
  -Dprotobuf_WITH_ZLIB="$with_zlib" \
  -Dprotobuf_INSTALL=ON

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! Protobuf installed to: $install_dir"
