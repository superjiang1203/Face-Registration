#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_orbbec.sh

Environment:
  JOBS  (optional) Parallel build jobs
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

install_dir="$project_root/thirdparty/OrbbecSDK"
source_dir="$project_root/thirdparty/src/OrbbecSDK_v2-2.7.2-rc"
build_dir="$project_root/thirdparty/build/OrbbecSDK_v2-2.7.2-rc-linux"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$source_dir" ]]; then
  echo "OrbbecSDK source directory not found: $source_dir" >&2
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
  -DBUILD_SHARED_LIBS=OFF \
  -DSPDLOG_BUILD_SHARED=OFF \
  -DOB_BUILD_EXAMPLES=OFF \
  -DOB_BUILD_TESTS=OFF \
  -DOB_BUILD_DOCS=OFF \
  -DOB_BUILD_TOOLS=OFF

cmake --build "$build_dir" --target install --parallel "$jobs"

echo "Success! OrbbecSDK installed to: $install_dir"
echo "Set OrbbecSDK_DIR to: $install_dir/lib"
