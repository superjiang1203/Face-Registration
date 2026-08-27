#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./install_thirdparty.sh [build|check|clean|rebuild] [--purge-install]

Actions:
  build         Build+install VTK/ITK/OpenCV/Open3D/Orbbec (default)
  check         Check toolchain and installed cmake config files
  clean         Remove build directories (*-linux*)
  rebuild       clean + build

Options:
  --purge-install  Also remove installed prefixes under thirdparty/ (vtk/itk/opencv/OrbbecSDK/open3d)

Environment:
  Qt6_DIR  (required by build_vtk.sh) directory containing Qt6Config.cmake
  JOBS     parallel build jobs (default: 16 if not set)
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 1
  fi
}

log() {
  echo "$@"
}

warn() {
  echo "$@" >&2
}

tar_list_root_dir() {
  local archive="$1"
  local root=""
  root="$(tar -taf "$archive" 2>/dev/null | head -n 1 | cut -d/ -f1 || true)"
  if [[ -z "$root" ]]; then
    root="$(tar -tf "$archive" 2>/dev/null | head -n 1 | cut -d/ -f1 || true)"
  fi
  echo "$root"
}

tar_extract_dir_strip1_safe() {
  local archive="$1"
  local dst="$2"
  local root="$3"
  local tmp="${dst}.tmp.$$"

  rm -rf "$tmp"
  mkdir -p "$tmp"

  if ! tar -xaf "$archive" -C "$tmp" --strip-components=1 "$root" >/dev/null 2>&1; then
    rm -rf "$tmp"
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  mv "$tmp" "$dst"
  return 0
}

install_galaxy_sdk() {
  require_cmd tar
  local tgz="$project_root/thirdparty/packages/linux/Galaxy_camera.tar.gz"
  local dst="$project_root/thirdparty/GalaxySDK"
  if [[ ! -f "$tgz" ]]; then
    warn "Skip (missing): $tgz"
    return 0
  fi

  log "Installing GalaxySDK (Linux): $tgz -> $dst"
  if ! tar_extract_dir_strip1_safe "$tgz" "$dst" "Galaxy_camera"; then
    warn "Failed to install GalaxySDK: $tgz"
  fi
}

install_onnxruntime() {
  require_cmd tar
  local tgz=""
  for p in "$project_root/thirdparty/packages/linux"/onnxruntime-*.tgz; do
    [[ -f "$p" ]] || continue
    tgz="$p"
    break
  done
  if [[ -z "$tgz" ]]; then
    warn "Skip (missing): $project_root/thirdparty/packages/linux/onnxruntime-*.tgz"
    return 0
  fi

  local dst="$project_root/thirdparty/onnxruntime"
  local root
  root="$(tar_list_root_dir "$tgz")"
  if [[ -z "$root" ]]; then
    warn "Skip (cannot list): $tgz"
    return 0
  fi

  log "Installing ONNXRuntime (Linux): $tgz -> $dst"
  if ! tar_extract_dir_strip1_safe "$tgz" "$dst" "$root"; then
    warn "Failed to install ONNXRuntime: $tgz"
    return 0
  fi

  if [[ -f "$dst/lib/libonnxruntime_providers_cuda.so" ]]; then
    if ! ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so\.9'; then
      warn "ONNXRuntime CUDA provider detected, but libcudnn.so.9 is missing."
      warn "  Install cuDNN (CUDA 12): sudo apt-get -y install cudnn9-cuda-12 && sudo ldconfig"
    fi
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

if [[ -z "${JOBS:-}" ]]; then
  export JOBS=16
fi

action="build"
purge_install="false"
for a in "$@"; do
  case "$a" in
    build|check|clean|rebuild) action="$a" ;;
    --purge-install) purge_install="true" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $a" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "$action" == "check" ]]; then
  require_cmd cmake
  require_cmd ninja
  require_cmd gcc
  require_cmd g++
  require_cmd git
  require_cmd pkg-config

  echo "Environment:"
  echo "  JOBS   $JOBS"
  echo "  Qt6_DIR ${Qt6_DIR:-<not set>}"
  if [[ -n "${Qt6_DIR:-}" && -f "${Qt6_DIR}/Qt6Config.cmake" ]]; then
    echo "  Qt6_DIR OK"
  else
    echo "  Qt6_DIR not set or invalid (missing Qt6Config.cmake)" >&2
  fi
  echo ""

  vtk_cfg="$(ls -1 "$project_root/thirdparty/vtk/lib/cmake/"vtk-*/vtk-config.cmake 2>/dev/null | head -n 1 || true)"
  itk_cfg="$(ls -1 "$project_root/thirdparty/itk/lib/cmake/"ITK-*/ITKConfig.cmake 2>/dev/null | head -n 1 || true)"
  cv_cfg="$(ls -1 "$project_root/thirdparty/opencv/lib/cmake/"opencv*/OpenCVConfig.cmake 2>/dev/null | head -n 1 || true)"
  o3d_cfg="$(ls -1 "$project_root/thirdparty/open3d/lib/cmake/"Open3D*/Open3DConfig.cmake 2>/dev/null | head -n 1 || true)"
  ob_cfg="$project_root/thirdparty/OrbbecSDK/lib/OrbbecSDKConfig.cmake"

  echo "Installed CMake configs:"
  if [[ -n "$vtk_cfg" ]]; then echo "  VTK: $vtk_cfg"; else echo "  VTK: MISSING" >&2; fi
  if [[ -n "$itk_cfg" ]]; then echo "  ITK: $itk_cfg"; else echo "  ITK: MISSING" >&2; fi
  if [[ -n "$cv_cfg" ]]; then echo "  OpenCV: $cv_cfg"; else echo "  OpenCV: MISSING" >&2; fi
  if [[ -n "$o3d_cfg" ]]; then echo "  Open3D: $o3d_cfg"; else echo "  Open3D: MISSING" >&2; fi
  if [[ -f "$ob_cfg" ]]; then echo "  OrbbecSDK: $ob_cfg"; else echo "  OrbbecSDK: MISSING" >&2; fi
  exit 0
fi

clean_build() {
  local build_root="$project_root/thirdparty/build"
  if [[ -d "$build_root" ]]; then
    find "$build_root" -maxdepth 1 -type d -name '*-linux*' -exec rm -rf {} + || true
  fi
}

purge_prefixes() {
  rm -rf \
    "$project_root/thirdparty/vtk" \
    "$project_root/thirdparty/itk" \
    "$project_root/thirdparty/opencv" \
    "$project_root/thirdparty/OrbbecSDK" \
    "$project_root/thirdparty/open3d" || true
}

if [[ "$action" == "clean" || "$action" == "rebuild" ]]; then
  clean_build
  if [[ "$purge_install" == "true" ]]; then
    purge_prefixes
  fi
  if [[ "$action" == "clean" ]]; then
    echo "Clean done."
    exit 0
  fi
fi

echo "Environment:"
echo "  JOBS   Parallel build jobs (current: $JOBS)"
echo "  Qt6_DIR ${Qt6_DIR:-<not set>}"
echo "  ninja  (required) build generator"

install_galaxy_sdk
install_onnxruntime

"$script_dir/build_vtk.sh"
"$script_dir/build_itk.sh"
"$script_dir/build_opencv.sh"
"$script_dir/build_open3d.sh"
"$script_dir/build_orbbec.sh"

echo "All third-party libraries installed successfully!"
