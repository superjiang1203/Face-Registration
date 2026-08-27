#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./packages.sh list
  ./packages.sh unpack
  ./packages.sh pack

Notes:
  - This script is for Linux usage. It operates on thirdparty/packages/linux/.
  - unpack extracts archives from thirdparty/packages/linux/ into thirdparty/
  - pack creates archives from installed prefixes under thirdparty/ into thirdparty/packages/linux/
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

action="${1:-}"
if [[ -z "$action" || "$action" == "-h" || "$action" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(resolve_project_root "$script_dir")"
thirdparty_dir="$project_root/thirdparty"
packages_dir="$thirdparty_dir/packages"
win_dir="$packages_dir/windows"
linux_dir="$packages_dir/linux"

mkdir -p "$win_dir" "$linux_dir"

if [[ "$action" == "list" ]]; then
  echo "Windows packages: $win_dir"
  ls -lah "$win_dir" || true
  echo ""
  echo "Linux packages: $linux_dir"
  ls -lah "$linux_dir" || true
  exit 0
fi

if [[ "$action" == "unpack" ]]; then
  require_cmd tar

  extract_dir_strip1() {
    local archive="$1"
    local dst="$2"
    local root="$3"
    rm -rf "$dst"
    mkdir -p "$dst"
    tar -xaf "$archive" -C "$dst" --strip-components=1 "$root" >/dev/null 2>&1
  }

  extract_archive() {
    local archive="$1"
    local dst="$2"
    rm -rf "$dst"
    mkdir -p "$dst"
    tar -xaf "$archive" -C "$dst" >/dev/null 2>&1
  }

  list_root_dir() {
    local archive="$1"
    local root=""
    root="$(tar -taf "$archive" 2>/dev/null | head -n 1 | cut -d/ -f1 || true)"
    if [[ -z "$root" ]]; then
      root="$(tar -tzf "$archive" 2>/dev/null | head -n 1 | cut -d/ -f1 || true)"
    fi
    if [[ -z "$root" ]]; then
      root="$(tar -tf "$archive" 2>/dev/null | head -n 1 | cut -d/ -f1 || true)"
    fi
    echo "$root"
  }

  galaxy_tgz="$linux_dir/Galaxy_camera.tar.gz"
  if [[ -f "$galaxy_tgz" ]]; then
    log "Extracting: $galaxy_tgz -> $thirdparty_dir/GalaxySDK"
    if ! extract_dir_strip1 "$galaxy_tgz" "$thirdparty_dir/GalaxySDK" "Galaxy_camera"; then
      warn "Failed to extract: $galaxy_tgz"
    fi
  else
    warn "Skip (missing): $galaxy_tgz"
  fi

  ort_tgz=""
  for p in "$linux_dir"/onnxruntime-*.tgz; do
    [[ -f "$p" ]] || continue
    ort_tgz="$p"
    break
  done
  if [[ -n "$ort_tgz" ]]; then
    ort_root="$(list_root_dir "$ort_tgz")"
    if [[ -z "$ort_root" ]]; then
      warn "Skip (cannot list): $ort_tgz"
    else
      log "Extracting: $ort_tgz -> $thirdparty_dir/onnxruntime"
      if ! extract_dir_strip1 "$ort_tgz" "$thirdparty_dir/onnxruntime" "$ort_root"; then
        warn "Failed to extract: $ort_tgz"
      fi
    fi
  else
    warn "Skip (missing): $linux_dir/onnxruntime-*.tgz"
  fi

  opencv_itk_vtk_tgz="$linux_dir/opencv_itk_vtk-linux.tgz"
  if [[ -f "$opencv_itk_vtk_tgz" ]]; then
    log "Extracting: $opencv_itk_vtk_tgz -> $thirdparty_dir"
    if ! tar -xaf "$opencv_itk_vtk_tgz" -C "$thirdparty_dir" >/dev/null 2>&1; then
      warn "Failed to extract: $opencv_itk_vtk_tgz"
    fi
  else
    warn "Skip (missing): $opencv_itk_vtk_tgz"
  fi

  open3d_tgz="$linux_dir/open3d-linux.tgz"
  if [[ -f "$open3d_tgz" ]]; then
    log "Extracting: $open3d_tgz -> $thirdparty_dir"
    if ! tar -xaf "$open3d_tgz" -C "$thirdparty_dir" >/dev/null 2>&1; then
      warn "Failed to extract: $open3d_tgz"
    fi
  else
    warn "Skip (missing): $open3d_tgz"
  fi

  orbbec_tgz="$linux_dir/OrbbecSDK-linux.tgz"
  if [[ -f "$orbbec_tgz" ]]; then
    log "Extracting: $orbbec_tgz -> $thirdparty_dir"
    if ! tar -xaf "$orbbec_tgz" -C "$thirdparty_dir" >/dev/null 2>&1; then
      warn "Failed to extract: $orbbec_tgz"
    fi
  else
    warn "Skip (missing): $orbbec_tgz"
  fi

  for tgz in "$linux_dir"/*.tgz "$linux_dir"/*.tar.gz; do
    [[ -f "$tgz" ]] || continue
    case "$(basename "$tgz")" in
      Galaxy_camera.tar.gz|onnxruntime-*.tgz|opencv_itk_vtk-linux.tgz|open3d-linux.tgz|OrbbecSDK-linux.tgz) continue ;;
    esac
    log "Extracting (generic): $tgz -> $thirdparty_dir"
    if ! tar -xaf "$tgz" -C "$thirdparty_dir" >/dev/null 2>&1; then
      warn "Failed to extract: $tgz"
    fi
  done

  for z in "$linux_dir"/*.zip; do
    [[ -f "$z" ]] || continue
    if [[ "$(basename "$z")" == Galaxy_*Linux*.zip ]]; then
      continue
    fi
    require_cmd unzip
    log "Extracting: $z -> $thirdparty_dir"
    if ! unzip -o "$z" -d "$thirdparty_dir" >/dev/null 2>&1; then
      warn "Failed to extract: $z"
    fi
  done
  log "Done."
  exit 0
fi

if [[ "$action" == "pack" ]]; then
  require_cmd tar
  out_dir="$linux_dir"

  if [[ -d "$thirdparty_dir/vtk" && -d "$thirdparty_dir/itk" && -d "$thirdparty_dir/opencv" ]]; then
    echo "Packing vtk/itk/opencv -> $out_dir/opencv_itk_vtk-linux.tgz"
    tar -czf "$out_dir/opencv_itk_vtk-linux.tgz" -C "$thirdparty_dir" vtk itk opencv
  fi
  if [[ -d "$thirdparty_dir/open3d" ]]; then
    echo "Packing open3d -> $out_dir/open3d-linux.tgz"
    tar -czf "$out_dir/open3d-linux.tgz" -C "$thirdparty_dir" open3d
  fi
  if [[ -d "$thirdparty_dir/OrbbecSDK" ]]; then
    echo "Packing OrbbecSDK -> $out_dir/OrbbecSDK-linux.tgz"
    tar -czf "$out_dir/OrbbecSDK-linux.tgz" -C "$thirdparty_dir" OrbbecSDK
  fi

  echo "Done."
  exit 0
fi

echo "Unknown command." >&2
usage >&2
exit 1
