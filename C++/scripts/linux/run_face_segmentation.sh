#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
config_file="${CONFIG_FILE:-C++/config/runtime.yml}"
threads="${THREADS:-4}"
args=(--config "$config_file" --target-locator sapiens_seg --threads "$threads" --camera-backend orbbec)
[[ -n "${CAMERA_SN:-}" ]] && args+=(--camera-sn "$CAMERA_SN")
cd "$project_root"
exec ./build/linux-Release/C++/face_camera_pipeline "${args[@]}"
