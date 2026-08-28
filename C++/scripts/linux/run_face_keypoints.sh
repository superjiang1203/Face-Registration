#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
config_file="${CONFIG_FILE:-C++/config/runtime.yml}"
args=(--config "$config_file" --manual-roi --camera-backend orbbec)
[[ -n "${CAMERA_SN:-}" ]] && args+=(--camera-sn "$CAMERA_SN")
cd "$project_root"
exec ./build/linux-Release/C++/face_camera_pipeline "${args[@]}"
