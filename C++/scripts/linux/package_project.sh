#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
output="${1:-$(dirname "$project_root")/face_registration-linux-transfer.tar.gz}"

tar -czf "$output" \
  --exclude='.git' \
  --exclude='build' \
  --exclude='thirdparty/build' \
  --exclude='output' \
  -C "$(dirname "$project_root")" "$(basename "$project_root")"
echo "Created: $output"
