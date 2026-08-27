#!/usr/bin/env bash
set -euo pipefail

mode="${1:-package}"
force="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"

case "$mode" in
  package)
    archive="$project_root/thirdparty/packages/linux/OrbbecSDK-linux.tgz"
    destination="$project_root/thirdparty/OrbbecSDK"
    root_name="OrbbecSDK"
    ;;
  source)
    archive="$project_root/thirdparty/src/OrbbecSDK_v2-2.7.2-rc.zip"
    destination="$project_root/thirdparty/src/OrbbecSDK_v2-2.7.2-rc"
    root_name="OrbbecSDK_v2-2.7.2-rc"
    ;;
  *) echo "Usage: $0 [package|source] [--force]" >&2; exit 2 ;;
esac

[[ -f "$archive" ]] || { echo "Archive not found: $archive" >&2; exit 1; }
if [[ -e "$destination" && "$force" != "--force" ]]; then
  echo "Destination exists: $destination (pass --force to replace it)" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [[ "$archive" == *.zip ]]; then
  command -v unzip >/dev/null || { echo "Install unzip first" >&2; exit 1; }
  unzip -q "$archive" -d "$tmp"
else
  tar -xzf "$archive" -C "$tmp"
fi
[[ -d "$tmp/$root_name" ]] || { echo "Expected archive root missing: $root_name" >&2; exit 1; }
if [[ -e "$destination" ]]; then rm -rf -- "$destination"; fi
mkdir -p "$(dirname "$destination")"
mv "$tmp/$root_name" "$destination"
echo "Orbbec SDK extracted to: $destination"
