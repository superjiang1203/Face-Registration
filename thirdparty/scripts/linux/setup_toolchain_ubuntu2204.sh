#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./setup_toolchain_ubuntu2204.sh check
  ./setup_toolchain_ubuntu2204.sh install [--with-cudnn]

What it does:
  check   Prints versions and reports missing tools/packages
  install Installs required packages via apt (Ubuntu 22.04)

Notes:
  - This script does not install Qt. Qt is expected to be installed separately.
  - For VTK Qt build, export Qt6_DIR to a directory containing Qt6Config.cmake.
  - Open3D (Open3D-main) requires CMake >= 3.24. Ubuntu 22.04 apt provides 3.22.1 by default.
  - cuDNN is optional and only needed if you want ONNXRuntime CUDA provider (GPU) on Linux.
EOF
}

mode="${1:-check}"
shift || true
with_cudnn="false"
for a in "$@"; do
  case "$a" in
    --with-cudnn) with_cudnn="true" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $a" >&2; usage >&2; exit 1 ;;
  esac
done
if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$mode" != "check" && "$mode" != "install" ]]; then
  usage >&2
  exit 1
fi

pkg_list=(
  build-essential
  cmake
  ninja-build
  binutils
  git
  pkg-config
  ca-certificates
  python3-pip
  python3-venv
  libssl-dev
  libcurl4-openssl-dev
  libglew-dev
  libzmq3-dev
  xz-utils
  libeigen3-dev
  libopenblas-dev
  liblapack-dev
  liblapacke-dev
  unzip
  libgl1-mesa-dev
  mesa-common-dev
  libx11-dev
  libxext-dev
  libxrender-dev
  libxt-dev
  libxrandr-dev
  libxi-dev
  libxinerama-dev
  libxcursor-dev
  libgtk-3-dev
)

print_versions() {
  echo "System:"
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -a 2>/dev/null || true
  else
    cat /etc/os-release || true
  fi
  echo ""

  echo "Toolchain:"
  (gcc --version 2>/dev/null || echo "gcc: MISSING") | head -n 1
  (g++ --version 2>/dev/null || echo "g++: MISSING") | head -n 1
  (cmake --version 2>/dev/null || echo "cmake: MISSING") | head -n 1
  (ninja --version 2>/dev/null || echo "ninja: MISSING") | head -n 1
  (nm --version 2>/dev/null || echo "nm: MISSING (install: sudo apt install -y binutils)") | head -n 1
  if command -v ispc >/dev/null 2>&1; then
    (ispc --version 2>/dev/null || echo "ispc: UNKNOWN") | head -n 1
  else
    echo "ispc: MISSING (optional for Open3D BUILD_ISPC_MODULE=ON; install via snap: sudo snap install ispc)"
  fi
  (make --version 2>/dev/null || echo "make: MISSING") | head -n 1
  (git --version 2>/dev/null || echo "git: MISSING") | head -n 1
  (pkg-config --version 2>/dev/null || echo "pkg-config: MISSING") | head -n 1
  if command -v pkg-config >/dev/null 2>&1; then
    if pkg-config --exists glew >/dev/null 2>&1; then
      echo "glew (pkg-config): $(pkg-config --modversion glew 2>/dev/null || echo "UNKNOWN")"
    else
      echo "glew (pkg-config): MISSING"
    fi
  else
    echo "glew (pkg-config): UNKNOWN (pkg-config missing)"
  fi
  echo ""

  if command -v cmake >/dev/null 2>&1; then
    cmake_ver="$(cmake --version | head -n 1 | awk '{print $3}')"
    if [[ -n "$cmake_ver" ]]; then
      required="3.24.0"
      if [[ "$(printf '%s\n%s\n' "$required" "$cmake_ver" | sort -V | head -n 1)" != "$required" ]]; then
        echo "CMake WARNING: Open3D requires >= $required (current: $cmake_ver)"
        echo "  Recommended: pip3 install --user \"cmake>=3.24\"  (then export PATH=~/.local/bin:\$PATH)"
        echo ""
      fi
    fi
  fi

  if [[ -x "$HOME/.local/bin/cmake" ]]; then
    local_cmake_ver="$("$HOME/.local/bin/cmake" --version 2>/dev/null | head -n 1 | awk '{print $3}' || true)"
    if [[ -n "${local_cmake_ver:-}" && ( ":$PATH:" != *":$HOME/.local/bin:"* ) ]]; then
      echo "CMake NOTE: $HOME/.local/bin is not on PATH. If you installed CMake via pip, consider:"
      echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
      echo "  (Detected user CMake version: ${local_cmake_ver})"
      echo ""
    fi
  fi

  echo "Qt:"
  if [[ -n "${Qt6_DIR:-}" ]]; then
    if [[ -f "${Qt6_DIR}/Qt6Config.cmake" ]]; then
      echo "Qt6_DIR OK: $Qt6_DIR"
    else
      echo "Qt6_DIR WRONG (missing Qt6Config.cmake): $Qt6_DIR"
    fi
  else
    echo "Qt6_DIR not set"
  fi
  echo ""

  echo "CUDA/cuDNN (optional, needed for ONNXRuntime CUDA provider):"
  (nvidia-smi --version 2>/dev/null || echo "nvidia-smi: MISSING") | head -n 2
  (nvcc --version 2>/dev/null | grep -E 'release|Cuda compilation tools' | head -n 1 || echo "nvcc: MISSING (CUDA toolkit not installed)") | head -n 1
  if ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so\.9'; then
    echo "libcudnn.so.9: OK"
  else
    echo "libcudnn.so.9: MISSING"
    echo "  Install (CUDA 12): sudo apt-get -y install cudnn9-cuda-12 && sudo ldconfig"
  fi

  echo ""
  echo "PostgreSQL (optional, needed for server runtime/build with libpq):"
  (psql --version 2>/dev/null || echo "psql: MISSING") | head -n 1
  (pg_config --version 2>/dev/null || echo "pg_config: MISSING") | head -n 1
  if dpkg -s libpq-dev >/dev/null 2>&1; then
    echo "libpq-dev: OK ($(dpkg-query -W -f='${Version}' libpq-dev 2>/dev/null || echo 'UNKNOWN'))"
  else
    echo "libpq-dev: MISSING (install: sudo apt install -y libpq-dev)" >&2
  fi
  if dpkg -s libpq5 >/dev/null 2>&1; then
    echo "libpq5: $(dpkg-query -W -f='${Version}' libpq5 2>/dev/null || echo 'UNKNOWN')"
  fi
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files 2>/dev/null | grep -q '^postgresql\.service'; then
      echo "postgresql.service: $(systemctl is-active postgresql 2>/dev/null || echo 'unknown')"
    fi
  fi
}

missing_pkgs=()
for p in "${pkg_list[@]}"; do
  if ! dpkg -s "$p" >/dev/null 2>&1; then
    missing_pkgs+=("$p")
  fi
done

if [[ "$mode" == "check" ]]; then
  print_versions
  if [[ "${#missing_pkgs[@]}" -eq 0 ]]; then
    echo "APT packages: OK"
  else
    echo "APT packages missing:"
    printf '  - %s\n' "${missing_pkgs[@]}"
    echo ""
    echo "Install with:"
    echo "  sudo apt update"
    echo "  sudo apt install -y ${missing_pkgs[*]}"
    exit 1
  fi
  exit 0
fi

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run install mode with sudo:" >&2
  echo "  sudo ./setup_toolchain_ubuntu2204.sh install [--with-cudnn]" >&2
  exit 1
fi

apt update
apt install -y "${pkg_list[@]}"

if [[ "$with_cudnn" == "true" ]]; then
  if apt-cache show cudnn9-cuda-12 >/dev/null 2>&1; then
    apt install -y cudnn9-cuda-12 || true
    ldconfig || true
  else
    echo "cuDNN NOTE: cudnn9-cuda-12 package not found in current APT sources." >&2
    echo "  If you need ONNXRuntime GPU, add NVIDIA CUDA repo for Ubuntu 22.04, then install cudnn9-cuda-12." >&2
  fi
fi

print_versions
echo "Done."
