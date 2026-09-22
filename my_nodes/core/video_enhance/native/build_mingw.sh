#!/usr/bin/env bash
set -euo pipefail

# Builds the three MIT-licensed Windows artifacts of the DNR3 DLSS worker with
# a Linux MinGW-w64 cross toolchain:
#
#   bin/dlss5nr_bridge.dll    NGX bridge (feature aware: SR / NR / SR+NR)
#   bin/dlss5nr_host.exe      stdin/stdout DNR3 transport, run under Wine
#   bin/nvngx.dll_comfy.dll   caller shim, loaded next to the bridge
#
# The shim and the bridge live together in bin/, because the bridge resolves
# the shim from its own directory - the user's NVIDIA runtime directory only
# ever holds NVIDIA binaries. This script never downloads, copies or patches an
# NVIDIA file. See ../THIRD_PARTY_NOTICES.md for provenance.
#
# The bridge and the host need the posix-thread variant of the toolchain
# (std::mutex is missing from the default win32 model): install
#   g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix
# and let x86_64-w64-mingw32-g++-posix be used, or point CXX at an equivalent
# `*-posix` compiler.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src_dir="$root_dir/src"
bin_dir="$root_dir/bin"
compiler=${CXX:-x86_64-w64-mingw32-g++-posix}
if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: $compiler not found; install a MinGW-w64 posix toolchain" >&2
    exit 1
fi
mkdir -p "$bin_dir"

common_flags=(-std=c++17 -O2 -static -static-libgcc -static-libstdc++)

# NGX bridge: exports dlss5nr_init3 / dlss5nr_process_v3 / dlss5nr_shutdown.
"$compiler" "${common_flags[@]}" -shared \
    "$src_dir/dlss5nr_bridge.cpp" \
    -o "$bin_dir/dlss5nr_bridge.dll" \
    -ld3d12 -ldxgi -lole32 -luuid -ldwmapi -luser32 -lkernel32
echo "built $bin_dir/dlss5nr_bridge.dll"

# DNR3 transport: a console app, so -municode keeps wmain available.
"$compiler" "${common_flags[@]}" -municode \
    "$src_dir/dlss5nr_host.cpp" \
    -o "$bin_dir/dlss5nr_host.exe" \
    -lole32 -luser32 -lkernel32
echo "built $bin_dir/dlss5nr_host.exe"

# Caller shim: a snippet-ABI trampoline that must stay a real CALL/RET in this
# module, so it is built without whole-program optimization.
"$compiler" "${common_flags[@]}" -shared \
    "$src_dir/caller_shim.cpp" \
    -o "$bin_dir/nvngx.dll_comfy.dll" \
    -ld3d12 -lole32 -luuid
echo "built $bin_dir/nvngx.dll_comfy.dll"

# Build stamp: lets tests and users tell these DNR3 artifacts apart from a
# stale DNR2 build, and from binaries compiled before the current sources.
stamp="$bin_dir/build-info.txt"
{
    echo "protocol=DNR3 (dlss5nr_init3 / dlss5nr_process_v3)"
    echo "compiler=$compiler"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "$src_dir/dlss5nr_bridge.cpp" "$src_dir/dlss5nr_host.cpp" "$src_dir/caller_shim.cpp" \
        | awk -v src="$src_dir/" '{ sub(src, "", $2); print $1 "  source-sha256 " $2 }'
} > "$stamp"
echo "wrote $stamp"
