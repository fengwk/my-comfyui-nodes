# Third-party notices

This directory contains code vendored from another project, plus the build
script for it. Our own code is everything else in
`my_nodes/core/video_enhance/`.

## ComfyUI-DLSS5-NR (RH-RunningHub)

- Upstream: the MIT-licensed `ComfyUI-DLSS5-NR` project ("RH-RunningHub",
  ComfyUI-DLSS5-NR contributors).
- Pinned commit: `a01887d489b8053af07673dd277079f412dc9022`
- License: MIT (reproduced below).

Vendored files, and what changed:

| File | Changes |
| --- | --- |
| `src/dlss5nr_bridge.cpp` | Adapted for DNR3: header/feature handling, feature-gated NGX loading, new exports, shim loaded next to the bridge. |
| `src/dlss5nr_host.cpp` | Adapted for DNR3: `DNR3` magic, feature field, feature-flag validation, feature-aware init call, new bridge export check. |
| `src/caller_shim.cpp` | Unmodified except the provenance comment. |
| `build_mingw.sh` | New in this repository: builds bridge, host and shim (upstream's `src/build_host_mingw.sh` only built the first two). |

### Modifications in detail

`src/dlss5nr_bridge.cpp`

- Upstream had no feature selection at all: `dlss5nr_init()` took no feature
  argument and neural rendering was switched on implicitly by the parameters of
  each process call, while the header field `profile` was read and then never
  used. DNR3 replaces that: the new export
  `dlss5nr_init3(gpu_index, runtime_dir, features, err, err_cap)` takes an
  explicit feature bitmask (`FEATURE_SR = 0x1`, `FEATURE_NR = 0x2`) and the
  bridge stays bound to it for the whole session, so a session cannot switch
  features after NGX was initialized.
- NGX components are loaded per feature: a super-resolution-only session loads
  only the NGX core and never `nvngx_dlssnr.dll` or the caller shim. Feature 1
  covers both enlargement and native DLAA (1.0x, `perf_quality` 5) and is
  stored in its own handle so an SR-only session does not recreate it every
  frame. A neural-rendering-only session creates and evaluates only feature 18
  at native resolution. SR+NR runs feature 1 first (DLAA or SR) and feeds that
  output to the feature-18 post-pass.
- The caller shim is resolved from the directory of `dlss5nr_bridge.dll`
  (via `GetModuleHandleExW`), not from `<runtime>/caller/nvngx.dll_comfy.dll`.
- `AllocateFrameResources` allocates only the output texture the session uses,
  and the read-back/barrier path selects the matching surface
  (`OutputTexture()`).
- Exports removed: `dlss5nr_init`, `dlss5nr_process` (legacy image ABI) and
  `dlss5nr_process_v2`, all replaced by `dlss5nr_init3` / `dlss5nr_process_v3`,
  so a stale DNR2 bridge is rejected instead of silently ignoring requests.
- `dlss5nr_version()` reports `0.5.0-dnr3-feature-flags`.

`src/dlss5nr_host.cpp`

- Header magic is `DNR3` (`0x33524E44`) and the former `profile` field carries
  the feature bitmask, so the header keeps its 72-byte size.
- The host validates the feature bits (rejecting zero and unknown bits) before
  loading NGX, rejects neural rendering without super resolution at a scaled
  size (equal dimensions with super resolution stay valid as native DLAA), and
  passes the bits to `dlss5nr_init3`.
- It requires the `dlss5nr_init3` / `dlss5nr_process_v3` exports and fails with
  an explicit "stale bridge" message when they are missing.

Nothing else in these files was touched; the upstream MIT headers are kept
verbatim at the top of each file.

## NVIDIA DLSS/NGX runtime binaries

`_nvngx.dll`, `nvngx_dlss.dll` and `nvngx_dlssnr*.dll` are proprietary NVIDIA
files. They are **not** part of this repository and are **never** downloaded,
copied or redistributed by this code: the user installs them (usually into
`<ComfyUI>/models/dlss5`) and the worker only validates that the files the
selected features need are present. The DLSS/NGX runtime is subject to NVIDIA's
own license terms.

## MIT license (ComfyUI-DLSS5-NR)

```
MIT License

Copyright (c) 2026 ComfyUI-DLSS5-NR contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
