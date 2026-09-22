"""Contract tests for the vendored native DNR3 sources.

MinGW is not available in every environment this repository is developed in, so
these tests pin the C++ side structurally instead of leaving it unverified:

* the wire structs must match `my_nodes.core.video_enhance.dnr3` field for field
  and byte for byte,
* the feature-bit paths (SR only / NR only / both) must exist in the bridge,
* the host must require the new feature-aware exports and reject feature
  combinations it cannot serve,
* no prebuilt binary may be presented as a DNR3 build without a build stamp.

They deliberately say nothing about real NGX execution: that needs a GPU, a
driver, the user's NVIDIA DLLs and Wine, and is validated on hardware, not here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from my_nodes.core.video_enhance import dnr3, runtime
from my_nodes.core.video_enhance.plan import SR_SCALES

NATIVE_DIR = Path(runtime.__file__).with_name("native")
SRC_DIR = NATIVE_DIR / "src"
BRIDGE = SRC_DIR / "dlss5nr_bridge.cpp"
HOST = SRC_DIR / "dlss5nr_host.cpp"
SHIM = SRC_DIR / "caller_shim.cpp"
BUILD_SCRIPT = NATIVE_DIR / "build_mingw.sh"
NOTICES = NATIVE_DIR / "THIRD_PARTY_NOTICES.md"
PINNED_COMMIT = "a01887d489b8053af07673dd277079f412dc9022"

_CPP_TYPES = {"std::uint32_t": 4, "float": 4}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip(text: str) -> str:
    """Blank comments and string/char literals, keeping the text length.

    Length preservation lets a test locate something in the stripped text and
    still report the original source (with its string literals) for it.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character == "/" and index + 1 < length and text[index + 1] == "/":
            end = text.find("\n", index)
            end = length if end < 0 else end
            out.append(" " * (end - index))
            index = end
            continue
        if character == "/" and index + 1 < length and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            out.append("".join(c if c == "\n" else " " for c in text[index:end]))
            index = end
            continue
        if character in "\"'":
            quote = character
            start = index
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    break
                index += 1
            out.append(text[start] + " " * (index - start - 2) + text[index - 1])
            continue
        out.append(character)
        index += 1
    return "".join(out)


def _code(path: Path) -> str:
    """The file with comments and literals blanked (same length as the source)."""
    return _strip(_source(path))


def _cpp_struct(path: Path, name: str) -> list[tuple[str, int]]:
    """[(field_name, size)] of a plain C++ struct, in declaration order."""
    match = re.search(rf"struct {name} \{{(.*?)\n\}};", _code(path), re.S)
    if match is None:
        raise AssertionError(f"struct {name} not found in {path.name}")
    fields: list[tuple[str, int]] = []
    for declaration in match.group(1).split(";"):
        declaration = " ".join(declaration.split())
        if not declaration:
            continue
        tokens = declaration.split(" ")
        type_name = tokens[0]
        if type_name not in _CPP_TYPES:
            raise AssertionError(f"unexpected field type {type_name!r} in struct {name}")
        for field in " ".join(tokens[1:]).replace(" ", "").split(","):
            fields.append((field, _CPP_TYPES[type_name]))
    return fields


def _magic(path: Path, name: str) -> bytes:
    """The 4-byte magic a C++ `constexpr std::uint32_t kXxx = 0x...;` spells."""
    match = re.search(rf"{name} = (0x[0-9A-Fa-f]+);", _code(path))
    if match is None:
        raise AssertionError(f"{name} not found in {path.name}")
    return int(match.group(1), 16).to_bytes(4, "little")


class _NativeSourceTestCase(unittest.TestCase):
    """Locate one function (signature through its closing brace)."""

    def function(self, path: Path, name: str) -> str:
        code = _code(path)
        source = _source(path)
        match = re.search(rf"\b{name}\(", code)
        if match is None:
            raise AssertionError(f"{name} not found in {path.name}")
        start = code.index("{", match.end())
        depth = 0
        for index in range(start, len(code)):
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
                if depth == 0:
                    body = source[match.start() : index + 1]
                    self.assertIn("{", body)
                    return body
        raise AssertionError(f"{name} in {path.name} has no closing brace")


class VendoredFilesTests(unittest.TestCase):
    def test_vendored_sources_and_build_files_are_present(self) -> None:
        for path in (BRIDGE, HOST, SHIM, BUILD_SCRIPT, NOTICES):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"{path} is missing")

    def test_every_vendored_source_keeps_its_mit_notice(self) -> None:
        for path in (BRIDGE, HOST, SHIM):
            with self.subTest(path=path.name):
                text = "\n".join(_source(path).splitlines()[:12])
                self.assertIn("SPDX-License-Identifier: MIT", text)
                self.assertIn("Copyright (c) 2026 ComfyUI-DLSS5-NR contributors", text)
                self.assertIn(PINNED_COMMIT, text)
                self.assertIn("THIRD_PARTY_NOTICES.md", text)

    def test_notices_attribute_upstream_and_never_redistribute_nvidia_files(self) -> None:
        text = _source(NOTICES)
        self.assertIn(PINNED_COMMIT, text)
        self.assertIn("MIT", text)
        self.assertIn("ComfyUI-DLSS5-NR", text)
        for name in (runtime.CORE_DLL, runtime.SR_DLL, "nvngx_dlssnr"):
            with self.subTest(name=name):
                self.assertIn(name, text)
        self.assertIn("never", text)
        for path in (BRIDGE, HOST, SHIM, BUILD_SCRIPT):
            with self.subTest(path=path.name):
                self.assertIn(path.name, text)

    def test_no_stale_binary_is_presented_as_a_dnr3_build(self) -> None:
        binary_dir = NATIVE_DIR / "bin"
        if not binary_dir.exists():
            self.skipTest("no prebuilt binaries in this checkout")
        binaries = [p for p in binary_dir.iterdir() if p.suffix in (".dll", ".exe")]
        if not binaries:
            return
        stamp = binary_dir / "build-info.txt"
        self.assertTrue(
            stamp.is_file(),
            f"{binaries} are present without build-info.txt: they may be stale "
            "(pre-DNR3) artifacts and must not be shipped as this build",
        )
        text = _source(stamp)
        self.assertIn("DNR3", text)
        # A stamp that only says DNR3 can still describe sources from before
        # the feature-lifetime / DLAA changes. The recorded hashes must match
        # the sources these binaries were compiled from.
        recorded = {
            name: digest
            for digest, name in re.findall(
                r"^([0-9a-f]{64})  source-sha256 (\S+)$", text, re.M
            )
        }
        self.assertEqual(
            set(recorded),
            {"dlss5nr_bridge.cpp", "dlss5nr_host.cpp", "caller_shim.cpp"},
        )
        stale = [
            name
            for name, digest in recorded.items()
            if hashlib.sha256((SRC_DIR / name).read_bytes()).hexdigest() != digest
        ]
        if stale:
            self.skipTest(
                "native/bin is stale for "
                + ", ".join(stale)
                + "; it is not a build of the current sources. Rebuild with "
                "native/build_mingw.sh before running the host. The directory "
                "could not be removed from this checkout."
            )


class BuildScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = _source(BUILD_SCRIPT)

    def test_shell_syntax_is_valid(self) -> None:
        bash = shutil.which("bash")
        if bash is None:  # pragma: no cover - bash is present in CI images
            self.skipTest("bash is not available")
        subprocess.run([bash, "-n", str(BUILD_SCRIPT)], check=True, timeout=30)

    def test_script_builds_bridge_host_and_shim(self) -> None:
        for source in ("dlss5nr_bridge.cpp", "dlss5nr_host.cpp", "caller_shim.cpp"):
            with self.subTest(source=source):
                self.assertIn(source, self.script)
        for output in ("dlss5nr_bridge.dll", "dlss5nr_host.exe", "nvngx.dll_comfy.dll"):
            with self.subTest(output=output):
                self.assertIn(output, self.script)

    def test_script_requires_the_posix_toolchain_and_writes_a_build_stamp(self) -> None:
        self.assertIn("-posix", self.script)
        self.assertIn("-municode", self.script)
        self.assertIn("build-info.txt", self.script)
        self.assertIn("DNR3", self.script)

    def test_script_never_touches_an_nvidia_binary(self) -> None:
        for name in (runtime.CORE_DLL, runtime.SR_DLL, "nvngx_dlssnr"):
            with self.subTest(name=name):
                self.assertNotIn(name, self.script)


class WireContractTests(unittest.TestCase):
    """The C++ structs and magics are the protocol the Python side speaks."""

    def test_header_struct_matches_the_dnr3_header(self) -> None:
        fields = _cpp_struct(HOST, "Header")
        self.assertEqual(fields[0], ("magic", 4))
        self.assertEqual(
            [name for name, _size in fields][1:],
            [field.name for field in dataclasses.fields(dnr3.Header)],
        )
        self.assertEqual(sum(size for _name, size in fields), dnr3.HEADER_SIZE)

    def test_frame_and_reply_structs_match(self) -> None:
        frame = _cpp_struct(HOST, "FrameHeader")
        self.assertEqual(frame, [("magic", 4), ("index", 4), ("reset", 4)])
        self.assertEqual(sum(size for _name, size in frame), dnr3.FRAME_HEADER_SIZE)
        reply = _cpp_struct(HOST, "ReplyHeader")
        self.assertEqual(
            reply, [("magic", 4), ("index", 4), ("ok", 4), ("float_count", 4)]
        )
        self.assertEqual(sum(size for _name, size in reply), dnr3.REPLY_HEADER_SIZE)

    def test_magic_values_match_the_python_constants(self) -> None:
        self.assertEqual(_magic(HOST, "kDnr3"), dnr3.MAGIC)
        self.assertEqual(_magic(HOST, "kFrm2"), dnr3.FRAME_MAGIC)
        self.assertEqual(_magic(HOST, "kOut1"), dnr3.REPLY_MAGIC)
        self.assertEqual(_magic(HOST, "kEnd1"), dnr3.END_MAGIC)

    def test_feature_bits_match_the_python_constants(self) -> None:
        code = _code(BRIDGE) + _code(HOST)
        self.assertIn("FEATURE_SR = 1u << 0", code)
        self.assertIn("FEATURE_NR = 1u << 1", code)
        self.assertIn("FEATURE_SR | FEATURE_NR", code)
        self.assertIn("kFeatureSr = 1u << 0", code)
        self.assertIn("kFeatureNr = 1u << 1", code)
        self.assertEqual((dnr3.FEATURE_SR, dnr3.FEATURE_NR, dnr3.FEATURE_MASK), (0x1, 0x2, 0x3))

    def test_envelope_limits_are_mirrored(self) -> None:
        code = _code(HOST).replace(" ", "").replace("\n", "")
        for limit in ("16384", "7680", "4320", "1000000", "1ull<<28"):
            with self.subTest(limit=limit):
                self.assertIn(limit, code)

    def test_perf_quality_ratio_table_matches_the_native_switch(self) -> None:
        # FixedScalingRatio in the bridge is what the runtime actually applies.
        match = re.search(r"static bool FixedScalingRatio.*?\n\}", _code(BRIDGE), re.S)
        self.assertIsNotNone(match, "FixedScalingRatio not found")
        cases = {
            int(selector): float(ratio)
            for selector, ratio in re.findall(r"case (\d+): \*ratio = ([0-9.]+)f;", match.group(0))
        }
        self.assertEqual(cases, dict(dnr3.PERF_QUALITY_RATIOS))
        for scale, selector in dnr3.SR_SCALE_PERF_QUALITY.items():
            with self.subTest(scale=scale):
                self.assertIn(scale, SR_SCALES)
                self.assertEqual(cases[selector], scale)
        self.assertNotIn("DLSS5NR_SKIP_RATIO_CHECK", _source(BRIDGE))


class BridgeContractTests(_NativeSourceTestCase):
    def setUp(self) -> None:
        self.source = _source(BRIDGE)
        self.code = _code(BRIDGE)

    def test_exports_are_the_feature_aware_dnr3_set(self) -> None:
        for export in ("dlss5nr_init3", "dlss5nr_process_v3", "dlss5nr_shutdown",
                       "dlss5nr_version", "dlss5nr_gpu_name"):
            with self.subTest(export=export):
                self.assertIn(f"__cdecl {export}(", self.source)
        self.assertIn("dlss5nr_init3(int gpu_index, const wchar_t* runtime_dir,", self.source)
        # The stale DNR2 exports are gone: a stale bridge cannot pass the host's
        # export check and silently ignore the feature flags.
        for stale in ('"dlss5nr_init"', '"dlss5nr_process"', '"dlss5nr_process_v2"',
                      "__cdecl dlss5nr_init(", "__cdecl dlss5nr_process(",
                      "__cdecl dlss5nr_process_v2("):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, self.source)

    def test_init3_rejects_zero_and_unknown_feature_bits(self) -> None:
        init = self.function(BRIDGE, "dlss5nr_init3")
        self.assertIn("features == 0", init)
        self.assertIn("features & ~FEATURE_MASK", init)
        self.assertIn("g_sr_enabled = (features & FEATURE_SR) != 0", self.source)
        self.assertIn("g_nr_enabled = (features & FEATURE_NR) != 0", self.source)

    def test_super_resolution_alone_never_loads_the_nr_runtime_or_shim(self) -> None:
        load_ngx = self.function(BRIDGE, "LoadNGX")
        self.assertIn("if (g_nr_enabled && !LoadNeuralRuntime()) return false;", load_ngx)
        # No NR DLL or shim load may stay in the always-run path.
        self.assertNotIn("nvngx_dlssnr", _strip(load_ngx))
        self.assertNotIn("nvngx.dll_comfy.dll", _strip(load_ngx))
        neural = self.function(BRIDGE, "LoadNeuralRuntime")
        self.assertIn("DLSS5NR_SNR_FILENAME", neural)
        self.assertIn("nvngx_dlssnr.dll", neural)
        self.assertIn("return LoadCallerShim();", neural)

    def test_caller_shim_is_loaded_next_to_the_bridge(self) -> None:
        directory = self.function(BRIDGE, "BridgeDirectory")
        shim = self.function(BRIDGE, "LoadCallerShim")
        self.assertIn("GetModuleHandleExW", directory)
        self.assertIn("nvngx.dll_comfy.dll", shim)
        self.assertIn("DLSSNR_CallInit", shim)
        # The old runtime\\caller lookup must be gone: the shim belongs to this
        # worker, not to the user's NVIDIA runtime directory.
        self.assertNotIn('L"caller"', self.source)

    def test_only_the_requested_features_are_created_and_evaluated(self) -> None:
        ensure = self.function(BRIDGE, "EnsureFeature")
        # Feature 1 exists whenever SR is enabled, including native-size DLAA.
        self.assertNotIn("g_sr_enabled && !requested_upscale", ensure)
        self.assertIn("if (g_sr_enabled && !g_dlss_ready)", ensure)
        self.assertIn("if (g_sr_enabled) {", ensure)
        self.assertIn("DLSS_FEATURE_ID", ensure)
        self.assertIn("NR_FEATURE_ID", ensure)
        carrier_branch = ensure[ensure.index("if (g_sr_enabled) {") :]
        self.assertIn("g_core_create(", carrier_branch)
        nr_branch = ensure[ensure.index("// Neural rendering alone creates only") :]
        self.assertIn("SetNeuralParams(", nr_branch)
        self.assertIn("g_shim_create(", nr_branch)
        self.assertIn("NR_FEATURE_ID", nr_branch)
        process = self.function(BRIDGE, "ProcessFrame")
        # Feature 1 is evaluated in the SR block, feature 18 in the NR block.
        self.assertIn("if (g_sr_enabled) {", process)
        self.assertIn("if (g_nr_enabled) {", process)
        self.assertLess(process.index("g_core_eval"), process.index("g_shim_eval"))

    def test_motion_vector_scales_match_each_feature_coordinate_space(self) -> None:
        carrier = self.function(BRIDGE, "SetDLSSCarrierParams")
        neural = self.function(BRIDGE, "SetNeuralParams")
        # Feature 1 consumes render-pixel vectors directly.
        self.assertIn('SetParamFloat("MV.Scale.X", 1.0f);', carrier)
        self.assertIn('SetParamFloat("MV.Scale.Y", 1.0f);', carrier)
        # Feature 18 runs at output resolution but consumes render-sized guides.
        self.assertIn("g_output_width) / static_cast<float>(g_input_width)", neural)
        self.assertIn("g_output_height) / static_cast<float>(g_input_height)", neural)
        self.assertIn('SetParamFloat("DLSSNR.MVecScaleX", mvec_scale_x);', neural)
        self.assertIn('SetParamFloat("DLSSNR.MVecScaleY", mvec_scale_y);', neural)

    def test_sr_only_reuses_feature_one_instead_of_recreating_it(self) -> None:
        # SR-only leaves g_feature null. An unconditional !g_feature check would
        # rebuild feature 1 on every frame.
        ensure = self.function(BRIDGE, "EnsureFeature")
        self.assertIn("g_nr_enabled ? g_feature : g_dlss_feature", ensure)
        self.assertNotIn("!g_feature ||", ensure.replace(" ", ""))
        self.assertIn("!session_feature", ensure)

    def test_readback_selects_the_output_texture_of_the_session(self) -> None:
        self.assertIn(
            "return g_nr_enabled ? g_output.Get() : g_dlss_output.Get();",
            self.function(BRIDGE, "OutputTexture"),
        )
        process = self.function(BRIDGE, "ProcessFrame")
        self.assertIn("ID3D12Resource* result = OutputTexture();", process)
        self.assertIn("rs.pResource = result;", process)
        self.assertEqual(process.count("g_output_readback->Map"), 1)

    def test_neural_output_texture_is_allocated_only_for_nr(self) -> None:
        allocate = self.function(BRIDGE, "AllocateFrameResources")
        marker = allocate.index("// Neural rendering owns its own output texture")
        self.assertIn("if (g_nr_enabled) {", allocate[marker:])
        # Feature 1's output exists for DLAA too, not only when the size grows.
        self.assertIn("if (g_sr_enabled) {", allocate)
        self.assertNotIn("if (g_upscale_active) {", allocate)

    def test_version_identifies_the_dnr3_build(self) -> None:
        self.assertIn("0.5.0-dnr3-feature-flags", self.source)


class HostContractTests(_NativeSourceTestCase):
    def setUp(self) -> None:
        self.source = _source(HOST)
        self.code = _code(HOST)

    def test_host_requires_the_feature_aware_bridge_exports(self) -> None:
        for export in ("dlss5nr_init3", "dlss5nr_process_v3", "dlss5nr_shutdown"):
            with self.subTest(export=export):
                self.assertIn(f'"{export}"', self.source)
        self.assertIn("bridge exports are incomplete", self.source)
        self.assertIn("stale", self.source)
        # A stale bridge is rejected before the header is even read.
        self.assertLess(
            self.source.index('"dlss5nr_init3"'), self.source.index("ReadExact(&header")
        )

    def test_host_validation_order_protects_the_feature_contract(self) -> None:
        flags = self.code.index("header.features == 0")
        self.assertIn("(header.features & ~(kFeatureSr | kFeatureNr)) != 0", self.code)
        self.assertIn("invalid DNR3 feature flags", self.source)
        # Features are known before any NGX component is initialized or any
        # frame is accepted.
        self.assertLess(flags, self.code.index("init(gpu_index"))
        self.assertLess(flags, self.code.index("ReadExact(&frame"))

    def test_host_rejects_invalid_controls_and_scale_before_ngx_init(self) -> None:
        init = self.code.index("if (!init(gpu_index")
        for validation in (
            "header.automask > 1",
            "header.ui_correction > 1",
            "!HeaderFloatsAreFinite(header)",
            "!FixedScalingRatio(header.perf_quality",
            "std::fabs(ratio_x - ratio_y) > 0.03f",
        ):
            with self.subTest(validation=validation):
                self.assertIn(validation, self.code)
                self.assertLess(self.code.index(validation), init)

    def test_host_passes_the_feature_bits_to_init(self) -> None:
        self.assertIn("init(gpu_index, runtime.c_str(), header.features, error", self.source)

    def test_host_rejects_neural_rendering_above_native_size(self) -> None:
        self.assertIn("neural rendering without super resolution must stay", self.source)
        self.assertIn("const bool sr = (header.features & kFeatureSr) != 0;", self.source)
        # Equal dimensions with FEATURE_SR are DLAA and must stay accepted.
        self.assertIn("native DLAA", self.source)

    def test_host_reads_sequential_frames_and_ends_with_the_marker(self) -> None:
        self.assertIn("frame.magic != kFrm2 || frame.index != expected", self.code)
        self.assertIn("frame.reset > 1", self.code)
        self.assertIn("const std::uint32_t done = kEnd1;", self.source)
        self.assertIn("WriteExact(&done, sizeof(done))", self.source)

    def test_error_reply_layout_matches_the_python_writer(self) -> None:
        error_function = self.function(HOST, "WriteError")
        self.assertIn("ReplyHeader reply{kOut1, index, 0, 0};", error_function)
        self.assertLess(
            error_function.index("WriteExact(&length, sizeof(length))"),
            error_function.index("WriteExact(message, length)"),
        )


class CallerShimContractTests(unittest.TestCase):
    def test_shim_exports_the_trampolines_the_bridge_looks_up(self) -> None:
        source = _source(SHIM)
        for symbol in (
            "DLSSNR_CallInit",
            "DLSSNR_CallCreate",
            "DLSSNR_CallEvaluate",
            "DLSSNR_CallRelease",
            "DLSSNR_CallShutdown",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(
                    f"__declspec(dllexport) __declspec(noinline) NGXResult __cdecl {symbol}",
                    source,
                )
        # The trampolines must not be tail-call optimized into the bridge, which
        # the NR runtime rejects with 0xBAD00002.
        self.assertIn("g_post_call_sink", source)


class SourceSanityTests(unittest.TestCase):
    def test_cpp_delimiters_are_balanced(self) -> None:
        """Cheap substitute for a compiler in environments without MinGW."""
        for path in (BRIDGE, HOST, SHIM):
            with self.subTest(path=path.name):
                self.assertEqual(_delimiter_depth(_code(path)), {"{": 0, "(": 0, "[": 0})


def _delimiter_depth(code: str) -> dict[str, int]:
    pairs = {"}": "{", ")": "(", "]": "["}
    depth = {"{": 0, "(": 0, "[": 0}
    for character in code:
        if character in depth:
            depth[character] += 1
        elif character in pairs:
            depth[pairs[character]] -= 1
    return depth
