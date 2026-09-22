from __future__ import annotations

import dataclasses
import io
import struct
import unittest

import numpy as np

from my_nodes.core.video_enhance import dnr3
from my_nodes.core.video_enhance.plan import SR_SCALES


def _header(**overrides) -> dnr3.Header:
    """A valid SR-only session header (2x, the most common request)."""
    values = {
        "input_width": 4,
        "input_height": 6,
        "output_width": 8,
        "output_height": 12,
        "warmup_frames": 0,
        "frame_count": 2,
        "perf_quality": 0,
        "features": dnr3.FEATURE_SR,
        "preset": 0,
        "style": 0,
        "automask": True,
        "ui_correction": False,
        "intensity": 1.0,
        "tone": 1.0,
        "structure": 1.5,
        "skin": -1.0,
        "global_tone": -1.0,
    }
    values.update(overrides)
    return dnr3.Header(**values)


class HeaderLayoutTests(unittest.TestCase):
    def test_wire_layout_keeps_the_upstream_fixed_size(self) -> None:
        # 4s + 12 uint32 + 5 float32: the feature bitmask replaced the field
        # upstream left unused, so the header size must not change.
        self.assertEqual(dnr3.HEADER.size, 72)
        self.assertEqual(dnr3.FRAME_HEADER.size, 12)
        self.assertEqual(dnr3.REPLY_HEADER.size, 16)
        self.assertEqual(dnr3.END_MAGIC, b"END1")
        self.assertEqual(dnr3.MAGIC, b"DNR3")

    def test_pack_places_the_feature_bits_in_the_profile_slot(self) -> None:
        packed = _header(features=dnr3.FEATURE_SR | dnr3.FEATURE_NR).pack()
        self.assertEqual(packed[:4], b"DNR3")
        (features,) = struct.unpack_from("<I", packed, 4 + 7 * 4)
        self.assertEqual(features, 3)

    def test_dataclass_fields_match_the_declared_wire_order(self) -> None:
        names = [field.name for field in dataclasses.fields(dnr3.Header)]
        self.assertEqual(
            names,
            [
                "input_width",
                "input_height",
                "output_width",
                "output_height",
                "warmup_frames",
                "frame_count",
                "perf_quality",
                "features",
                "preset",
                "style",
                "automask",
                "ui_correction",
                "intensity",
                "tone",
                "structure",
                "skin",
                "global_tone",
            ],
        )

    def test_round_trip_preserves_every_field(self) -> None:
        header = _header(
            features=dnr3.FEATURE_SR | dnr3.FEATURE_NR,
            warmup_frames=1,
            ui_correction=True,
            automask=False,
            intensity=0.75,
            global_tone=0.5,
        )
        self.assertEqual(dnr3.Header.parse(header.pack()), header)

    def test_parse_rejects_short_and_stale_headers(self) -> None:
        with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
            dnr3.Header.parse(b"\x00" * 71)
        self.assertIn("72 bytes", str(raised.exception))
        # A DNR2 payload must never be accepted as DNR3.
        stale = bytearray(_header().pack())
        stale[:4] = b"DNR2"
        with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
            dnr3.Header.parse(bytes(stale))
        self.assertIn("magic", str(raised.exception))

    def test_parse_reports_validation_failures_as_protocol_errors(self) -> None:
        broken = bytearray(_header().pack())
        struct.pack_into("<I", broken, 4 + 7 * 4, 0)  # features = 0
        with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
            dnr3.Header.parse(bytes(broken))
        self.assertIn("features", str(raised.exception))


class FeatureFlagTests(unittest.TestCase):
    def test_each_supported_combination_is_accepted(self) -> None:
        for features in (dnr3.FEATURE_SR, dnr3.FEATURE_NR, dnr3.FEATURE_SR | dnr3.FEATURE_NR):
            with self.subTest(features=features):
                header = _nr_header() if not features & dnr3.FEATURE_SR else _header(features=features)
                self.assertEqual(header.features, features)
                self.assertEqual(header.sr_enabled, bool(features & dnr3.FEATURE_SR))
                self.assertEqual(header.nr_enabled, bool(features & dnr3.FEATURE_NR))

    def test_zero_and_unknown_bits_are_rejected(self) -> None:
        for features in (0, 0b100, 0b111, -1):
            with self.subTest(features=features):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    dnr3.check_features(features)

    def test_non_integer_features_are_rejected(self) -> None:
        for features in (True, 1.0, "1", None):
            with self.subTest(features=features):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    dnr3.check_features(features)


def _nr_header(**overrides) -> dnr3.Header:
    """A valid neural-rendering-only header: native size, native quality."""
    values = {
        "input_width": 5,
        "input_height": 3,
        "output_width": 5,
        "output_height": 3,
        "perf_quality": dnr3.NATIVE_PERF_QUALITY,
        "features": dnr3.FEATURE_NR,
    }
    values.update(overrides)
    return _header(**values)


class HeaderValidationTests(unittest.TestCase):
    def test_dimension_envelope_is_enforced(self) -> None:
        cases = (
            {"input_width": 0},
            {"input_height": 0},
            {"output_width": dnr3.MAX_DIM + 1, "output_height": 1, "perf_quality": 5},
            {"output_width": 8192, "output_height": 8192, "perf_quality": 5},
            {"input_width": dnr3.MAX_DIM, "input_height": dnr3.MAX_DIM},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    _header(**overrides)

    def test_frame_and_warmup_bounds(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(frame_count=0)
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(frame_count=dnr3.MAX_FRAMES + 1)
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(frame_count=2, warmup_frames=3)
        self.assertEqual(_header(frame_count=2, warmup_frames=2).warmup_frames, 2)

    def test_perf_quality_must_be_a_known_selector(self) -> None:
        for quality in (4, 6, -1, True):
            with self.subTest(quality=quality):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    _header(perf_quality=quality)

    def test_super_resolution_requires_the_ratio_of_its_selector(self) -> None:
        # perf_quality 0 means 2.0x: a 1.5x output must be rejected up front.
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            _header(perf_quality=0, output_width=6, output_height=9)
        self.assertIn("perf_quality", str(raised.exception))
        # The matching combination stays valid.
        self.assertEqual(_header(perf_quality=0).output_width, 8)

    def test_feature_sr_at_native_size_is_dlaa(self) -> None:
        # FEATURE_SR + perf_quality 5 + equal dimensions is native DLAA, not NR.
        header = _header(
            perf_quality=dnr3.NATIVE_PERF_QUALITY, output_width=4, output_height=6
        )
        self.assertTrue(header.sr_enabled)
        self.assertFalse(header.nr_enabled)
        self.assertEqual(header.perf_quality, dnr3.NATIVE_PERF_QUALITY)
        # A scaled output still cannot claim the native selector.
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(perf_quality=dnr3.NATIVE_PERF_QUALITY, output_width=8, output_height=12)

    def test_aspect_ratio_must_be_preserved(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            _header(output_width=8, output_height=9)
        self.assertIn("aspect", str(raised.exception))

    def test_neural_rendering_alone_stays_native(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            _nr_header(output_width=10, output_height=6)
        self.assertIn("native resolution", str(raised.exception))
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            _nr_header(perf_quality=0)
        self.assertIn("native", str(raised.exception))

    def test_dlaa_then_neural_rendering_stays_at_native_size(self) -> None:
        header = _header(
            features=dnr3.FEATURE_SR | dnr3.FEATURE_NR,
            perf_quality=dnr3.NATIVE_PERF_QUALITY,
            output_width=4,
            output_height=6,
        )
        self.assertTrue(header.sr_enabled and header.nr_enabled)
        self.assertEqual((header.output_width, header.output_height), (4, 6))
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(
                features=dnr3.FEATURE_SR | dnr3.FEATURE_NR,
                perf_quality=dnr3.NATIVE_PERF_QUALITY,
            )

    def test_parse_rejects_header_booleans_other_than_zero_or_one(self) -> None:
        for name, offset in (("automask", 4 + 10 * 4), ("ui_correction", 4 + 11 * 4)):
            broken = bytearray(_header().pack())
            struct.pack_into("<I", broken, offset, 2)
            with self.subTest(name=name):
                with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
                    dnr3.Header.parse(bytes(broken))
                self.assertIn(name, str(raised.exception))
                self.assertIn("0 or 1", str(raised.exception))

    def test_super_resolution_and_neural_rendering_use_the_carrier_ratio(self) -> None:
        header = _header(features=dnr3.FEATURE_SR | dnr3.FEATURE_NR, perf_quality=2,
                         output_width=6, output_height=9)
        self.assertEqual((header.output_width, header.output_height), (6, 9))
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(features=dnr3.FEATURE_SR | dnr3.FEATURE_NR, perf_quality=2)

    def test_non_finite_strengths_are_rejected(self) -> None:
        for name in ("intensity", "tone", "structure", "skin", "global_tone"):
            with self.subTest(name=name):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    _header(**{name: float("nan")})

    def test_types_are_checked(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(automask=1)
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(preset=-1)
        with self.assertRaises(dnr3.Dnr3ValidationError):
            _header(frame_count=True)

    def test_payload_counts_follow_the_declared_dimensions(self) -> None:
        header = _header()
        self.assertEqual(header.input_pixels, 24)
        self.assertEqual(header.input_floats, 72)
        self.assertEqual(header.motion_words, 48)
        self.assertEqual(header.output_floats, 8 * 12 * 3)


class PerfQualityMappingTests(unittest.TestCase):
    def test_every_node_scale_maps_to_a_selector(self) -> None:
        for scale in SR_SCALES:
            with self.subTest(scale=scale):
                quality = dnr3.perf_quality_for_scale(scale)
                self.assertIn(quality, dnr3.PERF_QUALITY_RATIOS)
                self.assertAlmostEqual(dnr3.PERF_QUALITY_RATIOS[quality], scale, places=3)

    def test_unknown_scales_are_rejected(self) -> None:
        for scale in (1.25, 4.0):
            with self.subTest(scale=scale):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    dnr3.perf_quality_for_scale(scale)

    def test_native_selector_means_native_size(self) -> None:
        self.assertEqual(dnr3.PERF_QUALITY_RATIOS[dnr3.NATIVE_PERF_QUALITY], 1.0)


class FrameAndReplyTests(unittest.TestCase):
    def test_frame_header_round_trip(self) -> None:
        raw = dnr3.pack_frame_header(7, True)
        self.assertEqual(len(raw), dnr3.FRAME_HEADER_SIZE)
        self.assertEqual(dnr3.parse_frame_header(raw), (7, True))
        self.assertEqual(dnr3.parse_frame_header(dnr3.pack_frame_header(0, False)), (0, False))

    def test_frame_header_rejects_bad_magic_and_flags(self) -> None:
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.parse_frame_header(b"XXXX" + struct.pack("<II", 0, 0))
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.parse_frame_header(dnr3.FRAME_HEADER.pack(dnr3.FRAME_MAGIC, 0, 2))
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.parse_frame_header(b"\x00" * 11)

    def test_reply_header_round_trip_and_rejection(self) -> None:
        self.assertEqual(dnr3.parse_reply_header(dnr3.pack_reply_header(3, 96)), (3, True, 96))
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.parse_reply_header(dnr3.REPLY_HEADER.pack(b"XXXX", 3, 1, 96))
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.parse_reply_header(dnr3.REPLY_HEADER.pack(dnr3.REPLY_MAGIC, 3, 1, 96)[:12])

    def test_error_reply_round_trip(self) -> None:
        raw = dnr3.pack_error_reply(4, "no NR runtime")
        index, ok, float_count = dnr3.parse_reply_header(raw[: dnr3.REPLY_HEADER_SIZE])
        self.assertEqual((index, ok, float_count), (4, False, 0))
        length = dnr3.check_error_length(raw[dnr3.REPLY_HEADER_SIZE :][:4])
        text = raw[dnr3.REPLY_HEADER_SIZE + 4 :].decode("utf-8")
        self.assertEqual((length, text), (len(text), "no NR runtime"))

    def test_error_text_is_bounded(self) -> None:
        raw = dnr3.pack_error_reply(0, "x" * 100000)
        self.assertEqual(dnr3.check_error_length(raw[16:20]), dnr3.MAX_ERROR_BYTES)
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.check_error_length(struct.pack("<I", dnr3.MAX_ERROR_BYTES + 1))

    def test_pack_frame_header_requires_a_valid_index_and_bool_reset(self) -> None:
        for index in (-1, True, "0"):
            with self.subTest(index=index):
                with self.assertRaises(dnr3.Dnr3ValidationError):
                    dnr3.pack_frame_header(index, False)
        # 1 and 0 are not bools; bool() would have silently accepted them.
        for reset in (1, 0, "yes", None):
            with self.subTest(reset=reset):
                with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
                    dnr3.pack_frame_header(0, reset)
                self.assertIn("reset", str(raised.exception))


class PayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.linspace(0.0, 1.0, 4 * 6 * 3, dtype=np.float32).reshape((6, 4, 3))

    def test_rgb_payload_is_float32_and_round_trips(self) -> None:
        payload = dnr3.rgb_payload(self.rgb, 4, 6)
        self.assertEqual(len(payload), 4 * 6 * 3 * 4)
        decoded = dnr3.decode_rgb(payload, 4, 6)
        self.assertEqual(decoded.shape, (6, 4, 3))
        np.testing.assert_allclose(decoded, self.rgb)

    def test_rgb_contract_rejects_wrong_dtype_and_shape(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            dnr3.rgb_payload(self.rgb.astype(np.float64), 4, 6)
        self.assertIn("float32", str(raised.exception))
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            dnr3.rgb_payload(self.rgb.reshape((4, 6, 3)), 4, 6)
        self.assertIn("shape", str(raised.exception))

    def test_rgb_values_must_be_finite(self) -> None:
        broken = self.rgb.copy()
        broken[0, 0, 0] = np.inf
        with self.assertRaises(dnr3.Dnr3ValidationError):
            dnr3.rgb_payload(broken, 4, 6)
        with self.assertRaises(dnr3.Dnr3ProtocolError):
            dnr3.decode_rgb(np.full((6, 4, 3), np.nan, dtype=np.float32).tobytes(), 4, 6)

    def test_decode_rgb_rejects_wrong_payload_size(self) -> None:
        with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
            dnr3.decode_rgb(b"\x00" * 16, 4, 6)
        self.assertIn("288", str(raised.exception))

    def test_rgb_payload_is_a_copy_of_non_contiguous_input(self) -> None:
        strided = np.ascontiguousarray(self.rgb)[:, ::-1, :]
        payload = dnr3.rgb_payload(strided, 4, 6)
        np.testing.assert_allclose(
            dnr3.decode_rgb(payload, 4, 6), np.ascontiguousarray(self.rgb[:, ::-1, :])
        )

    def test_motion_accepts_fp16_and_uint16_bits(self) -> None:
        motion = np.zeros((6, 4, 2), dtype=np.float16)
        motion[2, 1, 0] = np.float16(0.5)
        payload = dnr3.motion_payload(motion, 4, 6)
        self.assertEqual(len(payload), 6 * 4 * 2 * 2)
        self.assertEqual(np.frombuffer(payload, dtype="<u2")[2 * 4 * 2 + 1 * 2], motion[2, 1, 0].view(np.uint16))
        self.assertEqual(dnr3.motion_payload(motion.view(np.uint16), 4, 6), payload)

    def test_motion_none_means_static_frames(self) -> None:
        payload = dnr3.motion_payload(None, 4, 6)
        self.assertEqual(len(payload), 6 * 4 * 2 * 2)
        self.assertFalse(np.frombuffer(payload, dtype="<u2").any())

    def test_float16_motion_rejects_non_finite_values(self) -> None:
        for value in (np.float16(np.inf), np.float16(np.nan)):
            motion = np.zeros((6, 4, 2), dtype=np.float16)
            motion[0, 0, 1] = value
            with self.subTest(value=str(value)):
                with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
                    dnr3.motion_payload(motion, 4, 6)
                self.assertIn("non-finite", str(raised.exception))

    def test_motion_contract_is_enforced(self) -> None:
        with self.assertRaises(dnr3.Dnr3ValidationError) as raised:
            dnr3.motion_payload(np.zeros((6, 4, 2), dtype=np.float32), 4, 6)
        self.assertIn("float16", str(raised.exception))
        with self.assertRaises(dnr3.Dnr3ValidationError):
            dnr3.motion_payload(np.zeros((4, 6, 2), dtype=np.float16), 4, 6)


class ReadExactTests(unittest.TestCase):
    def test_reads_across_chunk_boundaries(self) -> None:
        self.assertEqual(dnr3.read_exact(io.BytesIO(b"abcdef"), 6, "data"), b"abcdef")

    def test_short_stream_is_a_protocol_error(self) -> None:
        with self.assertRaises(dnr3.Dnr3ProtocolError) as raised:
            dnr3.read_exact(io.BytesIO(b"ab"), 4, "reply header")
        self.assertIn("reply header", str(raised.exception))
