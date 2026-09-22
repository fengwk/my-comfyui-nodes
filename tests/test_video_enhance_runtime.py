from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_nodes.core.video_enhance import FEATURE_NR, FEATURE_SR, RuntimeFileError, runtime

from .video_enhance_fixtures import create_runtime_dir, fake_worker_command

SR_ONLY = FEATURE_SR
NR_ONLY = FEATURE_NR
SR_NR = FEATURE_SR | FEATURE_NR
DISPLAY_ENV = {"DISPLAY": ":99"}


class RuntimeFileResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_super_resolution_alone_needs_core_and_dlss_only(self) -> None:
        directory = create_runtime_dir(self.root / "sr", SR_ONLY)
        files = runtime.resolve_runtime_files(directory, SR_ONLY)
        self.assertEqual(files.features, SR_ONLY)
        self.assertEqual(files.core, directory / runtime.CORE_DLL)
        self.assertEqual(files.sr, directory / runtime.SR_DLL)
        self.assertIsNone(files.nr)
        self.assertIsNone(files.nr_name)
        # The neural-rendering file is absent on purpose above and must not
        # matter: this is what "SR alone" has to guarantee.
        self.assertFalse((directory / runtime.NR_DLL).exists())

    def test_neural_rendering_alone_needs_core_and_nr_only(self) -> None:
        directory = create_runtime_dir(self.root / "nr", NR_ONLY)
        files = runtime.resolve_runtime_files(directory, NR_ONLY)
        self.assertIsNone(files.sr)
        self.assertEqual(files.nr, directory / runtime.NR_DLL)
        self.assertEqual(files.nr_name, runtime.NR_DLL)
        self.assertFalse((directory / runtime.SR_DLL).exists())

    def test_both_features_require_both_runtimes(self) -> None:
        directory = create_runtime_dir(self.root / "both", SR_NR)
        files = runtime.resolve_runtime_files(directory, SR_NR)
        self.assertIsNotNone(files.sr)
        self.assertIsNotNone(files.nr)

    def test_per_lineage_nr_build_wins_over_the_universal_one(self) -> None:
        directory = create_runtime_dir(self.root / "rtx30", NR_ONLY, nr_name=runtime.NR_DLL_RTX30)
        (directory / runtime.NR_DLL).write_bytes(b"universal")
        self.assertEqual(
            runtime.resolve_runtime_files(directory, NR_ONLY).nr_name, runtime.NR_DLL_RTX30
        )
        # Without the lineage build the universal name is used.
        (directory / runtime.NR_DLL_RTX30).unlink()
        self.assertEqual(runtime.resolve_runtime_files(directory, NR_ONLY).nr_name, runtime.NR_DLL)

    def test_explicit_nr_name_must_be_a_plain_file_name(self) -> None:
        directory = create_runtime_dir(self.root / "plain", NR_ONLY)
        (directory / "custom_nr.dll").write_bytes(b"custom build")
        self.assertEqual(
            runtime.resolve_runtime_files(directory, NR_ONLY, nr_dll="custom_nr.dll").nr_name,
            "custom_nr.dll",
        )
        for name in ("../nvngx_dlssnr.dll", "caller/nvngx.dll", "", ".."):
            with self.subTest(name=name):
                with self.assertRaises(RuntimeFileError):
                    runtime.resolve_runtime_files(directory, NR_ONLY, nr_dll=name)

    def test_missing_directory_core_and_feature_files_are_actionable(self) -> None:
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.resolve_runtime_files(self.root / "nope", SR_ONLY)
        self.assertIn("does not exist", str(raised.exception))
        self.assertIn("models/dlss5", str(raised.exception))

        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.resolve_runtime_files(empty, SR_ONLY)
        self.assertIn(runtime.CORE_DLL, str(raised.exception))

        missing_sr = create_runtime_dir(self.root / "no-sr", SR_ONLY, skip=(runtime.SR_DLL,))
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.resolve_runtime_files(missing_sr, SR_ONLY)
        self.assertIn(runtime.SR_DLL, str(raised.exception))
        self.assertIn("super resolution is enabled", str(raised.exception))

        missing_nr = create_runtime_dir(self.root / "no-nr", NR_ONLY, skip=(runtime.NR_DLL,))
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.resolve_runtime_files(missing_nr, NR_ONLY)
        self.assertIn(runtime.NR_DLL, str(raised.exception))
        self.assertIn("neural rendering is enabled", str(raised.exception))

    def test_invalid_feature_flags_are_rejected(self) -> None:
        directory = create_runtime_dir(self.root / "flags", SR_ONLY)
        with self.assertRaises(runtime.Dnr3Error):
            runtime.resolve_runtime_files(directory, 0)
        with self.assertRaises(runtime.Dnr3Error):
            runtime.resolve_runtime_files(directory, 0b1000)


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def files(self, features: int, *, nr_name: str | None = None) -> runtime.RuntimeFiles:
        directory = create_runtime_dir(self.root / f"runtime-{features}", features, nr_name=nr_name)
        return runtime.resolve_runtime_files(directory, features)

    def test_display_is_required_and_named(self) -> None:
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.build_environment(files=self.files(SR_ONLY), base_env={})
        self.assertIn("DISPLAY", str(raised.exception))
        with self.assertRaises(RuntimeFileError):
            runtime.build_environment(files=self.files(SR_ONLY), base_env={"DISPLAY": "   "})

    def test_display_and_user_variables_are_preserved(self) -> None:
        env = runtime.build_environment(
            files=self.files(SR_ONLY),
            base_env={"DISPLAY": ":7", "WINEDLLOVERRIDES": "custom", "PATH": "/usr/bin"},
        )
        self.assertEqual(env["DISPLAY"], ":7")
        self.assertEqual(env["WINEDLLOVERRIDES"], "custom")  # user override wins
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_required_dxvk_vkd3d_and_nvapi_variables_are_set(self) -> None:
        env = runtime.build_environment(files=self.files(SR_NR), base_env=DISPLAY_ENV)
        for name, value in runtime.REQUIRED_ENVIRONMENT.items():
            with self.subTest(name=name):
                self.assertEqual(env[name], value)
        self.assertEqual(env["DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE"], "on")
        self.assertEqual(env["DXVK_ENABLE_NVAPI"], "1")
        self.assertEqual(env["WINEDLLOVERRIDES"], "d3d12,d3d12core,nvapi64,dxgi=n,b")
        self.assertEqual(env["DXVK_LOG_LEVEL"], "none")
        self.assertEqual(env["VKD3D_DEBUG"], "none")

    def test_inherited_ld_library_path_is_removed(self) -> None:
        env = runtime.build_environment(
            files=self.files(SR_ONLY),
            base_env={**DISPLAY_ENV, "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"},
        )
        self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_nr_only_variables_follow_the_features(self) -> None:
        sr_env = runtime.build_environment(
            files=self.files(SR_ONLY),
            base_env={**DISPLAY_ENV, "DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE": "off",
                      "DLSS5NR_SNR_FILENAME": "stale.dll"},
        )
        # A stale selection from the parent environment must not survive.
        self.assertNotIn("DLSS5NR_SNR_FILENAME", sr_env)
        self.assertEqual(sr_env["DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE"], "off")

        nr_files = self.files(NR_ONLY, nr_name=runtime.NR_DLL_RTX30)
        nr_env = runtime.build_environment(files=nr_files, base_env=DISPLAY_ENV)
        self.assertEqual(nr_env["DLSS5NR_SNR_FILENAME"], runtime.NR_DLL_RTX30)
        self.assertEqual(nr_env["DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE"], "on")

    def test_wine_prefix_and_gpu_index_overrides(self) -> None:
        env = runtime.build_environment(
            files=self.files(NR_ONLY), base_env=DISPLAY_ENV, wine_prefix="/tmp/prefix", gpu_index=2
        )
        self.assertEqual(env["WINEPREFIX"], "/tmp/prefix")
        self.assertEqual(env["DLSS5NR_GPU_INDEX"], "2")
        # The documented environment variable is honoured as well.
        env = runtime.build_environment(
            files=self.files(NR_ONLY),
            base_env={**DISPLAY_ENV, runtime.WINEPREFIX_ENV_VAR: "/tmp/other"},
        )
        self.assertEqual(env["WINEPREFIX"], "/tmp/other")
        # Without an override the inherited value (or absence) is left alone.
        env = runtime.build_environment(files=self.files(NR_ONLY), base_env=DISPLAY_ENV)
        self.assertNotIn("WINEPREFIX", env)


class WineResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_explicit_override_wins_and_env_override_is_used(self) -> None:
        fake_wine = self.root / "wine"
        fake_wine.write_text("#!/bin/sh\n")
        self.assertEqual(runtime.find_wine(str(fake_wine), env={}), str(fake_wine))
        self.assertEqual(
            runtime.find_wine(env={runtime.WINE_ENV_VAR: str(fake_wine)}), str(fake_wine)
        )

    def test_missing_override_and_missing_wine_are_actionable(self) -> None:
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.find_wine(str(self.root / "nope"), env={})
        self.assertIn("was not found", str(raised.exception))
        with mock.patch.object(runtime.shutil, "which", return_value=None), mock.patch.object(
            runtime, "WINEHQ_STABLE_WINE", str(self.root / "nope")
        ):
            with self.assertRaises(RuntimeFileError) as raised:
                runtime.find_wine(env={})
        self.assertIn("Wine was not found", str(raised.exception))
        self.assertIn(runtime.WINE_ENV_VAR, str(raised.exception))

    @unittest.skipUnless(shutil.which("wine"), "wine is not installed")
    def test_system_wine_is_found_through_path(self) -> None:
        self.assertEqual(runtime.find_wine(env={}), shutil.which("wine"))


class HostDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_host_executable_matches_current_sources_and_rejects_a_stale_stamp(self) -> None:
        host = runtime.host_executable()
        self.assertTrue(host.is_file())
        self.assertTrue(runtime._current_build(host.parent))

        # A source change without a matching rebuild must make the packaged
        # executable unusable rather than silently launching an old protocol.
        with mock.patch.object(runtime, "_current_build", return_value=False):
            with self.assertRaises(RuntimeFileError) as raised:
                runtime.host_executable()
        self.assertIn("build_mingw.sh", str(raised.exception))
        self.assertIn("stale", str(raised.exception))

    def test_wine_driver_describes_the_host_command_and_environment(self) -> None:
        host = self.root / runtime.HOST_EXE_NAME
        host.write_bytes(b"MZ")
        runtime_dir = create_runtime_dir(self.root / "runtime", NR_ONLY)
        prefix = self.root / "driver-prefix"
        system32 = prefix / "drive_c" / "windows" / "system32"
        system32.mkdir(parents=True)
        for name in ("d3d12.dll", "nvapi64.dll"):
            (system32 / name).write_bytes(b"MZ")
        driver = runtime.HostDriver.wine(
            runtime_dir=runtime_dir,
            features=NR_ONLY,
            host=host,
            wine=sys.executable,
            wine_prefix=prefix,
            env=DISPLAY_ENV,
            gpu_index=1,
        )
        self.assertEqual(driver.command, (sys.executable, str(host), str(runtime_dir)))
        self.assertEqual(driver.cwd, str(self.root))
        self.assertEqual(driver.files.features, NR_ONLY)
        self.assertEqual(driver.env["DLSS5NR_GPU_INDEX"], "1")
        self.assertEqual(driver.env["DISPLAY"], ":99")
        self.assertEqual(driver.env["DLSS5NR_SNR_FILENAME"], runtime.NR_DLL)

    def _prefix(self, name: str, *, files: tuple[str, ...] = ("d3d12.dll", "nvapi64.dll")) -> Path:
        system32 = self.root / name / "drive_c" / "windows" / "system32"
        system32.mkdir(parents=True)
        for filename in files:
            (system32 / filename).write_bytes(b"MZ")
        return self.root / name

    def test_wine_prefix_precedence_is_argument_then_env_then_home(self) -> None:
        explicit = self.root / "explicit"
        dedicated = self.root / "dedicated"
        inherited = self.root / "inherited"
        with mock.patch.object(runtime.Path, "home", return_value=self.root / "home"):
            self.assertEqual(
                runtime.resolve_wine_prefix(
                    explicit,
                    env={
                        runtime.WINEPREFIX_ENV_VAR: str(dedicated),
                        "WINEPREFIX": str(inherited),
                    },
                ),
                explicit.resolve(),
            )
            self.assertEqual(
                runtime.resolve_wine_prefix(
                    env={runtime.WINEPREFIX_ENV_VAR: str(dedicated), "WINEPREFIX": str(inherited)}
                ),
                dedicated.resolve(),
            )
            self.assertEqual(
                runtime.resolve_wine_prefix(env={"WINEPREFIX": str(inherited)}),
                inherited.resolve(),
            )
            self.assertEqual(
                runtime.resolve_wine_prefix(env={}),
                (self.root / "home" / ".wine").resolve(),
            )

    def test_wine_prefix_preflight_names_every_missing_dll(self) -> None:
        missing_dir = self.root / "absent-prefix"
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.check_wine_prefix(missing_dir)
        self.assertIn(str(missing_dir), str(raised.exception))
        self.assertIn("vkd3d-proton", str(raised.exception))
        self.assertIn("dxvk-nvapi", str(raised.exception))
        self.assertFalse(missing_dir.exists())

        prefix = self._prefix("partial", files=())
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.check_wine_prefix(prefix)
        message = str(raised.exception)
        self.assertIn(str(prefix / "drive_c" / "windows" / "system32" / "d3d12.dll"), message)
        self.assertIn(str(prefix / "drive_c" / "windows" / "system32" / "nvapi64.dll"), message)
        self.assertIn("vkd3d-proton", message)
        self.assertIn("dxvk-nvapi", message)
        # Preflight must not create the missing files.
        self.assertEqual(list((prefix / "drive_c" / "windows" / "system32").iterdir()), [])

        complete = self._prefix("complete")
        self.assertEqual(runtime.check_wine_prefix(complete), complete)

    def test_wine_driver_checks_the_prefix_before_launch_and_publishes_it(self) -> None:
        host = self.root / runtime.HOST_EXE_NAME
        host.write_bytes(b"MZ")
        runtime_dir = create_runtime_dir(self.root / "runtime-prefix", SR_ONLY)
        prefix = self._prefix("ready")
        driver = runtime.HostDriver.wine(
            runtime_dir=runtime_dir,
            features=SR_ONLY,
            host=host,
            wine=sys.executable,
            wine_prefix=prefix,
            env={**DISPLAY_ENV, "WINEPREFIX": "relative-prefix"},
        )
        self.assertEqual(driver.env["WINEPREFIX"], str(prefix.resolve()))
        self.assertTrue(Path(driver.env["WINEPREFIX"]).is_absolute())

        with self.assertRaises(RuntimeFileError) as raised:
            runtime.HostDriver.wine(
                runtime_dir=runtime_dir,
                features=SR_ONLY,
                host=host,
                wine=sys.executable,
                wine_prefix=self._prefix("missing-nvapi", files=("d3d12.dll",)),
                env=DISPLAY_ENV,
            )
        self.assertIn("nvapi64.dll", str(raised.exception))

    def test_direct_driver_does_not_require_a_wine_prefix(self) -> None:
        runtime_dir = create_runtime_dir(self.root / "runtime-direct", SR_ONLY)
        driver = runtime.HostDriver.direct(
            fake_worker_command("ok"),
            runtime_dir=runtime_dir,
            features=SR_ONLY,
            env=DISPLAY_ENV,
        )
        self.assertNotIn("WINEPREFIX", driver.env)

    def test_wine_driver_requires_a_host_file_and_display(self) -> None:
        runtime_dir = create_runtime_dir(self.root / "runtime", SR_ONLY)
        prefix = self._prefix("launch-prefix")
        with self.assertRaises(RuntimeFileError):
            runtime.HostDriver.wine(
                runtime_dir=runtime_dir, features=SR_ONLY, host=self.root / "absent.exe",
                wine=sys.executable, wine_prefix=prefix, env=DISPLAY_ENV,
            )
        host = self.root / runtime.HOST_EXE_NAME
        host.write_bytes(b"MZ")
        with self.assertRaises(RuntimeFileError) as raised:
            runtime.HostDriver.wine(
                runtime_dir=runtime_dir, features=SR_ONLY, host=host, wine=sys.executable,
                wine_prefix=prefix, env={},
            )
        self.assertIn("DISPLAY", str(raised.exception))

    def test_direct_driver_is_a_passthrough_with_runtime_validation(self) -> None:
        runtime_dir = create_runtime_dir(self.root / "runtime", SR_ONLY)
        driver = runtime.HostDriver.direct(
            fake_worker_command("ok"),
            runtime_dir=runtime_dir,
            features=SR_ONLY,
            env={**DISPLAY_ENV, "CUSTOM": "1"},
        )
        self.assertEqual(driver.command, fake_worker_command("ok"))
        self.assertEqual(driver.env["CUSTOM"], "1")
        self.assertIsNone(driver.cwd)
        with self.assertRaises(RuntimeFileError):
            runtime.HostDriver.direct((), runtime_dir=runtime_dir, features=SR_ONLY)
        # A direct driver still validates the runtime files of its features.
        with self.assertRaises(RuntimeFileError):
            runtime.HostDriver.direct(
                fake_worker_command("ok"), runtime_dir=runtime_dir, features=NR_ONLY
            )

    def test_direct_driver_without_env_inherits_the_parent_environment(self) -> None:
        runtime_dir = create_runtime_dir(self.root / "runtime", SR_ONLY)
        os.environ["DNR3_RUNTIME_TEST_MARKER"] = "yes"
        self.addCleanup(os.environ.pop, "DNR3_RUNTIME_TEST_MARKER", None)
        driver = runtime.HostDriver.direct(
            fake_worker_command("ok"), runtime_dir=runtime_dir, features=SR_ONLY
        )
        self.assertEqual(driver.env["DNR3_RUNTIME_TEST_MARKER"], "yes")
