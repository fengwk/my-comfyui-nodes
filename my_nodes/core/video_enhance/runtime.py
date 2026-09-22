"""Runtime layout and launch environment for the DNR3 DLSS host.

Two things live here, both feature aware:

* `resolve_runtime_files` checks exactly the files the requested features need
  in the user's runtime directory (default `<ComfyUI>/models/dlss5`, supplied
  by the caller). Super resolution needs `nvngx_dlss.dll`, neural rendering
  needs its own `nvngx_dlssnr*.dll`, and neither feature ever requires the
  other's file. The NVIDIA binaries stay user supplied: nothing here downloads
  or copies them.
* `build_environment` / `HostDriver` describe how to start the vendored host
  under Wine: DISPLAY is preserved (and its absence is an actionable error),
  the Wine prefix is resolved and checked before launch (vkd3d-proton
  `d3d12.dll` and dxvk-nvapi `nvapi64.dll` must already be installed; nothing
  here creates or mutates a prefix), WINE can be overridden explicitly, the
  DXVK/vkd3d/NVAPI variables the NGX core needs are set, an inherited
  `LD_LIBRARY_PATH` is dropped, and the selected neural-rendering DLL name is
  passed to the bridge. There is no Xvfb manager in this version.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from my_nodes.core.video_enhance.dnr3 import FEATURE_NR, FEATURE_SR, Dnr3Error, check_features

CORE_DLL = "_nvngx.dll"
SR_DLL = "nvngx_dlss.dll"
NR_DLL = "nvngx_dlssnr.dll"
NR_DLL_RTX30 = "nvngx_dlssnr_rtx30.dll"
# Preference order for the neural-rendering runtime: a per-lineage build wins
# over the universal one when the user dropped it next to it.
NR_DLL_CANDIDATES: tuple[str, ...] = (NR_DLL_RTX30, NR_DLL)
HOST_EXE_NAME = "dlss5nr_host.exe"

WINE_ENV_VAR = "DLSS5_WINE"
WINEPREFIX_ENV_VAR = "DLSS5_WINEPREFIX"
WINEHQ_STABLE_WINE = "/opt/wine-stable/bin/wine"

# Variables the NGX core needs under Wine. Existing values win, so a user can
# still point at a different DXVK/vkd3d deployment.
REQUIRED_ENVIRONMENT: Mapping[str, str] = {
    "DXVK_ENABLE_NVAPI": "1",
    "DXVK_LOG_LEVEL": "none",
    "VKD3D_DEBUG": "none",
    "WINEDEBUG": "-all",
    # Prefer DXVK-NVAPI / vkd3d-proton DLLs over Wine's builtin D3D12: a
    # builtin-only override hides the physical NVIDIA adapter from NvAPI.
    "WINEDLLOVERRIDES": "d3d12,d3d12core,nvapi64,dxgi=n,b",
    # NGX's own file/console sinks must never reach our binary stdout stream.
    "DLSS5NR_DISABLE_OTHER_SINKS": "1",
}
# Only meaningful while neural rendering is enabled.
NR_ENVIRONMENT: Mapping[str, str] = {"DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE": "on"}


class RuntimeFileError(Dnr3Error):
    """A required runtime file or launch prerequisite is missing."""


@dataclass(frozen=True)
class RuntimeFiles:
    """The user supplied runtime files, resolved for one feature combination."""

    features: int
    directory: Path
    core: Path
    sr: Path | None
    nr: Path | None
    nr_name: str | None


def _base_env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is None:
        return dict(os.environ)
    return {str(key): str(value) for key, value in env.items()}


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeFileError(f"{what} is missing: {path}")
    return path


def resolve_runtime_files(
    runtime_dir: str | os.PathLike[str],
    features: int,
    *,
    nr_dll: str | None = None,
) -> RuntimeFiles:
    """Validate the runtime directory against the requested features.

    Fails with a message naming the one file that is missing and who supplies
    it; a feature that is off is never required, so super-resolution-only runs
    do not need a neural-rendering DLL and vice versa.
    """
    features = check_features(features)
    directory = Path(os.fspath(runtime_dir)).expanduser()
    if not directory.is_dir():
        raise RuntimeFileError(
            f"DLSS runtime directory does not exist: {directory}. Point it at a folder "
            f"(usually <ComfyUI>/models/dlss5) holding the NVIDIA runtime DLLs you installed."
        )
    core = _require_file(
        directory / CORE_DLL,
        f"NGX core {CORE_DLL} (copy it from an NVIDIA Windows driver package; it is never downloaded)",
    )
    sr: Path | None = None
    nr: Path | None = None
    nr_name: str | None = None
    if features & FEATURE_SR:
        sr = _require_file(
            directory / SR_DLL,
            f"DLSS super-resolution runtime {SR_DLL} (needed because super resolution is enabled)",
        )
    if features & FEATURE_NR:
        nr_name = _select_nr_name(directory, nr_dll)
        nr = _require_file(
            directory / nr_name,
            f"neural-rendering runtime {nr_name} (needed because neural rendering is enabled)",
        )
    return RuntimeFiles(
        features=features,
        directory=directory,
        core=core,
        sr=sr,
        nr=nr,
        nr_name=nr_name,
    )


def _select_nr_name(directory: Path, nr_dll: str | None) -> str:
    """Pick the neural-rendering DLL to load, never a path outside `directory`."""
    if nr_dll is not None:
        name = str(nr_dll)
        if not name or name != os.path.basename(name) or ".." in name:
            raise RuntimeFileError(
                f"neural-rendering DLL name must be a plain file name, got {nr_dll!r}"
            )
        return name
    for candidate in NR_DLL_CANDIDATES:
        if (directory / candidate).is_file():
            return candidate
    return NR_DLL


def _current_build(binary_dir: Path) -> bool:
    """True when bin/build-info.txt records the hashes of the current sources."""
    stamp = binary_dir / "build-info.txt"
    if not stamp.is_file():
        return False
    recorded: dict[str, str] = {}
    for line in stamp.read_text(encoding="utf-8").splitlines():
        digest, marker, name = (line.split(maxsplit=2) + ["", ""])[:3]
        if marker == "source-sha256" and len(digest) == 64:
            recorded[name] = digest
    sources = binary_dir.parent / "src"
    expected = ("dlss5nr_bridge.cpp", "dlss5nr_host.cpp", "caller_shim.cpp")
    if set(recorded) != set(expected):
        return False
    return all(
        hashlib.sha256((sources / name).read_bytes()).hexdigest() == recorded[name]
        for name in expected
    )


def host_executable() -> Path:
    """The vendored Windows host that runs under Wine."""
    path = Path(__file__).with_name("native") / "bin" / HOST_EXE_NAME
    if not path.is_file() or not _current_build(path.parent):
        raise RuntimeFileError(
            f"DLSS host executable is missing or stale: {path}. Build the native artifacts "
            f"with native/build_mingw.sh (a binary whose build-info.txt does not match the "
            f"current sources is not this build)."
        )
    return path


_PREFIX_SYSTEM32 = Path("drive_c") / "windows" / "system32"
# Native DLLs a usable prefix must already contain. Wine's builtin copies are
# not enough: vkd3d-proton supplies D3D12 and dxvk-nvapi supplies NVAPI.
_PREFIX_REQUIRED_DLLS: tuple[str, ...] = ("d3d12.dll", "nvapi64.dll")


def resolve_wine_prefix(
    override: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Pick the Wine prefix: argument, DLSS5_WINEPREFIX, WINEPREFIX, ~/.wine."""
    environment = _base_env(env)
    if override is not None:
        selected: str | os.PathLike[str] = override
    elif environment.get(WINEPREFIX_ENV_VAR):
        selected = environment[WINEPREFIX_ENV_VAR]
    elif environment.get("WINEPREFIX"):
        selected = environment["WINEPREFIX"]
    else:
        selected = Path.home() / ".wine"
    return Path(os.fspath(selected)).expanduser().resolve()


def check_wine_prefix(prefix: str | os.PathLike[str]) -> Path:
    """Require an existing prefix with vkd3d-proton and dxvk-nvapi installed.

    Does not run wineboot, download anything, or write into the prefix. The
    returned path is absolute so the child environment cannot drift.
    """
    root = Path(os.fspath(prefix)).expanduser()
    if not root.is_absolute():
        root = root.resolve()
    if not root.is_dir():
        raise RuntimeFileError(
            f"Wine prefix does not exist: {root}. Create it and install "
            "vkd3d-proton and dxvk-nvapi before starting the DLSS host; this "
            "code will not run wineboot or modify a prefix."
        )
    system32 = root / _PREFIX_SYSTEM32
    missing = [
        str(system32 / name)
        for name in _PREFIX_REQUIRED_DLLS
        if not (system32 / name).is_file()
    ]
    if missing:
        listed = ", ".join(missing)
        raise RuntimeFileError(
            f"Wine prefix {root} is missing required native DLLs: {listed}. "
            "Install vkd3d-proton (d3d12.dll) and dxvk-nvapi (nvapi64.dll) into "
            "drive_c/windows/system32. This code does not download or install them."
        )
    return root


def find_wine(
    override: str | None = None, *, env: Mapping[str, str] | None = None
) -> str:
    """Resolve the wine binary: explicit override, env, WineHQ stable, PATH."""
    environment = _base_env(env)
    for candidate in (override, environment.get(WINE_ENV_VAR)):
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved is None and Path(candidate).is_file():
            resolved = candidate
        if resolved is None:
            raise RuntimeFileError(f"configured Wine binary was not found: {candidate}")
        return resolved
    if Path(WINEHQ_STABLE_WINE).is_file():
        return WINEHQ_STABLE_WINE
    found = shutil.which("wine")
    if found:
        return found
    raise RuntimeFileError(
        "Wine was not found. Install Wine (9.0 or newer recommended) or point "
        f"{WINE_ENV_VAR} at a wine binary."
    )


def build_environment(
    *,
    files: RuntimeFiles,
    base_env: Mapping[str, str] | None = None,
    wine_prefix: str | os.PathLike[str] | None = None,
    gpu_index: int = 0,
) -> dict[str, str]:
    """Build the environment for the Wine host of this feature combination."""
    env = _base_env(base_env)
    if not env.get("DISPLAY", "").strip():
        raise RuntimeFileError(
            "DISPLAY is not set, and the DLSS host needs an X display for Wine's D3D12 "
            "presentation. Start ComfyUI from a desktop session or set DISPLAY explicitly; "
            "this version does not start an Xvfb server for you."
        )
    # A worker LD_LIBRARY_PATH (for example a CUDA toolkit) breaks the NVIDIA
    # shims on the Wine side.
    env.pop("LD_LIBRARY_PATH", None)
    for name, value in REQUIRED_ENVIRONMENT.items():
        env.setdefault(name, value)
    if files.features & FEATURE_NR:
        for name, value in NR_ENVIRONMENT.items():
            env.setdefault(name, value)
        if files.nr_name is None:
            raise RuntimeFileError("neural rendering is enabled but no NR runtime was selected")
        env["DLSS5NR_SNR_FILENAME"] = files.nr_name
    else:
        # Never let a stale value from the parent environment pick another DLL.
        env.pop("DLSS5NR_SNR_FILENAME", None)
    env["DLSS5NR_GPU_INDEX"] = str(int(gpu_index))
    # Publish only an explicitly selected prefix. HostDriver.wine always passes
    # the absolute prefix it already checked; leaving WINEPREFIX unset here
    # keeps environment construction independent of a real prefix.
    if wine_prefix is not None:
        env["WINEPREFIX"] = os.fspath(wine_prefix)
    elif WINEPREFIX_ENV_VAR in env and env.get(WINEPREFIX_ENV_VAR):
        env["WINEPREFIX"] = os.fspath(env[WINEPREFIX_ENV_VAR])
    return env


@dataclass(frozen=True)
class HostDriver:
    """Everything needed to start the DNR3 host: files, argv and environment."""

    files: RuntimeFiles
    command: tuple[str, ...]
    env: Mapping[str, str]
    cwd: str | None = None

    @classmethod
    def wine(
        cls,
        *,
        runtime_dir: str | os.PathLike[str],
        features: int,
        nr_dll: str | None = None,
        wine: str | None = None,
        wine_prefix: str | os.PathLike[str] | None = None,
        gpu_index: int = 0,
        env: Mapping[str, str] | None = None,
        host: str | os.PathLike[str] | None = None,
    ) -> HostDriver:
        """Run the vendored host under Wine, with the runtime-specific shim."""
        files = resolve_runtime_files(runtime_dir, features, nr_dll=nr_dll)
        base = _base_env(env)
        executable = Path(host).expanduser() if host is not None else host_executable()
        _require_file(executable, f"DLSS host executable {HOST_EXE_NAME}")
        # Fail before any process is spawned. Direct drivers skip this: tests
        # and custom hosts do not need a Wine prefix.
        prefix = check_wine_prefix(resolve_wine_prefix(wine_prefix, env=base))
        child_env = build_environment(
            files=files, base_env=base, wine_prefix=prefix, gpu_index=gpu_index
        )
        # The child must see the absolute prefix that was actually checked,
        # never a relative value inherited from the parent.
        child_env["WINEPREFIX"] = os.fspath(prefix)
        return cls(
            files=files,
            command=(find_wine(wine, env=base), str(executable), str(files.directory)),
            env=child_env,
            cwd=str(executable.parent),
        )

    @classmethod
    def direct(
        cls,
        command: Sequence[str],
        *,
        runtime_dir: str | os.PathLike[str],
        features: int,
        nr_dll: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> HostDriver:
        """Run an arbitrary DNR3 host command (tests, custom deployments).

        The runtime files are still validated, because the protocol requires a
        real host that knows how to load them.
        """
        if not command:
            raise RuntimeFileError("host command must not be empty")
        return cls(
            files=resolve_runtime_files(runtime_dir, features, nr_dll=nr_dll),
            command=tuple(str(part) for part in command),
            env=_base_env(env),
            cwd=None if cwd is None else os.fspath(cwd),
        )
