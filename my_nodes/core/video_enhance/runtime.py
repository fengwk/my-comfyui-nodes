"""Runtime layout and launch environment for the DNR3 DLSS host.

Two things live here, both feature aware:

* `resolve_runtime_files` checks exactly the files the requested features need
  in the user's runtime directory (default `<ComfyUI>/models/dlss5`, supplied
  by the caller). Super resolution needs `nvngx_dlss.dll`, neural rendering
  needs its own `nvngx_dlssnr*.dll`, and neither feature ever requires the
  other's file. The NVIDIA binaries stay user supplied: nothing here downloads
  or copies them.
* `build_environment` / `HostDriver` describe how to start the vendored host
  under Wine: an existing DISPLAY is preserved, and its absence is resolved by
  `VirtualDisplay`, which starts one private Xvfb for that launch and stops it
  when the launch ends. The Wine prefix is resolved and checked before launch
  (vkd3d-proton `d3d12.dll` and dxvk-nvapi `nvapi64.dll` must already be
  installed; nothing here creates or mutates a prefix), WINE can be overridden
  explicitly, the DXVK/vkd3d/NVAPI variables the NGX core needs are set, an
  inherited `LD_LIBRARY_PATH` is dropped, and the selected neural-rendering DLL
  name is passed to the bridge.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
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
# Private Xvfb range. A desktop session normally uses :0 or :1, and fixed
# service displays such as :99 stay outside this range.
_VIRTUAL_DISPLAY_MIN = 200
_VIRTUAL_DISPLAY_MAX = 299
_VIRTUAL_DISPLAY_READY_SECONDS = 5.0

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
    """Build the environment for the Wine host of this feature combination.

    `DISPLAY` is preserved when the caller already has one. It is not invented
    here: a missing display is supplied by `VirtualDisplay` at process launch,
    so this function stays free of processes and temporary files.
    """
    env = _base_env(base_env)
    if not env.get("DISPLAY", "").strip():
        raise RuntimeFileError(
            "DISPLAY is not set. The Wine launch must attach a VirtualDisplay "
            "before building the host environment."
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


class VirtualDisplay:
    """One private Xvfb used only while a Wine DLSS launch has no DISPLAY.

    An existing display is never replaced or stopped. A display created here is
    stopped by `close`, including when Wine fails to start. Concurrent launches
    receive different display numbers, so one launch cannot stop another's
    server.
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        self.env = dict(env)
        self._process: subprocess.Popen[bytes] | None = None
        self._owned = False
        self._auth_directory: tempfile.TemporaryDirectory | None = None

    @property
    def owned(self) -> bool:
        """True when this object started the X server it publishes."""
        return self._owned

    def start(self) -> dict[str, str]:
        """Return the launch environment, starting Xvfb only when necessary."""
        if self.env.get("DISPLAY", "").strip():
            return self.env
        executable = shutil.which("Xvfb")
        if executable is None:
            raise RuntimeFileError(
                "DISPLAY is not set and Xvfb was not found. Install Xvfb "
                "(xorg-server-xvfb) so the DLSS host can initialize Wine's D3D12 "
                "presentation; this code does not use a desktop session."
            )
        number, lock = _reserve_display_number()
        try:
            self._auth_directory = tempfile.TemporaryDirectory(prefix="dlss-xauth-")
            authority = Path(self._auth_directory.name) / "Xauthority"
            # Xauthority stores big-endian length-prefixed fields. FamilyLocal
            # restricts this cookie to the host and the selected display.
            fields = (socket.gethostname().encode(), str(number).encode(),
                      b"MIT-MAGIC-COOKIE-1", os.urandom(16))
            record = struct.pack(">H", 256)
            record += b"".join(struct.pack(">H", len(field)) + field for field in fields)
            with authority.open("xb") as auth_file:
                os.chmod(authority, 0o600)
                auth_file.write(record)
            self._process = subprocess.Popen(
                [
                    executable,
                    f":{number}",
                    "-screen",
                    "0",
                    "1024x768x24",
                    "-nolisten",
                    "tcp",
                    "-noreset",
                    # Xvfb's zero-rate RandR modes trigger a divide by zero in
                    # some DXVK DXGI builds. Wine's fallback modes avoid it.
                    "-extension",
                    "RANDR",
                    "-auth",
                    str(authority),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._owned = True
            _wait_until_ready(self._process, number, lock)
        except BaseException:
            self.close()
            raise
        finally:
            lock.close()
        # Only the child needs this endpoint. Do not publish it to ComfyUI.
        self.env["DISPLAY"] = f":{number}"
        self.env["XAUTHORITY"] = str(authority)
        return self.env

    def close(self) -> None:
        """Stop an Xvfb started by this object; idempotent and quiet."""
        process, self._process = self._process, None
        self._owned = False
        if self._auth_directory is not None:
            self._auth_directory.cleanup()
            self._auth_directory = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        else:
            process.wait(timeout=0)


def _reserve_display_number() -> tuple[int, socket.socket]:
    """Reserve one unused display and hold its lock socket until Xvfb binds it."""
    for number in range(_VIRTUAL_DISPLAY_MIN, _VIRTUAL_DISPLAY_MAX + 1):
        if Path(f"/tmp/.X{number}-lock").exists() or Path(f"/tmp/.X11-unix/X{number}").exists():
            continue
        lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            lock.bind(f"\0dlss5-xvfb-{number}")
        except OSError:
            lock.close()
            continue
        return number, lock
    raise RuntimeFileError(
        f"no free X display in :{_VIRTUAL_DISPLAY_MIN}-:{_VIRTUAL_DISPLAY_MAX}; "
        "close stale Xvfb processes before starting another DLSS host"
    )


def _wait_until_ready(process: subprocess.Popen[bytes], number: int, lock: socket.socket) -> None:
    """Wait until the reserved display accepts connections, then release its lock."""
    deadline = time.monotonic() + _VIRTUAL_DISPLAY_READY_SECONDS
    ready = False
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeFileError(
                    f"Xvfb exited with code {process.returncode} before display :{number} was ready"
                )
            if Path(f"/tmp/.X11-unix/X{number}").exists():
                ready = True
                return
            time.sleep(0.02)
    finally:
        lock.close()
        if not ready and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    raise RuntimeFileError(f"Xvfb did not open display :{number} within {_VIRTUAL_DISPLAY_READY_SECONDS:.0f}s")


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
