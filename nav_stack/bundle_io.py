# bundle_io.py
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


log = logging.getLogger("bundle_io")


# ----------------------------- TempAreaFile helper -----------------------------

@dataclass
class TempAreaFile:
    """
    Keeps a temp .area file on disk until you explicitly clean it up.
    ZED positional tracking uses a file path; we keep the file alive for the process lifetime.
    """
    path: str

    def cleanup(self) -> None:
        try:
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
                log.info("Deleted temp area file: %s", self.path)
        except Exception as e:
            log.warning("Failed to delete temp area file '%s': %s", self.path, e)

    def __enter__(self) -> "TempAreaFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def __del__(self) -> None:
        # Best-effort; do not rely solely on __del__ for cleanup.
        try:
            self.cleanup()
        except Exception:
            pass


# ----------------------------- ZED helpers -----------------------------

def _try_import_pyzed():
    try:
        import pyzed.sl as sl  # type: ignore
        return sl
    except Exception:
        return None


def _zed_is_available(zed_camera: Any) -> bool:
    return (zed_camera is not None) and hasattr(zed_camera, "enable_positional_tracking")


def _enable_positional_tracking_with_area(
    zed_camera: Any,
    area_path: str,
    *,
    pt_params: Optional[Any] = None,
) -> Any:
    """
    Enables ZED positional tracking using an area file at area_path.
    - Sets enable_area_memory=True
    - Sets area_file_path=area_path
    - Calls enable_positional_tracking(pt_params)

    pt_params can be passed in from your existing code to preserve all other settings.
    """
    sl = _try_import_pyzed()
    if sl is None:
        raise RuntimeError("pyzed.sl not importable; cannot enable ZED positional tracking.")

    if pt_params is None:
        # Safe defaults; if you already set these elsewhere, pass your pt_params in.
        pt_params = sl.PositionalTrackingParameters()

    # Required behavior:
    pt_params.enable_area_memory = True
    pt_params.area_file_path = str(area_path)

    err = zed_camera.enable_positional_tracking(pt_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"enable_positional_tracking failed: {err}")

    log.info("Enabled ZED positional tracking with area memory: %s", area_path)
    return pt_params


# ----------------------------- NPZ helpers -----------------------------

_RESERVED_KEYS = {"area_u8", "bundle_version"}


def _npz_load_to_dict(npz_path: str, *, allow_pickle: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=allow_pickle) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def _atomic_savez(bundle_path: str, data: Dict[str, Any]) -> None:
    """
    Atomic write:
      write to bundle_path + ".tmp" then os.replace().
    Uses a file handle so the temp name doesn't need .npz extension.
    """
    os.makedirs(os.path.dirname(bundle_path) or ".", exist_ok=True)
    tmp_path = bundle_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, bundle_path)
    finally:
        # If something failed before replace(), try to delete the temp file.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ----------------------------- Public API -----------------------------

def save_bundle(
    bundle_path: str,
    zed_camera: Any,
    *,
    bundle_version: int = 1,
    merge_existing: bool = True,
    **my_arrays: Any,
) -> None:
    """
    Required behavior:
      - If ZED available: export area memory to temp .area (zed_camera.save_area_map),
        read raw bytes and store as uint8 array in key 'area_u8' (lossless).
      - Store all existing keys exactly as-is + 'bundle_version'
      - Atomic write: bundle_path.tmp then os.replace()

    Practical addition (helps multi-process setups):
      - merge_existing=True preserves keys already present in bundle.npz that you don't pass this call.
        (e.g., zed process updates area_u8; slam process updates map arrays)
    """
    if any(k in _RESERVED_KEYS for k in my_arrays.keys()):
        bad = [k for k in my_arrays.keys() if k in _RESERVED_KEYS]
        raise ValueError(f"Refusing to overwrite reserved bundle keys: {bad}")

    data: Dict[str, Any] = {}

    # Optional merge to prevent clobbering.
    if merge_existing and os.path.isfile(bundle_path):
        try:
            existing = _npz_load_to_dict(bundle_path, allow_pickle=True)
            data.update(existing)
            log.info("Merged existing bundle keys from %s", bundle_path)
        except Exception as e:
            log.warning("Failed to merge existing bundle '%s' (continuing): %s", bundle_path, e)

    # Add/overwrite with user-provided arrays (keys unchanged).
    data.update(my_arrays)

    # Always set bundle_version (overwrite OK).
    data["bundle_version"] = np.array(int(bundle_version), dtype=np.int32)

    # Export and embed ZED area memory, if possible.
    if _zed_is_available(zed_camera) and hasattr(zed_camera, "save_area_map"):
        tmp_area_fd, tmp_area_path = tempfile.mkstemp(prefix="zed_area_export_", suffix=".area")
        os.close(tmp_area_fd)
        try:
            sl = _try_import_pyzed()
            if sl is None:
                raise RuntimeError("pyzed.sl not importable; cannot export ZED area map.")

            err = zed_camera.save_area_map(tmp_area_path)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"save_area_map failed: {err}")

            with open(tmp_area_path, "rb") as f:
                area_bytes = f.read()

            data["area_u8"] = np.frombuffer(area_bytes, dtype=np.uint8).copy()
            log.info("Embedded ZED area memory into bundle (%d bytes).", len(area_bytes))

        except Exception as e:
            # Do not fail the entire save if area export fails; keep map artifacts safe.
            log.exception("Failed to export/embed ZED area memory (continuing): %s", e)
        finally:
            try:
                if os.path.exists(tmp_area_path):
                    os.remove(tmp_area_path)
            except Exception:
                pass
    else:
        log.info("ZED not available; saving bundle without area_u8.")

    _atomic_savez(bundle_path, data)
    log.info("Saved bundle: %s (keys=%d)", bundle_path, len(data))


def load_bundle(
    bundle_path: str,
    zed_camera: Any,
    *,
    pt_params: Optional[Any] = None,
    enable_tracking: bool = True,
) -> Tuple[Dict[str, Any], Optional[TempAreaFile]]:
    """
    Required behavior:
      - Load bundle.npz, restore arrays/metadata (keys unchanged).
      - If area_u8 exists and ZED is available:
          - Write bytes back to a temp .area file byte-for-byte
          - Set pt_params.enable_area_memory=True and pt_params.area_file_path=temp_area_path
          - Call zed_camera.enable_positional_tracking(pt_params)
          - Keep temp file alive until shutdown; return TempAreaFile handle for cleanup.
    """
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    data = _npz_load_to_dict(bundle_path, allow_pickle=True)

    temp_area: Optional[TempAreaFile] = None

    area_u8 = data.get("area_u8", None)
    if (
        enable_tracking
        and area_u8 is not None
        and _zed_is_available(zed_camera)
        and hasattr(zed_camera, "enable_positional_tracking")
    ):
        try:
            # Pick a temp dir near the bundle if possible (often convenient for debugging/permissions).
            preferred_dir = os.path.dirname(os.path.abspath(bundle_path)) or None
            try:
                tmp = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="zed_area_restore_",
                    suffix=".area",
                    delete=False,
                    dir=preferred_dir,
                )
            except Exception:
                tmp = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix="zed_area_restore_",
                    suffix=".area",
                    delete=False,
                )

            tmp_area_path = tmp.name
            with tmp:
                # Byte-for-byte write
                u8 = np.asarray(area_u8, dtype=np.uint8).reshape(-1)
                tmp.write(u8.tobytes())
                tmp.flush()
                os.fsync(tmp.fileno())

            temp_area = TempAreaFile(tmp_area_path)
            log.info("Restored temp ZED area file from bundle: %s", tmp_area_path)

            _enable_positional_tracking_with_area(
                zed_camera,
                tmp_area_path,
                pt_params=pt_params,
            )

        except Exception as e:
            log.exception("Failed to restore/enable ZED area memory from bundle (continuing): %s", e)
            if temp_area is not None:
                temp_area.cleanup()
            temp_area = None
    else:
        if area_u8 is None:
            log.info("Bundle has no area_u8; skipping ZED relocalization restore.")
        elif not _zed_is_available(zed_camera):
            log.info("ZED not available; skipping enable_positional_tracking from area_u8.")
        else:
            log.info("enable_tracking=False; skipping ZED positional tracking enable.")

    # Return arrays unchanged (including area_u8 and bundle_version present in dict).
    # Your pipeline can ignore reserved keys, or you can filter them at the call site.
    return data, temp_area


def load_map_any_format(
    *,
    bundle_path: str = "bundle.npz",
    zed_camera: Any = None,
    old_npz_path: str = "test.npz",
    old_area_path: str = "saved_map.area",
    pt_params: Optional[Any] = None,
    enable_tracking: bool = True,
    auto_migrate: bool = True,
) -> Tuple[Dict[str, Any], Optional[TempAreaFile]]:
    """
    Backward-compatible entry point.

    Order:
      1) If bundle exists: load_bundle(bundle_path, ...)
      2) Else fall back to old behavior:
          - load old_npz_path (map artifacts)
          - if ZED available and old_area_path exists: enable positional tracking with that area file
      3) Optionally auto-migrate: after successful old-format load, write bundle.npz for next runs.
    """
    # 1) New format
    if os.path.isfile(bundle_path):
        try:
            log.info("Loading bundle format: %s", bundle_path)
            return load_bundle(
                bundle_path,
                zed_camera,
                pt_params=pt_params,
                enable_tracking=enable_tracking,
            )
        except Exception as e:
            log.exception("Failed to load bundle '%s'; falling back to old format: %s", bundle_path, e)

    # 2) Old format fallback
    data: Dict[str, Any] = {}
    temp_area: Optional[TempAreaFile] = None

    if os.path.isfile(old_npz_path):
        try:
            data = _npz_load_to_dict(old_npz_path, allow_pickle=True)
            log.info("Loaded old map NPZ: %s (keys=%d)", old_npz_path, len(data))
        except Exception as e:
            log.exception("Failed to load old map NPZ '%s': %s", old_npz_path, e)

    if enable_tracking and _zed_is_available(zed_camera) and os.path.isfile(old_area_path):
        try:
            log.info("Enabling ZED area memory from old .area: %s", old_area_path)
            _enable_positional_tracking_with_area(zed_camera, old_area_path, pt_params=pt_params)
        except Exception as e:
            log.exception("Failed to enable ZED tracking from old area '%s': %s", old_area_path, e)

    # 3) Auto-migrate if we successfully loaded something old
    if auto_migrate and (data or os.path.isfile(old_area_path)):
        try:
            log.info("Auto-migrating old format -> bundle: %s", bundle_path)
            save_bundle(
                bundle_path,
                zed_camera,
                merge_existing=True,
                **data,
            )
        except Exception as e:
            log.exception("Auto-migration to bundle failed (continuing): %s", e)

    return data, temp_area
