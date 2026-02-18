import numpy as np
from scipy.spatial.transform import Rotation as R

# Pose helpers
def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    quat, translation = pose[:4], pose[4:7]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = translation
    return T

def theta_y_from_R(Rm: np.ndarray) -> float:
    Rm = np.asarray(Rm, dtype=np.float64)
    if Rm.shape == (4, 4):
        Rm = Rm[:3, :3]
    if Rm.shape != (3, 3):
        raise ValueError(f"Expected (3,3) or (4,4), got {Rm.shape}")
    offset = 0
    # np.pi / 2.0
    angle = np.arctan2(Rm[0, 2], Rm[0, 0]) + offset
    return np.arctan2(np.sin(angle), np.cos(angle))



# Terminal waitkey
import sys, termios, tty, select, os, atexit, time

# Map a few common escape sequences to X11-style keycodes (like OpenCV on Linux)
_X11_CODES = {
    b"\x1b[A": 65362,  # Up
    b"\x1b[B": 65364,  # Down
    b"\x1b[C": 65363,  # Right
    b"\x1b[D": 65361,  # Left
}

_old_attrs = None

def _enable_raw():
    """Put terminal into cbreak/no-echo once, restore on exit."""
    global _old_attrs
    if _old_attrs is not None or not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
    _old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)                           # non-canonical, returns per char
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~termios.ECHO                   # turn off echo
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    atexit.register(_disable_raw)

def _disable_raw():
    global _old_attrs
    if _old_attrs is None or not sys.stdin.isatty():
        return
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _old_attrs)
    _old_attrs = None

def waitKey(ms: int) -> int:
    """
    Terminal equivalent of cv2.waitKey for Linux/macOS.

    Args:
        ms: Timeout in milliseconds.
            - ms == 0  -> **non-blocking** (returns immediately if no key)
            - ms > 0   -> wait up to ms for a key

    Returns:
        int: key code or -1 if no key.
             ASCII keys -> ord(char)
             Arrow keys -> 65362/65364/65363/65361 (Up/Down/Right/Left)
    """
    if not sys.stdin.isatty():
        # No interactive TTY; nothing to read
        return -1

    _enable_raw()
    fd = sys.stdin.fileno()

    # Non-blocking when ms == 0, otherwise wait up to ms
    timeout = 0.0 if ms == 0 else (ms / 1000.0)

    r, _, _ = select.select([fd], [], [], timeout)
    if not r:
        return -1

    # Read first byte
    b = os.read(fd, 1)

    # Handle escape sequences (arrows, etc.) by draining immediately-available bytes
    if b == b"\x1b":
        # Drain any bytes that arrived as part of the same sequence without blocking
        while True:
            r, _, _ = select.select([fd], [], [], 0)
            if not r:
                break
            b += os.read(fd, 1)

        if b in _X11_CODES:
            return _X11_CODES[b]
        # If it's just ESC alone (e.g., user pressed Esc), return 27 like OpenCV
        if b == b"\x1b":
            return 27

        # Fallback: if it's some other ESC sequence, return last byte
        return b[-1]

    # Regular ASCII: return last byte as int (compatible with `& 0xFF`)
    return b[-1]