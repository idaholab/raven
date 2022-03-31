import sys
import platform


def get_platform() -> str:
    """Get FMU binary platform folder name."""
    system = platform.system()
    is_64bits = sys.maxsize > 2 ** 32
    platforms = {"Windows": "win", "Linux": "linux", "Darwin": "darwin"}
    return platforms.get(system, "unknown") + "64" if is_64bits else "32"


def get_lib_extension() -> str:
    """Get FMU library platform extension."""
    platforms = {"Darwin": "dylib", "Linux": "so", "Windows": "dll"}
    return platforms.get(platform.system(), "")

