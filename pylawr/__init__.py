from .field import RadarField

try:
    from importlib.metadata import version as _version
except ImportError:
    from importlib_metadata import version as _version

try:
    __version__ = _version("pylawr")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__all__ = ['RadarField', ]