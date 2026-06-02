from importlib.metadata import PackageNotFoundError, version

from dohl import damage, ebp, lifespan, stockbridge

try:
    __version__ = version("dohl")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["damage", "ebp", "lifespan", "stockbridge", "__version__"]
