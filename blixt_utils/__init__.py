from . import io
from . import misc
from . import plotting
from . import signal_analysis

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass