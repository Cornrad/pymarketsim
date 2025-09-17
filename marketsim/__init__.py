"""Top level package for :mod:`marketsim`.

The original package initialised a number of heavy submodules eagerly.  Two
problems followed from that approach:

* ``SimulatorSampledArrivalCustom`` was re-exported even though the
  corresponding module is not present in the repository, which caused
  ``ImportError`` during package import; and
* importing deep modules while the package was still initialising made it easy
  to trigger circular import issues when those modules relied on absolute
  ``marketsim.`` imports.

To keep ``import marketsim`` lightweight and robust we only perform the
relatively small re-exports that are guaranteed to exist.  The sampled arrival
simulator remains available from ``marketsim.simulator``.
"""

from .simulator.simulator import Simulator
from .market.market import Market
from .fourheap.fourheap import FourHeap
from .fourheap.order_queue import OrderQueue

__version__ = "0.1.0"

__all__ = [
    "Simulator",
    "Market",
    "FourHeap",
    "OrderQueue",
]
