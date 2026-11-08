"""Simple forecasting references with explicitly defined behavior."""

from .persistence import persistence

__all__ = ["persistence"]
from .seasonal_average import SeasonalAverage

__all__.append("SeasonalAverage")
