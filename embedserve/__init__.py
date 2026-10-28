"""EmbedServe: a two-tier, quantized embedding store for DLRM-style recommenders."""

from embedserve.quantize import dequantize_int8, quantize_int8
from embedserve.tiers import HotTier, InMemoryColdTier, RocksDBColdTier, TierManager

__all__ = [
    "HotTier",
    "InMemoryColdTier",
    "RocksDBColdTier",
    "TierManager",
    "dequantize_int8",
    "quantize_int8",
]
__version__ = "0.1.0"
