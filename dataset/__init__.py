"""
Dataset readers for affordance segmentation benchmarks.

Available datasets:
- ReasonAff: Reasoning-based affordance segmentation (HuggingFace format)
- UMD: UMD Part-Affordance Dataset (preprocessed image/mask pairs)
"""

from .reasonaff_reader import ReasonAffDataset
from .umd_reader import UmdDataset

__all__ = ["ReasonAffDataset", "UmdDataset"]
