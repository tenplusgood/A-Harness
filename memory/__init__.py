"""
Hierarchical memory system for the Affordance Detection Agent.

Provides:
- MemoryManager: Inner/outer memory with RAG retrieval, deduplication, eviction
- MemoryEntry: Full inner-memory record for each detection sample
- CompactMemoryEntry: Compressed outer-memory summary for evicted entries
- StrategyInsight: Distilled strategy insight in the experience pool
- ExperiencePool: Collects observations and distills strategic insights
- prepare_templates: Commonsense template bank construction utilities
"""

from .manager import (
    MemoryManager,
    MemoryEntry,
    CompactMemoryEntry,
    StrategyInsight,
    ExperiencePool,
)

__all__ = [
    "MemoryManager",
    "MemoryEntry",
    "CompactMemoryEntry",
    "StrategyInsight",
    "ExperiencePool",
]
