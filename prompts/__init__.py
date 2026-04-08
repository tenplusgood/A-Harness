"""
Centralized prompt definitions for the Affordance Detection Agent.

All LLM prompts used across the system are defined here, organized by module:
- system: Agent-level system prompt
- detection: Detection skill prompts (Qwen VLM)
- dreamer: Dreamer skill prompts (interaction imagination)
- web_search: Web search skill prompts
- memory: Memory management prompts (dedup, eviction, distillation)
"""

from .system import SYSTEM_PROMPT, build_system_prompt
from .detection import (
    DETECTION_SYSTEM_PROMPT_TEMPLATE,
    DETECTION_USER_PROMPT_TEMPLATE,
    DETECTION_VERIFY_PROMPT_TEMPLATE,
)
from .dreamer import (
    DREAMER_SYSTEM_PROMPT_SINGLE,
    DREAMER_MULTI_TARGET_PROMPT,
    DREAMER_ANALYSIS_SYSTEM_PROMPT,
)
from .web_search import (
    SEARCH_QUERY_SYSTEM_PROMPT,
    SEARCH_QUERY_USER_PROMPT_TEMPLATE,
    SEARCH_ANALYSIS_SYSTEM_PROMPT_TEMPLATE,
    SEARCH_FALLBACK_SYSTEM_PROMPT,
)
from .memory import (
    EXPERIENCE_DISTILL_PROMPT_TEMPLATE,
    MEMORY_DEDUP_PROMPT_TEMPLATE,
    MEMORY_EVICTION_PROMPT_TEMPLATE,
)
