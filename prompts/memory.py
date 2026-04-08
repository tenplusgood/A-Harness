"""
Memory management prompts for the Affordance Detection Agent.

These prompts are used by the MemoryManager and ExperiencePool for:
1. Experience distillation — extracting strategic insights from execution history
2. Memory deduplication — judging whether two samples are duplicates
3. Memory eviction — deciding which entries to remove when at capacity
"""


EXPERIENCE_DISTILL_PROMPT_TEMPLATE = """You are a strategy insight extractor for an affordance detection agent.
Given new execution observations and the existing experience pool, produce updated strategic insights.

**Metric note**: IoU (Intersection over Union) measures mask overlap — **higher is better** (range 0–1). What counts as high or low depends on the dataset difficulty; judge by comparing across the observations below.

{existing_block}New observations ({num_observations} samples):
{obs_text}

Tasks:
1. Extract NEW insights from the observations (patterns, lessons, tips)
2. STRENGTHEN existing insights that are confirmed by new evidence (increase confidence)
3. WEAKEN or REMOVE insights contradicted by new evidence
4. MERGE similar insights into more concise ones

Output a JSON array of insights. Each insight:
{{"id": "ins_NNN", "category": "tool_strategy|task_pattern|failure_lesson|general_wisdom", "content": "...", "confidence": 0.0-1.0, "tags": ["tag1", "tag2"]}}

Keep at most {max_insights} insights total. Quality over quantity.
Output ONLY the JSON array."""


MEMORY_DEDUP_PROMPT_TEMPLATE = """You are a memory deduplication judge for an affordance agent.
Decide whether these two samples are duplicates (same affordance intent and target part/object).

Similarity score (lexical): {similarity:.3f}

NEW sample:
- task: {new_task}
- object: {new_object}
- tool chain: {new_tools}
- iou: {new_iou}

OLD sample:
- task: {old_task}
- object: {old_object}
- tool chain: {old_tools}
- iou: {old_iou}

Answer only one word: DUPLICATE or DIFFERENT."""


MEMORY_EVICTION_PROMPT_TEMPLATE = """You are a memory manager for an affordance detection agent.
The memory window has {num_entries} entries but the max capacity is {max_size}.
You must select exactly {num_to_remove} entries to DELETE.
{dist_note}{verified_note}
**Deletion guidelines:**
1. Remove unverified entries (no IoU) first — they have no outcome signal.
2. Remove redundant entries (same task type and object) — among duplicates, keep the one with the most informative IoU (either the best or the worst).
3. Keep both positive examples (high IoU) AND negative examples (low IoU) — both carry learning signal. Negative examples help avoid repeating failed strategies.
4. As a last resort, remove the oldest entries.

**Prefer diversity.** Keep entries covering different task types, objects, tool strategies, AND different IoU outcomes (both successes and failures).

Current memory entries:
{entries_text}

Respond with ONLY a JSON array of {num_to_remove} indices to delete, e.g. [0, 3, 5].
No explanation, just the JSON array."""
