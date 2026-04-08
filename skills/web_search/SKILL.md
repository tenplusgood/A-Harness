---
name: web_search
description: >
  Web search tool that retrieves concise textual knowledge for an affordance task:
  object parts, affordance mechanism, and how humans interact with the object.
  The decision model should CUSTOMIZE the search by providing a search strategy
  (search_focus, target_part, interaction_type) based on its
  analysis of the task. This produces much more targeted and useful results
  than a generic search.
  Returns: affordance_name, part_name, object_name, reasoning (text insights),
  plus searchable evidence (queries, results, crawled URLs).
characteristics:
  - decision-model-driven search strategy for targeted results
  - retrieves textual evidence only (no image crawling or image analysis)
  - search queries are generated from the model's strategy, not just the raw task
  - moderate latency due to search and crawling
when_to_use:
  - object part or affordance mechanism is unclear
  - task involves unfamiliar objects or uncommon interactions
  - decision model needs external evidence before detection
when_not_to_use:
  - task is straightforward and confidence is high
  - already have sufficient visual and textual evidence from current context
---

## Purpose

This tool performs a **decision-model-directed web search** to help the agent understand **how humans interact with a specific object** in an affordance task. The decision model analyzes the task first, then provides a customized search strategy that guides:

1. **What to search for** — targeted queries based on the model's understanding of the task
2. **What knowledge to extract** — structural, functional, or spatial information about the object

## Inputs

| Parameter          | Type   | Required | Description |
|-------------------|--------|----------|-------------|
| `question`        | string | ✅       | A specific question about the affordance — be precise |
| `task`            | string | optional | Full task description / affordance reasoning context |
| `image_hint`      | string | optional | Brief textual description of the scene or object |
| `search_focus`    | string | optional | Main search direction (e.g., "cup handle grasping ergonomics") |
| `target_part`     | string | optional | Hypothesized target part (e.g., "handle") |
| `interaction_type`| string | optional | How humans interact (e.g., "power grasp", "push", "twist") |
| `knowledge_type`  | string | optional | Knowledge needed: "structural", "functional", "spatial" |

## Example: Task-Directed Search

**Task:** "Identify the part of a cup used for grasping"

**Good call** (model provides strategy):
```json
{
  "question": "What part of a cup should be grasped when picking it up to drink?",
  "search_focus": "cup handle grasping ergonomics",
  "target_part": "handle",
  "interaction_type": "power grasp"
}
```

**Generic call** (less effective):
```json
{
  "question": "cup affordance"
}
```

## Outputs (JSON)

| Field              | Type     | Description |
|-------------------|----------|-------------|
| `affordance_name` | string   | The inferred action (e.g., "pressing", "turning", "sitting") |
| `part_name`       | string   | The specific object part (e.g., "handle", "lever", "seat") |
| `object_name`     | string   | The object involved (e.g., "faucet", "door", "bench") |
| `reasoning`       | string   | Key textual insights from web content about the interaction |
| `search_queries`  | string[] | Generated search queries |
| `search_results`  | object[] | Aggregated search result entries |
| `crawled_urls`    | string[] | URLs whose textual content was crawled |

## When to Use

- You are **uncertain** which part of the object is the interaction target
- The task involves an **unfamiliar object** or **uncommon affordance**
- You need **external textual evidence** before detection

## When NOT to Use

- The task is straightforward (e.g., "sit on the chair" → obviously the seat)
- You already have high confidence about the target object and part
