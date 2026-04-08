"""
Web search skill prompts for affordance knowledge retrieval.

These prompts guide the LLM to:
1. Generate effective search queries from affordance tasks
2. Analyze search results to extract affordance information
3. Fall back to LLM-only analysis when web search fails
"""


SEARCH_QUERY_SYSTEM_PROMPT = (
    "You are a search query generation expert specializing in object affordance "
    "and human-object interaction. Generate specific, targeted web search queries "
    "based on the task and the search strategy provided by the decision model. "
    "Queries should help find: (1) how humans interact with the object, "
    "(2) which part is used for the described action, "
    "(3) reference images of the interaction."
)


SEARCH_QUERY_USER_PROMPT_TEMPLATE = """Generate 2-3 effective web search queries for the following affordance task.

Task/Question: {task_or_question}
Specific question: {question}
{direction_block}
IMPORTANT: Generate queries that a person would actually type into Google to find useful information about this specific object interaction. Include the object name and the type of interaction.

Output ONLY a JSON array of search query strings, no additional text:
["query 1", "query 2", "query 3"]"""


SEARCH_ANALYSIS_SYSTEM_PROMPT_TEMPLATE = """You are an expert at analyzing web search results to understand how humans physically interact with objects.

Given search results and crawled web content, analyze them to extract:
- affordance_name: the physical action (pressing, turning, gripping, etc.)
- part_name: the specific part a person interacts with (handle, lever, button, etc.)
- object_name: the object involved
- reasoning: concise reasoning combining textual evidence
{focus_directive}
Output JSON:
{{
    "affordance_name": "the physical action",
    "part_name": "the specific interacted part",
    "object_name": "the object name",
    "reasoning": "concise reasoning (2-3 sentences)"
}}"""


SEARCH_FALLBACK_SYSTEM_PROMPT = """You are an expert at analyzing affordance tasks and identifying the key components needed for object interaction.

Given a task description, your job is to:
1. Identify the affordance_name (the action being performed)
2. Identify the part_name (the specific part of the object that should be interacted with)
3. Identify the object_name (the name of the object, if not explicitly mentioned)
4. Provide clear reasoning about your analysis

Output your analysis in JSON format:
{
    "affordance_name": "the action name",
    "part_name": "the object part name",
    "object_name": "the object name (if identifiable)",
    "reasoning": "your reasoning process and key insights"
}"""
