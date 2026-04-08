"""
Detection skill prompts for Qwen3-VL vision-language model.

These prompts guide Qwen3-VL to:
1. Predict bounding boxes and key-points for affordance regions
2. Verify the quality of bbox/point annotations
"""


DETECTION_SYSTEM_PROMPT_TEMPLATE = """Given the image(s) of a scene, the task is to predict the interactive region (affordance) of the object that matches the provided task instruction. The task instruction is "{object_part}". Affordance = the specific region a human would physically interact with to accomplish the described action.

You may receive the following inputs in order:
- **Reference images** (optional): Supplementary images for visual context — interaction visualizations, similar exemplar scenes and their ground-truth masks, or other visual references. These are for understanding only — do NOT output coordinates for them.
- **Target scene image**: The main image you need to analyze. Output coordinates ONLY for this image.
- **Context instructions** (optional): Analysis and guidance specifying WHICH target(s) to detect, HOW MANY, and what to focus on. **You MUST follow these instructions.**

**Follow these reasoning steps**:
1. Read the task instruction carefully — determine whether it requires detecting ONE target or MULTIPLE targets. Singular phrasing ("the handle", "where to grip") means one; plural phrasing ("all buttons", "every handle") means multiple. If context instructions specify the count, follow them.
2. Identify the key components of the object in the target scene image (e.g., shape, features, possible points of interaction).
3. If reference images are provided, study them to understand how the object is interacted with in relation to the given affordance type. Then return to the target scene image.
4. If context instructions are provided, follow them to determine which object/part to focus on.
5. Ground the target part(s) in the image. The bounding box should be as tight as possible. Each bounding box corresponds to exactly ONE key point — the point must be accurately located on the solid surface of the target part inside that bounding box.
   - Key points seed a segmentation model (SAM2) — they MUST land directly ON the solid pixel surface of the target part, not on background or empty space.
   - Do NOT default to the bounding box center. The center often falls on empty space (hole inside a handle, gap between legs, background behind a thin bar).
   - For hollow/ring-shaped parts (handle, ring, loop): point on the SOLID MATERIAL you would grip, NOT in the hole.
   - For thin/elongated parts (edge, rim, blade): point on the thickest visible section.
   - Self-check: "If I draw a small circle around this point, would MOST pixels belong to the target part?" If not, move the point.

**CRITICAL - TARGET COUNT DETERMINATION (read the task carefully)**:
- The number of targets depends on what the TASK DESCRIPTION asks for. Do NOT blindly detect every similar-looking object in the scene.
- **Singular tasks** ("which handle would you grip", "the handle of the kettle", "where would you place your hand"): detect exactly ONE target — the specific instance the task refers to.
- **Plural / exhaustive tasks** ("all drawer handles", "every button on the remote", "the seat sections of the bench"): detect ALL matching instances, including partially occluded ones.
- **Ambiguous cases**: if the context instructions specify a target count or identify which instance(s) to focus on, follow that guidance.
- When multiple similar objects exist but the task singles out one (e.g., "the top suitcase" among a stack), detect ONLY the one the task refers to.

**CRITICAL - POINT PLACEMENT (affects segmentation quality)**:
- The point is a seed for a segmentation model (SAM2). SAM2 generates a mask by growing outward from this point. If the point lands on the wrong surface, the entire mask will be wrong.
- The point MUST land directly ON the visible pixel surface of the target object part — not on background, not on empty space, not on a different object.
- Do NOT simply use the geometric center of the bounding box as the point. The bbox center often falls on empty space (e.g., the hole inside a handle, the gap between chair legs, the background behind a thin bar).
- Instead, visually identify the most representative solid surface area of the target part and place the point there.
- For hollow/ring-shaped parts (cup handle, ring, loop): place the point on the SOLID MATERIAL of the handle (the bar/arc you would grip), NOT in the hole or gap inside the loop.
- For thin/elongated parts (edge, rim, handle bar, blade): place the point on the thickest visible section of that part.
- For concave parts (bowl interior, cup interior): place the point inside the concave surface, not on the rim.
- Self-check: "If I draw a small circle around this point, would MOST pixels in that circle belong to the target part?" If not, move the point.

**Coordinate system & units**:
- Image origin is the top-left corner (0, 0).
- x increases to the right; y increases downward.
- Use relative coordinates in range [0, 1000], where (0, 0) is top-left and (1000, 1000) is bottom-right.
- All x, y values MUST be integers between 0 and 1000.

**Output format**:
### Thinking
thinking process
### Output
{{
    "task": "the task instruction",
    "object_name": "the name of the object",
    "object_part": "the [object part] of the [object name] (e.g. the blade of the shears)",
    "part_bbox": [[x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max], ...],
    "key_points": [[x1, y1], [x2, y2], ...]
}}"""


DETECTION_USER_PROMPT_TEMPLATE = """{context_section}Now analyze the target scene image and output your thinking process followed by the JSON result."""


DETECTION_VERIFY_PROMPT_TEMPLATE = """You are verifying the quality of bounding box and point annotations for an affordance detection task.

Task: {object_part}

The image shows green bounding boxes and green star points annotated on the scene. There are {num_bboxes} bbox(es) and {num_points} point(s).

Check ALL of the following:
1. **Target count**: Does the number of detected targets match what the task asks for? If the task refers to a single specific object (e.g., "the handle of the kettle", "which suitcase's handle"), there should be exactly 1 bbox+point. If it asks for multiple (e.g., "all buttons"), there should be multiple. Mark incorrect if the count is wrong.
2. **Point accuracy**: Does EACH green star land directly ON the visible SOLID SURFACE (actual material pixels) of the target part? NOT on background, empty space, a hole inside a handle, or a different object. For handle/loop shapes, the point must be on the grip bar itself, not in the air gap.
3. **Bbox tightness**: Is each bounding box reasonably tight around the target part only, without including unrelated objects or excessive background?

Respond with EXACTLY this JSON format:
{{
    "correct": true/false,
    "feedback": "description of issues if any, empty string if correct"
}}

Be strict: mark as incorrect if target count is wrong, if ANY point is on background/empty space/wrong object, or if a bbox is excessively loose."""
