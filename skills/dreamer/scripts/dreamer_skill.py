

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from prompts.dreamer import (
    DREAMER_SYSTEM_PROMPT_SINGLE,
    DREAMER_MULTI_TARGET_PROMPT,
    DREAMER_ANALYSIS_SYSTEM_PROMPT,
)

try:
    import requests
except ImportError as e:
    raise ImportError(" requests，：pip install requests") from e

from PIL import Image


# Reference / output directory
REFERENCE_DIR = Path(__file__).parent.parent / "reference"

# API 
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
PROMPT_GEN_MODEL = os.getenv("DREAMER_PROMPT_MODEL", "gpt-4o")
IMAGE_GEN_MODEL = os.getenv("DREAMER_IMAGE_MODEL", "qwen-image-edit")


def _load_api_config() -> Dict[str, Any]:
    """Load API configuration from config.py or environment variables."""
    config: Dict[str, Any] = {}
    try:
        from pathlib import Path as _P
        config_path = _P(__file__).resolve().parent.parent.parent / "config.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
                config["API_KEY"] = getattr(cfg, "API_KEY", None)
                config["API_BASE_URL"] = getattr(cfg, "API_BASE_URL", None)
                config["DEFAULT_MODEL"] = getattr(cfg, "DEFAULT_MODEL", None)
    except Exception as e:
        print(f"[dreamer] Warning: Failed to load config.py: {e}")

    # Environment variable fallback
    if not config.get("API_KEY"):
        config["API_KEY"] = os.environ.get("API_KEY", API_KEY)
    if not config.get("API_BASE_URL"):
        config["API_BASE_URL"] = os.environ.get("API_BASE_URL", API_BASE_URL)
    if not config.get("DEFAULT_MODEL"):
        config["DEFAULT_MODEL"] = os.environ.get("DEFAULT_MODEL", PROMPT_GEN_MODEL)
    return config


def _image_to_base64(image_path: str, max_side: int = 1280) -> str:
    """ base64 。"""
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Step 1: Call GPT-4o to generate editing prompts (supports multi-target)
def _generate_editing_prompts(
    image_path: str,
    task: str,
    object_name: Optional[str] = None,
) -> List[str]:
    """
     GPT-4o API 。
    ，。

    Returns:
        ，
    """
    config = _load_api_config()
    api_key = config.get("API_KEY", API_KEY)
    api_base = config.get("API_BASE_URL", API_BASE_URL)

    img_base64 = _image_to_base64(image_path)

    user_text = f"The given TASK is:\n{task}"
    if object_name:
        user_text += f"\nTarget object: {object_name}"

    messages = [
        {"role": "system", "content": DREAMER_MULTI_TARGET_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": PROMPT_GEN_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    print(f"[dreamer] Task: {task}")
    if object_name:
        print(f"[dreamer] Object: {object_name}")
    print(f"[dreamer] Generating editing prompt(s) with {PROMPT_GEN_MODEL}...")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        #  JSON 
        prompts = _parse_multi_prompts(content)

        print(f"[dreamer] Generated {len(prompts)} editing prompt(s):")
        for idx, p in enumerate(prompts):
            print(f"  [{idx}] {p[:150]}{'...' if len(p) > 150 else ''}")

        return prompts

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        print(f"[dreamer] {PROMPT_GEN_MODEL} returned HTTP {status}")
        return []
    except requests.exceptions.Timeout:
        print(f"[dreamer] {PROMPT_GEN_MODEL} timeout")
        return []
    except requests.exceptions.ConnectionError as e:
        print(f"[dreamer] Connection error: {e}")
        return []
    except Exception as e:
        print(f"[dreamer] API error: {e}")
        return []


def _parse_multi_prompts(content: str) -> List[str]:
    """
     GPT-4o 。
     JSON  {"prompts": [...]} 。
    """
    #  JSON
    json_match = re.search(r'\{[\s\S]*"prompts"\s*:\s*\[[\s\S]*\][\s\S]*\}', content)
    if json_match:
        try:
            data = json.loads(json_match.group())
            prompts = data.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass

    #  markdown code block  JSON
    code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            prompts = data.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass

    #  JSON ，
    content = content.strip().strip('"').strip("'")
    if content:
        return [content]

    return []


# Step 2: Call qwen-image to generate interaction image
def _generate_image_with_qwen(
    editing_prompt: str,
    image_path: str,
    save_path: str,
) -> Optional[str]:
    """
     qwen-image-edit  API， + 。
     multipart form 。

    Args:
        editing_prompt:  GPT-4o 
        image_path: 
        save_path: 

    Returns:
        ， None
    """
    api_key = API_KEY
    api_base = API_BASE_URL

    img = Image.open(image_path)
    w, h = img.size
    max_side = 1024
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    print(f"[dreamer] Generating edited image with {IMAGE_GEN_MODEL}...")
    print(f"[dreamer] Input image: {w}x{h} -> {img.size[0]}x{img.size[1]}")

    try:
        # qwen-image-edit  multipart form 
        resp = requests.post(
            f"{api_base}/images/edits",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"image": ("image.png", img_bytes, "image/png")},
            data={
                "model": IMAGE_GEN_MODEL,
                "prompt": editing_prompt,
                "n": "1",
                "size": "1024x1024",
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()

        img_data = result.get("data", [{}])[0]

        #  URL 
        if "url" in img_data:
            img_url = img_data["url"]
            print(f"[dreamer] Downloading generated image...")
            img_resp = requests.get(img_url, timeout=60)
            if img_resp.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(img_resp.content)
                gen_img = Image.open(save_path)
                print(f"[dreamer] ✅ Image generated: {gen_img.size[0]}x{gen_img.size[1]}, saved to {save_path}")
                return save_path

        #  base64 
        if "b64_json" in img_data:
            img_decoded = base64.b64decode(img_data["b64_json"])
            with open(save_path, "wb") as f:
                f.write(img_decoded)
            gen_img = Image.open(save_path)
            print(f"[dreamer] ✅ Image generated (b64): {gen_img.size[0]}x{gen_img.size[1]}, saved to {save_path}")
            return save_path

        print(f"[dreamer] ⚠ API returned 200 but no image data found")
        return None

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        err_msg = ""
        try:
            err_msg = e.response.json().get("error", {}).get("message", "")[:100]
        except Exception:
            pass
        print(f"[dreamer] {IMAGE_GEN_MODEL} returned HTTP {status}: {err_msg}")
        return None
    except requests.exceptions.Timeout:
        print(f"[dreamer] {IMAGE_GEN_MODEL} timeout")
        return None
    except Exception as e:
        print(f"[dreamer] Image generation error: {e}")
        return None


# Step 3: Analyze the generated interaction image
ANALYSIS_SYSTEM_PROMPT = DREAMER_ANALYSIS_SYSTEM_PROMPT


def _analyze_interaction_image(
    original_image_path: str,
    generated_image_path: str,
    task: str,
    object_name: Optional[str] = None,
) -> str:
    """
     GPT-4o ，
    /。

    Returns:
        
    """
    config = _load_api_config()
    api_key = config.get("API_KEY", API_KEY)
    api_base = config.get("API_BASE_URL", API_BASE_URL)

    orig_b64 = _image_to_base64(original_image_path)
    gen_b64 = _image_to_base64(generated_image_path)

    user_text = f"The given TASK is: {task}"
    if object_name:
        user_text += f"\nTarget object: {object_name}"
    user_text += "\n\nImage 1 is the original scene. Image 2 is the edited image showing the interaction. Please analyze the interaction."

    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": PROMPT_GEN_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
    }

    print(f"[dreamer] Analyzing interaction image with {PROMPT_GEN_MODEL}...")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print(f"[dreamer] Interaction analysis ({len(content)} chars):")
        print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
        return content
    except Exception as e:
        print(f"[dreamer] Analysis failed: {e}")
        return ""


# Main entry point
def run_dreamer_skill(
    image_path: str,
    task: str,
    object_name: Optional[str] = None,
    editing_prompt: Optional[str] = None,
    dynamic_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dreamer skill entry point.

    When *editing_prompt* is provided by the decision model, it is used
    directly — no separate LLM call is made for prompt generation.  This
    keeps the entire reasoning chain inside the decision model's
    multi-turn conversation.

    Args:
        image_path: Scene image path.
        task: Affordance task description.
        object_name: Optional target object name.
        editing_prompt: Optional editing prompt written by the decision
            model.  When provided, skips the internal prompt generation
            step.
        dynamic_params: Optional dynamic strategy parameters.
    """
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ： task 
    if dynamic_params:
        print(f"[dreamer] Dynamic params received: {list(dynamic_params.keys())}")
        extra_context_parts = []
        for key, value in dynamic_params.items():
            readable_key = key.replace("_", " ")
            extra_context_parts.append(f"{readable_key}: {value}")
        if extra_context_parts:
            extra_context = " | ".join(extra_context_parts)
            task = f"{task} [Strategy guidance: {extra_context}]"
            print(f"[dreamer] Enriched task: {task[:150]}...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_stem = Path(image_path).stem

    # ---- Step 1: Editing prompt ----
    if editing_prompt and editing_prompt.strip():
        prompts = [editing_prompt.strip()]
        print(f"[dreamer] Using decision model's editing prompt: {prompts[0][:150]}...")
    else:
        prompts = _generate_editing_prompts(image_path, task, object_name)

    if not prompts:
        return {
            "error": "Failed to generate dreamer interaction prompt",
            "task": task,
            "object_name": object_name,
        }

    num_targets = len(prompts)
    print(f"[dreamer] {num_targets} target(s) for this task")

    # ---- Step 2:  qwen-image-edit  ----
    generated_image_paths = []
    for idx, prompt in enumerate(prompts):
        suffix = f"_t{idx}" if num_targets > 1 else ""
        img_save_path = str(REFERENCE_DIR / f"{img_stem}_dreamer_{timestamp}{suffix}.png")
        if num_targets > 1:
            print(f"\n[dreamer] --- Target {idx + 1}/{num_targets} ---")
        gen_path = _generate_image_with_qwen(prompt, image_path, img_save_path)
        generated_image_paths.append(gen_path)

    # ---- Step 3:  ----
    analyses = []
    for idx, gen_path in enumerate(generated_image_paths):
        if gen_path is not None:
            if num_targets > 1:
                print(f"\n[dreamer] --- Analyzing Target {idx + 1}/{num_targets} ---")
            analysis = _analyze_interaction_image(
                image_path, gen_path, task, object_name
            )
            analyses.append(analysis)
        else:
            analyses.append("")

    valid_analyses = [a for a in analyses if a]
    if num_targets > 1 and len(valid_analyses) > 1:
        combined_analysis = ""
        for idx, a in enumerate(analyses):
            if a:
                combined_analysis += f"\n[Target {idx + 1}]\n{a}\n"
        combined_analysis = combined_analysis.strip()
    elif valid_analyses:
        combined_analysis = valid_analyses[0]
    else:
        combined_analysis = ""

    # ---- Step 4:  JSON  ----
    ref_filename = f"{img_stem}_dreamer_{timestamp}.json"
    ref_path = REFERENCE_DIR / ref_filename

    ref_data = {
        "task": task,
        "object_name": object_name,
        "num_targets": num_targets,
        "dreamer_prompts": prompts,
        "interaction_analyses": analyses,
        "image_path": image_path,
        "generated_image_paths": generated_image_paths,
        "timestamp": timestamp,
        "model_prompt_gen": PROMPT_GEN_MODEL,
        "model_image_gen": IMAGE_GEN_MODEL,
        "model_analysis": PROMPT_GEN_MODEL,
    }

    try:
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, indent=2, ensure_ascii=False)
        print(f"[dreamer] Saved reference JSON to: {ref_path}")
    except Exception as e:
        print(f"[dreamer] Warning: Failed to save reference: {e}")
        ref_path = None

    success_count = sum(1 for p in generated_image_paths if p is not None)
    combined_prompt = "\n---\n".join(prompts)

    result = {
        "dreamer_prompts": prompts,
        "dreamer_prompt": combined_prompt,
        "interaction_description": combined_prompt,
        "interaction_analysis": combined_analysis,
        "num_targets": num_targets,
        "task": task,
        "object_name": object_name,
        "reference_file": str(ref_path) if ref_path else None,
        "generated_image_paths": [p for p in generated_image_paths if p],
    }

    # ： generated_image_path
    if generated_image_paths and generated_image_paths[0]:
        result["generated_image_path"] = generated_image_paths[0]

    if dynamic_params:
        result["dynamic_params_used"] = dynamic_params

    print(f"\n[dreamer] ✅ Complete: {success_count}/{num_targets} images generated + analyzed, "
          f"saved to dreamer/reference/")

    return result
