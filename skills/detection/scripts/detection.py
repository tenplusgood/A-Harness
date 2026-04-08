import argparse
import base64
import io
import json
import os
import re
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from prompts.detection import (
    DETECTION_SYSTEM_PROMPT_TEMPLATE,
    DETECTION_USER_PROMPT_TEMPLATE,
    DETECTION_VERIFY_PROMPT_TEMPLATE,
)

try:
    import requests
except ImportError as e:
    raise ImportError(
        " requests，：pip install requests"
    ) from e

SAM2ImagePredictor = None  # lazy import; resolved in _get_sam2()

import matplotlib.pyplot as plt

API_KEY = os.getenv("DETECTION_API_KEY", os.getenv("API_KEY", ""))
API_BASE_URL = os.getenv("DETECTION_API_BASE_URL", os.getenv("API_BASE_URL", ""))
MODEL_NAME = os.getenv("DETECTION_MODEL_NAME", "qwen3-vl-235b-a22b-instruct")

QWEN3VL_COORD_RANGE = 1000

_SAM2_SINGLETON: Optional["SAM2ImagePredictor"] = None
_SAM2_SINGLETON_KEY: Optional[str] = None

# Thread lock for SAM2/SAM3 GPU model — prevents race conditions when
# multiple evaluation workers share the same singleton model.
_SAM2_LOCK = threading.Lock()

_SAM3_SINGLETON: Optional[Any] = None  # (processor, model) tuple
_SAM3_SINGLETON_KEY: Optional[str] = None
_SAM3_LOCK = threading.Lock()


def _get_sam2(model_path: str) -> "SAM2ImagePredictor":
    """Load SAM2 predictor with singleton caching (thread-safe)."""
    global SAM2ImagePredictor, _SAM2_SINGLETON, _SAM2_SINGLETON_KEY
    with _SAM2_LOCK:
        key = str(model_path)
        if _SAM2_SINGLETON is not None and _SAM2_SINGLETON_KEY == key:
            return _SAM2_SINGLETON

        if SAM2ImagePredictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor as _cls
                SAM2ImagePredictor = _cls
            except ImportError as e:
                raise ImportError(
                    "Unable to import sam2. Please install the segment-anything-2 dependency first."
                ) from e

        _SAM2_SINGLETON = SAM2ImagePredictor.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_obj = getattr(_SAM2_SINGLETON, "model", None)
            if model_obj is not None and hasattr(model_obj, "to"):
                model_obj = model_obj.to(device)
                try:
                    if device.type == "cuda":
                        model_obj = model_obj.to(dtype=torch.bfloat16)
                except Exception:
                    pass
                setattr(_SAM2_SINGLETON, "model", model_obj)
        except Exception:
            pass
        _SAM2_SINGLETON_KEY = key
        return _SAM2_SINGLETON


def _resolve_sam3_path(model_path: str) -> str:
    """Resolve SAM3 model path: ModelScope download, local dir, or HF repo id.

    Supported formats:
        "modelscope::facebook/sam3"  — download from ModelScope to local cache
        "/absolute/local/path"       — use local directory directly
        "facebook/sam3"              — HuggingFace repo id (default)
    """
    if model_path.startswith("modelscope::"):
        repo_id = model_path[len("modelscope::"):]
        try:
            from modelscope import snapshot_download
        except ImportError:
            raise ImportError(
                "modelscope package is required for downloading from ModelScope. "
                "Install it with: pip install modelscope"
            )
        print(f"[SAM3] Downloading from ModelScope: {repo_id} ...")
        local_dir = snapshot_download(repo_id)
        print(f"[SAM3] ModelScope download complete → {local_dir}")
        return local_dir

    mp = Path(str(model_path))
    if mp.is_dir():
        print(f"[SAM3] Using local directory: {model_path}")
        return str(mp)

    return model_path


def _get_sam3(model_path: str = "facebook/sam3"):
    """Load SAM3 (processor, model) with singleton caching (thread-safe).

    model_path accepts:
        "modelscope::facebook/sam3"  — download from https://modelscope.cn
        "/local/path/to/sam3"        — pre-downloaded local weights
        "facebook/sam3"              — HuggingFace repo id (default)
    """
    global _SAM3_SINGLETON, _SAM3_SINGLETON_KEY
    with _SAM3_LOCK:
        key = str(model_path)
        if _SAM3_SINGLETON is not None and _SAM3_SINGLETON_KEY == key:
            return _SAM3_SINGLETON

        resolved = _resolve_sam3_path(model_path)

        from transformers import Sam3Processor, Sam3Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM3] Loading processor and model from {resolved} ...")
        processor = Sam3Processor.from_pretrained(resolved)
        model = Sam3Model.from_pretrained(resolved)
        try:
            model = model.to(device)
            if device.type == "cuda":
                model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass
        model.eval()
        _SAM3_SINGLETON = (processor, model)
        _SAM3_SINGLETON_KEY = key
        print(f"[SAM3] Loaded on {device}")
        return _SAM3_SINGLETON


def _configure_torch_for_inference() -> None:
    """Best-effort speedups for GPU inference."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def _sanitize_payload_preview(obj: Any, max_text_len: int = 900) -> Any:
    """Sanitize API payload for readable debug export (truncate text/base64)."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image"):
                header, _, data = v.partition(",")
                out[k] = {
                    "data_url_header": header[:80] + ("..." if len(header) > 80 else ""),
                    "base64_chars": len(data),
                }
            else:
                out[k] = _sanitize_payload_preview(v, max_text_len=max_text_len)
        return out
    if isinstance(obj, list):
        return [_sanitize_payload_preview(x, max_text_len=max_text_len) for x in obj]
    if isinstance(obj, str):
        if len(obj) <= max_text_len:
            return obj
        half = max_text_len // 2
        return obj[:half] + f"\n...[truncated {len(obj) - 2 * half} chars]...\n" + obj[-half:]
    return obj


def _dump_qwen_payload_preview(payload: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Dump one preview JSON for the Qwen detection request."""
    try:
        preview_dir = os.path.join(output_dir, "payload_previews")
        os.makedirs(preview_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(preview_dir, f"detection_qwen_request_{ts}.json")
        data = {
            "model_name": MODEL_NAME,
            "request_preview": _sanitize_payload_preview(payload),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        print(f"[Detection] Failed to dump payload preview: {e}")
        return None


def _dump_qwen_full_context(
    output_dir: str,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    task_context: Optional[str],
    reference_images: Optional[List[Dict[str, str]]],
    attempt: int = 0,
    correction_feedback: Optional[str] = None,
) -> Optional[str]:
    """Save the complete context Qwen receives for this sample.

    Saves the full text of every input component — system prompt, context
    instructions, user prompt, and reference image metadata — so that each
    sample's inputs can be inspected without re-running.

    File is saved as ``{image_stem}_qwen_context.json`` in *output_dir*.
    """
    try:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        path = os.path.join(output_dir, f"{image_stem}_qwen_context.json")

        ref_summary = []
        if reference_images:
            for ref in reference_images:
                ref_summary.append({
                    "label": ref.get("label", ""),
                    "path": ref.get("path", ""),
                })

        data: Dict[str, Any] = {
            "model": MODEL_NAME,
            "image_path": image_path,
            "attempt": attempt,
            "system_prompt": system_prompt,
            "context_instructions": task_context,
            "user_prompt": user_prompt,
            "reference_images": ref_summary if ref_summary else None,
        }
        if correction_feedback:
            data["correction_feedback"] = correction_feedback

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Detection] Full Qwen context saved to {path}")
        return path
    except Exception as e:
        print(f"[Detection] Failed to dump full context: {e}")
        return None


def save_image_with_points_and_box(
    image: Image.Image, points: List[List[float]], boxes: List[List[float]], save_path: str
) -> str:
    h, w = image.height, image.width
    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    for box in boxes:
        x0, y0 = box[0], box[1]
        ax.add_patch(plt.Rectangle((x0, y0), box[2] - x0, box[3] - y0, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    if points:
        pts = np.array(points)
        ax.scatter(pts[:, 0], pts[:, 1], color="green", marker="*", s=200, edgecolor="white", linewidth=1.0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return save_path


def save_image_with_mask(mask: np.ndarray, image: Image.Image, save_path: str, borders: bool = False) -> str:
    h, w = image.height, image.width
    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(w / dpi, h / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)

    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    mh, mw = mask.shape[-2:]
    mask_image = mask.reshape(mh, mw, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    _ = borders  # kept for API compatibility

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return save_path

def _relative1000_to_normalized(coord: float) -> float:
    return coord / QWEN3VL_COORD_RANGE


def _relative1000_to_pixel(coord: float, dimension: int) -> int:
    return int(coord / QWEN3VL_COORD_RANGE * dimension)


def _parse_bbox_and_points_from_text(text: str, image_width: int, image_height: int) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]]]:
    bboxes: List[List[float]] = []
    points: List[List[float]] = []

    # Remove markdown code blocks
    text_cleaned = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
    text_cleaned = text_cleaned.strip()

    # Extract JSON with balanced braces
    brace_count = 0
    start_idx = -1
    json_str = None
    for i, char in enumerate(text_cleaned):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text_cleaned[start_idx:i+1]
                break
    
    # Fallback regex
    if not json_str:
        json_match = re.search(r'\{[^{}]*"(?:bboxes?|part_bbox)"[^{}]*\}', text_cleaned, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(0)
    
    if json_str:
        try:
            parsed = json.loads(json_str)
            
            # Parse bboxes (support both v1 "bboxes"/"bbox" and v2 "part_bbox")
            if "bboxes" in parsed:
                bboxes_val = parsed["bboxes"]
                if isinstance(bboxes_val, list):
                    bboxes = bboxes_val
            elif "part_bbox" in parsed:
                bbox_val = parsed["part_bbox"]
                if isinstance(bbox_val, list):
                    if len(bbox_val) > 0 and isinstance(bbox_val[0], (int, float)):
                        bboxes = [bbox_val]
                    else:
                        bboxes = bbox_val
            elif "bbox" in parsed:
                bbox_val = parsed["bbox"]
                if isinstance(bbox_val, list):
                    if len(bbox_val) > 0 and isinstance(bbox_val[0], (int, float)):
                        bboxes = [bbox_val]
                    else:
                        bboxes = bbox_val
            
            # Parse points (support both v1 "points" and v2 "key_points")
            points_val = parsed.get("key_points") or parsed.get("points")

            def extract_points(val):
                """Recursively extract [x, y] pairs from nested lists."""
                result = []
                if isinstance(val, list):
                    if len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                        return [val]
                    elif len(val) > 2 and all(isinstance(x, (int, float)) for x in val):
                        for i in range(0, len(val) - 1, 2):
                            result.append([val[i], val[i + 1]])
                    else:
                        for item in val:
                            result.extend(extract_points(item))
                return result

            if isinstance(points_val, list):
                points = extract_points(points_val)

        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    # Fallback: text pattern matching
    if not bboxes:
        bbox_pattern = r'(?:part_)?bbox[:\s]+\[([^\]]+)\]'
        bbox_matches = re.findall(bbox_pattern, text, re.IGNORECASE)
        for match in bbox_matches:
            try:
                coords = [float(x.strip()) for x in match.split(",")]
                if len(coords) == 4:
                    bboxes.append(coords)
            except ValueError:
                pass

    if not points:
        points_pattern = r'(?:key_)?points?[:\s]+\[([^\]]+)\]'
        points_matches = re.findall(points_pattern, text, re.IGNORECASE | re.DOTALL)
        for match in points_matches:
            try:
                nested = re.findall(r'\[([^\]]+)\]', match)
                for n in nested:
                    coords = [float(x.strip()) for x in n.split(",")]
                    if len(coords) == 2:
                        points.append(coords)
            except ValueError:
                pass

    # Ensure types
    if not isinstance(bboxes, list):
        bboxes = []
    if not isinstance(points, list):
        points = []

    # Normalize coordinates to [0, 1]
    # Qwen3-VL outputs coordinates in 0-1000 relative system.
    # Detection logic:
    #   - All values <= 1.0          → already normalized (0-1)
    #   - Any value > 1 and <= 1000  → Qwen3-VL 0-1000 system → / 1000
    #   - Any value > 1000           → pixel coordinates → / dimension

    def _detect_coord_system(values: List[float]) -> str:
        """Detect whether coordinates are normalized, 0-1000, or pixel."""
        if not values:
            return "normalized"
        max_val = max(abs(v) for v in values)
        if max_val <= 1.0:
            return "normalized"
        elif max_val <= 1000:
            return "relative_1000"
        else:
            return "pixel"

    def _normalize_coord(val: float, dimension: int, system: str) -> float:
        """Convert coordinate to [0, 1] based on detected system."""
        if system == "normalized":
            return float(val)
        elif system == "relative_1000":
            return float(val) / QWEN3VL_COORD_RANGE
        else:  # pixel
            return float(val) / dimension

    # Normalize bboxes
    validated_bboxes: List[List[float]] = []
    for bbox in bboxes:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                raw = [float(v) for v in bbox]
                # Detect system using all 4 values
                system = _detect_coord_system(raw)
                
                x_min = _normalize_coord(raw[0], image_width, system)
                y_min = _normalize_coord(raw[1], image_height, system)
                x_max = _normalize_coord(raw[2], image_width, system)
                y_max = _normalize_coord(raw[3], image_height, system)
                
                # Ensure correct order
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                
                # Clamp to [0, 1]
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                x_max = max(0.0, min(1.0, x_max))
                y_max = max(0.0, min(1.0, y_max))
                
                if x_max > x_min and y_max > y_min:
                    validated_bboxes.append([x_min, y_min, x_max, y_max])
                    print(f"  bbox raw={[int(v) for v in raw]} ({system}) → normalized=[{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}]")
            except (ValueError, TypeError, IndexError):
                continue

    # Normalize points
    validated_points: List[List[float]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                raw = [float(v) for v in point]
                system = _detect_coord_system(raw)
                
                x = _normalize_coord(raw[0], image_width, system)
                y = _normalize_coord(raw[1], image_height, system)
                
                # Clamp to [0, 1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                validated_points.append([x, y])
                print(f"  point raw={[int(v) for v in raw]} ({system}) → normalized=[{x:.4f}, {y:.4f}]")
            except (ValueError, TypeError, IndexError):
                continue

    return validated_bboxes if validated_bboxes else None, validated_points if validated_points else None


def _qwen3vl_detection_and_pointing(
    image: Image.Image,
    object_part: str,
    task_context: Optional[str] = None,
    output_dir: Optional[str] = None,
    reference_images: Optional[List[Dict[str, str]]] = None,
    image_path: Optional[str] = None,
) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]], Optional[str]]:
    image_width, image_height = image.size

    system_prompt = DETECTION_SYSTEM_PROMPT_TEMPLATE.format(object_part=object_part)

    context_section = ""
    if task_context and task_context.strip():
        context_section = (
            "\n=== Context Instructions ===\n"
            f"{task_context.strip()}\n"
            "=== End of Context Instructions ===\n"
        )

    user_prompt = DETECTION_USER_PROMPT_TEMPLATE.format(context_section=context_section)

    # Convert image to base64
    image_base64 = _image_to_base64(image)
    
    # Prepare API request
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    user_content: List[Dict[str, Any]] = []

    if reference_images:
        user_content.append({
            "type": "text",
            "text": (
                "=== Reference Images (for understanding only — do NOT output coordinates for these) ==="
            ),
        })
        ref_added = 0
        for ref in reference_images:
            ref_label = str(ref.get("label", ""))
            ref_path = str(ref.get("path", ""))
            if not ref_path or not os.path.exists(ref_path):
                continue
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                w, h = ref_img.size
                max_side = 768
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    ref_img = ref_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                ref_img.save(buf, format="JPEG", quality=85)
                ref_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                if ref_label:
                    user_content.append({"type": "text", "text": ref_label})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"},
                })
                ref_added += 1
            except Exception as e:
                print(f"[Detection] Failed to embed reference image {ref_path}: {e}")
        if ref_added:
            print(f"[Detection] Embedded {ref_added} reference image(s) for Qwen")

    user_content.append({
        "type": "text",
        "text": "=== Target Scene Image (detect affordance in THIS image) ===",
    })
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    })
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    
    payload_preview_path = _dump_qwen_payload_preview(data, output_dir) if output_dir else None

    # Save full context for per-sample inspection
    if output_dir and image_path:
        _dump_qwen_full_context(
            output_dir=output_dir,
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_context=task_context,
            reference_images=reference_images,
        )

    # Make API call with retry + exponential backoff for concurrent requests
    import time as _time
    import random as _random

    MAX_RETRIES = 4
    output_text = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=300)

            # Handle rate limiting (429) and server errors (5xx) with retry
            if response.status_code == 429 or response.status_code >= 500:
                wait = min(2 ** attempt + _random.uniform(0.5, 1.5), 30)
                print(f"  [Detection API] {response.status_code} on attempt {attempt+1}/{MAX_RETRIES}, "
                      f"retrying in {wait:.1f}s ...")
                _time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            # Extract response text
            try:
                output_text = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                print(f"Error extracting response content: {e}")
                print(f"API response structure: {json.dumps(result, indent=2)[:1000]}")
                # Might be a transient malformed response — retry
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt + _random.uniform(0.5, 1.5)
                    print(f"  Retrying in {wait:.1f}s ...")
                    _time.sleep(wait)
                    continue
                return None, None, payload_preview_path

            # Success — break out of retry loop
            break

        except requests.exceptions.Timeout:
            print(f"  [Detection API] Timeout on attempt {attempt+1}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt + _random.uniform(0.5, 1.5)
                print(f"  Retrying in {wait:.1f}s ...")
                _time.sleep(wait)
                continue
            print(f"API call failed after {MAX_RETRIES} attempts: timeout")
            return None, None, payload_preview_path

        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            # Retry on connection errors and server-side errors
            if attempt < MAX_RETRIES - 1 and (status_code is None or status_code >= 500):
                wait = 2 ** attempt + _random.uniform(0.5, 1.5)
                print(f"  [Detection API] {type(e).__name__} on attempt {attempt+1}/{MAX_RETRIES}, "
                      f"retrying in {wait:.1f}s ...")
                _time.sleep(wait)
                continue

            print(f"API call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.json()
                    print(f"Error response: {json.dumps(error_body, indent=2)}")
                except Exception:
                    print(f"Error response text: {e.response.text[:500]}")
            return None, None, payload_preview_path
    else:
        # All retries exhausted
        print(f"API call failed after {MAX_RETRIES} attempts (all retries exhausted)")
        return None, None, payload_preview_path

    if output_text is None:
        return None, None, payload_preview_path

    # Debug logging
    print(f"\n[Detection API Call]")
    print(f"  System prompt length: {len(system_prompt)} chars")
    print(f"  User prompt length: {len(user_prompt)} chars")
    if task_context:
        print(f"  Task context provided: {len(task_context)} chars")
        print(f"  Task context preview: {task_context[:200]}...")
    else:
        print(f"  No task context provided")
    if attempt > 0:
        print(f"  ✅ Succeeded after {attempt + 1} attempt(s)")
    print(f"  API raw response (first 500 chars): {output_text[:500]}")

    # Parse bbox and points (will be converted from 0-1000 to 0-1)
    bboxes, points = _parse_bbox_and_points_from_text(output_text, image_width, image_height)

    # Try to extract model-reported count for validation
    try:
        parsed_for_count = json.loads(re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', output_text, flags=re.DOTALL | re.IGNORECASE).strip())
        model_count = parsed_for_count.get("count", None)
    except Exception:
        model_count = None

    if bboxes is None or points is None:
        print(f"Warning: Failed to parse bboxes or points from API response.")
        print(f"Full API response: {output_text}")
        print(f"Parsed bboxes: {bboxes}, Parsed points: {points}")
        print(f"Image dimensions: width={image_width}, height={image_height}")
    else:
        print(f"Successfully parsed: {len(bboxes)} bbox(es), {len(points)} point(s)")
        if model_count is not None:
            print(f"  Model self-reported count: {model_count}")
            if model_count != len(bboxes):
                print(f"  ⚠️ Count mismatch: model said {model_count} but output {len(bboxes)} bbox(es)")
        if bboxes:
            print(f"  First bbox (normalized 0-1): {bboxes[0]}")
        if points:
            print(f"  First point (normalized 0-1): {points[0]}")

    return bboxes, points, payload_preview_path


def _verify_bbox_point_placement(
    annotated_image: Image.Image,
    original_image: Image.Image,
    object_part: str,
    bboxes: List[List[float]],
    points: List[List[float]],
) -> Tuple[bool, str]:
    """Ask Qwen to verify whether the annotated bboxes and points are correctly placed.

    Sends the annotated image (with drawn bboxes + points) to Qwen and asks it
    to check whether each point is on the correct target part surface.

    Returns:
        (is_correct, feedback): is_correct=True means the annotation is OK.
            feedback is a text description of what's wrong (empty if correct).
    """
    import time as _time
    import random as _random

    verify_prompt = DETECTION_VERIFY_PROMPT_TEMPLATE.format(
        object_part=object_part,
        num_bboxes=len(bboxes),
        num_points=len(points),
    )

    annotated_b64 = _image_to_base64(annotated_image)

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{annotated_b64}"}},
                {"type": "text", "text": verify_prompt},
            ]},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        if resp.status_code == 429 or resp.status_code >= 500:
            print(f"  [Verify] API returned {resp.status_code}, skipping verification")
            return True, ""
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        print(f"  [Verify] Response: {text[:300]}")

        # Parse JSON from response
        clean = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL | re.IGNORECASE).strip()
        parsed = json.loads(clean)
        is_correct = bool(parsed.get("correct", True))
        feedback = str(parsed.get("feedback", ""))
        return is_correct, feedback

    except Exception as e:
        print(f"  [Verify] Verification call failed: {e}, assuming correct")
        return True, ""


def spotter_detection(
    image: Image.Image,
    object_part: str,
    output_dir: str,
    save_rex_visualization: Optional[str] = None,
    task_context: Optional[str] = None,
    reference_images: Optional[List[Dict[str, str]]] = None,
    max_verify_retries: int = 1,
    image_path: Optional[str] = None,
) -> Tuple[Optional[List[List[float]]], Optional[List[List[float]]], Optional[str]]:
    image_width, image_height = image.size
    cumulative_feedback = ""

    for attempt in range(1 + max_verify_retries):
        # Build augmented task_context with previous feedback if retrying
        ctx = task_context
        if cumulative_feedback:
            correction = (
                f"\n\n=== CORRECTION from previous attempt ===\n"
                f"Your previous annotation was reviewed and found INCORRECT:\n"
                f"{cumulative_feedback}\n"
                f"Please re-annotate with corrected point placement. "
                f"Pay extra attention to placing each point ON the actual solid surface of the target part.\n"
                f"=== End of correction ==="
            )
            ctx = f"{ctx}\n{correction}" if ctx else correction

        bboxes, points, payload_preview_path = _qwen3vl_detection_and_pointing(
            image,
            object_part,
            task_context=ctx,
            output_dir=output_dir,
            reference_images=reference_images if attempt == 0 else None,
            image_path=image_path,
        )

        if bboxes is None or points is None:
            print(f"Error: No bounding box or points received from {MODEL_NAME} API.")
            return None, None, payload_preview_path

        # Convert to pixel coordinates for visualization
        bboxes_pixel = []
        for bbox in bboxes:
            if len(bbox) == 4:
                bboxes_pixel.append([
                    float(bbox[0]) * image_width, float(bbox[1]) * image_height,
                    float(bbox[2]) * image_width, float(bbox[3]) * image_height,
                ])
        points_pixel = []
        for point in points:
            if len(point) == 2:
                points_pixel.append([
                    float(point[0]) * image_width, float(point[1]) * image_height,
                ])

        # Save visualization
        vis_path = save_rex_visualization or f"{output_dir}/image_with_rex_grouding.png"
        save_image_with_points_and_box(image, points_pixel, bboxes_pixel, save_path=vis_path)
        print(f"Detection visualization saved to {vis_path}")

        # No more retries available — accept whatever we have
        if attempt >= max_verify_retries:
            break

        # Verification: send annotated image back for review
        try:
            annotated_img = Image.open(vis_path).convert("RGB")
        except Exception:
            break

        print(f"  [Verify] Checking bbox+point quality (attempt {attempt + 1}) ...")
        is_correct, feedback = _verify_bbox_point_placement(
            annotated_img, image, object_part, bboxes, points,
        )

        if is_correct:
            print(f"  [Verify] ✅ Annotations verified as correct")
            break
        else:
            print(f"  [Verify] ❌ Issues found: {feedback[:200]}")
            cumulative_feedback += f"Attempt {attempt + 1}: {feedback}\n"
            # Loop will retry with corrective feedback

    return bboxes, points, payload_preview_path


def spotter_segmentation(
    sam2_model: "SAM2ImagePredictor",
    image: Image.Image,
    bboxes: Optional[List[List[float]]],
    points: Optional[List[List[float]]],
    output_dir: str,
    device: torch.device,
    save_mask_image: Optional[str] = None,
    save_visualization: Optional[str] = None,
) -> Optional[np.ndarray]:
    if bboxes is None or points is None:
        print("Skipping SAM2 inference due to missing bboxes or points.")
        return None

    print("Processing with SAM2...")
    try:
        with _SAM2_LOCK, torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            image_tensor = np.array(image.convert("RGB"))
            image_height, image_width = image_tensor.shape[:2]
            sam2_model.set_image(image_tensor)

            mask_all = np.zeros((image_tensor.shape[0], image_tensor.shape[1]), dtype=bool)

            current_points = []
            if points:
                for point in points:
                    if len(point) == 2:
                        x_pixel = float(point[0]) * image_width
                        y_pixel = float(point[1]) * image_height
                        current_points.append([x_pixel, y_pixel])
            
            current_bboxes = []
            if bboxes:
                for bbox in bboxes:
                    if len(bbox) == 4:
                        current_bboxes.append([
                            float(bbox[0]) * image_width,
                            float(bbox[1]) * image_height,
                            float(bbox[2]) * image_width,
                            float(bbox[3]) * image_height,
                        ])

            paired: list = []
            n_bbox = len(current_bboxes)
            n_pts = len(current_points)

            if n_bbox == n_pts and n_pts > 0:
                paired = list(zip(current_bboxes, current_points))
            elif n_bbox > 0 and n_pts == 0:
                for bb in current_bboxes:
                    cx = (bb[0] + bb[2]) / 2.0
                    cy = (bb[1] + bb[3]) / 2.0
                    paired.append((bb, [cx, cy]))
            elif n_pts > 0 and n_bbox == 0:
                for pt in current_points:
                    paired.append((None, pt))
            elif n_bbox > 0 and n_pts > 0:
                shared = min(n_bbox, n_pts)
                for i in range(shared):
                    paired.append((current_bboxes[i], current_points[i]))
                for i in range(shared, n_bbox):
                    bb = current_bboxes[i]
                    cx = (bb[0] + bb[2]) / 2.0
                    cy = (bb[1] + bb[3]) / 2.0
                    paired.append((bb, [cx, cy]))
                for i in range(shared, n_pts):
                    paired.append((None, current_points[i]))

            print(f"  SAM2: Processing {len(paired)} target(s) (bboxes={n_bbox}, points={n_pts})")
            for bbox_item, point in paired:
                masks, scores, _ = sam2_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox_item if bbox_item is not None else None,
                    multimask_output=True,
                )
                scores_t = torch.tensor(scores)
                best_mask_idx = torch.argmax(scores_t)
                best_mask = masks[best_mask_idx].squeeze()
                mask_all = np.logical_or(mask_all, best_mask)

        mask = mask_all

        mask_image_array = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image_array)
        mask_save_path = save_mask_image or f"{output_dir}/mask.png"
        os.makedirs(os.path.dirname(os.path.abspath(mask_save_path)) or ".", exist_ok=True)
        mask_image.save(mask_save_path)
        print(f"Raw mask saved to {mask_save_path}")

        vis_save_path = save_visualization or f"{output_dir}/image_with_mask.png"
        save_image_with_mask(mask, image, save_path=vis_save_path, borders=False)
        print(f"Masked image saved to {vis_save_path}")

        return mask
    except Exception as e:
        print(f"Error running SAM2: {e}")
        import traceback
        traceback.print_exc()
        return None


def spotter_segmentation_sam3(
    sam3_assets: Any,  # (processor, model) tuple from _get_sam3()
    image: Image.Image,
    bboxes: Optional[List[List[float]]],
    points: Optional[List[List[float]]],
    output_dir: str,
    device: torch.device,
    save_mask_image: Optional[str] = None,
    save_visualization: Optional[str] = None,
) -> Optional[np.ndarray]:
    if bboxes is None and points is None:
        print("Skipping SAM3 inference due to missing bboxes and points.")
        return None

    print("Processing with SAM3...")
    processor, model = sam3_assets
    image_width, image_height = image.size

    # Convert normalized [0,1] → pixel coordinates
    def _norm_to_pixel_bbox(b):
        return [
            float(b[0]) * image_width,
            float(b[1]) * image_height,
            float(b[2]) * image_width,
            float(b[3]) * image_height,
        ]

    def _norm_to_pixel_point(p):
        return [float(p[0]) * image_width, float(p[1]) * image_height]

    pixel_bboxes = [_norm_to_pixel_bbox(b) for b in (bboxes or []) if len(b) == 4]
    pixel_points = [_norm_to_pixel_point(p) for p in (points or []) if len(p) == 2]

    # Build paired (bbox, point) list same as SAM2 path
    paired: list = []
    n_bbox, n_pts = len(pixel_bboxes), len(pixel_points)
    if n_bbox == n_pts and n_pts > 0:
        paired = list(zip(pixel_bboxes, pixel_points))
    elif n_bbox > 0 and n_pts == 0:
        for bb in pixel_bboxes:
            cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            paired.append((bb, [cx, cy]))
    elif n_pts > 0 and n_bbox == 0:
        for pt in pixel_points:
            paired.append((None, pt))
    else:
        shared = min(n_bbox, n_pts)
        for i in range(shared):
            paired.append((pixel_bboxes[i], pixel_points[i]))
        for i in range(shared, n_bbox):
            bb = pixel_bboxes[i]
            paired.append((bb, [(bb[0]+bb[2])/2, (bb[1]+bb[3])/2]))
        for i in range(shared, n_pts):
            paired.append((None, pixel_points[i]))

    print(f"  SAM3: Processing {len(paired)} target(s) (bboxes={n_bbox}, points={n_pts})")

    try:
        mask_all = np.zeros((image_height, image_width), dtype=bool)

        with _SAM3_LOCK:
            for bbox_item, point in paired:
                try:
                    if bbox_item is not None:
                        inputs = processor(
                            images=image,
                            input_boxes=[[bbox_item]],
                            input_boxes_labels=[[1]],
                            return_tensors="pt",
                        ).to(device)
                    else:
                        inputs = processor(
                            images=image,
                            input_points=[[[point[0], point[1]]]],
                            input_labels=[[1]],
                            return_tensors="pt",
                        ).to(device)

                    with torch.no_grad():
                        if device.type == "cuda":
                            with torch.autocast(device.type, dtype=torch.bfloat16):
                                outputs = model(**inputs)
                        else:
                            outputs = model(**inputs)

                    orig_sizes = inputs.get("original_sizes")
                    if orig_sizes is None:
                        orig_sizes = torch.tensor([[image_height, image_width]])
                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=0.3,
                        mask_threshold=0.5,
                        target_sizes=orig_sizes.tolist(),
                    )[0]

                    if results and len(results.get("masks", [])) > 0:
                        # Pick highest-score mask
                        scores = results.get("scores", [])
                        masks = results["masks"]
                        if len(scores) > 0:
                            if isinstance(scores, torch.Tensor):
                                best_idx = int(scores.cpu().argmax())
                            else:
                                scores_cpu = [s.cpu().item() if isinstance(s, torch.Tensor) else float(s) for s in scores]
                                best_idx = int(torch.tensor(scores_cpu).argmax())
                        else:
                            best_idx = 0
                        _m = masks[best_idx]
                        if isinstance(_m, torch.Tensor):
                            _m = _m.cpu()
                        m = np.array(_m)
                        if m.shape == (image_height, image_width):
                            mask_all = np.logical_or(mask_all, m.astype(bool))
                            print(f"  SAM3: mask merged, coverage={m.sum()/(image_height*image_width):.3f}")
                        else:
                            print(f"  SAM3: unexpected mask shape {m.shape}, skipping")
                    else:
                        # Fallback: draw a small ellipse around the point as a coarse mask
                        print(f"  SAM3: no mask returned for this prompt, using bbox region as fallback")
                        if bbox_item is not None:
                            x1, y1, x2, y2 = [int(v) for v in bbox_item]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(image_width, x2), min(image_height, y2)
                            mask_all[y1:y2, x1:x2] = True

                except Exception as e_inner:
                    print(f"  SAM3: error on one target: {e_inner}")
                    continue

        if not mask_all.any():
            print("  SAM3: all targets produced empty masks")
            return None

        # Save outputs (same as SAM2 path)
        mask_image_array = (mask_all * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image_array)
        mask_save_path = save_mask_image or f"{output_dir}/mask.png"
        os.makedirs(os.path.dirname(os.path.abspath(mask_save_path)) or ".", exist_ok=True)
        mask_pil.save(mask_save_path)
        print(f"Raw mask (SAM3) saved to {mask_save_path}")

        vis_save_path = save_visualization or f"{output_dir}/image_with_mask.png"
        save_image_with_mask(mask_all, image, save_path=vis_save_path, borders=False)
        print(f"Masked image (SAM3) saved to {vis_save_path}")

        return mask_all

    except Exception as e:
        print(f"Error running SAM3: {e}")
        import traceback
        traceback.print_exc()
        return None


def _format_dynamic_params_for_prompt(dynamic_params: Optional[Dict[str, Any]]) -> str:
    if not dynamic_params:
        return ""

    lines = ["\nAdditional Strategy Guidance from the Decision Model:"]
    for key, value in dynamic_params.items():
        #  snake_case 
        readable_key = key.replace("_", " ").title()
        lines.append(f"- {readable_key}: {value}")

    return "\n".join(lines)


def run_detection_skill(
    image_path: str,
    task: str,
    object_name: Optional[str] = None,
    sam2_model_path: str = "facebook/sam2.1-hiera-large",
    sam3_model_path: str = "facebook/sam3",
    save_mask_image: Optional[str] = None,
    save_visualization: Optional[str] = None,
    save_rex_visualization: Optional[str] = None,
    task_context: Optional[str] = None,
    dynamic_params: Optional[Dict[str, Any]] = None,
    reference_images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    sam_backend = os.getenv("SAM_BACKEND", "sam2").lower().strip()

    _configure_torch_for_inference()

    image = Image.open(image_path).convert("RGB")

    object_part = f"the part of the {object_name} that can satisfy: {task}" if object_name else task

    output_dir = os.path.dirname(os.path.abspath(save_visualization or save_mask_image or save_rex_visualization or image_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Detection] SAM backend: {sam_backend}")

    dynamic_prompt = _format_dynamic_params_for_prompt(dynamic_params)
    if dynamic_prompt:
        task_context = f"{task_context}\n{dynamic_prompt}" if task_context else dynamic_prompt
        print(f"[Detection] Dynamic params injected: {list(dynamic_params.keys())}")

    bboxes, points, payload_preview_path = spotter_detection(
        image, object_part, output_dir=output_dir, save_rex_visualization=save_rex_visualization,
        task_context=task_context, reference_images=reference_images,
        max_verify_retries=0,
        image_path=image_path,
    )
    if bboxes is None or points is None:
        return {
            "error": f"No bounding box or points received from {MODEL_NAME} API.",
            "stage": "qwen3vl_api",
            "object_part": object_part,
            "task_context_used": task_context,
            "dynamic_params_used": dynamic_params or {},
            "qwen_payload_preview_path": payload_preview_path,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sam_backend == "sam3":
        sam3_assets = _get_sam3(sam3_model_path)
        mask = spotter_segmentation_sam3(
            sam3_assets,
            image,
            bboxes,
            points,
            output_dir=output_dir,
            device=device,
            save_mask_image=save_mask_image,
            save_visualization=save_visualization,
        )
        seg_stage = "sam3"
    else:
        sam2 = _get_sam2(sam2_model_path)
        mask = spotter_segmentation(
            sam2,
            image,
            bboxes,
            points,
            output_dir=output_dir,
            device=device,
            save_mask_image=save_mask_image,
            save_visualization=save_visualization,
        )
        seg_stage = "sam2"

    if mask is None:
        return {"error": f"{seg_stage.upper()} segmentation failed.", "stage": seg_stage, "object_part": object_part}

    return {
        "mask_image_path": save_mask_image or f"{output_dir}/mask.png",
        "visualization_path": save_visualization or f"{output_dir}/image_with_mask.png",
        "rex_visualization_path": save_rex_visualization or f"{output_dir}/image_with_rex_grouding.png",
        "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "object_part": object_part,
        "bboxes": bboxes,
        "points": points,
        "task_context_used": task_context,
        "dynamic_params_used": dynamic_params or {},
        "qwen_payload_preview_path": payload_preview_path,
        "sam_backend": seg_stage,
    }


def main():
    parser = argparse.ArgumentParser(description="detection skill: Qwen3-vl-235b-a22b-instruct API grounding + SAM-2 segmentation")
    parser.add_argument("--image_path", type=str, required=True, help="Absolute path to RGB image")
    parser.add_argument("--task", type=str, required=True, help="Affordance task description")
    parser.add_argument("--object_name", type=str, default=None, help="Optional object category name")
    parser.add_argument("--sam2_model_path", type=str, default="facebook/sam2.1-hiera-large", help="SAM2 model path")
    parser.add_argument("--output_json", type=str, default=None, help="If set, save output JSON to this path")
    parser.add_argument("--output_mask", type=str, default=None, help="If set, save mask image to this path")
    parser.add_argument("--output_vis", type=str, default=None, help="If set, save visualization image to this path")
    parser.add_argument("--output_rex_vis", type=str, default=None, help="If set, save detection grounding visualization to this path")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    references_dir = os.path.join(os.path.dirname(script_dir), "references")
    os.makedirs(references_dir, exist_ok=True)

    out_mask = args.output_mask or os.path.join(references_dir, "mask.png")
    out_vis = args.output_vis or os.path.join(references_dir, "image_with_mask.png")
    out_rex_vis = args.output_rex_vis or os.path.join(references_dir, "image_with_rex_grouding.png")

    result = run_detection_skill(
        image_path=args.image_path,
        task=args.task,
        object_name=args.object_name,
        sam2_model_path=args.sam2_model_path,
        save_mask_image=out_mask,
        save_visualization=out_vis,
        save_rex_visualization=out_rex_vis,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()
