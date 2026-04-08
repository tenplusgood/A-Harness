"""
Affordance Agent - 
 API， function calling， rex-omni  sam-2 
"""

import json
import os
import base64
import io
import shutil
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from api.client import (
    APIClient,
    get_default_api_config,
    normalize_api_url,
    is_qwen35_model,
    get_qwen35_api_config,
)
from skills.registry import SkillRegistry
from prompts.system import build_system_prompt
from memory import MemoryManager, MemoryEntry, ExperiencePool


class AffordanceAgent:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
        output_dir: str = "output",
        tool_filter: Optional[set] = None,
        detection_backend: str = "qwen3vl_api",
        enable_memory: bool = True,
        memory_max_size: int = 80,
        memory_eviction_strategy: str = "model_decision",
        memory_eviction_model: Optional[str] = None,
        memory_retrieval_top_k: int = 20,
        memory_dedupe_top_k: int = 5,
        memory_duplicate_threshold: float = 0.6,
        memory_persist_dir: Optional[str] = None,
        memory_token_budget: int = 3000,
        max_tool_calls_per_tool: int = 8,
        detection_max_retries: int = 3,
        reliability_retry_limit: int = 3,
        min_reliable_sources_before_detection: int = 1,
        skill_prompt_mode: str = "on_demand",
        dump_request_payload_preview: bool = True,
        commonsense_templates_dir: Optional[str] = None,
        enable_commonsense_templates: bool = True,
        fixed_skill_chain: bool = False,
    ):

        if api_key is None or api_base_url is None:
            default_key, default_url = get_default_api_config()
            if api_key is None:
                api_key = default_key
            if api_base_url is None:
                api_base_url = default_url
        
        if api_base_url:
            api_base_url = normalize_api_url(api_base_url)
        
        if is_qwen35_model(model_name):
            print(f"[Agent] Detected Qwen3.5 model '{model_name}', switching to PAI-EAS config...")
            q_key, q_url, q_model = get_qwen35_api_config()
            if q_key and q_url:
                api_key = q_key
                api_base_url = normalize_api_url(q_url)
                if q_model:
                    self.model_name = q_model 
                    print(f"[Agent] Model name normalized to: {self.model_name}")
            else:
                print("[Agent] Warning: Qwen3.5 config not found in config.py, using provided/default credentials.")

        self.api_client = APIClient(api_key=api_key, base_url=api_base_url,
                                     max_retries=5, retry_delay=1.5)
        self.model_name = model_name
        self.output_dir = output_dir
        self.skill_registry = SkillRegistry(output_dir=self.output_dir, tool_filter=tool_filter, detection_backend=detection_backend)
        
        os.makedirs(self.output_dir, exist_ok=True)
        

        self.enable_memory = enable_memory
        if self.enable_memory:
            if memory_persist_dir is None:
                memory_persist_dir = os.path.join(self.output_dir, "memory")
            self.memory_manager = MemoryManager(
                max_size=memory_max_size,
                eviction_strategy=memory_eviction_strategy,
                eviction_model=(memory_eviction_model or self.model_name),
                retrieval_top_k=memory_retrieval_top_k,
                enqueue_dedupe_top_k=memory_dedupe_top_k,
                duplicate_similarity_threshold=memory_duplicate_threshold,
                persist_dir=memory_persist_dir,
                api_caller=self.api_client.call,
                context_token_budget=memory_token_budget,
                commonsense_templates_dir=commonsense_templates_dir,
                enable_commonsense_templates=enable_commonsense_templates,
            )
            self.skill_registry.set_memory_manager(self.memory_manager)
        else:
            self.memory_manager = None
        
        self._tool_results_storage: Dict[str, Dict[str, Any]] = {}
        self.max_tool_calls_per_tool = max(1, int(max_tool_calls_per_tool))
        raw_detection_max_retries = int(detection_max_retries)
        self.detection_max_retries = max(0, min(3, raw_detection_max_retries))
        if raw_detection_max_retries != self.detection_max_retries:
            print(
                f"[Agent] detection_max_retries={raw_detection_max_retries} is capped to "
                f"{self.detection_max_retries} based on evaluation findings."
            )
        self.reliability_retry_limit = max(0, int(reliability_retry_limit))
        self.min_reliable_sources_before_detection = max(0, int(min_reliable_sources_before_detection))
        self.skill_prompt_mode = skill_prompt_mode
        self.dump_request_payload_preview = bool(dump_request_payload_preview)
        self.payload_preview_dir = os.path.join(self.output_dir, "payload_previews")
        self.fixed_skill_chain = bool(fixed_skill_chain)
        if self.dump_request_payload_preview:
            os.makedirs(self.payload_preview_dir, exist_ok=True)
    
    def clear_memory(self, clear_experience: bool = False) -> None:
        if self.enable_memory and self.memory_manager:
            self.memory_manager.clear_memory(
                clear_outer=True,
                clear_experience=clear_experience,
            )
        else:
            print("[Agent] Memory is not enabled, nothing to clear.")
    
    def flush_experience_pool(self) -> None:
        if self.enable_memory and self.memory_manager:
            self.memory_manager.experience_pool.distill_insights()
            print("[Agent] Experience pool distillation complete.")
        else:
            print("[Agent] Memory is not enabled.")
    
    def _prepare_tool_result_for_api(self, skill_result: Dict[str, Any], function_name: str) -> Dict[str, Any]:

        api_result = skill_result.copy()
        
        if function_name == "sam2" and "mask" in api_result:
            mask = api_result.get("mask")
            mask_shape = api_result.get("mask_shape", [])
            
            if isinstance(mask, list) and len(mask) > 0:
                estimated_size = len(str(mask))
                max_size = 1000000  
                
                if estimated_size > max_size:
                    api_result.pop("mask", None)
                    api_result["mask_summary"] = {
                        "shape": mask_shape,
                        "total_pixels": mask_shape[0] * mask_shape[1] if len(mask_shape) == 2 else 0,
                        "note": "Full mask data removed to avoid API size limits. Use mask_image_path or visualization_path for the actual mask."
                    }
        
        return api_result
    
    def _encode_image(self, image_path: str) -> str:
        max_edge = int(os.getenv("AFFORDANCE_MAX_IMAGE_EDGE", "1024"))
        jpeg_quality = int(os.getenv("AFFORDANCE_JPEG_QUALITY", "90"))
        max_b64_len = int(os.getenv("AFFORDANCE_MAX_B64_LEN", "240000"))

        #  JPEG （ PNG）
        from PIL import Image

        orig = Image.open(image_path).convert("RGB")
        w0, h0 = orig.size

        # Adaptive compression loop to satisfy strict request-size limits.
        # We progressively shrink the max_edge until base64 length fits.
        current_max_edge = max_edge
        last_b64: Optional[str] = None
        for _ in range(8):
            img = orig
            if max(w0, h0) > 0:
                scale = min(1.0, float(current_max_edge) / float(max(w0, h0)))
            else:
                scale = 1.0
            if scale < 1.0:
                new_w = max(1, int(round(w0 * scale)))
                new_h = max(1, int(round(h0 * scale)))
                img = orig.resize((new_w, new_h), Image.BICUBIC)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            last_b64 = b64
            if len(b64) <= max_b64_len:
                return b64

            # Too large -> shrink further
            current_max_edge = max(64, int(current_max_edge * 0.75))

        # If we still exceed the limit, return the smallest one we could make.
        return last_b64 or ""
    
    def _get_image_mime_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    def _extract_tool_images(self, skill_result: Dict[str, Any], function_name: str) -> List[Dict[str, str]]:
        """
        Extract image paths from tool results for visual feedback to the decision model.
        
        Returns:
            List of dicts with 'path' and 'label' keys
        """
        images = []
        
        if function_name == "dreamer":
            for img_path in skill_result.get("generated_image_paths", []):
                if img_path and os.path.exists(img_path):
                    images.append({
                        "path": img_path,
                        "label": (
                            "This image shows the object interacting with a person "
                            "in relation to the given affordance type, for your reference."
                        ),
                    })
        
        elif function_name == "zoom_in":
            zoomed_path = skill_result.get("zoomed_image_path", "")
            if zoomed_path and os.path.exists(zoomed_path):
                images.append({
                    "path": zoomed_path,
                    "label": (
                        "This is a zoomed-in close-up of a specific region "
                        "in the target scene, for detailed inspection."
                    ),
                })
        
        return images

    def _summarize_tool_result(self, skill_result: Dict[str, Any], function_name: str) -> str:
        """
        Generate a concise, structured summary of a tool's result
        for the decision model to reason about.
        
        Args:
            skill_result: Raw result dict from the tool
            function_name: Name of the tool
            
        Returns:
            A concise summary string
        """
        if "error" in skill_result:
            return f"[{function_name}] ❌ Error: {skill_result['error']}"
        
        lines = [f"[{function_name}] Results:"]
        
        if function_name == "dreamer":
            num_targets = skill_result.get("num_targets", 1)
            lines.append(f"  • Detected {num_targets} target(s)")

            for idx, p in enumerate(skill_result.get("dreamer_prompts", [])):
                lines.append(f"  • Interaction prompt [{idx}]: {p[:150]}")

            analysis = skill_result.get("interaction_analysis", "")
            if analysis:
                lines.append(f"  • Interaction analysis:")
                for line in analysis.strip().split('\n'):
                    line = line.strip()
                    if line:
                        lines.append(f"    {line}")

            gen_paths = [p for p in skill_result.get("generated_image_paths", []) if p]
            lines.append(f"  • Generated {len(gen_paths)} interaction image(s) (shown below)")
            for gp in gen_paths:
                lines.append(f"  • Image path: {gp}")
        
        elif function_name == "web_search":
            aff = skill_result.get("affordance_name", "")
            part = skill_result.get("part_name", "")
            obj = skill_result.get("object_name", "")
            reasoning = skill_result.get("reasoning", "")
            
            if aff:
                lines.append(f"  • Affordance: {aff}")
            if part:
                lines.append(f"  • Target part: {part}")
            if obj:
                lines.append(f"  • Object: {obj}")
            if reasoning:
                lines.append(f"  • Reasoning: {reasoning[:300]}")
            
            n_results = len(skill_result.get("search_results", []))
            n_urls = len(skill_result.get("crawled_urls", []))
            lines.append(f"  • Sources: {n_results} search results, {n_urls} pages crawled")
        
        elif function_name == "zoom_in":
            crop = skill_result.get("crop_region_pixel", [])
            orig = skill_result.get("original_size", [])
            cropped = skill_result.get("cropped_size", [])
            if crop:
                lines.append(f"  • Cropped region (pixels): ({crop[0]},{crop[1]})-({crop[2]},{crop[3]})")
            if orig:
                lines.append(f"  • Original image: {orig[0]}x{orig[1]}")
            if cropped:
                lines.append(f"  • Zoomed view: {cropped[0]}x{cropped[1]}")
            zoomed_path = skill_result.get("zoomed_image_path", "")
            if zoomed_path:
                lines.append(f"  • Zoomed image path: {zoomed_path}")
            lines.append(f"  • Zoomed image shown below for visual analysis")
            lines.append(f"  • Note: this is a cropped view; detection operates on the original image")
        
        elif function_name == "detection":
            stage = skill_result.get("stage", "")
            if stage:
                lines.append(f"  • Stage: {stage}")
            obj_part = skill_result.get("object_part", "")
            if obj_part:
                lines.append(f"  • Detected part: {obj_part}")
            bboxes = skill_result.get("bboxes")
            points = skill_result.get("points")
            if bboxes is not None:
                lines.append(f"  • Bboxes (normalized [0,1]): {self._fmt_coords(bboxes)}")
            if points is not None:
                lines.append(f"  • Points (normalized [0,1]): {self._fmt_coords(points)}")
            mask_path = skill_result.get("mask_image_path", "")
            if mask_path:
                lines.append(f"  • Mask saved: {os.path.basename(mask_path)}")
        
        elif function_name == "load_skill_doc":
            skill_name = skill_result.get("skill_name", "")
            skill_dir = skill_result.get("skill_dir", "")
            truncated = bool(skill_result.get("truncated", False))
            content = str(skill_result.get("content", "") or "")
            lines.append(f"  • Loaded doc for: {skill_name or 'unknown'} ({skill_dir or 'n/a'})")
            if content:
                lines.append(f"  • Doc length: {len(content)} chars")
                lines.append(f"  • Preview: {content[:220].replace(chr(10), ' ')}")
            lines.append(f"  • Truncated: {truncated}")
        
        return "\n".join(lines)

    def _extract_key_insights_from_tool(self, skill_result: Dict[str, Any], function_name: str) -> Optional[str]:
        """
        Extract structured key insights from a helper tool's result.
        
        These insights will be automatically injected into detection's task_context
        to ensure Qwen3-VL receives all information from prior skill calls,
        regardless of how well the decision LLM synthesizes them.
        
        Returns:
            A compact insight string, or None if nothing useful.
        """
        if "error" in skill_result:
            return None
        
        lines = []
        
        if function_name == "web_search":
            aff = skill_result.get("affordance_name", "")
            part = skill_result.get("part_name", "")
            obj = skill_result.get("object_name", "")
            reasoning = skill_result.get("reasoning", "")
            
            lines.append("[web_search findings]")
            if aff:
                lines.append(f"  Affordance: {aff}")
            if part:
                lines.append(f"  Target part: {part}")
            if obj:
                lines.append(f"  Object: {obj}")
            if reasoning:
                lines.append(f"  Reasoning: {reasoning[:400]}")
        
        elif function_name == "dreamer":
            analysis = skill_result.get("interaction_analysis", "")
            num_targets = skill_result.get("num_targets", 1)
            
            lines.append("[dreamer findings]")
            lines.append(f"  Detected {num_targets} target(s)")
            if analysis:
                lines.append(f"  Interaction analysis: {analysis[:500]}")
            
            # Include dreamer prompts as they describe the imagined interaction
            for idx, p in enumerate(skill_result.get("dreamer_prompts", [])[:2]):
                lines.append(f"  Interaction scenario [{idx}]: {p[:200]}")
        
        elif function_name == "zoom_in":
            crop = skill_result.get("crop_region_pixel", [])
            orig = skill_result.get("original_size", [])
            zoom_purpose = skill_result.get("zoom_purpose", "")
            
            lines.append("[zoom_in findings]")
            if crop:
                lines.append(f"  Examined region (pixels): ({crop[0]},{crop[1]})-({crop[2]},{crop[3]})")
            if orig:
                lines.append(f"  In image {orig[0]}x{orig[1]}")
            if zoom_purpose:
                lines.append(f"  Purpose: {zoom_purpose}")
            lines.append("  Note: zoomed view was examined for fine details")
        
        elif function_name == "load_skill_doc":
            skill_name = skill_result.get("skill_name", "")
            content = str(skill_result.get("content", "") or "")
            lines.append("[load_skill_doc findings]")
            if skill_name:
                lines.append(f"  Skill: {skill_name}")
            # Keep only compact guidance-like lines to avoid huge context injection.
            key_lines = []
            for raw in content.splitlines():
                s = raw.strip()
                if not s:
                    continue
                if s.lower().startswith(("when_to_use", "when_not_to_use", "inputs", "outputs")):
                    key_lines.append(s)
                elif s.startswith(("-", "|")) and len(key_lines) < 6:
                    key_lines.append(s)
                if len(key_lines) >= 8:
                    break
            if key_lines:
                lines.append("  Key guidance:")
                lines.extend([f"    {x}" for x in key_lines])
        
        if len(lines) <= 1:
            return None
        return "\n".join(lines)

    def _build_skill_result_for_qwen(
        self, skill_result: Dict[str, Any], function_name: str
    ) -> Optional[str]:
        """Build a concise skill-result line for Qwen's [B] section.

        Image-based skills (dreamer, zoom_in) deliver their results via
        reference_images with descriptive labels.  Here we only add a short
        text note so Qwen knows what those images represent.
        Text-based skills (web_search) include their actual findings.
        """
        if "error" in skill_result:
            return None

        if function_name == "dreamer":
            analysis = skill_result.get("interaction_analysis", "")
            n = skill_result.get("num_targets", 1)
            parts = [f"[dreamer] {n} interaction image(s) provided as reference."]
            if analysis:
                parts.append(f"Interaction analysis: {analysis[:400]}")
            return " ".join(parts)

        if function_name == "zoom_in":
            crop = skill_result.get("crop_region_pixel", [])
            orig = skill_result.get("original_size", [])
            line = "[zoom_in] A zoomed-in close-up is provided as reference."
            if crop and orig:
                line += (
                    f" Region ({crop[0]},{crop[1]})-({crop[2]},{crop[3]})"
                    f" in {orig[0]}x{orig[1]} image."
                )
            return line

        if function_name == "web_search":
            aff = skill_result.get("affordance_name", "")
            part = skill_result.get("part_name", "")
            obj = skill_result.get("object_name", "")
            reasoning = skill_result.get("reasoning", "")
            pieces = ["[web_search]"]
            if obj:
                pieces.append(f"Object: {obj}.")
            if part:
                pieces.append(f"Target part: {part}.")
            if aff:
                pieces.append(f"Affordance: {aff}.")
            if reasoning:
                pieces.append(f"Findings: {reasoning[:400]}")
            return " ".join(pieces) if len(pieces) > 1 else None

        return None

    def _build_reflection_message(
        self,
        tool_summaries: List[str],
        tool_images: List[Dict[str, str]],
        detection_called: bool,
        reliability_report: Optional[Dict[str, Any]] = None,
        tool_call_counts: Optional[Dict[str, int]] = None,
        max_tool_calls_per_tool: Optional[int] = None,
        max_images: int = 4,
        commonsense_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a reflection message after tool calls.

        Always fires — whether detection has been called or not — so the
        decision model can reason about new evidence at every step.
        """
        if not tool_summaries and not tool_images:
            return None

        from PIL import Image

        content_parts: List[Dict[str, Any]] = []

        if commonsense_context and commonsense_context.strip():
            content_parts.append({"type": "text", "text": commonsense_context.strip()})

        summary_text = "**Tool Results Summary:**\n\n" + "\n\n".join(tool_summaries)
        content_parts.append({"type": "text", "text": summary_text})

        is_glm = "glm" in self.model_name.lower()
        if not is_glm:
            images_added = 0
            for img_info in tool_images[:max_images]:
                img_path = img_info["path"]
                label = img_info["label"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size
                    max_side = 768
                    if max(w, h) > max_side:
                        scale = max_side / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    content_parts.append({"type": "text", "text": label})
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                    images_added += 1
                except Exception as e:
                    print(f"[Warning] Failed to embed tool image {img_path}: {e}")
        else:
            images_added = 0 # GLM skips images

        reliability_block = ""
        if reliability_report:
            reliability_block = (
                "\n\n### Reliability Gate (agent-side check)\n"
                + reliability_report.get("report_text", "")
            )

        tool_budget_block = ""
        if tool_call_counts and max_tool_calls_per_tool:
            budget_lines = [
                f"- {k}: {v}/{max_tool_calls_per_tool}"
                for k, v in sorted(tool_call_counts.items())
            ]
            tool_budget_block = (
                "\n\n### Tool Call Budget\n"
                + ("\n".join(budget_lines) if budget_lines else "- No tools called yet")
            )

        if detection_called:
            context_note = (
                "\n\nThe above are the results from your latest tool calls. "
                "You have already called detection previously.\n\n"
                "**Reflect with the task in mind:**\n"
                "- Re-read the original task question. Does the detected "
                "region actually answer it?\n"
                "- A clean segmentation of the WRONG part is still wrong. "
                "Verify that the correct affordance part was targeted, not "
                "a different part of the same object.\n"
                "- If you have new evidence from tools, does it change your "
                "understanding of which part to detect?\n"
                "- You may call detection again with improved context, or "
                "call other tools to gather more information first."
                + reliability_block
                + tool_budget_block
            )
        else:
            context_note = (
                "\n\nThe above are the results from your tool calls. "
                "Before calling detection, think carefully about the task:\n"
                "- What SPECIFIC part is the task asking about? Not the "
                "whole object — which sub-part or affordance region?\n"
                "- Do your tool results help identify the correct target?\n\n"
                "Per-source reliability rubric:"
                "\n- High: directly supports target part/location for this exact task."
                "\n- Medium: partially relevant or indirect support."
                "\n- Low: weakly related, contradictory, or tool error."
                "\n\nIf reliability is insufficient, revise strategy and continue "
                "tool calls (you may call the same tool multiple times with "
                "different parameters)."
                + reliability_block
                + tool_budget_block
            )
        content_parts.append({"type": "text", "text": context_note})

        print(f"[Agent] Built reflection message: {len(tool_summaries)} summary(ies), "
              f"{images_added} image(s), detection_called={detection_called}")

        return {
            "role": "user",
            "content": content_parts,
        }

    def _embed_image_full_res(
        self, img_path: str, quality: int = 90
    ) -> Optional[str]:
        """Read an image at full resolution and return a base64-encoded JPEG.

        No resizing is applied so the decision model can inspect fine details
        such as point placement on small parts.
        """
        from PIL import Image as _PILImage

        try:
            img = _PILImage.open(img_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[Agent] Failed to encode image {img_path}: {e}")
            return None

    @staticmethod
    def _fmt_coords(coords: list) -> str:
        """Format nested coordinate lists with 4 decimal places."""
        if not coords:
            return "[]"
        formatted = []
        for item in coords:
            if isinstance(item, (list, tuple)):
                formatted.append("[" + ", ".join(f"{v:.4f}" for v in item) + "]")
            else:
                formatted.append(f"{item:.4f}")
        return "[" + ", ".join(formatted) + "]"

    def _build_detection_verification_message(
        self,
        tool_results: List[Dict[str, Any]],
        tool_call_counts: Dict[str, int],
        max_tool_calls_per_tool: int,
        task: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Build a verification message after detection.

        Sends the annotated image (bbox+point) AND the mask visualization
        at full resolution so the decision model can see both the point
        placement and the resulting segmentation mask.
        """
        rex_vis_path = None
        mask_vis_path = None
        det_data: Dict[str, Any] = {}
        for tr in tool_results:
            if tr.get("name") != "detection":
                continue
            try:
                payload = json.loads(tr.get("content", "{}"))
            except Exception:
                continue
            candidate = payload.get("rex_visualization_path")
            if candidate and os.path.exists(candidate):
                rex_vis_path = candidate
                det_data = payload
                mask_candidate = payload.get("visualization_path")
                if mask_candidate and os.path.exists(mask_candidate):
                    mask_vis_path = mask_candidate
                break

        if not rex_vis_path:
            return None

        content_parts: List[Dict[str, Any]] = []

        # --- Image 1: bbox + point annotation (full resolution) ---
        rex_b64 = self._embed_image_full_res(rex_vis_path)
        if not rex_b64:
            return None
        # --- Semantic context ---
        prev_bboxes = det_data.get("bboxes", [])
        prev_points = det_data.get("points", [])
        obj_part = det_data.get("object_part", "")
        data_block = (
            f"Detection data (normalized [0,1]):\n"
            f"  object_part: {obj_part}\n"
            f"  bboxes: {self._fmt_coords(prev_bboxes)}\n"
            f"  points: {self._fmt_coords(prev_points)}"
        )
        detection_budget = max_tool_calls_per_tool - tool_call_counts.get("detection", 0)

        # Task reminder comes FIRST, before any images, so the model
        # evaluates with the question fresh in mind.
        content_parts.append({
            "type": "text",
            "text": (
                f"[Detection Verification]\n"
                f"Re-read the task before inspecting the result:\n"
                f"**Task: {task}**\n\n"
                f"{data_block}\n\n"
                "Verify in this exact order — reject if ANY check fails:\n\n"
                "─── CHECK 1: SEMANTIC CORRECTNESS (most important) ───\n"
                "Ask yourself: does the detected region actually ANSWER "
                "the task question? A perfectly segmented WRONG part is "
                "still WRONG.\n"
                "- Re-read the task. What specific part/affordance is being "
                "asked about? (e.g. 'handle to pull open' ≠ 'drawer face')\n"
                "- Look at where the bbox/point lands in Image 1. Is that "
                "the correct object part, or a different part of the same "
                "object, or even a different object entirely?\n"
                "- Common mistakes: detecting the body instead of the handle, "
                "the spout instead of the knob, the blade instead of the "
                "grip, or the whole object instead of the specific part.\n"
                "- If the wrong part is detected, it MUST be rejected — no "
                "matter how clean the segmentation looks.\n\n"
                "─── CHECK 2: BBOX + POINT QUALITY ───\n"
                "- ONE bbox per discrete target region, ONE point per bbox.\n"
                "- Multiple separate targets → multiple compact bboxes, not "
                "one oversized bbox spanning gaps.\n"
                "- Each bbox must tightly enclose the FULL extent of the "
                "target affordance part.\n"
                "- Each point must land directly ON the solid surface of "
                "the target part — not on background, holes, gaps, or "
                "adjacent objects.\n"
                "- Target count must match the task: singular = 1, "
                "plural = multiple.\n\n"
                "─── CHECK 3: MASK QUALITY ───\n"
                "- The mask must cover ONLY the target affordance region "
                "and must cover ALL of it.\n"
                "- ANY spillover onto background, other objects, or "
                "non-target areas is WRONG.\n"
                "- If the mask covers a completely wrong region, the point "
                "placement or target identification is incorrect.\n\n"
                "**Accept ONLY if ALL three checks pass.**"
            ),
        })

        is_glm = "glm" in self.model_name.lower()
        if not is_glm:
            # --- Image 1: bbox + point annotation (full resolution) ---
            content_parts.append({
                "type": "text",
                "text": "[Image 1 — Bbox + Point Annotations]",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{rex_b64}"},
            })

            # --- Image 2: mask visualization (full resolution) ---
            if mask_vis_path:
                mask_b64 = self._embed_image_full_res(mask_vis_path)
                if mask_b64:
                    content_parts.append({
                        "type": "text",
                        "text": "[Image 2 — Segmentation Mask Overlay]",
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{mask_b64}"},
                    })
        else:
            content_parts.append({
                "type": "text",
                "text": "[Note: Verification images omitted for this model. Please rely on the bbox/point coordinates and your confidence in the detection tool's reliability.]"
            })

        # --- Action guidance ---
        content_parts.append({
            "type": "text",
            "text": (
                "**If the result is WRONG** (wrong target, bad bbox, bad "
                "mask), you have options:\n"
                "  1. Call `detection` again with corrective `task_context` "
                "— clearly state what was wrong and what the correct target "
                "is.\n"
                "  2. Call other tools first (zoom_in, web_search, dreamer) "
                "to better understand the target, then re-detect.\n"
                "  3. Reflect: was the target misidentified? Was the "
                "affordance part confused with the object body? Rethink "
                "the task semantics.\n\n"
                f"(Remaining detection budget: {detection_budget})"
            ),
        })

        img_count = 1 + (1 if mask_vis_path else 0)
        print(f"[Agent] Built detection verification: {img_count} image(s) at full resolution")
        return {"role": "user", "content": content_parts}

    def _evaluate_source_reliability(
        self,
        helper_tool_results: List[Dict[str, Any]],
        memory_context: str = "",
        commonsense_templates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate reliability of memory + tool + commonsense sources with explicit scoring.

        Reliability standard:
        - high (score >= 0.75): directly relevant + concrete evidence
        - medium (0.45 ~ 0.74): partially relevant, still uncertain
        - low (< 0.45): weak/noisy/failed
        """
        source_lines: List[str] = []
        source_details: List[Dict[str, Any]] = []
        reliable_sources = 0
        checked_sources = 0
        high_quality_sources = 0

        def _level_from_score(score: float) -> str:
            if score >= 0.75:
                return "high"
            if score >= 0.45:
                return "medium"
            return "low"

        def _append_source(name: str, score: float, rationale: str, action: str) -> None:
            nonlocal reliable_sources, checked_sources, high_quality_sources
            level = _level_from_score(score)
            checked_sources += 1
            if level in ("high", "medium"):
                reliable_sources += 1
            if level == "high":
                high_quality_sources += 1
            source_lines.append(
                f"- {name}: {level} ({score:.2f}) | rationale: {rationale} | next: {action}"
            )
            source_details.append(
                {
                    "source": name,
                    "level": level,
                    "score": round(score, 3),
                    "rationale": rationale,
                    "recommended_action": action,
                }
            )

        # Memory reliability
        mem_text = memory_context or ""
        mem_lower = mem_text.lower()
        if mem_text.strip():
            score = 0.2
            rationale_bits: List[str] = []
            if "verified cases" in mem_lower or "verified" in mem_lower:
                score += 0.35
                rationale_bits.append("contains verified history")
            if "iou" in mem_lower or "p@50" in mem_lower:
                score += 0.2
                rationale_bits.append("has quantitative metrics")
            if "unverified" in mem_lower or "⊘" in mem_text:
                score -= 0.15
                rationale_bits.append("includes unverified trajectories")
            if "strategy comparison" in mem_lower or "tool chain" in mem_lower:
                score += 0.15
                rationale_bits.append("has strategy/tool evidence")
            score = max(0.0, min(1.0, score))
            rationale = ", ".join(rationale_bits) if rationale_bits else "memory present but weakly structured"
            action = "prioritize verified/metric-backed entries; avoid relying on unverified-only cues"
            _append_source("memory", score, rationale, action)
        else:
            source_lines.append("- memory: unavailable")
            source_details.append(
                {
                    "source": "memory",
                    "level": "none",
                    "score": 0.0,
                    "rationale": "no retrieved memory context",
                    "recommended_action": "use tools to gather evidence",
                }
            )

        # Commonsense templates reliability
        cs_templates = commonsense_templates or []
        if cs_templates:
            avg_rel = sum(t.get("relevance_score", 0) for t in cs_templates) / max(len(cs_templates), 1)
            cs_score = min(1.0, avg_rel * 1.2)
            n_high = sum(1 for t in cs_templates if (t.get("relevance_score", 0) or 0) >= 0.4)
            rationale_parts = [f"{len(cs_templates)} template(s) retrieved", f"avg_relevance={avg_rel:.2f}"]
            if n_high:
                rationale_parts.append(f"{n_high} high-relevance")
            _append_source(
                "commonsense_templates",
                cs_score,
                ", ".join(rationale_parts),
                "if low relevance: templates may not match current task — rely more on tool evidence",
            )
        else:
            source_lines.append("- commonsense_templates: none retrieved (below relevance threshold)")

        # Helper tools reliability
        for item in helper_tool_results:
            tool_name = item.get("tool", "unknown")
            result = item.get("result", {}) or {}
            if "error" in result:
                _append_source(tool_name, 0.05, "tool returned error", "retry with revised parameters or switch tool")
                continue

            if tool_name == "web_search":
                score = 0.2
                if result.get("part_name"):
                    score += 0.25
                if result.get("reasoning"):
                    score += 0.2
                if len(result.get("crawled_urls", []) or []) >= 2:
                    score += 0.3
                score = min(1.0, score)
                _append_source(
                    "web_search",
                    score,
                    "part/reasoning/textual-source coverage determines evidence quality",
                    "if low: refine search_focus and rerun",
                )
            elif tool_name == "dreamer":
                score = 0.2
                if result.get("interaction_analysis"):
                    score += 0.3
                if len(result.get("generated_image_paths", []) or []) > 0:
                    score += 0.2
                if result.get("num_targets") is not None:
                    score += 0.1
                score = min(1.0, score)
                _append_source(
                    "dreamer",
                    score,
                    "interaction analysis + generated visual hypothesis quality",
                    "if low: rerun with clearer interaction_type/target hints",
                )
            elif tool_name == "zoom_in":
                score = 0.2
                crop = result.get("crop_region_pixel")
                cropped = result.get("cropped_size", [])
                if isinstance(crop, list) and len(crop) == 4:
                    score += 0.3
                if isinstance(cropped, list) and len(cropped) == 2 and cropped[0] > 20 and cropped[1] > 20:
                    score += 0.2
                if result.get("zoom_purpose"):
                    score += 0.1
                score = min(1.0, score)
                _append_source(
                    "zoom_in",
                    score,
                    "local detail coverage and usable crop quality",
                    "if low: select a better bbox and zoom again",
                )
            elif tool_name == "load_skill_doc":
                score = 0.25
                content = str(result.get("content", "") or "")
                if result.get("skill_name"):
                    score += 0.15
                if not result.get("truncated"):
                    score += 0.1
                if any(k in content.lower() for k in ["when_to_use", "when_not_to_use", "inputs", "outputs"]):
                    score += 0.2
                score = min(1.0, score)
                _append_source(
                    "load_skill_doc",
                    score,
                    "tool usage guidance quality (policy source, not direct visual evidence)",
                    "use it to improve tool strategy, not as final localization evidence",
                )
            else:
                _append_source(
                    tool_name,
                    0.4,
                    "unknown source type; conservative estimate",
                    "validate with another tool before relying on it",
                )

        # Sufficiency decision:
        # 1) enough reliable sources count
        # 2) at least one high-quality source (or reliability threshold raised by user config later)
        min_required = max(1, int(self.min_reliable_sources_before_detection))
        sufficient = (reliable_sources >= min_required) and (high_quality_sources >= 1)
        overall = "sufficient" if sufficient else "insufficient"

        report_text = (
            "Reliability check before detection:\n"
            f"- reliable sources (medium/high): {reliable_sources}\n"
            f"- high-quality sources: {high_quality_sources}\n"
            f"- checked sources: {checked_sources}\n"
            f"- threshold: at least {min_required} reliable source(s) and >=1 high-quality source\n"
            f"- overall: {overall}\n"
            + "\n".join(source_lines)
        )
        return {
            "reliable_sources": reliable_sources,
            "high_quality_sources": high_quality_sources,
            "checked_sources": checked_sources,
            "overall": overall,
            "report_text": report_text,
            "source_details": source_details,
            "min_required": min_required,
        }

    def _create_system_prompt(self) -> str:
        """
        
        
        Returns:
            
        """
        skill_index = self.skill_registry.get_skill_index_for_prompt(
            ["detection", "web_search", "dreamer", "zoom_in"]
        )

        mode = (self.skill_prompt_mode or "on_demand").strip().lower()
        if mode == "auto":
            model_lower = (self.model_name or "").lower()
            if any(k in model_lower for k in ("qwen", "llama", "mistral")):
                mode = "inline"
            else:
                mode = "on_demand"

        if mode == "inline":
            guidance = self.skill_registry.get_skill_guidance_for_prompt(
                ["detection", "web_search", "dreamer", "zoom_in"]
            )
            return build_system_prompt(guidance, skill_index=skill_index)

        # on_demand / unknown modes: keep concise index only.
        return build_system_prompt(None, skill_index=skill_index)

    def _sanitize_payload_for_preview(self, obj: Any, max_text_len: int = 900) -> Any:
        """
        Build a readable payload preview:
        - truncate long text
        - replace base64 image URLs with size metadata
        """
        if isinstance(obj, dict):
            sanitized: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "url" and isinstance(v, str) and v.startswith("data:image"):
                    header, _, data = v.partition(",")
                    sanitized[k] = {
                        "data_url_header": header[:80] + ("..." if len(header) > 80 else ""),
                        "base64_chars": len(data),
                    }
                else:
                    sanitized[k] = self._sanitize_payload_for_preview(v, max_text_len=max_text_len)
            return sanitized
        if isinstance(obj, list):
            return [self._sanitize_payload_for_preview(x, max_text_len=max_text_len) for x in obj]
        if isinstance(obj, str):
            compact = obj
            if len(compact) <= max_text_len:
                return compact
            head = compact[: max_text_len // 2]
            tail = compact[- max_text_len // 2 :]
            return f"{head}\n...[truncated {len(compact) - len(head) - len(tail)} chars]...\n{tail}"
        return obj

    def _dump_request_payload_preview(self, request_data: Dict[str, Any], iteration: int) -> Optional[str]:
        """Dump request payload preview to output dir for debugging/visualization."""
        if not self.dump_request_payload_preview:
            return None
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(self.payload_preview_dir, f"decision_model_iter{iteration}_{ts}.json")
            payload = {
                "model_name": self.model_name,
                "iteration": iteration,
                "request_preview": self._sanitize_payload_for_preview(request_data),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path
        except Exception as e:
            print(f"[Warning] Failed to dump payload preview: {e}")
            return None
    
    @staticmethod
    def _classify_user_message(text: str) -> str:
        """Classify a non-initial user message by its content for trace readability."""
        lower = text.lower()
        if "reliability" in lower and ("gate" in lower or "deferred" in lower or "insufficient" in lower):
            return "reliability_gate"
        if "detection verification" in lower or "verify the annotations" in lower:
            return "detection_verification"
        if "tool results summary" in lower or "tool retry budget" in lower:
            return "reflection"
        if "reliability" in lower or "source reliability" in lower:
            return "reflection_with_reliability"
        return "follow_up"

    def _dump_decision_trace(
        self,
        result: Dict[str, Any],
        image_path: str,
        task: str,
        object_name: Optional[str],
        memory_context: str,
        commonsense_templates: Optional[List[Dict[str, Any]]],
    ) -> Optional[str]:
        """Save a human-readable multi-turn decision trace for one sample.

        The output JSON captures each turn of the decision model conversation:
        system prompt, user message (with metadata instead of base64 images),
        assistant reasoning, tool calls, tool results, and follow-up turns.
        """
        if not self.dump_request_payload_preview:
            return None
        try:
            messages = result.get("messages", [])
            if not messages:
                return None

            image_stem = Path(image_path).stem
            path = os.path.join(self.output_dir, f"{image_stem}_decision_trace.json")

            turns: List[Dict[str, Any]] = []
            turn_idx = 0
            is_first_user = True
            standard_keys = {
                "image_path", "task", "object_name", "task_context",
                "question", "bbox", "points", "skill_name", "max_chars",
            }

            for msg in messages:
                role = msg.get("role", "unknown")
                turn_entry: Dict[str, Any] = {"turn": turn_idx, "role": role}

                if role == "system":
                    content = msg.get("content", "")
                    turn_entry["content"] = content[:2000] + ("..." if len(content) > 2000 else "")

                elif role == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        # Subsequent user messages are usually reflection /
                        # reliability-gate messages — keep more text.
                        tag = "initial_task" if is_first_user else self._classify_user_message(content)
                        turn_entry["message_type"] = tag
                        turn_entry["content"] = content[:5000] + ("..." if len(content) > 5000 else "")
                    elif isinstance(content, list):
                        tag = "initial_task" if is_first_user else "reflection_with_images"
                        turn_entry["message_type"] = tag
                        parts_summary: List[str] = []
                        text_parts: List[str] = []
                        img_count = 0
                        for part in content:
                            if part.get("type") == "text":
                                text = part.get("text", "")
                                text_parts.append(text)
                                parts_summary.append(f"[text] {text[:2000]}{'...' if len(text) > 2000 else ''}")
                            elif part.get("type") == "image_url":
                                img_count += 1
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    header = url.split(",")[0] if "," in url else url[:50]
                                    b64_len = len(url.split(",")[1]) if "," in url else 0
                                    parts_summary.append(f"[image #{img_count}] ({b64_len} base64 chars)")
                                else:
                                    parts_summary.append(f"[image #{img_count}] {url[:200]}")
                        turn_entry["content_parts"] = parts_summary
                        if not is_first_user:
                            full_text = "\n".join(text_parts)
                            if "Reliability" in full_text or "reliability" in full_text:
                                turn_entry["contains_reliability_assessment"] = True
                    is_first_user = False

                elif role == "assistant":
                    content = msg.get("content", "")
                    if content and isinstance(content, str) and content.strip():
                        turn_entry["reasoning"] = content.strip()

                    tool_calls_list = msg.get("tool_calls", [])
                    if tool_calls_list:
                        calls_summary = []
                        for tc in tool_calls_list:
                            func = tc.get("function", {})
                            name = func.get("name", "?")
                            try:
                                args = json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {})
                            except Exception:
                                args = {}
                            std_args = {}
                            dyn_args = {}
                            for k, v in args.items():
                                val = v
                                if isinstance(v, str) and len(v) > 1200:
                                    val = v[:1200] + "..."
                                elif isinstance(v, list) and len(str(v)) > 800:
                                    val = f"[list with {len(v)} items]"
                                if k in standard_keys:
                                    std_args[k] = val
                                else:
                                    dyn_args[k] = val
                            call_entry: Dict[str, Any] = {"tool": name, "arguments": std_args}
                            if dyn_args:
                                call_entry["dynamic_strategy_params"] = dyn_args
                            calls_summary.append(call_entry)
                        turn_entry["tool_calls"] = calls_summary

                elif role == "tool":
                    turn_entry["tool_name"] = msg.get("name", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if "error" in parsed:
                                err = parsed["error"]
                                turn_entry["error"] = err[:2000] + ("..." if len(str(err)) > 2000 else "")
                                if "reliability" in str(err).lower() or "deferred" in str(err).lower():
                                    turn_entry["is_reliability_gate"] = True
                            else:
                                summary = self._summarize_tool_result(parsed, msg.get("name", ""))
                                turn_entry["result_summary"] = summary
                                if msg.get("name") == "detection":
                                    for key in ("mask_image_path", "visualization_path", "object_part"):
                                        if key in parsed:
                                            turn_entry[key] = parsed[key]
                        except Exception:
                            turn_entry["result_raw"] = content[:2000] + ("..." if len(content) > 2000 else "")

                turns.append(turn_entry)
                turn_idx += 1

            # --- Build interaction_flow summary from turns ---
            flow_events: List[Dict[str, Any]] = []
            tool_call_counter: Dict[str, int] = {}
            all_dynamic_params: List[Dict[str, Any]] = []
            reliability_gate_count = 0
            reflection_count = 0
            verification_count = 0
            assistant_reasoning_snippets: List[str] = []

            for t in turns:
                role = t.get("role")
                if role == "assistant":
                    # Collect reasoning snippets (first 300 chars each)
                    reasoning = t.get("reasoning", "")
                    if reasoning:
                        assistant_reasoning_snippets.append(
                            reasoning[:300] + ("..." if len(reasoning) > 300 else "")
                        )
                    for tc in t.get("tool_calls", []):
                        tool_name = tc.get("tool", "?")
                        tool_call_counter[tool_name] = tool_call_counter.get(tool_name, 0) + 1
                        call_num = tool_call_counter[tool_name]
                        event: Dict[str, Any] = {
                            "event": "tool_call",
                            "tool": tool_name,
                            "call_number": call_num,
                            "arguments": tc.get("arguments", {}),
                        }
                        dyn = tc.get("dynamic_strategy_params")
                        if dyn:
                            event["dynamic_strategy_params"] = dyn
                            all_dynamic_params.append({"tool": tool_name, "call": call_num, "params": dyn})
                        flow_events.append(event)

                elif role == "tool":
                    tool_name = t.get("tool_name", "?")
                    event = {"event": "tool_result", "tool": tool_name}
                    if t.get("is_reliability_gate"):
                        event["event"] = "reliability_gate_triggered"
                        event["reason"] = t.get("error", "")[:500]
                        reliability_gate_count += 1
                    elif "error" in t:
                        event["status"] = "error"
                        event["error"] = t["error"][:300]
                    else:
                        event["status"] = "success"
                        event["summary"] = t.get("result_summary", "")[:400]
                        for k in ("mask_image_path", "visualization_path", "object_part"):
                            if k in t:
                                event[k] = t[k]
                    flow_events.append(event)

                elif role == "user":
                    msg_type = t.get("message_type", "")
                    if msg_type == "reflection" or msg_type == "reflection_with_images":
                        reflection_count += 1
                        event = {"event": "reflection", "count": reflection_count}
                        if t.get("contains_reliability_assessment"):
                            event["includes_reliability_assessment"] = True
                        flow_events.append(event)
                    elif msg_type == "reflection_with_reliability":
                        reflection_count += 1
                        event = {
                            "event": "reflection",
                            "count": reflection_count,
                            "includes_reliability_assessment": True,
                        }
                        flow_events.append(event)
                    elif msg_type == "reliability_gate":
                        flow_events.append({"event": "reliability_gate_user_prompt"})
                    elif msg_type == "detection_verification":
                        verification_count += 1
                        flow_events.append({"event": "detection_verification", "count": verification_count})

            # --- Memory context summary ---
            memory_section: Optional[Dict[str, Any]] = None
            if memory_context and memory_context.strip():
                mc = memory_context.strip()
                memory_section = {
                    "provided": True,
                    "char_length": len(mc),
                    "content": mc[:4000] + ("..." if len(mc) > 4000 else ""),
                }
            else:
                memory_section = {"provided": False}

            # --- Commonsense templates detail ---
            cs_section: Optional[List[Dict[str, Any]]] = None
            if commonsense_templates:
                cs_section = []
                for idx, tpl in enumerate(commonsense_templates, 1):
                    cs_section.append({
                        "exemplar": idx,
                        "object_name": tpl.get("object_name", "?"),
                        "affordance_part": tpl.get("affordance_part", "?"),
                        "question": str(tpl.get("question", ""))[:200],
                        "relevance_score": tpl.get("relevance_score"),
                        "image_path": tpl.get("image_path", ""),
                        "gt_path": tpl.get("gt_path", ""),
                    })

            # --- Dynamic params summary (from result) ---
            dynamic_params = result.get("dynamic_params_used", {})

            # --- Reliability reports collected during _call_llm_with_tools ---
            rel_reports = result.get("reliability_reports", [])

            trace = {
                "sample": image_stem,
                "task": task,
                "object_name": object_name,
                "image_path": image_path,
                "model": self.model_name,
                "total_turns": len(turns),
                "outcome": "success" if result.get("success") else result.get("error", "unknown"),

                "interaction_flow": {
                    "description": "High-level sequence of the decision model's actions",
                    "tool_call_counts": tool_call_counter,
                    "reliability_gates_triggered": reliability_gate_count,
                    "reflection_rounds": reflection_count,
                    "detection_verifications": verification_count,
                    "dynamic_strategy_params_used": all_dynamic_params if all_dynamic_params else None,
                    "events": flow_events,
                },

                "context_sources": {
                    "memory": memory_section,
                    "commonsense_templates": cs_section,
                    "dynamic_params_from_result": dynamic_params if dynamic_params else None,
                },

                "reliability_assessments": {
                    "description": (
                        "Each entry is a reliability check performed before detection or "
                        "during reflection. source_details shows per-source scoring with "
                        "rationale and recommended_action."
                    ),
                    "total_checks": len(rel_reports),
                    "checks": rel_reports if rel_reports else None,
                },

                "decision_model_reasoning": {
                    "description": "Snippets of the decision model's reasoning at each step",
                    "reasoning_steps": assistant_reasoning_snippets,
                },

                "turns": turns,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)
            return path

        except Exception as e:
            print(f"[Warning] Failed to dump decision trace: {e}")
            return None

    def _call_fixed_skill_chain(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        original_image_path: Optional[str] = None,
        task: str = "",
    ) -> Dict[str, Any]:
        """ skill （baseline ）。

         zoom_in → dreamer → web_search → detection ：
        1.  skill  tool_choice  GPT-4o ；
        2. ， tool_choice  GPT-4o  task_context  detection。
         LLM 。

        Args:
            messages: （ system + user ）
            tools: 
            original_image_path: （detection ）
            task: 
        """
        all_messages = messages.copy()
        _helper_tool_insights: List[str] = []
        _helper_tool_results: List[Dict[str, Any]] = []
        _helper_tool_images: List[Dict[str, str]] = []
        _payload_preview_paths: List[str] = []
        _tool_call_counts: Dict[str, int] = {}

        is_glm = "glm" in self.model_name.lower()

        FIXED_CHAIN = ["zoom_in", "dreamer", "web_search"]

        # ── Phase 1:  skills ─────────────────────────────────
        for skill_name in FIXED_CHAIN:
            print(f"[FixedChain] Forcing skill: {skill_name}")

            request_data: Dict[str, Any] = {
                "model": self.model_name,
                "messages": all_messages,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": skill_name}},
            }
            preview_path = self._dump_request_payload_preview(request_data, iteration=len(_payload_preview_paths))
            if preview_path:
                _payload_preview_paths.append(preview_path)

            try:
                response = self.api_client.call("chat/completions", request_data)
            except Exception as e:
                print(f"[FixedChain] API call failed for {skill_name}: {e}")
                #  skill，
                continue

            if "choices" not in response or not response["choices"]:
                print(f"[FixedChain] Invalid API response for {skill_name}, skipping.")
                continue

            message = response["choices"][0]["message"]
            all_messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                print(f"[FixedChain] LLM returned no tool_calls for {skill_name}, skipping.")
                continue

            #  tool call（）
            tool_call = tool_calls[0]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                function_args = {}

            #  image_path 
            if "image_path" in function_args:
                function_args["image_path"] = os.path.abspath(function_args["image_path"])

            _tool_call_counts[skill_name] = _tool_call_counts.get(skill_name, 0) + 1
            print(f"[Tool Call] {skill_name}({', '.join(f'{k}={v}' for k, v in function_args.items() if k != 'image_path')})")
            skill_result = self.skill_registry.call_skill(skill_name, **function_args)

            insight = self._extract_key_insights_from_tool(skill_result, skill_name)
            if insight:
                _helper_tool_insights.append(insight)
            _helper_tool_results.append({"tool": skill_name, "result": skill_result})
            imgs = self._extract_tool_images(skill_result, skill_name)
            _helper_tool_images.extend(imgs)

            api_result = self._prepare_tool_result_for_api(skill_result, skill_name)
            tool_result_msg = {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": skill_name,
                "content": json.dumps(api_result, ensure_ascii=False),
            }
            all_messages.append(tool_result_msg)

            if not hasattr(self, "_tool_results_storage"):
                self._tool_results_storage = {}
            self._tool_results_storage[tool_call["id"]] = skill_result

            #  reflection （ LLM ， detection ）
            summary = self._summarize_tool_result(skill_result, skill_name)
            print(f"[Agent] Tool summary:\n{summary}")
            reflection = self._build_reflection_message(
                [summary],
                imgs if not is_glm else [],
                detection_called=False,
                reliability_report=None,
                tool_call_counts=_tool_call_counts,
                max_tool_calls_per_tool=self.max_tool_calls_per_tool,
            )
            if reflection:
                all_messages.append(reflection)

        # ── Phase 2:  helper ， detection ────────────────
        if _helper_tool_insights:
            combined_insights = "\n\n".join(_helper_tool_insights)
            aggregation_msg = (
                "[Fixed Skill Chain] All helper skills (zoom_in, dreamer, web_search) "
                "have been executed. Aggregated findings from all skills:\n\n"
                + combined_insights
                + "\n\nPlease now call `detection` with a comprehensive `task_context` "
                "that incorporates all the above findings. Include all relevant "
                "reference images from dreamer and zoom_in in `reference_images`."
            )
            all_messages.append({"role": "user", "content": aggregation_msg})

        print("[FixedChain] Forcing skill: detection")
        request_data = {
            "model": self.model_name,
            "messages": all_messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "detection"}},
        }
        preview_path = self._dump_request_payload_preview(request_data, iteration=len(_payload_preview_paths))
        if preview_path:
            _payload_preview_paths.append(preview_path)

        try:
            response = self.api_client.call("chat/completions", request_data)
        except Exception as e:
            return {"error": f"API call failed for detection: {str(e)}", "messages": all_messages}

        if "choices" not in response or not response["choices"]:
            return {"error": "Invalid API response for detection", "messages": all_messages}

        message = response["choices"][0]["message"]
        all_messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {
                "error": "LLM returned no tool_calls for detection",
                "messages": all_messages,
                "decision_payload_preview_paths": _payload_preview_paths,
                "reliability_reports": [],
            }

        tool_call = tool_calls[0]
        try:
            function_args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            function_args = {}

        if "image_path" in function_args:
            function_args["image_path"] = os.path.abspath(function_args["image_path"])
        if original_image_path:
            given = function_args.get("image_path", "")
            if given != original_image_path:
                print(f"[FixedChain] Forcing detection image_path to original: {os.path.basename(original_image_path)}")
                function_args["image_path"] = original_image_path

        #  helper skill  reference_images
        # detection skill  reference_images  List[Dict[str, str]]，：{"path": ..., "label": ...}
        helper_img_paths = [img["path"] for img in _helper_tool_images if "path" in img and os.path.exists(img["path"])]
        if helper_img_paths:
            existing_refs = list(function_args.get("reference_images") or [])
            #  existing_refs  dict （LLM  dict ）
            normalized_existing: List[Dict[str, str]] = []
            seen_paths: set = set()
            for r in existing_refs:
                if isinstance(r, dict):
                    p = str(r.get("path", ""))
                    if p and p not in seen_paths:
                        seen_paths.add(p)
                        normalized_existing.append(r)
                else:
                    p = str(r)
                    if p and p not in seen_paths:
                        seen_paths.add(p)
                        normalized_existing.append({"path": p, "label": ""})
            #  helper skill  dict 
            helper_dicts: List[Dict[str, str]] = []
            for img in _helper_tool_images:
                p = img.get("path", "")
                if p and os.path.exists(p) and p not in seen_paths:
                    seen_paths.add(p)
                    helper_dicts.append({"path": p, "label": img.get("label", img.get("description", ""))})
            function_args["reference_images"] = helper_dicts + normalized_existing

        #  task_context
        if _helper_tool_insights:
            combined_ctx = "\n\n".join(_helper_tool_insights)
            existing_ctx = (function_args.get("task_context") or "").strip()
            function_args["task_context"] = (combined_ctx + "\n\n" + existing_ctx).strip() if existing_ctx else combined_ctx

        #  registry （agent ）
        function_args["_skip_registry_text_injection"] = True

        ref_count = len(function_args.get("reference_images") or [])
        ctx_len = len(function_args.get("task_context") or "")
        print(f"[FixedChain] Detection: task_context={ctx_len} chars, reference_images={ref_count}")

        skill_result = self.skill_registry.call_skill("detection", **function_args)
        if not hasattr(self, "_tool_results_storage"):
            self._tool_results_storage = {}
        self._tool_results_storage[tool_call["id"]] = skill_result

        #  detection 
        api_result = self._prepare_tool_result_for_api(skill_result, "detection")
        all_messages.append({
            "tool_call_id": tool_call["id"],
            "role": "tool",
            "name": "detection",
            "content": json.dumps(api_result, ensure_ascii=False),
        })

        return {
            "success": True,
            "final_response": "",
            "messages": all_messages,
            "decision_payload_preview_paths": _payload_preview_paths,
            "reliability_reports": [],
        }

    def _call_llm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_iterations: int = 25,
        original_image_path: Optional[str] = None,
        memory_context: str = "",
        commonsense_context: str = "",
        commonsense_templates: Optional[List[Dict[str, Any]]] = None,
        task: str = "",
    ) -> Dict[str, Any]:

        all_messages = messages.copy()
        iteration = 0
        detection_called = False  # Track whether detection has been called
        # Accumulate key insights from helper tools so that detection always
        # receives previous context, regardless of how well the LLM synthesizes.
        _helper_tool_insights: List[str] = []
        _helper_tool_results: List[Dict[str, Any]] = []
        _helper_tool_images: List[Dict[str, str]] = []
        _tool_call_counts: Dict[str, int] = {}
        _reliability_guard_count = 0
        _payload_preview_paths: List[str] = []
        _reliability_reports: List[Dict[str, Any]] = []
        
        while iteration < max_iterations:
            request_data = {
                "model": self.model_name,
                "messages": all_messages,
                "tools": tools,
            }
            # Only add tool_choice if not auto, or if we want to force it.
            # Some providers (like vveai/GLM) might fail with explicit "auto".
            if "glm" not in self.model_name.lower():
                 request_data["tool_choice"] = "auto"
            preview_path = self._dump_request_payload_preview(request_data, iteration=iteration)
            if preview_path:
                _payload_preview_paths.append(preview_path)
            
            #  API
            try:
                response = self.api_client.call("chat/completions", request_data)
            except Exception as e:
                return {"error": f"API call failed: {str(e)}"}
            
            if "choices" not in response or not response["choices"]:
                return {"error": "Invalid API response"}
            
            choice = response["choices"][0]
            message = choice["message"]
            
            all_messages.append(message)
            
            # Capture the LLM's reasoning text from this turn as a potential insight.
            # When the LLM explains its reasoning before calling tools, that analysis
            # is valuable context for detection.
            _assistant_text = message.get("content", "")
            if _assistant_text and isinstance(_assistant_text, str) and _assistant_text.strip() and not detection_called:
                _reasoning_snippet = _assistant_text.strip()
                if len(_reasoning_snippet) > 600:
                    _reasoning_snippet = _reasoning_snippet[:600] + "..."
                _helper_tool_insights.append(f"[decision model reasoning]\n  {_reasoning_snippet}")
            
            def _parse_tagged_tool_calls(text: str) -> List[Dict[str, Any]]:
                """
                Fallback parser for providers that don't return OpenAI-style `tool_calls`
                but instead embed tool calls in plain text like:
                  <tool_call>
                  <function=sam2>
                  <parameter=image_path>...</parameter>
                  ...
                  </function>
                  </tool_call>
                """
                if not text:
                    return []
                calls: List[Dict[str, Any]] = []
                # Find each <function=NAME>...</function> block
                for m in re.finditer(r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>", text, flags=re.DOTALL):
                    name = m.group("name").strip()
                    body = m.group("body")
                    args: Dict[str, Any] = {}
                    for pm in re.finditer(r"<parameter=(?P<key>[^>]+)>(?P<val>.*?)</parameter>", body, flags=re.DOTALL):
                        key = pm.group("key").strip()
                        val = pm.group("val").strip()
                        # Keep bbox/points as JSON-ish strings; everything else as plain string
                        args[key] = val
                    if name:
                        calls.append({"name": name, "arguments": args})
                return calls

            #  tool calls（ OpenAI tool_calls  fallback ）
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                fallback_calls = _parse_tagged_tool_calls(message.get("content") or "")
                if fallback_calls:
                    # Convert fallback calls to internal tool_call shape
                    tool_calls = [
                        {
                            "id": f"fallback_{iteration}_{i}",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                            },
                        }
                        for i, c in enumerate(fallback_calls)
                    ]

            if not tool_calls:
                #  tool calls —  detection 
                if not detection_called:
                    # detection ，
                    print("[Warning] LLM tried to finish without calling detection. Injecting reminder...")
                    all_messages.append({
                        "role": "user",
                        "content": "The `detection` tool has not been called yet. The segmentation mask is the final output of this task. Please proceed with calling `detection`."
                    })
                    iteration += 1
                    continue
                
                return {
                    "success": True,
                    "final_response": message.get("content", ""),
                    "messages": all_messages,
                    "decision_payload_preview_paths": _payload_preview_paths,
                    "reliability_reports": _reliability_reports,
                }
            
            #  tool calls
            tool_results = []
            tool_summaries = []         # Concise summaries of each tool's result
            tool_images_to_embed = []   # Images from tools for visual feedback
            reliability_gate_feedback: List[str] = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                try:
                    function_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    function_args = {}
                
                #  image_path 
                if "image_path" in function_args:
                    function_args["image_path"] = os.path.abspath(function_args["image_path"])
                
                # CRITICAL: detection ，
                #  LLM  zoom_in  mask 
                if function_name == "detection" and original_image_path:
                    given_path = function_args.get("image_path", "")
                    if given_path != original_image_path:
                        print(f"[Warning] detection was given image_path={os.path.basename(given_path)}, "
                              f"forcing to original: {os.path.basename(original_image_path)}")
                        function_args["image_path"] = original_image_path
                
                # The decision model composes the full detection context
                # (task_context + reference_images) — pass through directly.
                if function_name == "detection":
                    function_args["_skip_registry_text_injection"] = True
                    ref_count = len(function_args.get("reference_images") or [])
                    ctx_len = len(function_args.get("task_context") or "")
                    print(f"[Agent] Detection called with model-composed context: "
                          f"task_context={ctx_len} chars, reference_images={ref_count}")

                # --- Reliability gate before detection ---
                if function_name == "detection":
                    reliability_report = self._evaluate_source_reliability(
                        helper_tool_results=_helper_tool_results,
                        memory_context=memory_context,
                        commonsense_templates=commonsense_templates,
                    )
                    _reliability_reports.append({
                        "stage": "pre_detection_gate",
                        "iteration": iteration,
                        "overall": reliability_report["overall"],
                        "source_details": reliability_report.get("source_details", []),
                    })
                    has_any_context_source = bool((memory_context or "").strip()) or (len(_helper_tool_results) > 0)
                    if (
                        has_any_context_source
                        and reliability_report["overall"] != "sufficient"
                        and _reliability_guard_count < self.reliability_retry_limit
                    ):
                        _reliability_guard_count += 1
                        gate_msg = (
                            "Detection temporarily deferred due to insufficient source reliability. "
                            "Please evaluate each source reliability explicitly, then gather stronger evidence "
                            "(you may call helper tools again, including repeated calls with new strategies) "
                            "before calling detection.\n\n"
                            + reliability_report["report_text"]
                        )
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"error": gate_msg}, ensure_ascii=False),
                        })
                        reliability_gate_feedback.append(gate_msg)
                        continue

                # --- Per-tool retry limit ---
                tool_executed = False
                call_count = _tool_call_counts.get(function_name, 0)
                if function_name == "detection":
                    # Detection retries are capped to control token/time cost:
                    # 1 initial call + <=3 retries.
                    detection_max_calls = min(
                        self.max_tool_calls_per_tool,
                        self.detection_max_retries + 1,
                    )
                    if call_count >= detection_max_calls:
                        limit_msg = (
                            f"Tool '{function_name}' reached max retry limit "
                            f"({self.detection_max_retries} retries, {detection_max_calls} total calls) "
                            "for this sample. Please proceed with current result."
                        )
                        skill_result = {"error": limit_msg}
                    else:
                        _tool_call_counts[function_name] = call_count + 1
                        #  skill
                        print(f"[Tool Call] {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items() if k != 'image_path')})")
                        skill_result = self.skill_registry.call_skill(function_name, **function_args)
                        tool_executed = True
                elif call_count >= self.max_tool_calls_per_tool:
                    limit_msg = (
                        f"Tool '{function_name}' reached max retry limit "
                        f"({self.max_tool_calls_per_tool}) for this sample. "
                        "Please adjust strategy and choose another tool or proceed if ready."
                    )
                    skill_result = {"error": limit_msg}
                else:
                    _tool_call_counts[function_name] = call_count + 1
                    #  skill
                    print(f"[Tool Call] {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items() if k != 'image_path')})")
                    skill_result = self.skill_registry.call_skill(function_name, **function_args)
                    tool_executed = True
                
                # Track if detection was actually executed; reset verified
                # flag so the new result gets verified too.
                if function_name == "detection" and tool_executed:
                    detection_called = True
                    self._detection_verified = False
                
                # Generate concise summary for the decision model
                summary = self._summarize_tool_result(skill_result, function_name)
                tool_summaries.append(summary)
                print(f"[Agent] Tool summary:\n{summary}")
                
                # Collect images from helper tools for visual feedback
                if function_name != "detection":
                    extracted_imgs = self._extract_tool_images(skill_result, function_name)
                    tool_images_to_embed.extend(extracted_imgs)
                    _helper_tool_images.extend(extracted_imgs)
                    insight = self._extract_key_insights_from_tool(skill_result, function_name)
                    if insight:
                        _helper_tool_insights.append(insight)
                    _helper_tool_results.append({"tool": function_name, "result": skill_result})
                
                #  API （ API ）
                api_result = self._prepare_tool_result_for_api(skill_result, function_name)
                
                # （ skill_result ，）
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(api_result, ensure_ascii=False)
                })
                
                if not hasattr(self, "_tool_results_storage"):
                    self._tool_results_storage = {}
                self._tool_results_storage[tool_call["id"]] = skill_result
            
            #  tool （API  tool results  assistant message ）
            all_messages.extend(tool_results)

            if reliability_gate_feedback:
                all_messages.append({
                    "role": "user",
                    "content": (
                        "Before calling detection, assess reliability of memory/tool evidence "
                        "for this specific task. If evidence is weak, continue tool calls with a revised strategy.\n\n"
                        + "\n\n".join(reliability_gate_feedback)
                    ),
                })
            
            # Structured reflection after every tool call round — the decision
            # model always gets a chance to reason about new evidence, whether
            # or not detection has already been called.
            if tool_summaries:
                live_reliability_report = self._evaluate_source_reliability(
                    helper_tool_results=_helper_tool_results,
                    memory_context=memory_context,
                    commonsense_templates=commonsense_templates,
                )
                _reliability_reports.append({
                    "stage": "reflection",
                    "iteration": iteration,
                    "overall": live_reliability_report["overall"],
                    "source_details": live_reliability_report.get("source_details", []),
                })
                reflection_msg = self._build_reflection_message(
                    tool_summaries,
                    tool_images_to_embed,
                    detection_called,
                    reliability_report=live_reliability_report,
                    tool_call_counts=_tool_call_counts,
                    max_tool_calls_per_tool=self.max_tool_calls_per_tool,
                    commonsense_context=commonsense_context if commonsense_context else None,
                )
                if reflection_msg:
                    all_messages.append(reflection_msg)

            # Post-detection visual verification: show the bbox+point annotated image
            # to the decision model so it can judge whether the annotations are accurate.
            # If unsatisfied, the model can re-call detection with corrective feedback.
            if detection_called and not self._detection_verified:
                self._detection_verified = True
                verification_msg = self._build_detection_verification_message(
                    tool_results, _tool_call_counts, self.max_tool_calls_per_tool,
                    task=task,
                )
                if verification_msg:
                    all_messages.append(verification_msg)
                    # Allow the model one more turn: it can either approve (finish)
                    # or re-call detection.  We keep detection_called=True so the
                    # model CAN finish if satisfied; it just gets an extra turn.
            
            iteration += 1
        
        return {
            "error": f"Max iterations ({max_iterations}) reached",
            "messages": all_messages,
            "decision_payload_preview_paths": _payload_preview_paths,
            "reliability_reports": _reliability_reports,
        }
    
    def detect_affordance(
        self,
        image_path: str,
        task: str,
        object_name: Optional[str] = None,
        save_conversation: bool = True,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
         affordance 
        
        Args:
            image_path: （）
            task: （ "Find the part of the mug that can be grasped"）
            object_name: （，， task ）
            save_conversation: 
            sample_id: （ memory），
        
        Returns:
            
        """
        #  tool （）
        self._tool_results_storage = {}
        self._detection_verified = False
        
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        # ， skills 
        image_path = os.path.abspath(image_path)
        
        #  sample_id
        if sample_id is None:
            #  ID
            import hashlib
            sample_id = hashlib.md5(f"{image_path}:{task}".encode()).hexdigest()[:16]
        
        #  memory （token ，）
        # NOTE:  —  memory_manager （ worker），
        #        enable_memory 。
        memory_context = ""
        commonsense_templates: List[Dict[str, Any]] = []
        commonsense_context = ""
        if self.memory_manager:
            relevant_memories = self.memory_manager.get_relevant_memories(
                task=task,
                object_name=object_name,
                query_image_path=image_path,
            )
            memory_context = self.memory_manager.format_memories_for_context(
                relevant_memories,
                include_images=False,
                task=task,
                object_name=object_name,
            )
            # ：（task + image + GT）
            commonsense_templates = self.memory_manager.retrieve_common_sense_templates(
                task=task,
                object_name=object_name,
                query_image_path=image_path,
                top_k=2,
            )
            commonsense_context = self.memory_manager.format_common_sense_templates_for_context(
                commonsense_templates,
                token_budget=400,
            )
        
        # （ base64  URL， API ）
        #  API  base64 
        # Note: _encode_image() always encodes as JPEG for size efficiency, so force mime accordingly.
        image_base64 = self._encode_image(image_path)
        image_mime_type = "image/jpeg"
        
        # ： image_path ， tools
        task_text = f"Task: {task}\n\nImage path: {image_path}\n\nPlease analyze this image and use the available tools to detect the affordance."
        if object_name:
            task_text += f"\n\nObject name: {object_name}"
        
        #  memory 
        if memory_context:
            task_text = f"{memory_context}\n\n{task_text}"
        if commonsense_context:
            task_text = f"{commonsense_context}\n\n{task_text}"
        
        user_content = [
            {
                "type": "text",
                "text": task_text
            }
        ]

        # Check if model supports images (GLM-5 on vveai currently fails with images)
        is_glm = "glm" in self.model_name.lower()
        if not is_glm:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{image_base64}"
                }
            })
        else:
            user_content.append({
                "type": "text",
                "text": "\n[Note: Direct image input is disabled for this model due to API limitations. Please rely on the 'detection', 'dreamer', and 'web_search' tools to analyze the image content.]"
            })

        # （ + GT）。
        # GLM  image_url（vveai  glm-5 ）。
        extra_img_slots = 4  # 2 templates * (image + gt)
        used_slots = 0
        for tpl in commonsense_templates:
            if used_slots >= extra_img_slots:
                break
            q = str(tpl.get("question", "") or "")
            img_ref = str(tpl.get("image_path", "") or "")
            gt_ref = str(tpl.get("gt_path", "") or "")
            pair_text = (
                "[Template Reference] "
                f"object={tpl.get('object_name', 'N/A')}, affordance/part={tpl.get('affordance_part', 'N/A')}, "
                f"question={q[:180]}"
            )
            user_content.append({"type": "text", "text": pair_text})

            if (not is_glm) and used_slots < extra_img_slots and img_ref and os.path.exists(img_ref):
                try:
                    img_b64 = self._encode_image(img_ref)
                    user_content.append({
                        "type": "text",
                        "text": f"[Template Scene Image] {os.path.basename(img_ref)}",
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                    used_slots += 1
                except Exception:
                    pass

            if (not is_glm) and used_slots < extra_img_slots and gt_ref and os.path.exists(gt_ref):
                try:
                    gt_b64 = self._encode_image(gt_ref)
                    user_content.append({
                        "type": "text",
                        "text": f"[Template GT Mask / Answer] {os.path.basename(gt_ref)}",
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"},
                    })
                    used_slots += 1
                except Exception:
                    pass
        
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt()
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        tools = self.skill_registry.get_tools_for_api(strict_schema=is_glm)

        if self.fixed_skill_chain:
            print("[Agent] Fixed skill chain mode: zoom_in → dreamer → web_search → detection")
            result = self._call_fixed_skill_chain(
                messages,
                tools,
                original_image_path=image_path,
                task=task,
            )
        else:
            #  LLM（， detection  zoom_in ）
            result = self._call_llm_with_tools(
                messages,
                tools,
                original_image_path=image_path,
                memory_context=memory_context,
                commonsense_context=commonsense_context,
                commonsense_templates=commonsense_templates,
                task=task,
            )
        
        #  tool  result ，
        # ：_tool_results_storage  detect_affordance 
        if hasattr(self, "_tool_results_storage") and self._tool_results_storage:
            result["_tool_results_storage"] = self._tool_results_storage.copy()  #  copy 

        #  tool calls、summaries、LLM reasoning、  memory
        tool_calls_for_memory = []
        tool_summaries_for_memory = []
        llm_reasoning_for_memory = []
        decision_trace_for_memory = []  # 
        dynamic_params_used = {}  # 
        task_context_for_memory = None
        mask_path_for_memory = None
        strategy_reasoning_parts = []  # 
        
        #  tool calls  LLM reasoning
        turn_idx = 0
        for msg in result.get("messages", []):
            if msg.get("role") == "assistant":
                #  LLM 
                content = msg.get("content", "")
                has_tool_calls = bool(msg.get("tool_calls"))
                if content and isinstance(content, str) and content.strip():
                    llm_reasoning_for_memory.append(content.strip())
                    # ： assistant 
                    turn_idx += 1
                    trace_entry = f"[Turn {turn_idx}] {content.strip()[:300]}"
                    decision_trace_for_memory.append(trace_entry)
                    
                    # 1) assistant  tool_calls（）
                    # 2) /
                    content_lower = content.lower()
                    is_strategy = has_tool_calls or any(kw in content_lower for kw in [
                        "strategy", "approach", "because", "reason",
                        "i will", "i'll", "let me", "first",
                        "based on", "considering", "experience",
                        "memory", "similar", "previous", "recall",
                        "decide", "plan", "step", "next",
                    ])
                    if is_strategy:
                        strategy_reasoning_parts.append(content.strip()[:300])
                
                #  tool calls 
                if "tool_calls" in msg:
                    for tool_call in msg.get("tool_calls", []):
                        try:
                            args = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except Exception:
                            args = {}
                        func_name = tool_call["function"]["name"]
                        tool_calls_for_memory.append({
                            "name": func_name,
                            "arguments": args,
                        })
                        # （ schema ）
                        standard_keys = {"image_path", "task", "object_name", "task_context", "question", "bbox", "points"}
                        for k, v in args.items():
                            if k not in standard_keys:
                                dynamic_params_used[f"{func_name}.{k}"] = v
                        
            elif msg.get("role") == "tool":
                #  tool summary
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        tool_result = json.loads(content)
                        tool_name = msg.get("name", "unknown")
                        summary = self._summarize_tool_result(tool_result, tool_name)
                        tool_summaries_for_memory.append(summary)
                        
                        if tool_name == "detection" and isinstance(tool_result, dict):
                            task_context_for_memory = tool_result.get("task_context")
                            mask_path_for_memory = tool_result.get("mask_image_path")
                        
                        #  skill  dynamic_params_used
                        if isinstance(tool_result, dict) and tool_result.get("dynamic_params_used"):
                            for k, v in tool_result["dynamic_params_used"].items():
                                dynamic_params_used[f"{tool_name}.{k}"] = v
                    except Exception:
                        pass
        
        strategy_reasoning = ""
        if strategy_reasoning_parts:
            strategy_reasoning = " | ".join(strategy_reasoning_parts[:3])
            if len(strategy_reasoning) > 500:
                strategy_reasoning = strategy_reasoning[:500] + "..."
        
        strategy_reflection = None
        tool_chain = " → ".join(tc.get("name", "?") for tc in tool_calls_for_memory)
        if tool_chain:
            strategy_reflection = f"Used strategy: {tool_chain}."
            if task_context_for_memory:
                strategy_reflection += f" Task context: {task_context_for_memory[:100]}"
        
        #  memory（）
        if self.enable_memory and self.memory_manager:
            self.memory_manager.add_entry(
                sample_id=sample_id,
                image_path=image_path,
                task=task,
                object_name=object_name,
                tool_calls=tool_calls_for_memory,
                tool_summaries=tool_summaries_for_memory,
                task_context=task_context_for_memory,
                mask_path=mask_path_for_memory,
                llm_reasoning=llm_reasoning_for_memory,
                reasoning_context=result.get("final_response", ""),
                strategy_reasoning=strategy_reasoning,
                strategy_reflection=strategy_reflection,
                dynamic_params_used=dynamic_params_used if dynamic_params_used else None,
                decision_trace=decision_trace_for_memory,
            )

        #  result， / trajectory log 
        result["strategy_reasoning"] = strategy_reasoning
        result["strategy_reflection"] = strategy_reflection
        result["dynamic_params_used"] = dynamic_params_used if dynamic_params_used else {}
        result["decision_trace"] = decision_trace_for_memory
        result["_memory_tool_calls"] = tool_calls_for_memory
        result["_memory_tool_summaries"] = tool_summaries_for_memory
        result["_memory_llm_reasoning"] = llm_reasoning_for_memory
        result["_memory_task_context"] = task_context_for_memory
        result["_memory_mask_path"] = mask_path_for_memory

        os.makedirs(self.output_dir, exist_ok=True)
        
        if save_conversation:
            image_basename = Path(image_path).stem
            conversation_path = os.path.join(
                self.output_dir,
                f"{image_basename}_conversation.json"
            )
            with open(conversation_path, "w", encoding="utf-8") as f:
                json.dump(result.get("messages", []), f, indent=2, ensure_ascii=False)
            result["conversation_path"] = conversation_path

        trace_path = self._dump_decision_trace(
            result, image_path, task, object_name,
            memory_context, commonsense_templates,
        )
        if trace_path:
            result["decision_trace_path"] = trace_path

        #  skill  output ，
        artifacts: List[str] = []
        for msg in result.get("messages", []):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except Exception:
                continue

            for key in ("visualization_path", "mask_image_path"):
                path = payload.get(key)
                if not path or not isinstance(path, str):
                    continue
                if os.path.exists(path):
                    artifacts.append(path)

        copied_paths: List[str] = []
        for src in artifacts:
            dst = os.path.join(self.output_dir, os.path.basename(src))
            try:
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                copied_paths.append(dst)
            except Exception:
                continue

        if copied_paths:
            result["output_images"] = copied_paths
        
        return result


def main():
    """"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Affordance Detection Agent")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--task", type=str, required=True, help="Task instruction")
    parser.add_argument("--object_name", type=str, default=None, help="Object name (optional)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set API_KEY env var)")
    parser.add_argument("--api_url", type=str, default=None, help="API base URL (or set API_BASE_URL env var)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--skill_prompt_mode",
        type=str,
        default="on_demand",
        choices=["on_demand", "inline", "auto"],
        help="How SKILL.md guidance is injected into system prompt",
    )
    parser.add_argument(
        "--disable_payload_preview",
        action="store_true",
        help="Disable request payload preview dumps",
    )
    
    args = parser.parse_args()
    
    #  agent
    agent = AffordanceAgent(
        api_key=args.api_key,
        api_base_url=args.api_url,
        model_name=args.model,
        output_dir=args.output_dir,
        skill_prompt_mode=args.skill_prompt_mode,
        dump_request_payload_preview=(not args.disable_payload_preview),
    )
    
    print("=" * 60)
    print("Affordance Detection Agent")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    result = agent.detect_affordance(
        image_path=args.image_path,
        task=args.task,
        object_name=args.object_name
    )
    
    print("\n" + "=" * 60)
    if result.get("success"):
        print("✓ Detection completed!")
        print(f"\nFinal Response:\n{result.get('final_response', '')}")
        if "conversation_path" in result:
            print(f"\nConversation saved to: {result['conversation_path']}")
    else:
        print("✗ Detection failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
