"""
Skill registry for affordance agent.

- Reads skill metadata (name/description) from each `skills/*/SKILL.md`
- Exposes OpenAI/Claude-style tool schemas for function calling
- Executes the underlying Python skill implementations in `scripts/`
"""

from __future__ import annotations

import importlib.util
import os
import sys
import hashlib
import re
import copy
import threading
from dataclasses import dataclass

# Module-level lock for _import_runner.
# Each worker has its own SkillRegistry instance so a per-instance lock cannot
# guard against concurrent imports across workers.  A single module-level lock
# ensures only one thread at a time can register a module in sys.modules and
# execute it, preventing the "partial module" race condition.
_GLOBAL_IMPORT_LOCK = threading.Lock()
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Skill:
    """A single skill definition."""

    name: str
    description: str
    parameters: Dict[str, Any]
    runner: Callable[..., Dict[str, Any]]
    allow_dynamic_params: bool = False  #  LLM （）

def _load_local_model_config() -> Dict[str, Any]:
    """
    Best-effort load of local model config from config.py and env vars.
    Priority: env > config.py > None
    """
    cfg: Dict[str, Any] = {}
    try:
        import importlib.util as _ilu
        _cfg_path = Path(__file__).resolve().parent.parent / "config.py"
        if _cfg_path.exists():
            _spec = _ilu.spec_from_file_location("_user_config", _cfg_path)
            if _spec and _spec.loader:
                user_config = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(user_config)
                cfg["REX_OMNI_MODEL_PATH"] = getattr(user_config, "REX_OMNI_MODEL_PATH", None)
                cfg["QWEN3VL_MODEL_PATH"] = getattr(user_config, "QWEN3VL_MODEL_PATH", None)
                cfg["SAM2_MODEL_PATH"] = getattr(user_config, "SAM2_MODEL_PATH", None)
                cfg["SAM3_MODEL_PATH"] = getattr(user_config, "SAM3_MODEL_PATH", None)
                cfg["HF_HOME"] = getattr(user_config, "HF_HOME", None)
                cfg["HF_ENDPOINT"] = getattr(user_config, "HF_ENDPOINT", None)
    except Exception:
        pass

    cfg["REX_OMNI_MODEL_PATH"] = os.getenv("REX_OMNI_MODEL_PATH", cfg.get("REX_OMNI_MODEL_PATH"))
    cfg["QWEN3VL_MODEL_PATH"] = os.getenv("QWEN3VL_MODEL_PATH", cfg.get("QWEN3VL_MODEL_PATH"))
    cfg["SAM2_MODEL_PATH"] = os.getenv("SAM2_MODEL_PATH", cfg.get("SAM2_MODEL_PATH"))
    cfg["SAM3_MODEL_PATH"] = os.getenv("SAM3_MODEL_PATH", cfg.get("SAM3_MODEL_PATH"))
    cfg["HF_HOME"] = os.getenv("HF_HOME", cfg.get("HF_HOME"))
    cfg["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", cfg.get("HF_ENDPOINT"))
    return cfg


def _prefer_skill_model_dir(skills_dir: Path, skill_folder: str, fallback: str) -> str:
    """
    Prefer loading from `skills/<skill_folder>/model` if it exists and is non-empty.
    Otherwise fall back to a HF repo id or whatever `fallback` is.
    """
    model_dir = skills_dir / skill_folder / "model"
    try:
        if model_dir.exists() and model_dir.is_dir():
            # Non-empty directory check (any file/dir inside)
            if any(model_dir.iterdir()):
                return str(model_dir)
    except Exception:
        pass
    return fallback


class SkillRegistry:
    """
    Collects skills and provides:
    - JSON schema for LLM function calling
    - Actual execution entrypoints
    """

    def __init__(self, output_dir: Optional[str] = None, tool_filter: Optional[set] = None,
                 detection_backend: str = "qwen3vl_api"):
        """
        Args:
            output_dir: Where to place skill artifacts.
            tool_filter: Optional set of tool names to expose to LLM. None = default allowlist.
            detection_backend: Which backend to use for the detection skill.
                - "qwen3vl_api": Qwen3-VL-235B cloud API (default)
                - "rex_omni": Rex-Omni local model
        """
        self.base_dir = Path(__file__).resolve().parent.parent  # A-Harness/
        self.skills_dir = Path(__file__).resolve().parent  # A-Harness/skills/
        # Optional: restrict which tools are exposed to the LLM.
        # If None, use the default allowlist. If set, only expose tools in this set.
        self._tool_filter = tool_filter
        self._detection_backend = detection_backend
        # Where to place skill artifacts (visualizations/masks/json) if not explicitly overridden.
        # Default aligns with run_affordance_detection.py / AffordanceAgent default.
        self.output_dir = Path(output_dir).resolve() if output_dir else (self.base_dir / "output")
        self.skills: Dict[str, Skill] = {}
        self._model_cfg = _load_local_model_config()
        # Cache imported runner callables so we don't re-import modules on every tool call.
        # This is critical for performance since model singletons live at module scope.
        self._runner_cache: Dict[Tuple[str, str], Callable[..., Dict[str, Any]]] = {}
        # Kept for backward compatibility but _import_runner now uses the module-level
        # _GLOBAL_IMPORT_LOCK which protects across all SkillRegistry instances.
        self._import_lock = _GLOBAL_IMPORT_LOCK
        # Memory manager (set by AffordanceAgent)
        self._memory_manager = None
        # Parsed metadata from SKILL.md headers, keyed by canonical tool name.
        self._skill_meta_by_name: Dict[str, Dict[str, Any]] = {}
        # Mapping from tool name aliases to skill folder path.
        self._skill_folder_by_name: Dict[str, str] = {}

        # If user configured HF_HOME, set it early so downstream from_pretrained uses it.
        hf_home = self._model_cfg.get("HF_HOME")
        if hf_home:
            os.environ.setdefault("HF_HOME", str(hf_home))

        # Optional: use a HuggingFace Hub mirror endpoint (common on clusters).
        # Example: HF_ENDPOINT="https://hf-mirror.com"
        hf_endpoint = self._model_cfg.get("HF_ENDPOINT")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))

        # Register built-in skills
        self._register_detection()
        self._register_dreamer()
        # Optional helper skills (not exposed to LLM tools by default)
        self._register_web_search()
        # NOTE: Disabled per request — model should not see or use these tools.
        # self._register_rex_omni()
        # self._register_sam2()
        self._register_zoom_in()
        self._register_load_skill_doc()

    # Public API
    def get_tools_for_api(self, strict_schema: bool = False) -> List[Dict[str, Any]]:
        """Return tool (function) schemas for LLM APIs.

        IMPORTANT: Only expose a small, stable tool surface to the model.
        """
        exposed_tool_names = self._tool_filter or {"detection", "dreamer", "zoom_in", "web_search", "load_skill_doc"}
        tools = []
        for skill in self.skills.values():
            if skill.name not in exposed_tool_names:
                continue
            
            # Deep copy parameters to modify them safely
            params = copy.deepcopy(skill.parameters)
            if strict_schema:
                # Some providers (e.g. GLM) reject additionalProperties: True
                if "additionalProperties" in params:
                    params["additionalProperties"] = False
            
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": skill.name,
                        "description": skill.description,
                        "parameters": params,
                    },
                }
            )
        return tools

    def get_skill_guidance_for_prompt(self, tool_names: Optional[List[str]] = None) -> str:
        """Build a compact guidance block from SKILL.md YAML headers for prompt injection."""
        if tool_names is None:
            tool_names = ["detection", "web_search", "dreamer", "zoom_in"]

        lines: List[str] = []
        for name in tool_names:
            meta = self._skill_meta_by_name.get(name) or {}
            desc = str(meta.get("description", "")).strip()
            characteristics = self._normalize_yaml_list(meta.get("characteristics"))
            when_to_use = self._normalize_yaml_list(meta.get("when_to_use"))
            when_not_to_use = self._normalize_yaml_list(meta.get("when_not_to_use"))

            lines.append(f"- {name}:")
            if desc:
                lines.append(f"  - description: {desc}")
            if characteristics:
                lines.append(f"  - characteristics: {'; '.join(characteristics)}")
            if when_to_use:
                lines.append(f"  - use_when: {'; '.join(when_to_use)}")
            if when_not_to_use:
                lines.append(f"  - avoid_when: {'; '.join(when_not_to_use)}")

        return "\n".join(lines).strip()

    def get_skill_index_for_prompt(self, tool_names: Optional[List[str]] = None) -> str:
        """
        Build one-line skill index for system prompt.
        Format: - **tool** (`skills/<folder>/`) — short description
        """
        if tool_names is None:
            tool_names = ["detection", "web_search", "dreamer", "zoom_in"]

        lines: List[str] = []
        for name in tool_names:
            folder = self._skill_folder_by_name.get(name, name.replace("_", "-"))
            meta = self._skill_meta_by_name.get(name) or {}
            desc = str(meta.get("description", "")).strip()
            if desc:
                desc = " ".join(desc.split())
                if len(desc) > 120:
                    desc = desc[:117] + "..."
            else:
                desc = "No short description provided."
            lines.append(f"- **{name}** (`skills/{folder}/`) — {desc}")
        return "\n".join(lines)

    def get_skill_markdown(self, skill_name: str, max_chars: int = 6000) -> Dict[str, Any]:
        """Load full SKILL.md content for a given skill (on demand)."""
        canonical = skill_name.strip().replace("-", "_")
        folder = self._skill_folder_by_name.get(canonical)
        if not folder:
            return {
                "error": f"Unknown skill '{skill_name}'. Available: detection, web_search, dreamer, zoom_in",
            }
        skill_md = self.skills_dir / folder / "SKILL.md"
        if not skill_md.exists():
            return {"error": f"SKILL.md not found for '{canonical}' at {skill_md}"}

        content = skill_md.read_text(encoding="utf-8")
        clipped = content
        truncated = False
        if len(content) > max_chars:
            clipped = content[:max_chars]
            truncated = True
        return {
            "skill_name": canonical,
            "skill_dir": f"skills/{folder}/",
            "skill_md_path": str(skill_md),
            "truncated": truncated,
            "max_chars": int(max_chars),
            "content": clipped,
        }

    def set_memory_manager(self, memory_manager) -> None:
        """Set the memory manager for skills to access."""
        self._memory_manager = memory_manager
    
    def call_skill(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a registered skill.

        For skills with ``allow_dynamic_params=True``, any keys that are **not**
        defined in the skill's JSON schema are collected into a dict and passed
        to the runner as ``_dynamic_params``.  This enables the decision model
        to send extra strategy hints / configuration without changing the fixed
        skill interface.

        For skills with ``allow_dynamic_params=False`` (e.g. SAM-2), unknown
        keys are silently stripped to avoid TypeError.
        """
        if name not in self.skills:
            return {"error": f"Skill '{name}' not found"}
        try:
            skill = self.skills[name]

            # --- Separate known (schema-defined) params from dynamic ones ---
            known_keys = set(skill.parameters.get("properties", {}).keys())
            known_kwargs: Dict[str, Any] = {}
            dynamic_kwargs: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in known_keys or k.startswith("_"):
                    known_kwargs[k] = v
                else:
                    dynamic_kwargs[k] = v

            if dynamic_kwargs:
                if skill.allow_dynamic_params:
                    known_kwargs["_dynamic_params"] = dynamic_kwargs
                    print(
                        f"[SkillRegistry] Dynamic params for '{name}': "
                        f"{list(dynamic_kwargs.keys())}"
                    )
                else:
                    print(
                        f"[SkillRegistry] Stripping unknown params for '{name}' "
                        f"(dynamic extension disabled): {list(dynamic_kwargs.keys())}"
                    )

            # --- Memory context injection for detection (standalone mode only) ---
            # When called from the agent loop, `_skip_registry_text_injection`
            # is True — the agent has already built the context and no
            # additional injection is needed.  Commonsense templates and
            # skill results are consumed only by the decision model.
            skip_text = known_kwargs.pop("_skip_registry_text_injection", False)

            if name == "detection" and self._memory_manager is not None and not skip_text:
                task = known_kwargs.get("task", "")
                object_name = known_kwargs.get("object_name")
                detection_budget = max(500, self._memory_manager.context_token_budget // 3)
                detection_top_k = max(3, min(10, int(getattr(self._memory_manager, "retrieval_top_k", 20) // 2)))

                extra_ctx_parts: list = []

                relevant_memories = self._memory_manager.get_relevant_memories(
                    task=task,
                    object_name=object_name,
                    query_image_path=known_kwargs.get("image_path"),
                    top_k=detection_top_k,
                    token_budget=detection_budget,
                )
                if relevant_memories:
                    memory_context = self._memory_manager.format_memories_for_context(
                        relevant_memories,
                        include_images=False,
                        max_tool_calls_per_entry=1,
                        token_budget=detection_budget,
                    )
                    if memory_context:
                        extra_ctx_parts.append(memory_context)

                if extra_ctx_parts:
                    injected = "\n\n".join(extra_ctx_parts)
                    if "task_context" in known_kwargs and known_kwargs["task_context"]:
                        known_kwargs["task_context"] = f"{injected}\n\n{known_kwargs['task_context']}"
                    else:
                        known_kwargs["task_context"] = injected

            return skill.runner(**known_kwargs)
        except Exception as exc:  # pragma: no cover - surface errors to caller
            return {"error": f"Skill '{name}' failed ({type(exc).__name__}): {exc}"}

    # Internal helpers
    def _load_skill_meta(self, skill_folder: str) -> Dict[str, str]:
        """
        Parse name/description from SKILL.md front-matter.
        Expects:
        ---
        name: ...
        description: ...
        ---
        """
        skill_md = self.skills_dir / skill_folder / "SKILL.md"
        default_name = skill_folder.replace("-", "_")
        if not skill_md.exists():
            return {"name": default_name, "description": ""}

        content = skill_md.read_text(encoding="utf-8")
        parsed = self._parse_skill_header_yaml(content)
        name = str(parsed.get("name") or default_name).strip()
        description = str(parsed.get("description") or "").strip()

        canonical = name.replace("-", "_")
        self._skill_meta_by_name[canonical] = parsed
        self._skill_meta_by_name[name] = parsed
        self._skill_meta_by_name[skill_folder.replace("-", "_")] = parsed
        self._skill_folder_by_name[canonical] = skill_folder
        self._skill_folder_by_name[name.replace("-", "_")] = skill_folder
        self._skill_folder_by_name[skill_folder.replace("-", "_")] = skill_folder
        return {"name": name, "description": description}

    def _parse_skill_header_yaml(self, content: str) -> Dict[str, Any]:
        """Parse YAML front-matter from SKILL.md with safe fallbacks."""
        m = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", content, flags=re.DOTALL)
        if not m:
            return {}
        header = m.group(1)

        # Try PyYAML first (preferred)
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(header)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass

        # Fallback: simple key: value parser
        parsed: Dict[str, Any] = {}
        for line in header.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip().strip('"').strip("'")
        return parsed

    @staticmethod
    def _normalize_yaml_list(value: Any) -> List[str]:
        """Normalize YAML scalar/list to a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            # support "a; b; c" compact style
            if ";" in s:
                return [p.strip() for p in s.split(";") if p.strip()]
            return [s]
        return [str(value).strip()]

    def _auto_reference_path(self, skill_name: str, image_path: str, suffix: str) -> str:
        """
        Build a deterministic path for visual outputs.
        Save outputs to the registry output directory (default: A-Harness/output),
        regardless of where the input image lives.
        """
        safe_image = Path(image_path).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return str(self.output_dir / f"{safe_image}_{suffix}")

    def _import_runner(self, module_path: Path, func_name: str) -> Callable[..., Dict[str, Any]]:
        """Import a function from an arbitrary path (folders may contain hyphens).

        Thread-safe: a lock prevents concurrent threads from racing to load the
        same module (which causes the 'Function not found' error when one thread
        stores a partial module in sys.modules before exec_module finishes).
        """
        cache_key = (str(module_path.resolve()), str(func_name))
        # Fast path: already cached (no lock needed for reads after first load)
        cached = self._runner_cache.get(cache_key)
        if cached is not None:
            return cached

        with self._import_lock:
            # Double-check inside the lock in case another thread loaded it first
            cached = self._runner_cache.get(cache_key)
            if cached is not None:
                return cached

            # Use a deterministic unique module name derived from full path to avoid collisions,
            # and ensure the module can retain its global singletons across calls.
            h = hashlib.md5(str(module_path.resolve()).encode("utf-8")).hexdigest()[:12]
            module_name = f"_skill_{module_path.stem}_{h}"

            # If already imported in this interpreter, reuse it.
            existing = sys.modules.get(module_name)
            if existing is not None:
                runner = getattr(existing, func_name, None)
                if runner is None:
                    raise AttributeError(f"Function {func_name} not found in {module_path}")
                self._runner_cache[cache_key] = runner
                return runner

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                raise ImportError(f"Cannot load module at {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                # Remove the partial module so subsequent threads can retry.
                sys.modules.pop(module_name, None)
                raise
            runner = getattr(module, func_name, None)
            if runner is None:
                raise AttributeError(f"Function {func_name} not found in {module_path}")
            self._runner_cache[cache_key] = runner
            return runner

    # Skill registrations
    def _register_detection(self) -> None:
        """
        Register detection skill. Backend is controlled by self._detection_backend:
          - "qwen3vl_api": Qwen3-VL-235B cloud API (detection.py)
          - "rex_omni":    Rex-Omni local model (detection_skill.py)

        Supports dynamic parameter extension (``allow_dynamic_params=True``).
        The LLM can pass extra strategy keys (e.g. ``num_targets_hint``,
        ``detection_strategy``, ``focus_region``) which are forwarded to the
        detection backend and injected into its prompt.
        """
        meta = self._load_skill_meta("detection")
        backend = self._detection_backend
        print(f"[SkillRegistry] Detection backend: {backend}")

        if backend == "rex_omni":
            module_path = self.skills_dir / "detection" / "scripts" / "detection_skill.py"
        else:
            module_path = self.skills_dir / "detection" / "scripts" / "detection.py"

        run_detection_skill: Optional[Callable[..., Dict[str, Any]]] = None

        def runner(
            image_path: str,
            task: str,
            object_name: Optional[str] = None,
            task_context: Optional[str] = None,
            reference_images: Optional[List[Dict[str, str]]] = None,
            _dynamic_params: Optional[Dict[str, Any]] = None,
        ):
            nonlocal run_detection_skill
            if run_detection_skill is None:
                run_detection_skill = self._import_runner(module_path, "run_detection_skill")

            mask_path = self._auto_reference_path("detection", image_path, "mask.png")
            vis_path = self._auto_reference_path("detection", image_path, "mask_vis.png")
            rex_vis_path = self._auto_reference_path("detection", image_path, "rex_vis.png")

            sam2_model_path = self._model_cfg.get("SAM2_MODEL_PATH") or "facebook/sam2.1-hiera-large"
            # Guard: ignore accidental paths to `skills/**/model` directories.
            try:
                sam2_p = Path(str(sam2_model_path))
                if sam2_p.is_dir() and sam2_p.name == "model" and self.skills_dir in sam2_p.parents:
                    sam2_model_path = "facebook/sam2.1-hiera-large"
            except Exception:
                pass

            sam3_model_path = self._model_cfg.get("SAM3_MODEL_PATH") or "facebook/sam3"
            try:
                sam3_p = Path(str(sam3_model_path))
                if sam3_p.is_dir() and sam3_p.name == "model" and self.skills_dir in sam3_p.parents:
                    sam3_model_path = "facebook/sam3"
            except Exception:
                pass

            if backend == "rex_omni":
                rex_model_path = self._model_cfg.get("REX_OMNI_MODEL_PATH") or "IDEA-Research/Rex-Omni"
                try:
                    rex_p = Path(str(rex_model_path))
                    if rex_p.is_dir() and rex_p.name == "model" and self.skills_dir in rex_p.parents:
                        rex_model_path = "IDEA-Research/Rex-Omni"
                except Exception:
                    pass
                return run_detection_skill(
                    image_path=image_path,
                    task=task,
                    object_name=object_name,
                    rex_model_path=rex_model_path,
                    sam2_model_path=sam2_model_path,
                    sam3_model_path=sam3_model_path,
                    save_mask_image=mask_path,
                    save_visualization=vis_path,
                    save_rex_visualization=rex_vis_path,
                    task_context=task_context,
                    dynamic_params=_dynamic_params,
                )
            else:
                # qwen3vl_api backend — forward dynamic params
                return run_detection_skill(
                    image_path=image_path,
                    task=task,
                    object_name=object_name,
                    sam2_model_path=sam2_model_path,
                    sam3_model_path=sam3_model_path,
                    save_mask_image=mask_path,
                    save_visualization=vis_path,
                    save_rex_visualization=rex_vis_path,
                    task_context=task_context,
                    reference_images=reference_images,
                    dynamic_params=_dynamic_params,
                )

        parameters = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the target scene image (must be the ORIGINAL, not cropped/generated)",
                },
                "task": {
                    "type": "string",
                    "description": "Affordance task description",
                },
                "object_name": {
                    "type": "string",
                    "description": "Target object name if already known",
                },
                "task_context": {
                    "type": "string",
                    "description": (
                        "Your text instructions for the detection model (Qwen). "
                        "Compose clear, actionable guidance: which part(s) to detect, "
                        "how many targets, spatial location, distinguishing features, "
                        "and any insights from your analysis. This is the primary way "
                        "you control detection accuracy."
                    ),
                },
                "reference_images": {
                    "type": "array",
                    "description": (
                        "Optional list of reference images for the detection model's "
                        "visual context. Each item has 'path' (file path) and 'label' "
                        "(description). Examples: dreamer interaction images, "
                        "commonsense template scene/GT pairs. Only include images "
                        "you judge genuinely helpful for this specific detection."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Image file path"},
                            "label": {"type": "string", "description": "What this image shows"},
                        },
                    },
                },
            },
            "required": ["image_path", "task"],
            "additionalProperties": True,
        }

        desc_suffix = "Rex-Omni" if backend == "rex_omni" else "Qwen3-VL-235B API"
        self.skills["detection"] = Skill(
            name="detection",
            description=(
                meta["description"]
                or f"Detection + segmentation ({desc_suffix} → SAM2 mask). "
                "Qwen predicts bbox/points from your instructions + images; "
                "SAM-2 generates the mask. You compose the full context via "
                "task_context (text) and reference_images (visual references)."
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=True,
        )

    def _register_web_search(self) -> None:
        """
        Register a web-search helper tool.

        This tool uses an LLM API to intelligently analyze task descriptions,
        reason about affordances, and infer key information (affordance_name,
        part_name, object_name) to help with subsequent detection tasks.

        Supports dynamic parameter extension.
        """
        meta = self._load_skill_meta("web_search")
        module_path = self.skills_dir / "web_search" / "scripts" / "web_search_skill.py"

        run_web_search_skill: Optional[Callable[..., Dict[str, Any]]] = None

        #  key ： schema （ LLM ），
        #  dynamic_params  run_web_search_skill。
        _STRATEGY_KEYS = {
            "search_focus", "target_part",
            "interaction_type", "knowledge_type",
        }

        def runner(
            question: str,
            task: Optional[str] = None,
            image_hint: Optional[str] = None,
            _dynamic_params: Optional[Dict[str, Any]] = None,
            **extra_kwargs,
        ) -> Dict[str, Any]:
            nonlocal run_web_search_skill
            if run_web_search_skill is None:
                run_web_search_skill = self._import_runner(module_path, "run_web_search_skill")

            #  schema  +   dynamic_params
            merged_dynamic = dict(_dynamic_params or {})
            for key in _STRATEGY_KEYS:
                val = extra_kwargs.pop(key, None)
                if val is not None:
                    merged_dynamic[key] = val
            #  kwargs 
            merged_dynamic.update(extra_kwargs)

            return run_web_search_skill(
                question=question,
                task=task,
                image_hint=image_hint,
                dynamic_params=merged_dynamic if merged_dynamic else None,
            )

        parameters = {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A specific question about the affordance you want to understand. "
                        "Be precise: instead of 'cup affordance', write "
                        "'What part of a cup should be grasped when picking it up to drink?'"
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The full task context (e.g., 'press down handle to turn on faucet'). "
                        "Provides background for the search."
                    ),
                },
                "image_hint": {
                    "type": "string",
                    "description": "Brief description of the scene or object appearance for additional context.",
                },
                "search_focus": {
                    "type": "string",
                    "description": (
                        "Main search direction — what angle to search from. "
                        "E.g., 'cup handle grasping ergonomics', 'faucet lever mechanism', "
                        "'scissors finger placement and grip'. Guides query generation."
                    ),
                },
                "target_part": {
                    "type": "string",
                    "description": (
                        "The object part you hypothesize is the answer. "
                        "E.g., 'handle', 'trigger', 'blade'. Helps focus the search."
                    ),
                },
                "interaction_type": {
                    "type": "string",
                    "description": (
                        "How humans physically interact with the object. "
                        "E.g., 'power grasp', 'pinch grip', 'push', 'twist', 'squeeze'."
                    ),
                },
                "knowledge_type": {
                    "type": "string",
                    "description": (
                        "Type of knowledge needed: 'structural' (parts layout), "
                        "'functional' (how it works), 'spatial' (where the part is)."
                    ),
                },
            },
            "required": ["question"],
            "additionalProperties": True,
        }

        self.skills["web_search"] = Skill(
            name=meta["name"] or "web_search",
            description=(
                meta["description"]
                or (
                    "Web search tool that retrieves textual knowledge about the object's "
                    "parts and affordance mechanism. Returns affordance_name, part_name, "
                    "object_name, reasoning, and textual evidence sources. "
                    "Accepts additional dynamic parameters."
                )
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=True,
        )

    def _register_rex_omni(self) -> None:
        meta = self._load_skill_meta("rex_omni")
        module_path = self.skills_dir / "rex_omni" / "scripts" / "rex_omni_skill.py"
        run_rex_omini_skill = self._import_runner(module_path, "run_rex_omini_skill")

        def runner(
            image_path: str,
            task: str,
            object_name: Optional[str] = None,
            task_context: Optional[str] = None,
            _dynamic_params: Optional[Dict[str, Any]] = None,
        ):
            vis_path = self._auto_reference_path("rex-omni", image_path, "rex_vis.png")
            model_path = self._model_cfg.get("REX_OMNI_MODEL_PATH") or _prefer_skill_model_dir(
                self.skills_dir, "rex-omni", "IDEA-Research/Rex-Omni"
            )
            return run_rex_omini_skill(
                image_path=image_path,
                task=task,
                object_name=object_name,
                model_path=model_path,
                save_visualization=vis_path,
                task_context=task_context,
                dynamic_params=_dynamic_params,
            )

        parameters = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the scene image",
                },
                "task": {
                    "type": "string",
                    "description": "Affordance task description",
                },
                "object_name": {
                    "type": "string",
                    "description": "Optional object name if already known",
                },
                "task_context": {
                    "type": "string",
                    "description": (
                        "Analysis context from the decision model. Rex-Omni uses this "
                        "to refine its grounding query for better localization."
                    ),
                },
            },
            "required": ["image_path", "task"],
            "additionalProperties": True,
        }

        self.skills["rex_omni"] = Skill(
            name="rex_omni",
            description=(
                meta["description"]
                or (
                    "Open-vocabulary part grounding with Rex-Omni. Detects bounding boxes "
                    "and key points for the target object part in the scene image. "
                    "Accepts additional dynamic parameters (e.g. target_part, "
                    "interaction_type, focus_region) to refine the grounding query."
                )
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=True,
        )

    def _register_dreamer(self) -> None:
        meta = self._load_skill_meta("dreamer")
        module_path = self.skills_dir / "dreamer" / "scripts" / "dreamer_skill.py"
        run_dreamer_skill = self._import_runner(module_path, "run_dreamer_skill")

        def runner(
            image_path: str,
            task: str,
            object_name: Optional[str] = None,
            editing_prompt: Optional[str] = None,
            _dynamic_params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            return run_dreamer_skill(
                image_path=image_path,
                task=task,
                object_name=object_name,
                editing_prompt=editing_prompt,
                dynamic_params=_dynamic_params,
            )

        parameters = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the scene image",
                },
                "task": {
                    "type": "string",
                    "description": "Affordance task description",
                },
                "object_name": {
                    "type": "string",
                    "description": "Optional object name hint if already known",
                },
                "editing_prompt": {
                    "type": "string",
                    "description": (
                        "Image editing prompt describing the human-object interaction "
                        "to visualize (e.g. 'A person gripping the handle of the knife "
                        "with their right hand'). When provided, the dreamer uses this "
                        "prompt directly instead of generating one. Write a concrete, "
                        "vivid description of how a person physically interacts with "
                        "the target part."
                    ),
                },
            },
            "required": ["image_path", "task"],
            "additionalProperties": True,
        }

        self.skills["dreamer"] = Skill(
            name=meta["name"] or "dreamer",
            description=(
                meta["description"]
                or (
                    "Interaction visualizer: generates edited images showing how a person "
                    "interacts with the target object, then analyzes the interaction to "
                    "identify target parts, interaction method, and spatial layout. "
                    "Returns generated images and detailed analysis. "
                    "Accepts additional dynamic parameters."
                )
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=True,
        )

    def _register_sam2(self) -> None:
        meta = self._load_skill_meta("sam2")
        module_path = self.skills_dir / "sam2" / "scripts" / "sam2_skill.py"
        run_sam2_skill = self._import_runner(module_path, "run_sam2_skill")

        def runner(
            image_path: str,
            bbox: Optional[str] = None,
            points: Optional[str] = None,
        ):
            mask_path = self._auto_reference_path("SAM-2", image_path, "mask.png")
            vis_path = self._auto_reference_path("SAM-2", image_path, "mask_vis.png")
            model_path = self._model_cfg.get("SAM2_MODEL_PATH") or _prefer_skill_model_dir(
                self.skills_dir, "SAM-2", "facebook/sam2.1-hiera-large"
            )
            return run_sam2_skill(
                image_path=image_path,
                bbox=bbox,
                points=points,
                model_path=model_path,
                save_mask_image=mask_path,
                save_visualization=vis_path,
            )

        parameters = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the scene image",
                },
                "bbox": {
                    "type": "string",
                    "description": "Normalized bbox JSON string from rex_omni (e.g. \"[0.1,0.2,0.5,0.6]\")",
                },
                "points": {
                    "type": "string",
                    "description": "Normalized points JSON string from rex_omni (e.g. \"[[0.3,0.4]]\")",
                },
            },
            "required": ["image_path", "points"],
        }

        self.skills["sam2"] = Skill(
            name="sam2",
            description=meta["description"] or "Segmentation with SAM2 using bbox/points",
            parameters=parameters,
            runner=runner,
        )

    def _register_zoom_in(self) -> None:
        """
        Register zoom-in helper tool.

        This tool crops and saves the zoomed-in region of an image to
        skills/zoom-in/reference/ for debugging and visualization,
        and returns the cropped image path to the LLM for analysis.

        Supports dynamic parameter extension.
        """
        meta = self._load_skill_meta("zoom_in")
        reference_dir = self.skills_dir / "zoom_in" / "reference"

        def runner(
            image_path: str,
            bbox: str,
            _dynamic_params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            import json as _json
            from datetime import datetime

            # Parse bbox - supports both JSON string and plain list string
            try:
                if isinstance(bbox, str):
                    bbox_coords = _json.loads(bbox)
                else:
                    bbox_coords = bbox
                if not isinstance(bbox_coords, (list, tuple)) or len(bbox_coords) != 4:
                    return {"error": f"Invalid bbox format: expected [x_min, y_min, x_max, y_max], got {bbox}"}
                x_min, y_min, x_max, y_max = [float(v) for v in bbox_coords]
            except Exception as e:
                return {"error": f"Failed to parse bbox '{bbox}': {e}"}

            # Open image and crop
            try:
                from PIL import Image as PILImage
                img = PILImage.open(image_path).convert("RGB")
                w, h = img.size

                # Convert normalized [0,1] coords to pixel coords
                px_left = int(x_min * w) if x_min <= 1.0 else int(x_min)
                px_top = int(y_min * h) if y_min <= 1.0 else int(y_min)
                px_right = int(x_max * w) if x_max <= 1.0 else int(x_max)
                px_bottom = int(y_max * h) if y_max <= 1.0 else int(y_max)

                # Clamp to image bounds
                px_left = max(0, min(w, px_left))
                px_top = max(0, min(h, px_top))
                px_right = max(0, min(w, px_right))
                px_bottom = max(0, min(h, px_bottom))

                if px_right <= px_left or px_bottom <= px_top:
                    return {"error": f"Invalid crop region: ({px_left},{px_top})-({px_right},{px_bottom})"}

                cropped = img.crop((px_left, px_top, px_right, px_bottom))

                # Save to reference directory
                reference_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(image_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{stem}_zoom_{timestamp}.png"
                save_path = str(reference_dir / save_name)
                cropped.save(save_path)

                print(f"[zoom_in] Cropped region ({px_left},{px_top})-({px_right},{px_bottom}) from {w}x{h} image")
                print(f"[zoom_in] Saved zoomed image to: {save_path}")
                print(f"[zoom_in] Cropped size: {cropped.size[0]}x{cropped.size[1]}")

                result = {
                    "image_path": image_path,
                    "bbox": bbox,
                    "zoomed_image_path": save_path,
                    "crop_region_pixel": [px_left, px_top, px_right, px_bottom],
                    "original_size": [w, h],
                    "cropped_size": [cropped.size[0], cropped.size[1]],
                    "note": "Zoomed-in image saved for visual analysis.",
                }

                # Forward dynamic params info to result for context
                if _dynamic_params:
                    zoom_purpose = _dynamic_params.get("zoom_purpose", "")
                    if zoom_purpose:
                        result["zoom_purpose"] = zoom_purpose
                        print(f"[zoom_in] Purpose: {zoom_purpose}")

                return result
            except FileNotFoundError:
                return {"error": f"Image not found: {image_path}"}
            except Exception as e:
                return {"error": f"Failed to crop image: {e}"}

        parameters = {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the scene image",
                },
                "bbox": {
                    "type": "string",
                    "description": 'Normalized crop region JSON string, e.g. "[0.2, 0.3, 0.6, 0.8]"',
                },
            },
            "required": ["image_path", "bbox"],
            "additionalProperties": True,
        }

        self.skills["zoom_in"] = Skill(
            name="zoom_in",
            description=(
                meta["description"]
                or "Crops and enlarges a region of the image for closer visual inspection. "
                "Accepts additional dynamic parameters."
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=True,
        )

    def _register_load_skill_doc(self) -> None:
        """Register a meta tool to load SKILL.md content on demand."""

        def runner(skill_name: str, max_chars: int = 6000) -> Dict[str, Any]:
            return self.get_skill_markdown(skill_name=skill_name, max_chars=int(max_chars))
        parameters = {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Skill name to load in detail: detection, web_search, dreamer, zoom_in",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return from SKILL.md to control context size",
                },
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        }
        self.skills["load_skill_doc"] = Skill(
            name="load_skill_doc",
            description=(
                "Loads detailed guidance from a tool's SKILL.md on demand. "
                "Use this only when you need deeper instructions beyond the one-line skill index."
            ),
            parameters=parameters,
            runner=runner,
            allow_dynamic_params=False,
        )
