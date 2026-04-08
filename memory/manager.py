import json
import os
import sys
import hashlib
import base64
import io
import re
import time
import shutil
import threading
import requests
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from prompts.memory import (
    EXPERIENCE_DISTILL_PROMPT_TEMPLATE,
    MEMORY_DEDUP_PROMPT_TEMPLATE,
    MEMORY_EVICTION_PROMPT_TEMPLATE,
)



@dataclass
class MemoryEntry:
    """（inner memory） — """
    sample_id: str  # 
    image_path: str  # 
    task: str  # 
    object_name: Optional[str] = None  # 
    
    # Skill 
    tool_calls: List[Dict[str, Any]] = None  #  tool call 
    tool_summaries: List[str] = None  #  tool 
    
    # （ iou  precision_at_50， memory ）
    evaluation_metrics: Optional[Dict[str, float]] = None
    mask_path: Optional[str] = None  #  mask 
    
    reasoning_context: Optional[str] = None  # 
    task_context: Optional[str] = None  #  detection  task_context
    
    # LLM （ content）
    llm_reasoning: List[str] = None  # 
    
    timestamp: str = None  # 
    
    feature_vector: Optional[List[float]] = None  # SigLIP2 
    dino_feature_vector: Optional[List[float]] = None  # DINOv2 
    
    strategy_reasoning: Optional[str] = None  # （）
    strategy_reflection: Optional[str] = None  # （）
    dynamic_params_used: Optional[Dict[str, Any]] = None  # 
    decision_trace: List[str] = None  # （ llm_reasoning ）
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_summaries is None:
            self.tool_summaries = []
        if self.llm_reasoning is None:
            self.llm_reasoning = []
        if self.decision_trace is None:
            self.decision_trace = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CompactMemoryEntry:
    """（outer memory） — 
    
    ，，。
    ，。
    """
    sample_id: str
    task: str
    object_name: Optional[str] = None
    summary: str = ""  # 
    tool_chain: str = ""  # （， "dreamer → detection"）
    strategy_reasoning: Optional[str] = None  # （）
    iou: float = 0.0
    timestamp: str = ""
    tags: List[str] = field(default_factory=list)  # （）
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class StrategyInsight:
    """
    
    ，（），
    （/），（）。
    """
    insight_id: str  # 
    category: str  # : "tool_strategy" | "task_pattern" | "failure_lesson" | "general_wisdom"
    content: str  # （）
    evidence_samples: List[str] = field(default_factory=list)  #  ID
    confidence: float = 0.5  #  [0, 1]，/
    avg_iou: float = 0.0  #  IoU（，）
    success_count: int = 0  # 
    failure_count: int = 0  # 
    tags: List[str] = field(default_factory=list)  # （）
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.evidence_samples is None:
            self.evidence_samples = []
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


# （Experience Pool）

class ExperiencePool:
    """ — 、、
    
    ：
    - ，/
    - ，、
    - ，
    
     {persist_dir}/experience_pool.json
    """
    
    def __init__(
        self,
        persist_path: Optional[Path] = None,
        api_caller: Optional[Callable] = None,
        model: str = "gpt-4o-mini",
        max_insights: int = 50,
        distill_interval: int = 5,
    ):
        self.persist_path = persist_path
        self._api_caller = api_caller
        self._model = model
        self.max_insights = max_insights
        self.distill_interval = distill_interval  #  N 
        
        self.insights: List[StrategyInsight] = []
        self._pending_observations: List[Dict[str, Any]] = []
        self._total_samples_seen: int = 0
        
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def set_api_caller(self, api_caller: Callable) -> None:
        self._api_caller = api_caller
    
    def add_observation(
        self,
        sample_id: str,
        task: str,
        object_name: Optional[str],
        tool_chain: str,
        iou: float,
        strategy_reasoning: Optional[str] = None,
        dynamic_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """，"""
        self._pending_observations.append({
            "sample_id": sample_id,
            "task": task,
            "object_name": object_name or "",
            "tool_chain": tool_chain,
            "iou": iou,
            "strategy_reasoning": strategy_reasoning or "",
            "dynamic_params": dynamic_params or {},
            "timestamp": datetime.now().isoformat(),
        })
        self._total_samples_seen += 1
        
        if len(self._pending_observations) >= self.distill_interval:
            self.distill_insights()
    
    def distill_insights(self) -> None:
        """ LLM  pending observations 
        
        Only observations with verified IoU (> 0) are used for insight
        extraction. Observations with IoU = 0 are placeholders that haven't
        received real metrics yet — including them would distill noise.
        """
        if not self._pending_observations:
            return
        
        # Separate verified (real IoU) from unverified (IoU=0 placeholder)
        verified = [o for o in self._pending_observations if o["iou"] > 0]
        unverified = [o for o in self._pending_observations if o["iou"] <= 0]
        
        if unverified:
            print(f"[ExperiencePool] Skipping {len(unverified)} unverified observations "
                  f"(IoU=0, no reward signal)")
        
        if not verified:
            self._pending_observations = []
            self._save()
            return
        
        # Only distill from verified observations
        self._pending_observations = verified
        
        if self._api_caller is not None:
            self._distill_with_llm()
        else:
            self._distill_rule_based()
        
        self._pending_observations = []
        self._save()
    
    def _distill_with_llm(self) -> None:
        """LLM """
        existing_summary = ""
        if self.insights:
            lines = []
            for ins in self.insights:
                lines.append(
                    f"[{ins.insight_id}] ({ins.category}, conf={ins.confidence:.2f}) {ins.content}"
                )
            existing_summary = "\n".join(lines)
        
        obs_lines = []
        for obs in self._pending_observations:
            obs_lines.append(
                f"- task=\"{obs['task'][:80]}\" obj=\"{obs['object_name']}\" "
                f"tools={obs['tool_chain']} IoU={obs['iou']:.3f} "
                f"reason=\"{obs['strategy_reasoning'][:100]}\""
            )
        obs_text = "\n".join(obs_lines)
        
        existing_block = ""
        if existing_summary:
            existing_block = f"Existing insights:\n{existing_summary}\n\n"
        
        prompt = EXPERIENCE_DISTILL_PROMPT_TEMPLATE.format(
            existing_block=existing_block,
            num_observations=len(self._pending_observations),
            obs_text=obs_text,
            max_insights=self.max_insights,
        )
        
        try:
            response = self._api_caller(
                "chat/completions",
                {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            #  JSON 
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                new_insights = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    ins = StrategyInsight(
                        insight_id=str(item.get("id", f"ins_{len(new_insights)}")),
                        category=str(item.get("category", "general_wisdom")),
                        content=str(item.get("content", "")),
                        confidence=float(item.get("confidence", 0.5)),
                        tags=item.get("tags", []),
                        evidence_samples=[o["sample_id"] for o in self._pending_observations],
                        avg_iou=float(np.mean([o["iou"] for o in self._pending_observations])),
                        updated_at=datetime.now().isoformat(),
                    )
                    new_insights.append(ins)
                
                if new_insights:
                    self.insights = new_insights[:self.max_insights]
                    print(f"[ExperiencePool] LLM distilled {len(self.insights)} insights from {len(self._pending_observations)} observations")
                    return
        except Exception as e:
            print(f"[ExperiencePool] LLM distillation failed ({e}), falling back to rule-based")
        
        self._distill_rule_based()
    
    def _distill_rule_based(self) -> None:
        """（LLM ）
        
        Uses adaptive thresholds based on the distribution of observed IoUs
        rather than fixed values, so it works across datasets with different
        difficulty levels.
        """
        # Collect all observed IoUs (pending + historical insight averages) 
        # to compute adaptive thresholds
        all_ious = [obs["iou"] for obs in self._pending_observations if obs["iou"] > 0]
        for ins in self.insights:
            if ins.avg_iou > 0:
                all_ious.append(ins.avg_iou)
        
        if all_ious:
            sorted_ious = sorted(all_ious)
            n = len(sorted_ious)
            high_threshold = sorted_ious[min(n - 1, 3 * n // 4)]  # 75th percentile
            low_threshold = sorted_ious[max(0, n // 4)]  # 25th percentile
        else:
            high_threshold = 0.5
            low_threshold = 0.2
        
        for obs in self._pending_observations:
            iou = obs["iou"]
            chain = obs["tool_chain"]
            task_snippet = obs["task"][:60]
            
            # Adaptive rule: compare against the distribution
            if iou >= high_threshold and iou > 0:
                content = (
                    f"Strategy '{chain}' achieved IoU={iou:.3f} (above p75={high_threshold:.3f}) "
                    f"on task similar to '{task_snippet}'. This approach was relatively effective."
                )
                category = "tool_strategy"
                conf = min(0.9, 0.5 + iou * 0.4)
            elif iou <= low_threshold:
                content = (
                    f"Strategy '{chain}' achieved IoU={iou:.3f} (below p25={low_threshold:.3f}) "
                    f"on task similar to '{task_snippet}'. Consider alternative approaches."
                )
                category = "failure_lesson"
                conf = 0.4
            else:
                continue  # 
            
            merged = False
            for existing in self.insights:
                if existing.category == category and chain in existing.content:
                    existing.evidence_samples.append(obs["sample_id"])
                    n = len(existing.evidence_samples)
                    existing.avg_iou = (existing.avg_iou * (n - 1) + iou) / n
                    existing.confidence = min(0.95, existing.confidence + 0.05)
                    existing.updated_at = datetime.now().isoformat()
                    if iou >= high_threshold:
                        existing.success_count += 1
                    else:
                        existing.failure_count += 1
                    merged = True
                    break
            
            if not merged:
                ins = StrategyInsight(
                    insight_id=f"ins_{len(self.insights)}_{hashlib.md5(content.encode()).hexdigest()[:6]}",
                    category=category,
                    content=content,
                    confidence=conf,
                    evidence_samples=[obs["sample_id"]],
                    avg_iou=iou,
                    success_count=1 if iou >= high_threshold else 0,
                    failure_count=0 if iou >= high_threshold else 1,
                    tags=[t for t in chain.replace("→", ",").replace(" ", "").split(",") if t],
                )
                self.insights.append(ins)
        
        # Generate a tool usage distribution insight from all observations
        all_tools_seen: Dict[str, int] = {}
        for obs in self._pending_observations:
            for t in obs["tool_chain"].replace("→", ",").replace(" ", "").split(","):
                t = t.strip()
                if t and t != "detection":
                    all_tools_seen[t] = all_tools_seen.get(t, 0) + 1
        
        # Also count from existing insights
        total_obs = self._total_samples_seen
        available_helpers = {"web_search", "dreamer", "zoom_in"}
        unused_in_batch = available_helpers - set(all_tools_seen.keys())
        
        # Update or create a tool-diversity insight if some tools are rarely used
        if unused_in_batch and total_obs >= 10:
            unused_str = ", ".join(sorted(unused_in_batch))
            diversity_content = (
                f"Among recent observations, {unused_str} "
                f"{'was' if len(unused_in_batch) == 1 else 'were'} not used. "
                f"Each tool provides unique value: web_search brings external knowledge, "
                f"dreamer generates visual interaction hypotheses, zoom_in reveals fine details. "
                f"Consider whether untried tools could improve results for difficult cases."
            )
            # Check if a similar diversity insight already exists
            diversity_exists = any(
                "not used" in ins.content and ins.category == "general_wisdom"
                for ins in self.insights
            )
            if not diversity_exists:
                ins = StrategyInsight(
                    insight_id=f"ins_diversity_{hashlib.md5(diversity_content.encode()).hexdigest()[:6]}",
                    category="general_wisdom",
                    content=diversity_content,
                    confidence=0.5,
                    evidence_samples=[obs["sample_id"] for obs in self._pending_observations],
                    avg_iou=float(np.mean(all_ious)) if all_ious else 0.0,
                    tags=["tool_diversity", "exploration"],
                )
                self.insights.append(ins)
        
        # ：， top N
        self.insights.sort(key=lambda x: x.confidence, reverse=True)
        self.insights = self.insights[:self.max_insights]
        print(f"[ExperiencePool] Rule-based: {len(self.insights)} insights after distillation "
              f"(adaptive thresholds: high={high_threshold:.3f}, low={low_threshold:.3f})")
    
    def get_relevant_insights(
        self,
        task: str,
        object_name: Optional[str] = None,
        top_k: int = 10,
    ) -> List[StrategyInsight]:
        """"""
        if not self.insights:
            return []
        
        task_words = set(task.lower().split())
        if object_name:
            task_words.update(object_name.lower().split())
        
        scored = []
        for ins in self.insights:
            score = ins.confidence
            for tag in ins.tags:
                if tag.lower() in task_words:
                    score += 0.15
            content_words = set(ins.content.lower().split())
            overlap = len(task_words & content_words)
            if overlap > 0:
                score += 0.05 * overlap
            scored.append((score, ins))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ins for _, ins in scored[:top_k]]
    
    def format_for_context(
        self,
        task: str,
        object_name: Optional[str] = None,
        top_k: int = 8,
        token_budget: int = 800,
    ) -> str:
        """"""
        relevant = self.get_relevant_insights(task, object_name, top_k)
        if not relevant:
            return ""
        
        lines = [
            "🧠 **Experience Pool** (accumulated insights from past executions; "
            "IoU = mask overlap, higher is better — what counts as 'high' depends "
            "on the dataset):\n"
        ]
        chars_per_token = 3.5
        tokens_used = len(lines[0]) / chars_per_token
        
        for ins in relevant:
            evidence_note = ""
            if ins.evidence_samples:
                n_evidence = len(ins.evidence_samples)
                if ins.avg_iou > 0:
                    evidence_note = f", samples={n_evidence}, avg_IoU={ins.avg_iou:.3f}"
                else:
                    evidence_note = f", samples={n_evidence}"
            line = f"- [conf={ins.confidence:.2f}{evidence_note}] {ins.content}"
            line_tokens = len(line) / chars_per_token
            if tokens_used + line_tokens > token_budget:
                break
            lines.append(line)
            tokens_used += line_tokens
        
        if len(lines) <= 1:
            return ""
        
        lines.append("")
        return "\n".join(lines)
    
    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            data = {
                "version": 1,
                "total_samples_seen": self._total_samples_seen,
                "last_updated": datetime.now().isoformat(),
                "insights": [asdict(ins) for ins in self.insights],
                "pending_observations": self._pending_observations,
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ExperiencePool] Failed to save: {e}")
    
    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._total_samples_seen = data.get("total_samples_seen", 0)
            self._pending_observations = data.get("pending_observations", [])
            self.insights = []
            for d in data.get("insights", []):
                # Backward compat: old files used "avg_metric", now "avg_iou"
                if "avg_metric" in d and "avg_iou" not in d:
                    d["avg_iou"] = d.pop("avg_metric")
                elif "avg_metric" in d:
                    d.pop("avg_metric")
                self.insights.append(StrategyInsight(**d))
            print(f"[ExperiencePool] Loaded {len(self.insights)} insights, {self._total_samples_seen} samples seen")
        except Exception as e:
            print(f"[ExperiencePool] Failed to load: {e}")
    
    def clear(self) -> None:
        """"""
        n = len(self.insights)
        self.insights = []
        self._pending_observations = []
        self._total_samples_seen = 0
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
        print(f"[ExperiencePool] Cleared {n} insights")


class MemoryManager:
    """
    
    （inner memory）：， + ，
    （outer memory）：，，
    （experience pool）：，
    """
    
    #  token  token （ 4， 1.5，）
    _CHARS_PER_TOKEN = 3.5
    
    def __init__(
        self,
        max_size: int = 80,
        eviction_strategy: str = "model_decision",
        similarity_threshold: float = 0.7,
        duplicate_similarity_threshold: Optional[float] = None,
        retrieval_top_k: int = 20,
        enqueue_dedupe_top_k: int = 5,
        persist_dir: Optional[str] = None,
        api_caller: Optional[Callable] = None,
        eviction_model: Optional[str] = None,
        context_token_budget: int = 3000,
        experience_pool_max: int = 50,
        experience_distill_interval: int = 5,
        enable_commonsense_templates: bool = True,
        commonsense_max_per_pair: int = 2,
        commonsense_max_total: Optional[int] = None,
        datasets_root: Optional[str] = None,
        commonsense_templates_dir: Optional[str] = None,
        commonsense_auto_build: bool = False,
    ):
        """
        Args:
            max_size: 
            eviction_strategy:  ("similarity", "fifo", "model_decision")
            similarity_threshold: 
            duplicate_similarity_threshold: 
            retrieval_top_k:  memory  TopK
            enqueue_dedupe_top_k:  TopK
            persist_dir: 
            api_caller: LLM API 
            eviction_model: /
            context_token_budget: memory  token 
            experience_pool_max: 
            experience_distill_interval:  N 
            enable_commonsense_templates: （）
            commonsense_max_per_pair:  object-part 
            commonsense_max_total: （<=0  None ）
            datasets_root: （ datasets）
            commonsense_templates_dir: （ commonsense_templates）
            commonsense_auto_build: ， datasets （ False）
        """
        self.max_size = max_size
        self.eviction_strategy = eviction_strategy
        self.similarity_threshold = similarity_threshold
        self.duplicate_similarity_threshold = (
            duplicate_similarity_threshold
            if duplicate_similarity_threshold is not None
            else similarity_threshold
        )
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        self.enqueue_dedupe_top_k = max(1, int(enqueue_dedupe_top_k))
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._api_caller = api_caller
        self._eviction_model = eviction_model or "gpt-4o-mini"
        self.context_token_budget = context_token_budget
        # If LLM dedupe keeps failing (permission/quota), disable it and use threshold-only fallback.
        self._llm_dedupe_disabled = False
        self.enable_commonsense_templates = bool(enable_commonsense_templates)
        self.commonsense_max_per_pair = max(1, int(commonsense_max_per_pair))
        if commonsense_max_total is None:
            self.commonsense_max_total = None
        else:
            max_total = int(commonsense_max_total)
            self.commonsense_max_total = None if max_total <= 0 else max(1, max_total)
        _project_root = Path(__file__).resolve().parent.parent  # A-Harness/
        self.datasets_root = Path(datasets_root) if datasets_root else (_project_root / "data")
        self.commonsense_templates_dir = (
            Path(commonsense_templates_dir)
            if commonsense_templates_dir
            else (_project_root / "memory" / "commonsense_templates")
        )
        self.commonsense_auto_build = bool(commonsense_auto_build)
        
        self._memory: List[MemoryEntry] = []
        
        self._outer_memory: List[CompactMemoryEntry] = []
        self._last_outer_supplements: List[CompactMemoryEntry] = []  # （）
        self._thread_local = threading.local()  # per-thread storage to avoid race conditions
        #  skill （ + ）
        self._skill_profiles: Dict[str, Dict[str, Any]] = {}
        # ：/ (task, image, GT) 
        self._commonsense_templates: List[Dict[str, Any]] = []
        self._commonsense_ready: bool = False
        # CLIP image embeddings for hybrid (image+text) retrieval
        self._clip_embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self._clip_index: Optional[List[str]] = None  # template_id list
        self._clip_id_to_row: Optional[Dict[str, int]] = None  # tid -> row
        self._clip_model = None  # lazy-loaded CLIPModel
        self._clip_processor = None  # lazy-loaded CLIPProcessor
        self._clip_lock = threading.Lock()  # thread safety for SigLIP2 load/inference
        # DINOv2 image embeddings for enhanced visual retrieval
        self._dino_embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self._dino_index: Optional[List[str]] = None  # template_id list
        self._dino_id_to_row: Optional[Dict[str, int]] = None  # tid -> row
        self._dino_model = None  # lazy-loaded DINOv2
        self._dino_processor = None  # lazy-loaded DINOv2 processor
        self._dino_lock = threading.Lock()  # thread safety for DINOv2 load/inference
        
        exp_path = (self.persist_dir / "experience_pool.json") if self.persist_dir else None
        self.experience_pool = ExperiencePool(
            persist_path=exp_path,
            api_caller=api_caller,
            model=self._eviction_model,
            max_insights=experience_pool_max,
            distill_interval=experience_distill_interval,
        )
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
            # load prebuilt commonsense bank (if provided)
            self._load_commonsense_templates_from_disk()
    
    def set_api_caller(self, api_caller: Callable) -> None:
        """ API （ AffordanceAgent ）"""
        self._api_caller = api_caller
        self.experience_pool.set_api_caller(api_caller)

    # Skill profile maintenance (continuous)
    def _extract_result_features_from_summary(self, summary: str) -> List[str]:
        """Extract compact result features from tool summary text."""
        if not summary:
            return []
        feats: List[str] = []
        lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        for ln in lines:
            low = ln.lower()
            if "error" in low or "❌" in ln:
                feats.append("error_signal")
            if "reasoning" in low:
                feats.append("reasoning_output")
            if "image" in low or "mask" in low or "visual" in low:
                feats.append("visual_artifact")
            if "target" in low or "part" in low:
                feats.append("target_localization")
            if "source" in low or "search" in low:
                feats.append("external_evidence")
            m = re.match(r"^[\-\*\u2022]+\s*([^:]{2,40}):", ln)
            if m:
                feats.append(m.group(1).strip().lower().replace(" ", "_"))
        return feats

    def _iter_entry_tool_records(self, entry: MemoryEntry) -> List[Dict[str, Any]]:
        """Flatten one entry into aligned tool call + summary records."""
        records: List[Dict[str, Any]] = []
        calls = entry.tool_calls or []
        summaries = entry.tool_summaries or []
        n_calls = len(calls)
        for idx, call in enumerate(calls):
            tool = call.get("name", "?")
            args = call.get("arguments", {}) or {}
            summary = summaries[idx] if idx < len(summaries) else ""
            position = "middle"
            if idx == 0:
                position = "first"
            if idx == n_calls - 1:
                position = "last" if n_calls > 1 else "first_last"
            records.append(
                {
                    "tool": tool,
                    "args": args,
                    "summary": summary,
                    "position": position,
                    "entry_iou": (
                        entry.evaluation_metrics.get("iou")
                        if entry.evaluation_metrics and isinstance(entry.evaluation_metrics.get("iou"), (int, float))
                        else None
                    ),
                    "sample_id": entry.sample_id,
                }
            )
        return records

    def _rebuild_skill_profiles(self) -> None:
        """
        Rebuild per-skill profiles from current inner+outer memory.
        This runs after add/update/load/eviction to keep profiles fresh.
        """
        profile_tmp: Dict[str, Dict[str, Any]] = {}

        def _get_profile(name: str) -> Dict[str, Any]:
            if name not in profile_tmp:
                profile_tmp[name] = {
                    "skill_name": name,
                    "total_calls": 0,
                    "sample_ids": set(),
                    "position_counter": Counter(),
                    "arg_key_counter": Counter(),
                    "result_feature_counter": Counter(),
                    "verified_ious": [],
                    "error_calls": 0,
                    "last_updated": datetime.now().isoformat(),
                }
            return profile_tmp[name]

        # Inner memory: richest source
        for entry in self._memory:
            for rec in self._iter_entry_tool_records(entry):
                name = rec["tool"]
                p = _get_profile(name)
                p["total_calls"] += 1
                p["sample_ids"].add(rec["sample_id"])
                p["position_counter"][rec["position"]] += 1

                for k in (rec["args"] or {}).keys():
                    if k == "image_path":
                        continue
                    p["arg_key_counter"][str(k)] += 1

                features = self._extract_result_features_from_summary(rec["summary"])
                if not features and rec["summary"]:
                    features = ["summary_present"]
                for ft in features:
                    p["result_feature_counter"][ft] += 1
                if any(ft == "error_signal" for ft in features):
                    p["error_calls"] += 1

                if isinstance(rec["entry_iou"], (int, float)):
                    p["verified_ious"].append(float(rec["entry_iou"]))

        # Outer memory: keep long-term call-count continuity using tool_chain only
        for oe in self._outer_memory:
            tools = [t.strip() for t in (oe.tool_chain or "").replace("→", ",").split(",") if t.strip()]
            for t in tools:
                p = _get_profile(t)
                p["total_calls"] += 1
                p["sample_ids"].add(oe.sample_id)
                p["position_counter"]["outer_memory"] += 1
                if isinstance(oe.iou, (int, float)) and oe.iou > 0:
                    p["verified_ious"].append(float(oe.iou))

        # Finalize into JSON-serializable compact profiles
        finalized: Dict[str, Dict[str, Any]] = {}
        for name, p in profile_tmp.items():
            ious = p["verified_ious"]
            avg_iou = float(np.mean(ious)) if ious else None
            finalized[name] = {
                "skill_name": name,
                "total_calls": int(p["total_calls"]),
                "sample_count": int(len(p["sample_ids"])),
                "position_distribution": dict(p["position_counter"]),
                "top_arg_keys": [k for k, _ in p["arg_key_counter"].most_common(6)],
                "top_result_features": [k for k, _ in p["result_feature_counter"].most_common(6)],
                "error_rate": (float(p["error_calls"]) / float(p["total_calls"])) if p["total_calls"] > 0 else 0.0,
                "verified_case_count": int(len(ious)),
                "avg_iou_verified": avg_iou,
                "last_updated": datetime.now().isoformat(),
            }

        self._skill_profiles = finalized

    def _format_skill_profiles_for_context(self, token_budget: int = 500) -> str:
        """Format per-skill maintained profiles for decision-model context."""
        if not self._skill_profiles:
            return ""
        lines = ["🧩 **Skill Profiles (continuously maintained):**"]
        tokens_used = self._estimate_tokens(lines[0])
        # Sort by usage count desc
        items = sorted(
            self._skill_profiles.values(),
            key=lambda x: x.get("total_calls", 0),
            reverse=True,
        )
        for prof in items:
            skill = prof.get("skill_name", "unknown")
            calls = int(prof.get("total_calls", 0))
            samples = int(prof.get("sample_count", 0))
            avg_iou = prof.get("avg_iou_verified")
            iou_str = f"{avg_iou:.3f}" if isinstance(avg_iou, (int, float)) else "N/A"
            args = ", ".join(prof.get("top_arg_keys", [])[:3]) or "none"
            feats = ", ".join(prof.get("top_result_features", [])[:3]) or "none"
            err = float(prof.get("error_rate", 0.0))
            line = (
                f"- {skill}: used {calls} calls across {samples} samples; "
                f"avg IoU(verified)={iou_str}; error_rate={err:.2f}; "
                f"usage_features=[{args}]; result_features=[{feats}]"
            )
            t = self._estimate_tokens(line)
            if tokens_used + t > token_budget:
                break
            lines.append(line)
            tokens_used += t
        if len(lines) == 1:
            return ""
        lines.append("")
        return "\n".join(lines)

    # Common-sense template bank (dataset exemplars)
    def _commonsense_file_path(self) -> Optional[Path]:
        if not self.commonsense_templates_dir:
            return None
        return self.commonsense_templates_dir / "commonsense_templates.json"

    def _pair_key(self, object_name: str, affordance_part: str) -> str:
        return f"{object_name.strip().lower()}::{affordance_part.strip().lower()}"

    def _safe_text(self, s: Any) -> str:
        return str(s or "").strip()

    def _resolve_template_asset_path(self, p: str) -> str:
        """Resolve template asset path; support relative path under templates_dir."""
        p = self._safe_text(p)
        if not p:
            return ""
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str((self.commonsense_templates_dir / pp).resolve())

    def _load_clip_embeddings(self) -> bool:
        """Load pre-computed CLIP embeddings from disk (if available)."""
        emb_path = self.commonsense_templates_dir / "clip_embeddings.npy"
        idx_path = self.commonsense_templates_dir / "clip_index.json"
        if not emb_path.exists() or not idx_path.exists():
            return False
        try:
            self._clip_embeddings = np.load(str(emb_path)).astype(np.float32)
            with open(idx_path, "r", encoding="utf-8") as f:
                self._clip_index = json.load(f)
            self._clip_id_to_row = {
                tid: i for i, tid in enumerate(self._clip_index)
            }
            print(
                f"[Memory] Loaded CLIP embeddings: {self._clip_embeddings.shape[0]} vectors, "
                f"dim={self._clip_embeddings.shape[1]}"
            )
            return True
        except Exception as e:
            print(f"[Memory] Failed to load CLIP embeddings: {e}")
            self._clip_embeddings = None
            self._clip_index = None
            self._clip_id_to_row = None
            return False

    def _load_dino_embeddings(self) -> bool:
        """Load pre-computed DINOv2 embeddings from disk (if available)."""
        emb_path = self.commonsense_templates_dir / "dino_embeddings.npy"
        idx_path = self.commonsense_templates_dir / "dino_index.json"
        if not emb_path.exists() or not idx_path.exists():
            return False
        try:
            self._dino_embeddings = np.load(str(emb_path)).astype(np.float32)
            with open(idx_path, "r", encoding="utf-8") as f:
                self._dino_index = json.load(f)
            self._dino_id_to_row = {
                tid: i for i, tid in enumerate(self._dino_index)
            }
            print(
                f"[Memory] Loaded DINOv2 embeddings: {self._dino_embeddings.shape[0]} vectors, "
                f"dim={self._dino_embeddings.shape[1]}"
            )
            return True
        except Exception as e:
            print(f"[Memory] Failed to load DINOv2 embeddings: {e}")
            self._dino_embeddings = None
            self._dino_index = None
            self._dino_id_to_row = None
            return False

    def _ensure_clip_model(self):
        """Lazy-load SigLIP2 vision encoder for query-time image embedding."""
        if self._clip_model is not None:
            return
        with self._clip_lock:
            if self._clip_model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
                model_name = "google/siglip2-base-patch16-384"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._clip_model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
                self._clip_processor = AutoProcessor.from_pretrained(model_name)
                self._clip_device = device
                print(f"[Memory] SigLIP2 model loaded on {device}")
            except Exception as e:
                print(f"[Memory] Could not load SigLIP2 model: {e}")
                self._clip_model = False  # sentinel: tried and failed

    def _compute_clip_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Compute a normalised SigLIP2 image embedding for a single image."""
        self._ensure_clip_model()
        if self._clip_model is False or self._clip_model is None:
            return None
        try:
            import torch
            img = Image.open(image_path).convert("RGB")
            with self._clip_lock:
                inputs = self._clip_processor(images=img, return_tensors="pt").to(self._clip_device)
                with torch.no_grad():
                    emb = self._clip_model.get_image_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"[Memory] SigLIP2 embedding failed for {image_path}: {e}")
            return None

    def _compute_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute a normalised SigLIP2 text embedding.

        Since SigLIP2 maps text and images into the same vector space,
        the resulting embedding can be compared directly with image embeddings
        via cosine similarity (text-to-image retrieval).
        """
        self._ensure_clip_model()
        if self._clip_model is False or self._clip_model is None:
            return None
        try:
            import torch
            with self._clip_lock:
                inputs = self._clip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self._clip_device)
                with torch.no_grad():
                    emb = self._clip_model.get_text_features(**inputs)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"[Memory] SigLIP2 text embedding failed: {e}")
            return None

    def _compute_clip_similarities(self, query_emb: np.ndarray) -> Dict[str, float]:
        """Cosine similarity between *query_emb* and all stored template image embeddings.

        Works for both image embeddings and text embeddings as queries,
        since SigLIP2 places them in the same vector space.

        Returns dict mapping template_id → similarity score in [0, 1].
        """
        if self._clip_embeddings is None or self._clip_index is None:
            return {}
        stored_dim = self._clip_embeddings.shape[1]
        query_dim = query_emb.shape[0]
        if stored_dim != query_dim:
            print(
                f"[Memory] CLIP embedding dimension mismatch: stored={stored_dim}, "
                f"query={query_dim}. Re-run prepare_commonsense_templates.py to "
                f"rebuild embeddings. Falling back to text-only retrieval."
            )
            return {}
        sims = (self._clip_embeddings @ query_emb).flatten()
        sims = np.clip(sims, 0.0, 1.0)
        return {tid: float(sims[i]) for i, tid in enumerate(self._clip_index)}

    def _ensure_dino_model(self):
        """Lazy-load DINOv2 vision encoder for query-time image embedding."""
        if self._dino_model is not None:
            return
        with self._dino_lock:
            if self._dino_model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoImageProcessor
                model_name = "facebook/dinov2-base"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._dino_model = AutoModel.from_pretrained(model_name).to(device).eval()
                self._dino_processor = AutoImageProcessor.from_pretrained(model_name)
                self._dino_device = device
                print(f"[Memory] DINOv2 model loaded on {device}")
            except Exception as e:
                print(f"[Memory] Could not load DINOv2 model: {e}")
                self._dino_model = False  # sentinel: tried and failed

    def _compute_dino_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Compute a normalised DINOv2 CLS-token embedding for a single image."""
        self._ensure_dino_model()
        if self._dino_model is False or self._dino_model is None:
            return None
        try:
            import torch
            img = Image.open(image_path).convert("RGB")
            with self._dino_lock:
                inputs = self._dino_processor(images=img, return_tensors="pt").to(self._dino_device)
                with torch.no_grad():
                    outputs = self._dino_model(**inputs)
                    emb = outputs.last_hidden_state[:, 0]  # CLS token
                    emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"[Memory] DINOv2 embedding failed for {image_path}: {e}")
            return None

    def _compute_dino_similarities(self, query_emb: np.ndarray) -> Dict[str, float]:
        """Cosine similarity between *query_emb* and all stored DINOv2 template embeddings.

        Returns dict mapping template_id → similarity score in [0, 1].
        """
        if self._dino_embeddings is None or self._dino_index is None:
            return {}
        stored_dim = self._dino_embeddings.shape[1]
        query_dim = query_emb.shape[0]
        if stored_dim != query_dim:
            print(
                f"[Memory] DINOv2 embedding dimension mismatch: stored={stored_dim}, "
                f"query={query_dim}. Re-run prepare_commonsense_templates.py with "
                f"DINOv2 to rebuild. Skipping DINOv2 retrieval."
            )
            return {}
        sims = (self._dino_embeddings @ query_emb).flatten()
        sims = np.clip(sims, 0.0, 1.0)
        return {tid: float(sims[i]) for i, tid in enumerate(self._dino_index)}

    def _load_commonsense_templates_from_disk(self) -> bool:
        fp = self._commonsense_file_path()
        if not fp or not fp.exists():
            return False
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", []) if isinstance(data, dict) else []
            if not isinstance(entries, list):
                return False
            valid = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                if not e.get("image_path") or not e.get("gt_path"):
                    continue
                valid.append(e)
            self._commonsense_templates = valid
            self._commonsense_ready = True
            print(f"[Memory] Loaded {len(valid)} commonsense templates from disk")
            # Also load pre-computed CLIP embeddings if available
            self._load_clip_embeddings()
            # Also load pre-computed DINOv2 embeddings if available
            self._load_dino_embeddings()
            return True
        except Exception as e:
            print(f"[Memory] Failed to load commonsense templates: {e}")
            return False

    def _save_commonsense_templates_to_disk(self) -> None:
        fp = self._commonsense_file_path()
        if not fp:
            return
        try:
            payload = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "entries": self._commonsense_templates,
            }
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Failed to save commonsense templates: {e}")

    def _append_template_with_cap(
        self,
        templates: List[Dict[str, Any]],
        pair_counter: Dict[str, int],
        dataset_group: str,
        dataset_name: str,
        object_name: str,
        affordance_part: str,
        question: str,
        image_path: str,
        gt_path: str,
    ) -> bool:
        object_name = self._safe_text(object_name)
        affordance_part = self._safe_text(affordance_part)
        question = self._safe_text(question)
        if not object_name or not affordance_part or not question:
            return False
        if not image_path or not gt_path:
            return False
        if not (os.path.exists(image_path) and os.path.exists(gt_path)):
            return False

        key = self._pair_key(object_name, affordance_part)
        cap_key = f"{self._safe_text(dataset_group).lower()}::{key}"
        if pair_counter.get(cap_key, 0) >= self.commonsense_max_per_pair:
            return False
        if self.commonsense_max_total is not None and len(templates) >= self.commonsense_max_total:
            return False

        template_id = hashlib.md5(
            f"{dataset_name}|{object_name}|{affordance_part}|{question}|{image_path}|{gt_path}".encode("utf-8")
        ).hexdigest()[:16]
        templates.append(
            {
                "template_id": template_id,
                "object_name": object_name,
                "affordance_part": affordance_part,
                "question": question,
                "image_path": image_path,
                "gt_path": gt_path,
                "pair_key": key,
            }
        )
        pair_counter[cap_key] = pair_counter.get(cap_key, 0) + 1
        return True

    def _can_add_more_commonsense_templates(self, current_count: int) -> bool:
        """Whether template adding can continue under max_total constraint."""
        return (self.commonsense_max_total is None) or (current_count < self.commonsense_max_total)

    def _question_similarity(self, q1: str, q2: str) -> float:
        """Rough text similarity for near-duplicate filtering."""
        s1 = set(re.findall(r"[a-z0-9]+", self._safe_text(q1).lower()))
        s2 = set(re.findall(r"[a-z0-9]+", self._safe_text(q2).lower()))
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / max(len(s1 | s2), 1)

    def _infer_object_from_question(self, question: str) -> str:
        """Best-effort object extraction from natural language question."""
        q = self._safe_text(question).lower()
        if not q:
            return ""
        patterns = [
            r"part of the ([a-z0-9_\-\s]+?)(?:\s+(?:that|which|to|for)\b|[?.!,]|$)",
            r"of this ([a-z0-9_\-\s]+?)(?:\s+(?:that|which|to|for)\b|[?.!,]|$)",
            r"on the ([a-z0-9_\-\s]+?)(?:\s+(?:that|which|to|for)\b|[?.!,]|$)",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                obj = re.sub(r"\s+", " ", m.group(1)).strip()
                obj = obj.strip(" .,!?:;")
                if obj:
                    return obj
        return ""

    def _build_commonsense_templates_from_datasets(self) -> None:
        """Build representative templates from datasets/* (small, diverse bank)."""
        if not self.enable_commonsense_templates:
            self._commonsense_templates = []
            self._commonsense_ready = True
            return

        root = self.datasets_root
        if not root.exists():
            self._commonsense_templates = []
            self._commonsense_ready = True
            return

        # Candidate pools keyed by dataset-group and pair_key:
        # pools[dataset][pair_key] = [candidate, ...]
        pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        def _push_candidate(
            dataset_group: str,
            dataset_name: str,
            object_name: str,
            affordance_part: str,
            question: str,
            image_path: str,
            gt_path: str,
        ) -> None:
            object_name = self._safe_text(object_name)
            affordance_part = self._safe_text(affordance_part)
            question = self._safe_text(question)
            if not object_name or not affordance_part or not question:
                return
            if not image_path or not gt_path:
                return
            if not (os.path.exists(image_path) and os.path.exists(gt_path)):
                return
            pair_key = self._pair_key(object_name, affordance_part)
            pools[dataset_group][pair_key].append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": dataset_name,
                    "object_name": object_name,
                    "affordance_part": affordance_part,
                    "question": question,
                    "image_path": image_path,
                    "gt_path": gt_path,
                    "pair_key": pair_key,
                }
            )

        # 1) HANDAL_* preprocessed (samples.json)
        for split_name in ("HANDAL_hard", "HANDAL_easy", "HANDAL"):
            samples_json = root / split_name / "samples.json"
            if not samples_json.exists():
                continue
            try:
                with open(samples_json, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if not isinstance(items, list):
                    continue
                for item in items:
                    img = os.path.join(str(root / split_name), str(item.get("image_path", "")))
                    gt = os.path.join(str(root / split_name), str(item.get("mask_path", "")))
                    obj = item.get("category", "")
                    part = item.get("answer", "handle")
                    question = item.get("question", "")
                    _push_candidate(
                        "HANDAL", split_name, obj, part, question, img, gt
                    )
            except Exception:
                continue

        # 2) 3DOI samples_reasoning.json
        samples_3doi = root / "3doi" / "samples_reasoning.json"
        if samples_3doi.exists():
            try:
                with open(samples_3doi, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        base = str(root / "3doi")
                        img = os.path.join(base, str(item.get("image_path", "")))
                        gt = os.path.join(base, str(item.get("mask_path", "")))
                        obj = item.get("task_object_class", "")
                        part = item.get("answer", "")
                        question = item.get("question", "")
                        _push_candidate(
                            "3doi", "3doi", obj, part, question, img, gt
                        )
            except Exception:
                pass

        # 3) UMD_preprocessed* (derive from filenames)
        for umd_name in ("UMD_preprocessed", "UMD_preprocessed_first", "UMD_preprocessed_r1"):
            umd_root = root / umd_name
            if not umd_root.exists() or not umd_root.is_dir():
                continue
            try:
                for tool_dir in sorted(os.listdir(umd_root)):
                    tool_dir_path = umd_root / tool_dir
                    if not tool_dir_path.is_dir():
                        continue
                    obj = tool_dir.split("_")[0]
                    for fn in sorted(os.listdir(tool_dir_path)):
                        if not fn.endswith("_gt_mask.png"):
                            continue
                        affordance = fn.split("_gt_mask.png")[0].split("_")[-1]
                        base_name = "_".join(fn.split("_")[:-3])
                        img = str(tool_dir_path / f"{base_name}_rgb.jpg")
                        gt = str(tool_dir_path / fn)
                        q = f"Find the part of the object in the center of the image that can {affordance}"
                        _push_candidate(
                            "UMD", umd_name, obj, affordance, q, img, gt
                        )
            except Exception:
                continue

        # 4) AGD20K (seen test split)
        agd_ego = root / "AGD20K" / "AGD20K" / "Seen" / "testset" / "egocentric"
        agd_gt = root / "AGD20K" / "AGD20K" / "Seen" / "testset" / "GT"
        if agd_ego.exists() and agd_gt.exists():
            try:
                for affordance in sorted(os.listdir(agd_ego)):
                    aff_ego = agd_ego / affordance
                    aff_gt = agd_gt / affordance
                    if not aff_ego.is_dir():
                        continue
                    for obj in sorted(os.listdir(aff_ego)):
                        obj_ego = aff_ego / obj
                        obj_gt = aff_gt / obj
                        if not obj_ego.is_dir():
                            continue
                        for fname in sorted(os.listdir(obj_ego)):
                            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                                continue
                            img = str(obj_ego / fname)
                            gt = str(obj_gt / (Path(fname).stem + ".png"))
                            q = f"{affordance.replace('_', ' ')} the {obj}"
                            _push_candidate(
                                "AGD20K", "AGD20K_seen", obj, affordance, q, img, gt
                            )
            except Exception:
                pass

        # 5) reasonaff (vis_train + vis)
        for reason_sub in ("vis_train", "vis"):
            rr = root / "reasonaff" / reason_sub
            if not rr.exists() or not rr.is_dir():
                continue
            try:
                for fn in sorted(os.listdir(rr)):
                    if not fn.endswith("_meta.json"):
                        continue
                    stem = fn.replace("_meta.json", "")
                    meta_path = rr / fn
                    image_path = rr / f"{stem}_image.png"
                    gt_path = rr / f"{stem}_mask_gt.png"
                    if not image_path.exists() or not gt_path.exists():
                        continue
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        continue
                    q = self._safe_text(meta.get("problem", ""))
                    part = self._safe_text(meta.get("part_name", "")) or self._safe_text(meta.get("aff_name", ""))
                    obj = self._infer_object_from_question(q)
                    if not obj:
                        # fallback to a coarse object token if parse failed
                        obj = "unknown_object"
                    _push_candidate(
                        "reasonaff",
                        f"reasonaff_{reason_sub}",
                        obj,
                        part,
                        q,
                        str(image_path),
                        str(gt_path),
                    )
            except Exception:
                continue

        # ---- Coverage-first selection across datasets ----
        # Special handling for 3DOI (evaluation set):
        # - remove object-action pairs already present in non-3DOI datasets
        # - remove near-duplicate prompts vs non-3DOI templates
        # - cap 3DOI size close to reasonaff
        non_3doi_groups = [g for g in pools.keys() if g != "3doi"]
        non_3doi_pair_keys: set = set()
        non_3doi_questions: List[str] = []
        for g in non_3doi_groups:
            for pair_key, candidates in pools[g].items():
                if not candidates:
                    continue
                non_3doi_pair_keys.add(pair_key)
                non_3doi_questions.extend([str(c.get("question", "")) for c in candidates])

        if "3doi" in pools:
            filtered_3doi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for pair_key, candidates in pools["3doi"].items():
                if pair_key in non_3doi_pair_keys:
                    continue
                for c in candidates:
                    q = str(c.get("question", ""))
                    if any(self._question_similarity(q, qn) >= 0.9 for qn in non_3doi_questions):
                        continue
                    filtered_3doi[pair_key].append(c)
            pools["3doi"] = filtered_3doi

        ordered_groups = ["UMD", "reasonaff", "AGD20K", "HANDAL", "3doi"]
        # Keep only existing groups
        ordered_groups = [g for g in ordered_groups if g in pools]
        three_doi_cap = len(pools.get("reasonaff", {}))
        if three_doi_cap <= 0:
            three_doi_cap = 300

        templates: List[Dict[str, Any]] = []
        pair_counter: Dict[str, int] = {}
        group_counter: Dict[str, int] = defaultdict(int)

        def _take_candidate(c: Dict[str, Any]) -> bool:
            gname = c["dataset_group"]
            if gname == "3doi" and group_counter.get("3doi", 0) >= three_doi_cap:
                return False
            added = self._append_template_with_cap(
                templates,
                pair_counter,
                c["dataset_group"],
                c["dataset_name"],
                c["object_name"],
                c["affordance_part"],
                c["question"],
                c["image_path"],
                c["gt_path"],
            )
            if added:
                group_counter[gname] = group_counter.get(gname, 0) + 1
            return added

        # Pass 1: maximize pair coverage (prefer unseen pair across all groups)
        progress = True
        while progress and self._can_add_more_commonsense_templates(len(templates)):
            progress = False
            for g in ordered_groups:
                group_pairs = pools[g]
                picked = False
                for pair_key in sorted(group_pairs.keys()):
                    cap_key = f"{g.lower()}::{pair_key}"
                    if pair_counter.get(cap_key, 0) > 0:
                        continue
                    candidates = group_pairs[pair_key]
                    if not candidates:
                        continue
                    if _take_candidate(candidates[0]):
                        picked = True
                        progress = True
                        break
                if picked and (not self._can_add_more_commonsense_templates(len(templates))):
                    break

        # Pass 2: fill remaining with second examples (still round-robin groups)
        if self._can_add_more_commonsense_templates(len(templates)):
            progress = True
            while progress and self._can_add_more_commonsense_templates(len(templates)):
                progress = False
                for g in ordered_groups:
                    group_pairs = pools[g]
                    picked = False
                    for pair_key in sorted(group_pairs.keys()):
                        cap_key = f"{g.lower()}::{pair_key}"
                        if pair_counter.get(cap_key, 0) >= self.commonsense_max_per_pair:
                            continue
                        candidates = group_pairs[pair_key]
                        if not candidates:
                            continue
                        # pick by current count index, fallback last one
                        idx = min(pair_counter.get(cap_key, 0), len(candidates) - 1)
                        if _take_candidate(candidates[idx]):
                            picked = True
                            progress = True
                            break
                    if picked and (not self._can_add_more_commonsense_templates(len(templates))):
                        break

        self._commonsense_templates = templates
        self._commonsense_ready = True
        print(
            f"[Memory] Built commonsense template bank: {len(templates)} templates, "
            f"{len(pair_counter)} object-part pairs, groups={ordered_groups}, "
            f"3doi_cap={three_doi_cap}, 3doi_kept={group_counter.get('3doi', 0)}"
        )
        self._save_commonsense_templates_to_disk()

    def _ensure_commonsense_ready(self) -> None:
        if self._commonsense_ready:
            return
        if self._load_commonsense_templates_from_disk():
            return
        if self.commonsense_auto_build:
            self._build_commonsense_templates_from_datasets()
        else:
            self._commonsense_templates = []
            self._commonsense_ready = True

    def prepare_commonsense_templates(self, force_rebuild: bool = True) -> Dict[str, Any]:
        """Precompute commonsense templates and save to templates dir."""
        if not self.enable_commonsense_templates:
            return {"success": False, "error": "commonsense templates disabled"}
        if (not force_rebuild) and self._load_commonsense_templates_from_disk():
            return {
                "success": True,
                "templates_count": len(self._commonsense_templates),
                "templates_file": str(self._commonsense_file_path()) if self._commonsense_file_path() else "",
                "mode": "loaded_existing",
            }
        self._build_commonsense_templates_from_datasets()
        return {
            "success": True,
            "templates_count": len(self._commonsense_templates),
            "templates_file": str(self._commonsense_file_path()) if self._commonsense_file_path() else "",
            "mode": "rebuilt",
        }

    def materialize_commonsense_assets(self, rewrite_relative: bool = True) -> Dict[str, Any]:
        """
        Copy template image/GT assets under commonsense_templates_dir and optionally
        rewrite JSON paths as relative paths.
        """
        self._ensure_commonsense_ready()
        if not self._commonsense_templates:
            return {"success": False, "error": "no commonsense templates loaded"}

        templates_dir = self.commonsense_templates_dir
        images_dir = templates_dir / "images"
        gt_dir = templates_dir / "gt"
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_gt = 0
        missing = 0

        updated: List[Dict[str, Any]] = []
        for t in self._commonsense_templates:
            rec = dict(t)
            tid = str(rec.get("template_id") or hashlib.md5(json.dumps(rec, ensure_ascii=False).encode("utf-8")).hexdigest()[:16])

            src_img = self._resolve_template_asset_path(str(rec.get("image_path", "")))
            src_gt = self._resolve_template_asset_path(str(rec.get("gt_path", "")))

            if not src_img or not src_gt or (not os.path.exists(src_img)) or (not os.path.exists(src_gt)):
                missing += 1
                continue

            img_ext = Path(src_img).suffix or ".jpg"
            gt_ext = Path(src_gt).suffix or ".png"
            dst_img = images_dir / f"{tid}_img{img_ext}"
            dst_gt = gt_dir / f"{tid}_gt{gt_ext}"

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_images += 1
            if not dst_gt.exists():
                shutil.copy2(src_gt, dst_gt)
                copied_gt += 1

            if rewrite_relative:
                rec["image_path"] = os.path.relpath(dst_img, templates_dir)
                rec["gt_path"] = os.path.relpath(dst_gt, templates_dir)
            else:
                rec["image_path"] = str(dst_img)
                rec["gt_path"] = str(dst_gt)

            updated.append(rec)

        self._commonsense_templates = updated
        self._save_commonsense_templates_to_disk()

        return {
            "success": True,
            "templates_count": len(updated),
            "copied_images": copied_images,
            "copied_gt": copied_gt,
            "missing_source_entries": missing,
            "templates_file": str(self._commonsense_file_path()) if self._commonsense_file_path() else "",
            "templates_dir": str(self.commonsense_templates_dir),
            "rewrite_relative": rewrite_relative,
        }

    def retrieve_common_sense_templates(
        self,
        task: str,
        object_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 2,
        min_relevance: float = 0.15,
        image_weight: float = 0.6,
        text_weight: float = 0.4,
        dino_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Multi-modal RAG retrieval combining SigLIP2 and DINOv2 embeddings.

        SigLIP2 provides both text→image and image→image similarity.
        DINOv2 provides complementary image→image similarity with stronger
        spatial/structural understanding.

        When both encoders are available and a query image is given, the final
        score uses three channels:

            score = w_text * siglip_text_sim
                  + w_siglip_img * siglip_img_sim
                  + w_dino * dino_img_sim

        Weights are normalised so they sum to 1.  When DINOv2 embeddings are
        unavailable, falls back to SigLIP2-only scoring.  When neither is
        available, falls back to Jaccard text similarity (legacy behaviour).

        Args:
            task: Current task description.
            object_name: Optional object name hint.
            query_image_path: Path to the query image for image-based retrieval.
            top_k: Max number of templates to return.
            min_relevance: Minimum relevance score to include a template.
            image_weight: Weight for SigLIP2 image→image similarity.
            text_weight: Weight for SigLIP2 text→image similarity.
            dino_weight: Weight for DINOv2 image→image similarity.

        Returns:
            List of template dicts, each augmented with ``relevance_score``,
            ``text_score``, ``image_score`` and ``dino_score`` fields.
        """
        if not self.enable_commonsense_templates:
            return []
        self._ensure_commonsense_ready()
        if not self._commonsense_templates:
            return []

        has_siglip = (
            self._clip_embeddings is not None
            and self._clip_index is not None
        )
        has_dino = (
            self._dino_embeddings is not None
            and self._dino_index is not None
        )

        # -- SigLIP2 text→image similarity ------------------------------------
        text_sims: Dict[str, float] = {}
        use_siglip_text = False
        if has_siglip:
            q_text = self._safe_text(task)
            if object_name:
                q_text = f"{q_text} {self._safe_text(object_name)}"
            text_emb = self._compute_text_embedding(q_text)
            if text_emb is not None:
                text_sims = self._compute_clip_similarities(text_emb)
                use_siglip_text = True

        # -- SigLIP2 image→image similarity ------------------------------------
        img_sims: Dict[str, float] = {}
        use_siglip_img = False
        if has_siglip and query_image_path:
            img_emb = self._compute_clip_embedding(query_image_path)
            if img_emb is not None:
                img_sims = self._compute_clip_similarities(img_emb)
                use_siglip_img = True

        # -- DINOv2 image→image similarity ------------------------------------
        dino_sims: Dict[str, float] = {}
        use_dino_img = False
        if has_dino and query_image_path:
            dino_emb = self._compute_dino_embedding(query_image_path)
            if dino_emb is not None:
                dino_sims = self._compute_dino_similarities(dino_emb)
                use_dino_img = True

        use_semantic = use_siglip_text or use_siglip_img or use_dino_img
        mode_parts = []
        if use_siglip_text or use_siglip_img:
            mode_parts.append("siglip2")
        if use_dino_img:
            mode_parts.append("dinov2")
        mode = "+".join(mode_parts) if mode_parts else "jaccard_fallback"

        if use_semantic:
            active = []
            if use_siglip_text:
                active.append(f"siglip_text→img(w={text_weight:.2f})")
            if use_siglip_img:
                active.append(f"siglip_img→img(w={image_weight:.2f})")
            if use_dino_img:
                active.append(f"dino_img→img(w={dino_weight:.2f})")
            print(f"[Memory] Retrieval active — {', '.join(active)}")

        # -- Score every template ----------------------------------------------
        scored: List[Tuple[float, float, float, float, Dict[str, Any]]] = []

        if use_semantic:
            w_text = float(text_weight) if use_siglip_text else 0.0
            w_img = float(image_weight) if use_siglip_img else 0.0
            w_dino = float(dino_weight) if use_dino_img else 0.0
            w_sum = w_text + w_img + w_dino
            if w_sum > 0:
                w_text /= w_sum
                w_img /= w_sum
                w_dino /= w_sum

            for t in self._commonsense_templates:
                tid = t.get("template_id", "")
                ts = text_sims.get(tid, 0.0)
                is_ = img_sims.get(tid, 0.0)
                ds = dino_sims.get(tid, 0.0)
                combined = w_text * ts + w_img * is_ + w_dino * ds
                scored.append((combined, ts, is_, ds, t))
        else:
            # Fallback: legacy Jaccard text similarity
            q_task = self._safe_text(task)
            q_obj = self._safe_text(object_name)
            q_low = (q_task + " " + q_obj).lower()
            for t in self._commonsense_templates:
                text_score = self._compute_text_similarity(
                    q_task, object_name,
                    t.get("question", ""), t.get("object_name", None),
                )
                obj = str(t.get("object_name", "")).lower()
                part = str(t.get("affordance_part", "")).lower()
                if q_obj and q_obj.lower() == obj:
                    text_score += 0.25
                if part and part in q_low:
                    text_score += 0.15
                if obj and obj in q_low:
                    text_score += 0.15
                scored.append((text_score, text_score, 0.0, 0.0, t))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for combined, text_s, img_s, dino_s, t in scored[: max(1, int(top_k))]:
            if combined < min_relevance:
                break
            rec = dict(t)
            rec["image_path"] = self._resolve_template_asset_path(
                str(rec.get("image_path", ""))
            )
            rec["gt_path"] = self._resolve_template_asset_path(
                str(rec.get("gt_path", ""))
            )
            rec["relevance_score"] = round(combined, 4)
            rec["text_score"] = round(text_s, 4)
            rec["image_score"] = round(img_s, 4)
            rec["dino_score"] = round(dino_s, 4)
            selected.append(rec)

        if selected:
            print(
                f"[Memory] Commonsense retrieval ({mode}) returned {len(selected)} template(s) "
                f"(best relevance={selected[0]['relevance_score']:.4f}, "
                f"text={selected[0]['text_score']:.4f}, "
                f"siglip_img={selected[0]['image_score']:.4f}, "
                f"dino_img={selected[0]['dino_score']:.4f})"
            )
        return selected

    def format_common_sense_templates_for_context(
        self,
        templates: List[Dict[str, Any]],
        token_budget: int = 500,
    ) -> str:
        """Format commonsense exemplar templates into a compact context block.

        Each template line includes relevance scores and file paths so the
        decision model can both evaluate relevance and optionally pass images
        to detection via ``reference_images``.
        """
        if not templates:
            return ""
        lines = [
            "**Common-sense Template Exemplars** (RAG-retrieved from training data):",
            "(You may pass these images to detection via reference_images if you judge them helpful.)",
        ]
        tokens_used = sum(self._estimate_tokens(l) for l in lines)
        for i, t in enumerate(templates, 1):
            rel = t.get("relevance_score")
            rel_str = f", relevance={rel:.2f}" if isinstance(rel, (int, float)) else ""
            txt_s = t.get("text_score")
            img_s = t.get("image_score")
            if isinstance(txt_s, (int, float)) and isinstance(img_s, (int, float)):
                rel_str += f" (text={txt_s:.2f}, image={img_s:.2f})"
            img_p = t.get("image_path", "")
            gt_p = t.get("gt_path", "")
            path_str = ""
            if img_p:
                path_str += f" | scene_image: {img_p}"
            if gt_p:
                path_str += f" | gt_mask: {gt_p}"
            line = (
                f"- Template {i}: "
                f"object={t.get('object_name','?')}, affordance/part={t.get('affordance_part','?')}"
                f"{rel_str}; "
                f"Q: {str(t.get('question',''))[:140]}"
                f"{path_str}"
            )
            tt = self._estimate_tokens(line)
            if tokens_used + tt > token_budget:
                break
            lines.append(line)
            tokens_used += tt
        lines.append("")
        return "\n".join(lines)
    
    def _get_iou_distribution(self) -> Dict[str, float]:
        """Compute IoU distribution stats from current inner + outer memory.
        
        Returns dict with keys: median, mean, p25, p75, min, max, count.
        All values are 0.0 if no IoU data is available.
        """
        ious: List[float] = []
        for entry in self._memory:
            if entry.evaluation_metrics:
                v = entry.evaluation_metrics.get("iou")
                if isinstance(v, (int, float)):
                    ious.append(float(v))
        for oe in self._outer_memory:
            if isinstance(oe.iou, (int, float)) and oe.iou > 0:
                ious.append(float(oe.iou))
        if not ious:
            return {"median": 0.0, "mean": 0.0, "p25": 0.0, "p75": 0.0,
                    "min": 0.0, "max": 0.0, "count": 0}
        arr = sorted(ious)
        n = len(arr)
        return {
            "median": float(arr[n // 2]) if n % 2 else float((arr[n // 2 - 1] + arr[n // 2]) / 2),
            "mean": float(sum(arr) / n),
            "p25": float(arr[max(0, n // 4)]),
            "p75": float(arr[min(n - 1, 3 * n // 4)]),
            "min": float(arr[0]),
            "max": float(arr[-1]),
            "count": n,
        }
    
    def add_entry(
        self,
        sample_id: str,
        image_path: str,
        task: str,
        object_name: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_summaries: Optional[List[str]] = None,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        mask_path: Optional[str] = None,
        reasoning_context: Optional[str] = None,
        task_context: Optional[str] = None,
        llm_reasoning: Optional[List[str]] = None,
        feature_vector: Optional[List[float]] = None,
        dedupe_top_k: Optional[int] = None,
        strategy_reasoning: Optional[str] = None,
        strategy_reflection: Optional[str] = None,
        dynamic_params_used: Optional[Dict[str, Any]] = None,
        decision_trace: Optional[List[str]] = None,
    ) -> None:
        """ memory （ RAG /）
        
        ：
        1)  TopK （RAG ）
        2) 
        3) ：
        4) ：，
        
        ：
        - strategy_reasoning: 
        - strategy_reflection: 
        - dynamic_params_used: 
        - decision_trace: 
        """
        entry = MemoryEntry(
            sample_id=sample_id,
            image_path=image_path,
            task=task,
            object_name=object_name,
            tool_calls=tool_calls or [],
            tool_summaries=tool_summaries or [],
            evaluation_metrics=evaluation_metrics,
            mask_path=mask_path,
            reasoning_context=reasoning_context,
            task_context=task_context,
            llm_reasoning=llm_reasoning or [],
            feature_vector=feature_vector,
            strategy_reasoning=strategy_reasoning,
            strategy_reflection=strategy_reflection,
            dynamic_params_used=dynamic_params_used,
            decision_trace=decision_trace or [],
        )

        # ： RAG ，
        existing_idx_to_remove = self._find_duplicate_entry_index(
            new_entry=entry,
            top_k=dedupe_top_k or self.enqueue_dedupe_top_k,
        )
        if existing_idx_to_remove is not None and 0 <= existing_idx_to_remove < len(self._memory):
            removed = self._memory.pop(existing_idx_to_remove)
            print(
                f"[Memory] Duplicate detected. Dequeued old sample '{removed.sample_id}' "
                f"and enqueued new sample '{entry.sample_id}'."
            )

        # Auto-compute SigLIP2 image embedding for future retrieval
        if entry.feature_vector is None and entry.image_path:
            emb = self._compute_clip_embedding(entry.image_path)
            if emb is not None:
                entry.feature_vector = emb.tolist()

        # Auto-compute DINOv2 image embedding for future retrieval
        if entry.dino_feature_vector is None and entry.image_path:
            dino_emb = self._compute_dino_embedding(entry.image_path)
            if dino_emb is not None:
                entry.dino_feature_vector = dino_emb.tolist()

        self._memory.append(entry)
        
        if len(self._memory) > self.max_size:
            self._evict_entries()

        #  skill /
        self._rebuild_skill_profiles()
        
        # （， update_metrics ）
        iou = 0.0
        if evaluation_metrics and isinstance(evaluation_metrics.get("iou"), (int, float)):
            iou = float(evaluation_metrics["iou"])
        tool_chain = " → ".join(tc.get("name", "?") for tc in (tool_calls or []))
        self.experience_pool.add_observation(
            sample_id=sample_id,
            task=task,
            object_name=object_name,
            tool_chain=tool_chain or "none",
            iou=iou,
            strategy_reasoning=strategy_reasoning,
            dynamic_params=dynamic_params_used,
        )
        
        if self.persist_dir:
            self._save_to_disk()

    # Metrics update — called by evaluation scripts after ground-truth
    # comparison, so that memory entries carry real performance data.
    def update_entry_metrics(
        self,
        sample_id: str,
        evaluation_metrics: Dict[str, float],
    ) -> bool:
        """Update evaluation metrics for an existing memory entry.

        This is the critical bridge between evaluation and memory: after the
        evaluation script computes IoU / gIoU / cIoU / P@50, it calls this
        method so that future memory retrieval can consider result quality.

        Also re-submits the observation to the experience pool with the real
        IoU so that insights are based on actual performance, not placeholder 0.

        Args:
            sample_id: The unique sample identifier used when add_entry was called.
            evaluation_metrics: Dict of metric name → value, e.g. {"iou": 0.65, ...}.

        Returns:
            True if an entry was found and updated, False otherwise.
        """
        # Search inner memory
        for entry in self._memory:
            if entry.sample_id == sample_id:
                entry.evaluation_metrics = evaluation_metrics

                # Re-submit to experience pool with the real IoU
                iou = 0.0
                if isinstance(evaluation_metrics.get("iou"), (int, float)):
                    iou = float(evaluation_metrics["iou"])

                tool_chain = " → ".join(
                    tc.get("name", "?") for tc in (entry.tool_calls or [])
                )
                self.experience_pool.add_observation(
                    sample_id=sample_id,
                    task=entry.task,
                    object_name=entry.object_name,
                    tool_chain=tool_chain or "none",
                    iou=iou,
                    strategy_reasoning=entry.strategy_reasoning,
                    dynamic_params=entry.dynamic_params_used,
                )

                #  skill （）
                self._rebuild_skill_profiles()

                # Persist the update
                if self.persist_dir:
                    self._save_to_disk()

                return True

        return False

    def _has_verified_metrics(self, entry: MemoryEntry) -> bool:
        """Check if an entry has verified evaluation metrics (not placeholder)."""
        if not entry.evaluation_metrics:
            return False
        iou = entry.evaluation_metrics.get("iou")
        return isinstance(iou, (int, float))

    def _score_entry(
        self,
        query_task: str,
        query_object_name: Optional[str],
        entry: MemoryEntry,
        iou_dist: Optional[Dict[str, float]] = None,
        query_text_emb: Optional[np.ndarray] = None,
        query_img_emb: Optional[np.ndarray] = None,
        query_dino_emb: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """ (score, base_similarity)。
        
        Combines SigLIP2 (text→img + img→img) and DINOv2 (img→img) similarity
        when available.  Falls back to Jaccard text overlap when no embeddings
        exist.

        Entries with verified metrics get a significant bonus because they
        carry actual learning signal (the model knows if the strategy worked).
        Unverified entries are just trajectories with no reward signal.
        """
        entry_emb = None
        if entry.feature_vector is not None:
            entry_emb = np.array(entry.feature_vector, dtype=np.float32)

        entry_dino_emb = None
        if entry.dino_feature_vector is not None:
            entry_dino_emb = np.array(entry.dino_feature_vector, dtype=np.float32)

        has_siglip = entry_emb is not None and (query_text_emb is not None or query_img_emb is not None)
        has_dino = entry_dino_emb is not None and query_dino_emb is not None

        if has_siglip or has_dino:
            scores = []
            weights = []
            if has_siglip:
                if query_text_emb is not None:
                    ts = float(np.clip(query_text_emb @ entry_emb, 0.0, 1.0))
                    scores.append(ts)
                    weights.append(0.4)
                if query_img_emb is not None:
                    is_ = float(np.clip(query_img_emb @ entry_emb, 0.0, 1.0))
                    scores.append(is_)
                    weights.append(0.6)
            if has_dino:
                ds = float(np.clip(query_dino_emb @ entry_dino_emb, 0.0, 1.0))
                scores.append(ds)
                weights.append(0.5)
            w_sum = sum(weights)
            sim = sum(s * w for s, w in zip(scores, weights)) / w_sum if w_sum > 0 else 0.0
        else:
            sim = self._compute_text_similarity(
                query_task, query_object_name,
                entry.task, entry.object_name,
            )

        bonus = 0.0
        
        # Verified entries are much more valuable than unverified ones
        if self._has_verified_metrics(entry):
            bonus += 0.15  # significant bonus for having reward signal
            
            # Additional bonus based on relative IoU position
            iou_val = entry.evaluation_metrics.get("iou", 0)
            if isinstance(iou_val, (int, float)):
                dist = iou_dist or self._get_iou_distribution()
                if dist["count"] >= 3:
                    if iou_val >= dist["p75"]:
                        bonus += 0.08
                    elif iou_val >= dist["median"]:
                        bonus += 0.05
                    elif iou_val >= dist["p25"]:
                        bonus += 0.02
                elif iou_val > 0:
                    bonus += min(0.08, iou_val * 0.1)

        return sim + bonus, sim

    def _retrieve_scored_candidates(
        self,
        task: str,
        object_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: float = 0.0,
    ) -> List[Tuple[int, float, float, MemoryEntry]]:
        """RAG  memory（ idx, score, similarity, entry）。
        
        Combines SigLIP2 (text→img + img→img) and DINOv2 (img→img) when
        available.  Falls back to Jaccard text overlap otherwise.

        Entries with verified metrics are strongly preferred over unverified
        trajectory-only entries.
        """
        if not self._memory:
            return []

        k = max(1, int(top_k or self.retrieval_top_k))
        iou_dist = self._get_iou_distribution()  # compute once

        # Pre-compute SigLIP2 query embeddings once for all entries
        query_text_emb: Optional[np.ndarray] = None
        query_img_emb: Optional[np.ndarray] = None
        any_entry_has_emb = any(e.feature_vector is not None for e in self._memory)
        if any_entry_has_emb:
            q_text = self._safe_text(task)
            if object_name:
                q_text = f"{q_text} {self._safe_text(object_name)}"
            query_text_emb = self._compute_text_embedding(q_text)
            if query_image_path:
                query_img_emb = self._compute_clip_embedding(query_image_path)

        # Pre-compute DINOv2 query embedding once for all entries
        query_dino_emb: Optional[np.ndarray] = None
        any_entry_has_dino = any(e.dino_feature_vector is not None for e in self._memory)
        if any_entry_has_dino and query_image_path:
            query_dino_emb = self._compute_dino_embedding(query_image_path)

        scored: List[Tuple[int, float, float, MemoryEntry]] = []
        for idx, entry in enumerate(self._memory):
            score, sim = self._score_entry(
                task, object_name, entry,
                iou_dist=iou_dist,
                query_text_emb=query_text_emb,
                query_img_emb=query_img_emb,
                query_dino_emb=query_dino_emb,
            )
            if sim >= min_similarity:
                scored.append((idx, score, sim, entry))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _is_duplicate_with_llm(
        self,
        new_entry: MemoryEntry,
        candidate_entry: MemoryEntry,
        similarity: float,
    ) -> bool:
        """。。"""
        #  API 
        if self._api_caller is None or self._llm_dedupe_disabled:
            return similarity >= self.duplicate_similarity_threshold

        try:
            new_tools = " → ".join([tc.get("name", "?") for tc in (new_entry.tool_calls or [])]) or "none"
            old_tools = " → ".join([tc.get("name", "?") for tc in (candidate_entry.tool_calls or [])]) or "none"
            new_iou = (
                new_entry.evaluation_metrics.get("iou")
                if isinstance(new_entry.evaluation_metrics, dict)
                else None
            )
            old_iou = (
                candidate_entry.evaluation_metrics.get("iou")
                if isinstance(candidate_entry.evaluation_metrics, dict)
                else None
            )

            prompt = MEMORY_DEDUP_PROMPT_TEMPLATE.format(
                similarity=similarity,
                new_task=new_entry.task,
                new_object=new_entry.object_name or "N/A",
                new_tools=new_tools,
                new_iou=new_iou if isinstance(new_iou, (int, float)) else "N/A",
                old_task=candidate_entry.task,
                old_object=candidate_entry.object_name or "N/A",
                old_tools=old_tools,
                old_iou=old_iou if isinstance(old_iou, (int, float)) else "N/A",
            )

            response = self._api_caller(
                "chat/completions",
                {
                    "model": self._eviction_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 8,
                },
            )
            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .upper()
            )
            if "DUPLICATE" in content:
                return True
            if "DIFFERENT" in content:
                return False
        except Exception as e:
            err = str(e)
            # Permission/quota style errors are usually persistent; disable LLM dedupe to avoid repeated failing calls.
            hard_fail_keywords = (
                "403",
                "quota",
                "insuf",
                "no permission",
                "",
                "",
            )
            if any(k.lower() in err.lower() for k in hard_fail_keywords):
                self._llm_dedupe_disabled = True
                print(
                    "[Memory] LLM duplicate check disabled due to persistent auth/quota error. "
                    "Using threshold-only dedupe for subsequent samples."
                )
            print(f"[Memory] LLM duplicate check failed ({e}), fallback to threshold")

        return similarity >= self.duplicate_similarity_threshold

    def _find_duplicate_entry_index(
        self,
        new_entry: MemoryEntry,
        top_k: int,
    ) -> Optional[int]:
        """ memory 。"""
        if not self._memory:
            return None

        candidates = self._retrieve_scored_candidates(
            task=new_entry.task,
            object_name=new_entry.object_name,
            top_k=top_k,
            min_similarity=0.0,
        )
        if not candidates:
            return None

        for idx, _score, sim, entry in candidates:
            if sim < self.duplicate_similarity_threshold:
                continue
            if self._is_duplicate_with_llm(new_entry, entry, similarity=sim):
                return idx

        return None
    
    def _compress_to_outer(self, entry: MemoryEntry) -> CompactMemoryEntry:
        """（）"""
        tool_chain = " → ".join(tc.get("name", "?") for tc in (entry.tool_calls or []))
        iou = 0.0
        if entry.evaluation_metrics and isinstance(entry.evaluation_metrics.get("iou"), (int, float)):
            iou = float(entry.evaluation_metrics["iou"])
        
        parts = [f"Task: {entry.task[:80]}"]
        if entry.object_name:
            parts.append(f"Object: {entry.object_name}")
        parts.append(f"Strategy: {tool_chain or 'none'}")
        parts.append(f"IoU: {iou:.3f}")
        if entry.strategy_reasoning:
            parts.append(f"Reasoning: {entry.strategy_reasoning[:150]}")
        summary = "; ".join(parts)
        
        tags = []
        if entry.object_name:
            tags.append(entry.object_name.lower())
        for tc in (entry.tool_calls or []):
            name = tc.get("name", "")
            if name:
                tags.append(name)
        
        return CompactMemoryEntry(
            sample_id=entry.sample_id,
            task=entry.task,
            object_name=entry.object_name,
            summary=summary,
            tool_chain=tool_chain,
            strategy_reasoning=entry.strategy_reasoning,
            iou=iou,
            timestamp=entry.timestamp or "",
            tags=tags,
        )
    
    def _evict_entries(self) -> None:
        """ — 
        
        Priority: remove unverified entries (no metrics) first, then apply
        the configured eviction strategy for the remainder.
        """
        num_to_remove = len(self._memory) - self.max_size
        if num_to_remove <= 0:
            return
        
        # Phase 1: remove unverified (no reward signal) entries first — 
        # they occupy slots but provide no learning signal
        unverified_indices = [
            i for i, e in enumerate(self._memory)
            if not self._has_verified_metrics(e)
        ]
        remove_phase1 = set()
        for idx in unverified_indices:
            if len(remove_phase1) >= num_to_remove:
                break
            remove_phase1.add(idx)
        
        if remove_phase1:
            # Compress to outer (even unverified entries keep trajectory info)
            for idx in sorted(remove_phase1, reverse=True):
                self._outer_memory.append(self._compress_to_outer(self._memory[idx]))
                self._memory.pop(idx)
            num_to_remove -= len(remove_phase1)
            if num_to_remove <= 0:
                return
        
        # Phase 2: apply configured strategy for remaining removals
        if self.eviction_strategy == "fifo":
            for entry in self._memory[:num_to_remove]:
                self._outer_memory.append(self._compress_to_outer(entry))
            self._memory = self._memory[num_to_remove:]
        
        elif self.eviction_strategy == "similarity":
            self._evict_by_similarity(num_to_remove)
        
        elif self.eviction_strategy == "model_decision":
            self._evict_by_model(num_to_remove)
    
    def _evict_by_similarity(self, num_to_remove: int) -> None:
        """：，。
        
        Among highly similar pairs, keep the one with more informative IoU
        (either the best performer as positive example, or the worst as
        negative example). Both extremes carry learning signal — the middle
        is less informative.
        """
        similarities = []
        for i in range(len(self._memory)):
            for j in range(i + 1, len(self._memory)):
                sim = self._compute_text_similarity(
                    self._memory[i].task, self._memory[i].object_name,
                    self._memory[j].task, self._memory[j].object_name
                )
                similarities.append((i, j, sim))
        
        similarities.sort(key=lambda x: -x[2])
        
        def _get_iou(idx: int) -> Optional[float]:
            e = self._memory[idx]
            if self._has_verified_metrics(e):
                return float(e.evaluation_metrics.get("iou", 0))
            return None
        
        to_remove = set()
        for i, j, sim in similarities:
            if sim >= self.similarity_threshold and len(to_remove) < num_to_remove:
                iou_i = _get_iou(i)
                iou_j = _get_iou(j)
                
                # Both verified: keep the one with more extreme IoU 
                # (better positive or worse negative — both are informative)
                if iou_i is not None and iou_j is not None:
                    # Keep the one further from median (more informative)
                    dist = self._get_iou_distribution()
                    median = dist["median"] if dist["count"] >= 3 else 0.3
                    dist_i = abs(iou_i - median)
                    dist_j = abs(iou_j - median)
                    victim = i if dist_i < dist_j else j
                elif iou_i is not None:
                    victim = j  # j has no metrics, remove it
                elif iou_j is not None:
                    victim = i  # i has no metrics, remove it
                else:
                    victim = min(i, j)  # both unverified, remove older
                
                if victim not in to_remove:
                    to_remove.add(victim)
        
        #  FIFO 
        if len(to_remove) < num_to_remove:
            for idx in range(len(self._memory)):
                if idx not in to_remove and len(to_remove) < num_to_remove:
                    to_remove.add(idx)
        
        for idx in sorted(to_remove, reverse=True):
            self._outer_memory.append(self._compress_to_outer(self._memory[idx]))
            self._memory.pop(idx)
    
    def _evict_by_model(self, num_to_remove: int) -> None:
        """： LLM ，。
        
        （ prompt）：
        1. /，
        2. 
        3. 
        
         API ， similarity 。
        """
        if self._api_caller is None:
            print("[Memory] No API caller set for model_decision, falling back to similarity eviction")
            self._evict_by_similarity(num_to_remove)
            return
        
        entries_summary = []
        for idx, entry in enumerate(self._memory):
            iou_str = ""
            if entry.evaluation_metrics:
                iou_val = entry.evaluation_metrics.get("iou")
                if iou_val is not None:
                    iou_str = f", IoU={iou_val:.3f}"
            tools_used = [tc.get("name", "?") for tc in (entry.tool_calls or [])]
            tools_str = "→".join(tools_used) if tools_used else "none"
            entries_summary.append(
                f"[{idx}] task=\"{entry.task[:80]}\" obj=\"{entry.object_name or 'N/A'}\" "
                f"tools={tools_str}{iou_str} time={entry.timestamp}"
            )
        
        entries_text = "\n".join(entries_summary)
        
        # Provide IoU distribution context so the model can judge relatively
        dist = self._get_iou_distribution()
        dist_note = ""
        if dist["count"] >= 3:
            dist_note = (
                f"\nIoU distribution across memory: "
                f"min={dist['min']:.3f}, p25={dist['p25']:.3f}, "
                f"median={dist['median']:.3f}, p75={dist['p75']:.3f}, "
                f"max={dist['max']:.3f} (higher IoU = better performance).\n"
            )
        
        # Count verified/unverified for the prompt
        n_verified = sum(1 for e in self._memory if self._has_verified_metrics(e))
        n_unverified = len(self._memory) - n_verified
        verified_note = ""
        if n_unverified > 0:
            verified_note = (
                f"\n{n_verified} entries have verified IoU metrics, "
                f"{n_unverified} are unverified (no metrics). "
                f"Unverified entries carry less value.\n"
            )
        
        prompt = MEMORY_EVICTION_PROMPT_TEMPLATE.format(
            num_entries=len(self._memory),
            max_size=self.max_size,
            num_to_remove=num_to_remove,
            dist_note=dist_note,
            verified_note=verified_note,
            entries_text=entries_text,
        )
        
        try:
            response = self._api_caller(
                "chat/completions",
                {
                    "model": self._eviction_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            import re
            #  JSON 
            match = re.search(r'\[[\d\s,]+\]', content)
            if match:
                indices_to_remove = json.loads(match.group())
                valid_indices = set()
                for idx in indices_to_remove:
                    if isinstance(idx, int) and 0 <= idx < len(self._memory):
                        valid_indices.add(idx)
                
                if len(valid_indices) >= num_to_remove:
                    valid_indices = set(list(valid_indices)[:num_to_remove])
                elif len(valid_indices) > 0:
                    #  FIFO 
                    for idx in range(len(self._memory)):
                        if idx not in valid_indices and len(valid_indices) < num_to_remove:
                            valid_indices.add(idx)
                else:
                    raise ValueError("No valid indices parsed")
                
                removed_ids = [self._memory[i].sample_id for i in sorted(valid_indices)]
                print(f"[Memory] Model decided to remove {len(valid_indices)} entries: {removed_ids}")
                
                for idx in sorted(valid_indices, reverse=True):
                    self._outer_memory.append(self._compress_to_outer(self._memory[idx]))
                    self._memory.pop(idx)
                return
            else:
                raise ValueError(f"Could not parse index array from: {content[:200]}")
        
        except Exception as e:
            print(f"[Memory] Model eviction failed ({e}), falling back to similarity")
            self._evict_by_similarity(num_to_remove)
    
    def _compute_text_similarity(
        self,
        task1: str, obj1: Optional[str],
        task2: str, obj2: Optional[str]
    ) -> float:
        """（）"""
        def tokenize(text: str) -> set:
            if not text:
                return set()
            return set(text.lower().split())
        
        tokens1 = tokenize(task1) | tokenize(obj1 or "")
        tokens2 = tokenize(task2) | tokenize(obj2 or "")
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_tokens(self, text: str) -> int:
        """ token """
        return max(1, int(len(text) / self._CHARS_PER_TOKEN))
    
    def _search_outer_memory(
        self,
        task: str,
        object_name: Optional[str] = None,
        query_text_emb: Optional[np.ndarray] = None,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> List[CompactMemoryEntry]:
        """。

        When *query_text_emb* (SigLIP2) is provided, uses semantic similarity
        against outer memory entries (text→text via Jaccard since outer entries
        don't store image embeddings).  Falls back to Jaccard regardless.
        """
        if not self._outer_memory:
            return []
        
        dist = self._get_iou_distribution()
        scored: List[Tuple[float, CompactMemoryEntry]] = []
        for entry in self._outer_memory:
            sim = self._compute_text_similarity(task, object_name, entry.task, entry.object_name)
            iou_bonus = 0.0
            if isinstance(entry.iou, (int, float)) and entry.iou > 0:
                if dist["count"] >= 3:
                    if entry.iou >= dist["p75"]:
                        iou_bonus = 0.08
                    elif entry.iou >= dist["median"]:
                        iou_bonus = 0.05
                    elif entry.iou >= dist["p25"]:
                        iou_bonus = 0.02
                else:
                    iou_bonus = min(0.08, entry.iou * 0.1)
            score = sim + iou_bonus
            if sim >= min_similarity:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]
    
    def get_relevant_memories(
        self,
        task: str,
        object_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: float = 0.15,
        token_budget: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """ memory （，）
        
        ：
        1. （inner memory）
           （ SigLIP2  Jaccard ）
        2. ，（outer memory）
        3.  token 
        
        Args:
            task: 
            object_name: 
            query_image_path: （ SigLIP2 ）
            top_k: （None  self.retrieval_top_k）
            min_similarity: 
            token_budget: token 
        
        Returns:
             memory （）
        """
        if not self._memory and not self._outer_memory:
            return []
        
        if token_budget is None:
            token_budget = self.context_token_budget
        
        # 1) RAG  TopK （）
        effective_top_k = top_k if top_k is not None else self.retrieval_top_k
        scored_candidates = self._retrieve_scored_candidates(
            task=task,
            object_name=object_name,
            query_image_path=query_image_path,
            top_k=effective_top_k,
            min_similarity=min_similarity,
        )
        
        # 2.  token 
        selected: List[MemoryEntry] = []
        tokens_used = self._estimate_tokens("📚 **Relevant Historical Cases:**\n\n")
        
        for _idx, _score, _sim, entry in scored_candidates:
            if len(selected) >= effective_top_k:
                break
            entry_text = self._format_single_entry(entry, len(selected) + 1)
            entry_tokens = self._estimate_tokens(entry_text)
            
            if tokens_used + entry_tokens <= token_budget:
                selected.append(entry)
                tokens_used += entry_tokens
            else:
                entry_text_compact = self._format_single_entry(entry, len(selected) + 1, compact=True)
                compact_tokens = self._estimate_tokens(entry_text_compact)
                if tokens_used + compact_tokens <= token_budget:
                    selected.append(entry)
                    tokens_used += compact_tokens
                else:
                    break
        
        # 3. ，
        min_inner_results = min(3, effective_top_k)
        outer_supplements: List[CompactMemoryEntry] = []
        if len(selected) < min_inner_results and self._outer_memory:
            outer_results = self._search_outer_memory(
                task, object_name, top_k=5, min_similarity=0.1,
            )
            if outer_results:
                outer_supplements = outer_results

        # Store in thread-local to avoid race conditions in parallel workers
        self._thread_local.last_outer_supplements = outer_supplements
        self._last_outer_supplements = outer_supplements  # backward compat for serial mode
        
        return selected
    
    def _get_quality_indicator(self, entry: MemoryEntry, iou_dist: Optional[Dict[str, float]] = None) -> str:
        """Return a quantitative quality indicator based on IoU.
        
        Unverified entries (no metrics) are explicitly marked as such.
        Verified entries show IoU value and percentile rank.
        """
        if not self._has_verified_metrics(entry):
            return "[unverified]"
        iou = entry.evaluation_metrics.get("iou", 0)
        dist = iou_dist or self._get_iou_distribution()
        if dist["count"] < 3:
            return f"[IoU={iou:.3f}]"
        if iou >= dist["p75"]:
            return f"[IoU={iou:.3f}, >p75]"
        if iou >= dist["median"]:
            return f"[IoU={iou:.3f}, >median]"
        if iou >= dist["p25"]:
            return f"[IoU={iou:.3f}, <median]"
        return f"[IoU={iou:.3f}, <p25]"

    def _get_strategy_hint(self, entry: MemoryEntry, iou_dist: Optional[Dict[str, float]] = None) -> str:
        """Generate a factual observation about this entry's trajectory and outcome.

        Highlights both positive and negative examples for the decision model.
        """
        if not self._has_verified_metrics(entry):
            return ""

        iou = entry.evaluation_metrics.get("iou", 0)
        if not isinstance(iou, (int, float)):
            return ""

        tool_names = [tc.get("name", "?") for tc in (entry.tool_calls or [])]
        used_helpers = [t for t in tool_names if t != "detection"]
        detection_only = len(used_helpers) == 0

        strategy = "detection-only" if detection_only else f"{' + '.join(set(used_helpers))} → detection"
        
        # Add a relative hint based on distribution
        dist = iou_dist or self._get_iou_distribution()
        label = ""
        if dist["count"] >= 3:
            if iou < dist["p25"]:
                label = " — negative example, consider what went wrong"
            elif iou >= dist["p75"]:
                label = " — strong result, worth replicating"
        
        return f"({strategy}, IoU={iou:.3f}{label})"

    def _format_single_entry(
        self,
        entry: MemoryEntry,
        idx: int,
        compact: bool = False,
        max_tool_summaries: int = 3,
        iou_dist: Optional[Dict[str, float]] = None,
    ) -> str:
        """ memory （ token ）
        
        Args:
            entry: 
            idx: 
            compact: （ task_context）
            max_tool_summaries: 
            iou_dist: IoU distribution stats (from _get_iou_distribution)
        """
        quality = self._get_quality_indicator(entry, iou_dist=iou_dist)
        verified = self._has_verified_metrics(entry)
        
        if verified:
            lines = [f"{quality} **Case {idx}:**"]
        else:
            lines = [f"{quality} **Case {idx} (unverified — outcome unknown):**"]
        
        lines.append(f"- Task: {entry.task}")
        if entry.object_name:
            lines.append(f"- Object: {entry.object_name}")
        
        # （IoU  P@50，）+ 
        if verified:
            _useful = {"iou", "precision_at_50"}
            useful = {
                k: v for k, v in entry.evaluation_metrics.items()
                if k.lower() in _useful and isinstance(v, (int, float))
            }
            if useful:
                metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in useful.items())
                iou_val = entry.evaluation_metrics.get("iou")
                if iou_dist and iou_dist["count"] >= 3 and isinstance(iou_val, (int, float)):
                    if iou_val >= iou_dist["p75"]:
                        pos = "above p75"
                    elif iou_val >= iou_dist["median"]:
                        pos = "above median"
                    elif iou_val >= iou_dist["p25"]:
                        pos = "below median"
                    else:
                        pos = "below p25"
                    metrics_str += f" [{pos}]"
                lines.append(f"- Metrics (higher=better): {metrics_str}")
        
        if not compact:
            if entry.tool_calls:
                tool_names = [tc.get("name", "?") for tc in entry.tool_calls]
                lines.append(f"- Tool chain: {' → '.join(tool_names)}")
            
            # Strategy hint (factual, highlights positive and negative examples)
            hint = self._get_strategy_hint(entry, iou_dist=iou_dist)
            if hint:
                lines.append(f"- {hint}")
            
            if entry.strategy_reasoning:
                sr = entry.strategy_reasoning[:200]
                if len(entry.strategy_reasoning) > 200:
                    sr += "..."
                lines.append(f"- 💭 Strategy reasoning: {sr}")
            
            if entry.strategy_reflection:
                ref = entry.strategy_reflection[:150]
                if len(entry.strategy_reflection) > 150:
                    ref += "..."
                lines.append(f"- 🔄 Reflection: {ref}")
            
            if entry.dynamic_params_used:
                dp_str = ", ".join(f"{k}={v}" for k, v in entry.dynamic_params_used.items())
                if len(dp_str) > 150:
                    dp_str = dp_str[:150] + "..."
                lines.append(f"- ⚙️ Dynamic params: {dp_str}")
            
            # （， 3 ）
            if entry.decision_trace:
                lines.append(f"- 📝 Decision trace ({len(entry.decision_trace)} steps):")
                for step in entry.decision_trace[:3]:
                    step_truncated = step[:200] if len(step) > 200 else step
                    lines.append(f"    {step_truncated}")
                if len(entry.decision_trace) > 3:
                    lines.append(f"    ... ({len(entry.decision_trace) - 3} more steps)")
            
            # （ helper tool ，detection ）
            if entry.tool_summaries:
                for summary in entry.tool_summaries[:max_tool_summaries]:
                    summary_lines = [l.strip() for l in summary.strip().split('\n') if l.strip()]
                    if not summary_lines:
                        continue
                    header = summary_lines[0]
                    is_detection = header.startswith("[detection]")
                    if is_detection:
                        lines.append(f"  • {header}")
                    else:
                        # For helper tools, show key findings (up to 4 lines)
                        content_lines = summary_lines[:5]
                        lines.append(f"  • {content_lines[0]}")
                        for cl in content_lines[1:]:
                            truncated = cl[:150] + "..." if len(cl) > 150 else cl
                            lines.append(f"    {truncated}")
            
            # Task context
            if entry.task_context:
                tc_truncated = entry.task_context[:200]
                if len(entry.task_context) > 200:
                    tc_truncated += "..."
                lines.append(f"- Context: {tc_truncated}")
        else:
            # ： + hint +  + （）
            if entry.tool_calls:
                tool_names = [tc.get("name", "?") for tc in entry.tool_calls]
                lines.append(f"- Tools: {' → '.join(tool_names)}")
            hint = self._get_strategy_hint(entry, iou_dist=iou_dist)
            if hint:
                lines.append(f"- {hint}")
            if entry.strategy_reasoning:
                lines.append(f"- 💭 {entry.strategy_reasoning[:150]}")
            if entry.dynamic_params_used:
                dp_str = ", ".join(f"{k}={v}" for k, v in entry.dynamic_params_used.items())
                if len(dp_str) > 100:
                    dp_str = dp_str[:100] + "..."
                lines.append(f"- ⚙️ Params: {dp_str}")
        
        lines.append("")  # 
        return "\n".join(lines)
    
    def _build_memory_strategy_summary(self, memories: List[MemoryEntry]) -> str:
        """Produce a distribution-aware strategy summary from recalled memory entries.

        Presents aggregate statistics per strategy type with IoU distribution info.
        No fixed quality thresholds — the model judges relative performance by
        comparing across strategies and looking at the distribution.
        """
        if not memories:
            return ""

        total = 0
        all_ious: List[float] = []
        detection_only_count = 0
        detection_only_ious: List[float] = []
        # Track per-helper-tool combos: key = frozenset of helper names
        helper_combos: Dict[str, List[float]] = {}
        no_metrics_count = 0

        for entry in memories:
            iou = None
            if entry.evaluation_metrics:
                iou = entry.evaluation_metrics.get("iou")
            if not isinstance(iou, (int, float)):
                iou = None

            tool_names = [tc.get("name", "?") for tc in (entry.tool_calls or [])]
            used_helpers = sorted(set(t for t in tool_names if t != "detection"))
            total += 1

            if iou is not None:
                all_ious.append(iou)
            else:
                no_metrics_count += 1

            if len(used_helpers) == 0:
                detection_only_count += 1
                if iou is not None:
                    detection_only_ious.append(iou)
            else:
                combo_key = " + ".join(used_helpers)
                if combo_key not in helper_combos:
                    helper_combos[combo_key] = []
                if iou is not None:
                    helper_combos[combo_key].append(iou)

        lines = ["📊 **Strategy Comparison** (from recalled cases, IoU: higher=better):"]

        # Overall IoU distribution — factual, no judgment
        if all_ious:
            sorted_ious = sorted(all_ious)
            n = len(sorted_ious)
            median = sorted_ious[n // 2] if n % 2 else (sorted_ious[n // 2 - 1] + sorted_ious[n // 2]) / 2
            lines.append(
                f"- Overall ({n} cases): "
                f"min={sorted_ious[0]:.3f}, median={median:.3f}, "
                f"max={sorted_ious[-1]:.3f}, mean={sum(sorted_ious) / n:.3f}"
            )
        if no_metrics_count:
            lines.append(f"- {no_metrics_count} case(s) without metrics yet")

        # Detection-only strategy
        if detection_only_ious:
            avg = sum(detection_only_ious) / len(detection_only_ious)
            lines.append(
                f"- detection-only: {detection_only_count} cases, avg IoU={avg:.3f}"
            )
        elif detection_only_count > 0:
            lines.append(f"- detection-only: {detection_only_count} cases, no metrics yet")

        # Per-helper-tool combo breakdown
        for combo_key in sorted(helper_combos.keys()):
            combo_ious = helper_combos[combo_key]
            combo_total = len(combo_ious)
            if combo_ious:
                avg = sum(combo_ious) / len(combo_ious)
                lines.append(f"- {combo_key} → detection: {combo_total} cases, avg IoU={avg:.3f}")
            else:
                lines.append(f"- {combo_key} → detection: count only, no metrics yet")

        # Highlight best and worst cases for comparison
        if len(all_ious) >= 2:
            entries_with_iou = [
                e for e in memories
                if e.evaluation_metrics and isinstance(e.evaluation_metrics.get("iou"), (int, float))
            ]
            if entries_with_iou:
                best_entry = max(entries_with_iou, key=lambda e: e.evaluation_metrics["iou"])
                worst_entry = min(entries_with_iou, key=lambda e: e.evaluation_metrics["iou"])
                best_tools = " → ".join(tc.get("name", "?") for tc in (best_entry.tool_calls or []))
                worst_tools = " → ".join(tc.get("name", "?") for tc in (worst_entry.tool_calls or []))
                lines.append(
                    f"- Best case: IoU={best_entry.evaluation_metrics['iou']:.3f} "
                    f"({best_tools or 'unknown'})"
                )
                lines.append(
                    f"- Worst case: IoU={worst_entry.evaluation_metrics['iou']:.3f} "
                    f"({worst_tools or 'unknown'})"
                )

        # Per-individual-tool usage frequency and contribution
        all_available_tools = {"web_search", "dreamer", "zoom_in"}
        tool_ious: Dict[str, List[float]] = {t: [] for t in all_available_tools}
        tool_counts: Dict[str, int] = {t: 0 for t in all_available_tools}
        for entry in memories:
            tool_names_set = set(tc.get("name", "") for tc in (entry.tool_calls or []))
            iou = None
            if entry.evaluation_metrics:
                iou = entry.evaluation_metrics.get("iou")
            for t in all_available_tools:
                if t in tool_names_set:
                    tool_counts[t] += 1
                    if isinstance(iou, (int, float)):
                        tool_ious[t].append(iou)

        tool_parts = []
        for t in sorted(all_available_tools):
            cnt = tool_counts[t]
            ious = tool_ious[t]
            if ious:
                avg = sum(ious) / len(ious)
                tool_parts.append(f"{t}: {cnt}/{total} cases, avg IoU={avg:.3f}")
            elif cnt > 0:
                tool_parts.append(f"{t}: {cnt}/{total} cases, no metrics")
            else:
                tool_parts.append(f"{t}: not used in recalled cases")
        if tool_parts:
            lines.append("- Individual tool usage: " + "; ".join(tool_parts))

        # Flag tools that are underexplored — the model should be aware of untried options
        underexplored = [t for t in all_available_tools if tool_counts[t] <= 1]
        if underexplored and total >= 3:
            lines.append(
                f"- Note: {', '.join(sorted(underexplored))} "
                f"{'has' if len(underexplored) == 1 else 'have'} "
                f"rarely been tried for similar tasks — "
                f"their potential contribution is unknown."
            )

        return "\n".join(lines)

    def format_memories_for_context(
        self,
        memories: List[MemoryEntry],
        include_images: bool = False,
        max_tool_calls_per_entry: int = 3,
        token_budget: Optional[int] = None,
        task: Optional[str] = None,
        object_name: Optional[str] = None,
        query_image_path: Optional[str] = None,
    ) -> str:
        """ memory 
        
        ：
        1. （，）
        2. （ + ）
        3. （）
        4. （）
        
        Args:
            memories: 
            include_images: 
            max_tool_calls_per_entry: 
            token_budget: token 
            task: （）
            object_name: （）
            query_image_path: （）
        
        Returns:
            
        """
        if token_budget is None:
            token_budget = self.context_token_budget
        
        parts = []
        tokens_used = 0
        
        # ===  1 ： ===
        exp_budget = min(800, token_budget // 4)  #  1/4 
        if task:
            exp_text = self.experience_pool.format_for_context(
                task, object_name, top_k=8, token_budget=exp_budget,
            )
            if exp_text:
                exp_tokens = self._estimate_tokens(exp_text)
                parts.append(exp_text)
                tokens_used += exp_tokens

        # ===  1.5 ：（）===
        commonsense_templates: List[Dict[str, Any]] = []
        if task and self.enable_commonsense_templates:
            commonsense_templates = self.retrieve_common_sense_templates(
                task, object_name, query_image_path=query_image_path, top_k=2,
            )
            if commonsense_templates:
                cs_budget = min(500, max(120, token_budget // 6))
                cs_text = self.format_common_sense_templates_for_context(
                    commonsense_templates,
                    token_budget=cs_budget,
                )
                if cs_text:
                    cs_tokens = self._estimate_tokens(cs_text)
                    if tokens_used + cs_tokens <= token_budget:
                        parts.append(cs_text)
                        tokens_used += cs_tokens
        
        # ===  2 ： ===
        if memories:
            iou_dist = self._get_iou_distribution()
            # Count verified vs unverified among recalled cases
            n_verified = sum(1 for e in memories if self._has_verified_metrics(e))
            n_unverified = len(memories) - n_verified
            
            # Provide distribution context header so the model knows the scale
            verified_note = ""
            if n_unverified > 0:
                verified_note = (
                    f"\n_{n_verified} verified cases (with IoU), "
                    f"{n_unverified} unverified (trajectory only, no reward signal). "
                    f"Verified cases are the primary learning signal; "
                    f"unverified cases show strategy patterns but outcome is unknown._"
                )
            if iou_dist["count"] >= 3:
                dist_header = (
                    "📚 **Relevant Historical Cases (Inner Memory):**\n"
                    f"_IoU distribution (verified cases): "
                    f"min={iou_dist['min']:.3f}, p25={iou_dist['p25']:.3f}, "
                    f"median={iou_dist['median']:.3f}, p75={iou_dist['p75']:.3f}, "
                    f"max={iou_dist['max']:.3f} (higher=better)_"
                    f"{verified_note}\n\n"
                )
            else:
                if verified_note:
                    dist_header = f"📚 **Relevant Historical Cases (Inner Memory):**{verified_note}\n\n"
                else:
                    dist_header = "📚 **Relevant Historical Cases (Inner Memory):**\n\n"
            tokens_used += self._estimate_tokens(dist_header)
            parts.append(dist_header)
            
            for idx, entry in enumerate(memories, 1):
                full_text = self._format_single_entry(entry, idx, compact=False, max_tool_summaries=max_tool_calls_per_entry, iou_dist=iou_dist)
                full_tokens = self._estimate_tokens(full_text)
                
                if tokens_used + full_tokens <= token_budget:
                    parts.append(full_text)
                    tokens_used += full_tokens
                else:
                    compact_text = self._format_single_entry(entry, idx, compact=True, iou_dist=iou_dist)
                    compact_tokens = self._estimate_tokens(compact_text)
                    if tokens_used + compact_tokens <= token_budget:
                        parts.append(compact_text)
                        tokens_used += compact_tokens
                    else:
                        remaining = len(memories) - idx + 1
                        if remaining > 0:
                            parts.append(f"_(... {remaining} more similar cases omitted due to context length)_\n")
                        break
        
        # ===  3 ：（） ===
        # Prefer thread-local (safe for parallel workers), fall back to instance attr
        outer_supplements = getattr(self._thread_local, "last_outer_supplements",
                                    getattr(self, "_last_outer_supplements", []))
        if outer_supplements and tokens_used < token_budget:
            outer_header = "\n📦 **Archived Memory (Compressed Summaries):**\n"
            outer_header_tokens = self._estimate_tokens(outer_header)
            if tokens_used + outer_header_tokens < token_budget:
                parts.append(outer_header)
                tokens_used += outer_header_tokens
                
                for oe in outer_supplements:
                    line = (
                        f"- [{oe.object_name or 'N/A'}] {oe.tool_chain} → IoU={oe.iou:.3f}: "
                        f"{oe.summary[:120]}"
                    )
                    if oe.strategy_reasoning:
                        line += f" | 💭 {oe.strategy_reasoning[:80]}"
                    line += "\n"
                    line_tokens = self._estimate_tokens(line)
                    if tokens_used + line_tokens <= token_budget:
                        parts.append(line)
                        tokens_used += line_tokens
                    else:
                        break

        # ===  3.5 ：Skill （ + ）===
        if tokens_used < token_budget and self._skill_profiles:
            profile_budget = min(500, max(120, token_budget // 6))
            profile_text = self._format_skill_profiles_for_context(token_budget=profile_budget)
            if profile_text:
                profile_tokens = self._estimate_tokens(profile_text)
                if tokens_used + profile_tokens <= token_budget:
                    parts.append(profile_text)
                    tokens_used += profile_tokens
        
        if not memories and not outer_supplements and not parts and not commonsense_templates:
            return ""
        
        # ===  4 ： ===
        if memories:
            strategy_summary = self._build_memory_strategy_summary(memories)
            if strategy_summary:
                summary_tokens = self._estimate_tokens(strategy_summary)
                if tokens_used + summary_tokens <= token_budget:
                    parts.append("\n" + strategy_summary)
        
        return "\n".join(parts)
    
    def clear_memory(self, clear_outer: bool = True, clear_experience: bool = False) -> None:
        """
        
        Args:
            clear_outer: （ True）
            clear_experience: （ False，）
        """
        num_inner = len(self._memory)
        num_outer = len(self._outer_memory)
        self._memory = []
        
        if clear_outer:
            self._outer_memory = []
        
        #  skill ： outer memory，
        self._rebuild_skill_profiles()
        self._commonsense_ready = False
        
        if clear_experience:
            self.experience_pool.clear()
        
        if self.persist_dir:
            memory_file = self.persist_dir / "memory.json"
            if memory_file.exists():
                memory_file.unlink()
            outer_file = self.persist_dir / "outer_memory.json"
            if clear_outer and outer_file.exists():
                outer_file.unlink()
            print(f"[Memory] Cleared {num_inner} inner + {num_outer if clear_outer else 0} outer entries"
                  f"{' + experience pool' if clear_experience else ''}")
        else:
            print(f"[Memory] Cleared {num_inner} inner entries")

    def save(self) -> None:
        """（ entry ）"""
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        """ + """
        if not self.persist_dir:
            return
        
        memory_file = self.persist_dir / "memory.json"
        try:
            data = {
                "max_size": self.max_size,
                "eviction_strategy": self.eviction_strategy,
                "similarity_threshold": self.similarity_threshold,
                "duplicate_similarity_threshold": self.duplicate_similarity_threshold,
                "retrieval_top_k": self.retrieval_top_k,
                "enqueue_dedupe_top_k": self.enqueue_dedupe_top_k,
                "entries": [asdict(entry) for entry in self._memory],
                "skill_profiles": self._skill_profiles,
            }
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save inner memory to disk: {e}")
        
        outer_file = self.persist_dir / "outer_memory.json"
        try:
            outer_data = {
                "version": 1,
                "total_entries": len(self._outer_memory),
                "entries": [asdict(entry) for entry in self._outer_memory],
            }
            with open(outer_file, "w", encoding="utf-8") as f:
                json.dump(outer_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save outer memory to disk: {e}")

        self._save_commonsense_templates_to_disk()
    
    def _load_from_disk(self) -> None:
        """ + """
        if not self.persist_dir:
            return
        
        memory_file = self.persist_dir / "memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.max_size = data.get("max_size", self.max_size)
                self.eviction_strategy = data.get("eviction_strategy", self.eviction_strategy)
                self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
                self.duplicate_similarity_threshold = data.get(
                    "duplicate_similarity_threshold",
                    self.duplicate_similarity_threshold,
                )
                self.retrieval_top_k = max(1, int(data.get("retrieval_top_k", self.retrieval_top_k)))
                self.enqueue_dedupe_top_k = max(
                    1,
                    int(data.get("enqueue_dedupe_top_k", self.enqueue_dedupe_top_k)),
                )
                
                self._memory = []
                for entry_dict in data.get("entries", []):
                    entry = MemoryEntry(**entry_dict)
                    self._memory.append(entry)
                # ： skill_profiles 
                saved_profiles = data.get("skill_profiles")
                if isinstance(saved_profiles, dict):
                    self._skill_profiles = saved_profiles
                else:
                    self._rebuild_skill_profiles()
                
                print(f"[Memory] Loaded {len(self._memory)} inner memory entries from disk")
            except Exception as e:
                print(f"[Warning] Failed to load inner memory from disk: {e}")
        
        outer_file = self.persist_dir / "outer_memory.json"
        if outer_file.exists():
            try:
                with open(outer_file, "r", encoding="utf-8") as f:
                    outer_data = json.load(f)
                
                self._outer_memory = []
                for entry_dict in outer_data.get("entries", []):
                    self._outer_memory.append(CompactMemoryEntry(**entry_dict))
                
                print(f"[Memory] Loaded {len(self._outer_memory)} outer memory entries from disk")
            except Exception as e:
                print(f"[Warning] Failed to load outer memory from disk: {e}")
        # ， outer memory
        self._rebuild_skill_profiles()
        self._load_commonsense_templates_from_disk()
    
    def get_statistics(self) -> Dict[str, Any]:
        """ memory （ +  + ）"""
        avg_metrics = {}
        metrics_count = 0
        for entry in self._memory:
            if entry.evaluation_metrics:
                metrics_count += 1
                for k, v in entry.evaluation_metrics.items():
                    if k not in avg_metrics:
                        avg_metrics[k] = []
                    avg_metrics[k].append(v)
        
        for k in avg_metrics:
            avg_metrics[k] = float(np.mean(avg_metrics[k])) if avg_metrics[k] else 0.0
        
        exp_stats = {
            "total_insights": len(self.experience_pool.insights),
            "total_samples_seen": self.experience_pool._total_samples_seen,
            "pending_observations": len(self.experience_pool._pending_observations),
        }
        if self.experience_pool.insights:
            exp_stats["avg_confidence"] = float(np.mean([i.confidence for i in self.experience_pool.insights]))
            cat_counts = {}
            for i in self.experience_pool.insights:
                cat_counts[i.category] = cat_counts.get(i.category, 0) + 1
            exp_stats["category_distribution"] = cat_counts
        
        return {
            "inner_memory_entries": len(self._memory),
            "outer_memory_entries": len(self._outer_memory),
            "total_entries": len(self._memory) + len(self._outer_memory),
            "max_size": self.max_size,
            "eviction_strategy": self.eviction_strategy,
            "retrieval_top_k": self.retrieval_top_k,
            "enqueue_dedupe_top_k": self.enqueue_dedupe_top_k,
            "duplicate_similarity_threshold": self.duplicate_similarity_threshold,
            "entries_with_metrics": metrics_count,
            "average_metrics": avg_metrics,
            "experience_pool": exp_stats,
            "skill_profiles": {
                "count": len(self._skill_profiles),
                "profiles": self._skill_profiles,
            },
            "commonsense_templates": {
                "enabled": self.enable_commonsense_templates,
                "count": len(self._commonsense_templates),
                "max_per_pair": self.commonsense_max_per_pair,
                "max_total": self.commonsense_max_total if self.commonsense_max_total is not None else "unlimited",
                "datasets_root": str(self.datasets_root),
                "templates_dir": str(self.commonsense_templates_dir),
                "auto_build": self.commonsense_auto_build,
            },
        }

    # Visualization
    def _image_to_thumbnail_b64(self, img_path: str, max_side: int = 200) -> Optional[str]:
        """ base64 data-URI"""
        try:
            if not img_path or not os.path.exists(img_path):
                return None
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    def export_visualization(self, output_path: Optional[str] = None) -> str:
        """ memory  HTML 
        
        Args:
            output_path:  HTML ，None  persist_dir/memory_vis.html
        
        Returns:
             HTML 
        """
        if output_path is None:
            if self.persist_dir:
                output_path = str(self.persist_dir / "memory_vis.html")
            else:
                output_path = "memory_vis.html"
        
        stats = self.get_statistics()
        
        #  entry  HTML 
        cards_html = []
        for idx, entry in enumerate(self._memory):
            img_thumb = self._image_to_thumbnail_b64(entry.image_path)
            mask_thumb = self._image_to_thumbnail_b64(entry.mask_path)
            
            img_html = f'<img src="{img_thumb}" alt="input" />' if img_thumb else '<div class="no-img">No Image</div>'
            mask_html = f'<img src="{mask_thumb}" alt="mask" />' if mask_thumb else '<div class="no-img">No Mask</div>'
            
            # IoU bar
            iou = 0.0
            if entry.evaluation_metrics:
                iou = entry.evaluation_metrics.get("iou", 0.0)
            iou_pct = max(0, min(100, iou * 100))
            iou_color = "#4caf50" if iou >= 0.5 else "#ff9800" if iou >= 0.3 else "#f44336"
            
            # Metrics display
            metrics_html = ""
            if entry.evaluation_metrics:
                metrics_items = []
                for k, v in entry.evaluation_metrics.items():
                    if isinstance(v, float):
                        metrics_items.append(f'<span class="metric"><b>{k}</b>: {v:.4f}</span>')
                    else:
                        metrics_items.append(f'<span class="metric"><b>{k}</b>: {v}</span>')
                metrics_html = " &nbsp;|&nbsp; ".join(metrics_items)
            else:
                metrics_html = '<span class="metric" style="color:#999">No metrics</span>'
            
            # Tool calls trajectory
            tool_trajectory_html = ""
            if entry.tool_calls:
                steps = []
                for i, tc in enumerate(entry.tool_calls):
                    tool_name = tc.get("name", "?")
                    tool_args = tc.get("arguments", {})
                    args_display = {}
                    for k, v in tool_args.items():
                        if k == "image_path":
                            args_display[k] = os.path.basename(str(v))
                        elif isinstance(v, str) and len(v) > 80:
                            args_display[k] = v[:80] + "..."
                        else:
                            args_display[k] = v
                    args_str = ", ".join(f'{k}="{v}"' for k, v in args_display.items())
                    
                    icon = "🔍" if tool_name == "detection" else "🌐" if tool_name == "web_search" else "🔮" if tool_name == "dreamer" else "🔎" if tool_name == "zoom_in" else "🔧"
                    steps.append(f'<div class="tool-step"><span class="step-num">Step {i+1}</span> {icon} <b>{tool_name}</b>({args_str})</div>')
                tool_trajectory_html = "\n".join(steps)
            else:
                tool_trajectory_html = '<div class="tool-step" style="color:#999">No tool calls recorded</div>'
            
            # Tool summaries
            summaries_html = ""
            if entry.tool_summaries:
                sums = []
                for s in entry.tool_summaries:
                    escaped = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                    sums.append(f'<div class="summary-item">{escaped}</div>')
                summaries_html = "\n".join(sums)
            
            # LLM reasoning
            reasoning_html = ""
            if entry.llm_reasoning:
                r_items = []
                for i, r in enumerate(entry.llm_reasoning):
                    escaped = r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                    r_items.append(f'<div class="reasoning-item"><b>Round {i+1}:</b><br/>{escaped[:500]}{"..." if len(escaped) > 500 else ""}</div>')
                reasoning_html = "\n".join(r_items)
            
            # Task context
            tc_html = ""
            if entry.task_context:
                escaped = entry.task_context.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
                tc_html = f'<div class="task-context"><b>Task Context:</b> {escaped[:300]}{"..." if len(escaped) > 300 else ""}</div>'
            
            card = f"""
            <div class="card" id="entry-{idx}">
                <div class="card-header">
                    <span class="card-idx">#{idx}</span>
                    <span class="card-id">{entry.sample_id}</span>
                    <span class="card-time">{entry.timestamp}</span>
                    <div class="iou-bar-container">
                        <div class="iou-bar" style="width:{iou_pct}%;background:{iou_color}"></div>
                        <span class="iou-label">IoU: {iou:.4f}</span>
                    </div>
                </div>
                <div class="card-body">
                    <div class="images-row">
                        <div class="img-col">
                            <div class="img-label">Input Image</div>
                            {img_html}
                        </div>
                        <div class="img-col">
                            <div class="img-label">Pred Mask</div>
                            {mask_html}
                        </div>
                        <div class="info-col">
                            <div class="task-text"><b>Task:</b> {entry.task}</div>
                            <div class="obj-text"><b>Object:</b> {entry.object_name or "N/A"}</div>
                            <div class="metrics-row">{metrics_html}</div>
                        </div>
                    </div>
                    <details class="trajectory-section">
                        <summary>🛠 Tool Call Trajectory ({len(entry.tool_calls)} calls)</summary>
                        <div class="trajectory-content">
                            {tool_trajectory_html}
                        </div>
                    </details>
                    <details class="summaries-section">
                        <summary>📋 Tool Summaries ({len(entry.tool_summaries)} items)</summary>
                        <div class="summaries-content">
                            {summaries_html}
                        </div>
                    </details>
                    {"<details class='reasoning-section'><summary>🧠 LLM Reasoning (" + str(len(entry.llm_reasoning)) + " rounds)</summary><div class='reasoning-content'>" + reasoning_html + "</div></details>" if reasoning_html else ""}
                    {tc_html}
                </div>
            </div>
            """
            cards_html.append(card)
        
        avg_metrics_str = ""
        if stats.get("average_metrics"):
            items = []
            for k, v in stats["average_metrics"].items():
                items.append(f"<b>{k}</b>: {v:.4f}")
            avg_metrics_str = " &nbsp;|&nbsp; ".join(items)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory Visualization</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
h1 {{ text-align: center; margin-bottom: 10px; color: #1a73e8; }}
.stats {{ text-align: center; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
.stats .stat-item {{ display: inline-block; margin: 0 15px; }}
.stats .stat-value {{ font-size: 1.5em; font-weight: bold; color: #1a73e8; }}
.stats .stat-label {{ font-size: 0.85em; color: #666; }}
.avg-metrics {{ margin-top: 8px; font-size: 0.9em; color: #444; }}
.card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px; overflow: hidden; }}
.card-header {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #fafafa; border-bottom: 1px solid #eee; flex-wrap: wrap; }}
.card-idx {{ font-weight: bold; color: #1a73e8; font-size: 1.1em; min-width: 40px; }}
.card-id {{ font-family: monospace; font-size: 0.85em; color: #666; }}
.card-time {{ font-size: 0.8em; color: #999; margin-left: auto; }}
.iou-bar-container {{ width: 200px; height: 20px; background: #e0e0e0; border-radius: 10px; position: relative; overflow: hidden; }}
.iou-bar {{ height: 100%; border-radius: 10px; transition: width 0.3s; }}
.iou-label {{ position: absolute; top: 0; left: 0; right: 0; text-align: center; font-size: 0.75em; line-height: 20px; color: #fff; font-weight: bold; text-shadow: 0 0 2px rgba(0,0,0,0.5); }}
.card-body {{ padding: 12px 16px; }}
.images-row {{ display: flex; gap: 16px; align-items: flex-start; margin-bottom: 10px; flex-wrap: wrap; }}
.img-col {{ text-align: center; }}
.img-col img {{ max-width: 200px; max-height: 200px; border-radius: 4px; border: 1px solid #ddd; }}
.img-label {{ font-size: 0.8em; color: #888; margin-bottom: 4px; }}
.no-img {{ width: 200px; height: 120px; background: #eee; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 0.85em; }}
.info-col {{ flex: 1; min-width: 250px; }}
.task-text, .obj-text {{ font-size: 0.9em; margin-bottom: 4px; }}
.metrics-row {{ margin-top: 6px; }}
.metric {{ font-size: 0.85em; }}
details {{ margin-top: 8px; }}
summary {{ cursor: pointer; font-weight: bold; font-size: 0.9em; color: #1a73e8; padding: 4px 0; }}
summary:hover {{ color: #0d47a1; }}
.tool-step {{ padding: 4px 8px; margin: 2px 0; background: #f9f9f9; border-left: 3px solid #1a73e8; font-size: 0.85em; font-family: monospace; word-break: break-all; }}
.step-num {{ font-weight: bold; color: #1a73e8; }}
.summary-item {{ padding: 6px 8px; margin: 2px 0; background: #f0f7ff; border-radius: 4px; font-size: 0.85em; }}
.reasoning-item {{ padding: 6px 8px; margin: 4px 0; background: #fff8e1; border-radius: 4px; font-size: 0.85em; }}
.task-context {{ margin-top: 6px; padding: 6px 8px; background: #e8f5e9; border-radius: 4px; font-size: 0.85em; }}
.filter-bar {{ text-align: center; margin-bottom: 16px; }}
.filter-bar input {{ padding: 8px 16px; width: 400px; border: 1px solid #ddd; border-radius: 20px; font-size: 0.9em; outline: none; }}
.filter-bar input:focus {{ border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.2); }}
</style>
</head>
<body>
<h1>📚 Memory Visualization</h1>

<div class="stats">
    <div class="stat-item"><div class="stat-value">{stats['total_entries']}</div><div class="stat-label">Entries</div></div>
    <div class="stat-item"><div class="stat-value">{stats['max_size']}</div><div class="stat-label">Max Size</div></div>
    <div class="stat-item"><div class="stat-value">{stats.get('entries_with_metrics', 0)}</div><div class="stat-label">With Metrics</div></div>
    <div class="stat-item"><div class="stat-value">{stats['eviction_strategy']}</div><div class="stat-label">Eviction</div></div>
    {f'<div class="avg-metrics">Average Metrics: {avg_metrics_str}</div>' if avg_metrics_str else ''}
</div>

<div class="filter-bar">
    <input type="text" id="filterInput" placeholder="Filter by task, object, sample_id..." oninput="filterCards()" />
</div>

<div id="cardsContainer">
{"".join(cards_html)}
</div>

<script>
function filterCards() {{
    const q = document.getElementById('filterInput').value.toLowerCase();
    document.querySelectorAll('.card').forEach(card => {{
        card.style.display = card.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
}}
</script>
</body>
</html>
"""
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"[Memory] Visualization exported to: {output_path}")
        return output_path

    def export_trajectory_log(self, output_path: Optional[str] = None) -> str:
        """（JSON ，）
        
        Args:
            output_path:  JSON ，None  persist_dir/trajectory_log.json
        
        Returns:
            
        """
        if output_path is None:
            if self.persist_dir:
                output_path = str(self.persist_dir / "trajectory_log.json")
            else:
                output_path = "trajectory_log.json"
        
        log_entries = []
        for entry in self._memory:
            log_entry = {
                "sample_id": entry.sample_id,
                "timestamp": entry.timestamp,
                "task": entry.task,
                "object_name": entry.object_name,
                "image_path": entry.image_path,
                "mask_path": entry.mask_path,
                "tool_trajectory": [],
                "llm_reasoning": entry.llm_reasoning or [],
                "task_context": entry.task_context,
                "evaluation_metrics": entry.evaluation_metrics,
            }
            
            #  tool 
            for i, tc in enumerate(entry.tool_calls or []):
                step = {
                    "step": i + 1,
                    "tool": tc.get("name", "?"),
                    "arguments": tc.get("arguments", {}),
                    "summary": entry.tool_summaries[i] if i < len(entry.tool_summaries or []) else None,
                }
                log_entry["tool_trajectory"].append(step)
            
            log_entries.append(log_entry)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)
        
        print(f"[Memory] Trajectory log exported to: {output_path}")
        return output_path
