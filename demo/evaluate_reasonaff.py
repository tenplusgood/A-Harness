"""
： ReasonAff  affordance detection 

：
- gIoU: Generalized IoU ( IoU)
- cIoU: Complete IoU ( IoU，)
- P_{50}: Precision at IoU threshold 0.5
- P_{50-95}: Average Precision from IoU 0.5 to 0.95 ( 0.05)
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import datasets

import queue as _queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import load_from_disk with fallback
try:
    from datasets import load_from_disk
except ImportError:
    # Fallback for older versions or different installations
    if hasattr(datasets, 'load_from_disk'):
        load_from_disk = datasets.load_from_disk
    else:
        # Last resort: use datasets.load function
        load_from_disk = datasets.load

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import AffordanceAgent


def calculate_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """ IoU (Intersection over Union)
    
    Args:
        mask_pred:  (H, W) bool array
        mask_gt:  (H, W) bool array
    
    Returns:
        IoU  [0, 1]
    """
    if mask_pred.shape != mask_gt.shape:
        from PIL import Image as PILImage
        mask_pred_pil = PILImage.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), PILImage.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def calculate_intersection_and_union(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[int, int]:
    """（ cIoU ）
    
    Args:
        mask_pred:  (H, W) bool array
        mask_gt:  (H, W) bool array
    
    Returns:
        (intersection_pixels, union_pixels)
    """
    if mask_pred.shape != mask_gt.shape:
        from PIL import Image as PILImage
        mask_pred_pil = PILImage.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), PILImage.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    
    intersection = int(np.logical_and(mask_pred, mask_gt).sum())
    union = int(np.logical_or(mask_pred, mask_gt).sum())
    return intersection, union


def _ensure_shape_reasonaff(mask_pred: np.ndarray, mask_gt: np.ndarray) -> np.ndarray:
    """Resize mask_pred to match mask_gt shape if needed."""
    if mask_pred.shape != mask_gt.shape:
        from PIL import Image as PILImage
        mask_pred_pil = PILImage.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), PILImage.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    return mask_pred


def calculate_kld(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """KLD (Kullback-Leibler Divergence). （0 = ）。"""
    mask_pred = _ensure_shape_reasonaff(mask_pred, mask_gt)
    pred = mask_pred.astype(np.float64)
    gt = mask_gt.astype(np.float64)
    pred_sum, gt_sum = pred.sum(), gt.sum()
    if gt_sum == 0:
        return 0.0
    if pred_sum == 0:
        return float('inf')
    pred_norm = pred / pred_sum
    gt_norm = gt / gt_sum
    eps = np.finfo(np.float64).eps
    pred_norm = np.maximum(pred_norm, eps)
    gt_norm = np.maximum(gt_norm, eps)
    return float(np.sum(gt_norm * np.log(gt_norm / pred_norm)))


def calculate_sim(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """SIM (Similarity).  [0,1]，。"""
    mask_pred = _ensure_shape_reasonaff(mask_pred, mask_gt)
    pred = mask_pred.astype(np.float64)
    gt = mask_gt.astype(np.float64)
    pred_sum, gt_sum = pred.sum(), gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    return float(np.minimum(pred / pred_sum, gt / gt_sum).sum())


def calculate_nss(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """NSS (Normalized Scanpath Saliency). 。"""
    mask_pred = _ensure_shape_reasonaff(mask_pred, mask_gt)
    pred = mask_pred.astype(np.float64)
    gt = mask_gt.astype(np.bool_)
    if not gt.any():
        return 0.0
    pred_std = pred.std()
    if pred_std == 0:
        return 0.0
    pred_normalized = (pred - pred.mean()) / pred_std
    return float(pred_normalized[gt].mean())


def extract_mask_from_agent_result(
    result: Dict[str, Any],
    image_height: int,
    image_width: int
) -> Optional[np.ndarray]:
    """ agent  mask
    
    Args:
        result: agent.detect_affordance() 
        image_height: 
        image_width: 
    
    Returns:
         mask (H, W) bool array， None
    """
    #  _tool_results_storage （）
    tool_storage = result.get("_tool_results_storage", {})
    for tool_call_id, skill_result in tool_storage.items():
        #  mask （ sam2  detection  mask）
        if isinstance(skill_result, dict) and "mask" in skill_result and "mask_shape" in skill_result:
            mask_list = skill_result["mask"]
            mask_shape = skill_result["mask_shape"]
            
            #  mask （JSON），
            if isinstance(mask_list, str):
                try:
                    mask_list = json.loads(mask_list)
                except:
                    continue
            
            #  numpy array
            try:
                mask = np.array(mask_list, dtype=bool)
            except Exception as e:
                continue
            
            if mask.ndim == 1:
                # ， reshape
                if len(mask_shape) == 2:
                    try:
                        mask = mask.reshape(mask_shape)
                    except:
                        continue
                else:
                    continue
            elif mask.shape != tuple(mask_shape):
                #  reshape
                try:
                    mask = mask.reshape(mask_shape)
                except:
                    #  reshape ，
                    pass
            
            if mask.shape != (image_height, image_width):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((image_width, image_height), PILImage.NEAREST)
                mask = np.array(mask_pil) > 0
            
            return mask

        #  detection skill： mask_image_path / visualization_path（ mask ）
        if isinstance(skill_result, dict) and "mask_image_path" in skill_result:
            mask_path = skill_result.get("mask_image_path")
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask_img) > 0
                    
                    #  mask （）
                    mask_h, mask_w = mask_arr.shape
                    size_ratio = (mask_h * mask_w) / max(1, image_height * image_width)
                    if size_ratio < 0.1:
                        print(f"  Warning: mask from {os.path.basename(mask_path)} is much smaller "
                              f"({mask_w}x{mask_h}) than image ({image_width}x{image_height}), "
                              f"ratio={size_ratio:.3f}. This may be from a zoomed/cropped image. Skipping.")
                        continue
                    
                    if mask_arr.shape != (image_height, image_width):
                        mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
                        mask_arr = np.array(mask_img) > 0
                    return mask_arr
                except Exception:
                    pass
    
    # Fallback:  tool （）
    messages = result.get("messages", [])
    
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        
        try:
            payload = json.loads(content)
        except Exception:
            continue
        
        # 1)  sam2 /  detection  mask
        if "mask" in payload and "mask_shape" in payload:
            mask_list = payload["mask"]
            mask_shape = payload["mask_shape"]
            
            #  numpy array
            mask = np.array(mask_list, dtype=bool)
            
            if mask.shape != tuple(mask_shape):
                mask = mask.reshape(mask_shape)
            
            if mask.shape != (image_height, image_width):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((image_width, image_height), PILImage.NEAREST)
                mask = np.array(mask_pil) > 0
            
            return mask

        # 2)  detection： mask_image_path  PNG
        if "mask_image_path" in payload:
            mask_path = payload.get("mask_image_path")
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask_img) > 0
                    
                    #  mask 
                    mask_h, mask_w = mask_arr.shape
                    size_ratio = (mask_h * mask_w) / max(1, image_height * image_width)
                    if size_ratio < 0.1:
                        print(f"  Warning: mask from {os.path.basename(mask_path)} too small "
                              f"({mask_w}x{mask_h} vs {image_width}x{image_height}). Skipping.")
                        continue
                    
                    if mask_arr.shape != (image_height, image_width):
                        mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
                        mask_arr = np.array(mask_img) > 0
                    return mask_arr
                except Exception:
                    pass
    
    return None


def _load_resume_state_reasonaff(
    output_dir: str,
    predictions_dir: str,
    iou_thresholds: np.ndarray,
) -> Tuple[
    set,
    List[Dict[str, Any]],
    List[float],
    int,
    int,
    List[float],
    Dict[float, List[float]],
]:
    """Load existing metrics so evaluation can resume from previous progress."""
    all_samples_metrics_file = os.path.join(output_dir, "all_samples_metrics.json")
    loaded_metrics: List[Dict[str, Any]] = []

    if os.path.exists(all_samples_metrics_file):
        try:
            with open(all_samples_metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                loaded_metrics = data
                print(f"[Resume] Loaded {len(loaded_metrics)} samples from all_samples_metrics.json")
        except Exception as e:
            print(f"[Resume] Failed to load all_samples_metrics.json: {e}")

    if not loaded_metrics and os.path.exists(predictions_dir):
        for fn in sorted(os.listdir(predictions_dir)):
            if not fn.endswith("_prediction.json"):
                continue
            fp = os.path.join(predictions_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    loaded_metrics.append(payload)
            except Exception:
                continue
        if loaded_metrics:
            print(f"[Resume] Loaded {len(loaded_metrics)} samples from predictions/")

    processed_sample_ids: set = set()
    all_sample_metrics: List[Dict[str, Any]] = []
    ious: List[float] = []
    cumulative_intersection = 0
    cumulative_union = 0
    precisions_at_50: List[float] = []
    precisions_at_thresholds = {thresh: [] for thresh in iou_thresholds}

    for metric in loaded_metrics:
        sample_id = metric.get("sample_id")
        iou = metric.get("iou")
        inter = metric.get("sample_intersection", metric.get("intersection"))
        union = metric.get("sample_union", metric.get("union"))
        p50 = metric.get("precision_at_50")
        if sample_id in (None, ""):
            continue
        if not isinstance(iou, (int, float)):
            continue
        if not isinstance(inter, (int, float)):
            continue
        if not isinstance(union, (int, float)):
            continue

        sid = str(sample_id)
        if sid in processed_sample_ids:
            continue
        processed_sample_ids.add(sid)
        all_sample_metrics.append(metric)

        iou_f = float(iou)
        inter_i = int(inter)
        union_i = int(union)
        p50_f = float(p50) if isinstance(p50, (int, float)) else (1.0 if iou_f >= 0.5 else 0.0)

        ious.append(iou_f)
        cumulative_intersection += inter_i
        cumulative_union += union_i
        precisions_at_50.append(p50_f)

        p_all = metric.get("precisions_at_all_thresholds", {})
        for thresh in iou_thresholds:
            key = f"P_{thresh:.2f}"
            if isinstance(p_all, dict) and isinstance(p_all.get(key), (int, float)):
                val = float(p_all[key])
            else:
                val = 1.0 if iou_f >= thresh else 0.0
            precisions_at_thresholds[thresh].append(val)

    return (
        processed_sample_ids,
        all_sample_metrics,
        ious,
        cumulative_intersection,
        cumulative_union,
        precisions_at_50,
        precisions_at_thresholds,
    )


def _process_one_sample_reasonaff(
    idx: int,
    sample: Dict[str, Any],
    agent: AffordanceAgent,
    predictions_dir: str,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Process a single ReasonAff sample.

    Returns:
        (sample_index, metrics_dict) always; detection_failed=True in dict means IoU=0.
        Returns (sample_index, None) only for unrecoverable exceptions.
    """
    # Extract metadata upfront so exception handler can build a zero-metric dict
    sample_id = str(sample.get("sample_id", idx))
    image_path = os.path.join(predictions_dir, f"{idx}_image.png")

    try:
        sample_id = str(sample["sample_id"])
        image = sample["image"]           # PIL Image
        mask_gt = sample["mask_gt"]       # np.ndarray bool (H, W)
        image_height = sample["image_height"]
        image_width = sample["image_width"]
        task = sample["task"]
        object_name = sample.get("object_name")
        aff_name = sample.get("aff_name", "")
        part_name = sample.get("part_name", "")

        # Save image for agent
        image_path = os.path.join(predictions_dir, f"{idx}_image.png")
        image.save(image_path)

        def _zero_metric(failure_reason: str) -> Dict[str, Any]:
            """Return an IoU=0 sample dict for a failed detection."""
            gt_union = int(mask_gt.sum())
            return {
                "sample_id": sample_id,
                "sample_index": idx,
                "iou": 0.0,
                "intersection": 0,
                "union": gt_union if gt_union > 0 else 1,
                "precision_at_50": 0.0,
                "KLD": None,
                "SIM": 0.0,
                "NSS": 0.0,
                "task": task,
                "object_name": object_name,
                "aff_name": aff_name,
                "part_name": part_name,
                "tool_trajectory": [],
                "llm_reasoning": [],
                "strategy_reasoning": "",
                "strategy_reflection": "",
                "dynamic_params_used": {},
                "decision_trace": [],
                "image_path": image_path,
                "_memory_tool_calls": [],
                "_memory_tool_summaries": [],
                "_memory_llm_reasoning": [],
                "_memory_task_context": None,
                "_memory_mask_path": None,
                "_memory_reasoning_context": "",
                "detection_failed": True,
                "failure_reason": failure_reason,
            }

        # Run affordance detection
        result = agent.detect_affordance(
            image_path=image_path,
            task=task,
            object_name=object_name,
            save_conversation=False,
            sample_id=sample_id,
        )

        if not result.get("success"):
            reason = result.get('error', 'Detection failed')
            print(f"  FAILED sample {sample_id}: {reason}")
            return (idx, _zero_metric(reason))

        # Extract predicted mask
        mask_pred = extract_mask_from_agent_result(result, image_height, image_width)
        if mask_pred is None:
            reason = "No mask extracted from agent result (all detection calls returned no bbox/points)"
            print(f"  FAILED sample {sample_id}: {reason}")
            return (idx, _zero_metric(reason))

        # Ensure shapes match
        if mask_gt.shape != mask_pred.shape:
            from PIL import Image as PILImage
            mask_pred_pil = PILImage.fromarray(mask_pred.astype(np.uint8) * 255)
            mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), PILImage.NEAREST)
            mask_pred = np.array(mask_pred_pil) > 0

        # Compute metrics
        iou = calculate_iou(mask_pred, mask_gt)
        inter, union = calculate_intersection_and_union(mask_pred, mask_gt)
        kld = calculate_kld(mask_pred, mask_gt)
        sim = calculate_sim(mask_pred, mask_gt)
        nss = calculate_nss(mask_pred, mask_gt)
        p50 = 1.0 if iou >= 0.5 else 0.0

        # Extract tool trajectory
        tool_trajectory = []
        llm_reasoning_texts = []
        for msg in result.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and isinstance(content, str) and content.strip():
                    llm_reasoning_texts.append(content.strip()[:500])
                if "tool_calls" in msg:
                    for tc in msg.get("tool_calls", []):
                        try:
                            tool_args = json.loads(tc["function"]["arguments"]) if isinstance(
                                tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        except Exception:
                            tool_args = {}
                        display_args = {}
                        for k, v in tool_args.items():
                            if k == "image_path":
                                display_args[k] = os.path.basename(str(v))
                            elif isinstance(v, str) and len(v) > 120:
                                display_args[k] = v[:120] + "..."
                            else:
                                display_args[k] = v
                        tool_trajectory.append({
                            "step": len(tool_trajectory) + 1,
                            "tool": tc["function"]["name"],
                            "arguments": display_args,
                        })

        sm = {
            "sample_id": sample_id,
            "sample_index": idx,
            "iou": float(iou),
            "intersection": int(inter),
            "union": int(union),
            "precision_at_50": p50,
            "KLD": float(kld) if np.isfinite(kld) else None,
            "SIM": float(sim),
            "NSS": float(nss),
            "task": task,
            "object_name": object_name,
            "aff_name": aff_name,
            "part_name": part_name,
            "tool_trajectory": tool_trajectory,
            "llm_reasoning": llm_reasoning_texts,
            "strategy_reasoning": result.get("strategy_reasoning", ""),
            "strategy_reflection": result.get("strategy_reflection", ""),
            "dynamic_params_used": result.get("dynamic_params_used", {}),
            "decision_trace": result.get("decision_trace", []),
            "image_path": image_path,
            "_memory_tool_calls": result.get("_memory_tool_calls", []),
            "_memory_tool_summaries": result.get("_memory_tool_summaries", []),
            "_memory_llm_reasoning": result.get("_memory_llm_reasoning", []),
            "_memory_task_context": result.get("_memory_task_context"),
            "_memory_mask_path": result.get("_memory_mask_path"),
            "_memory_reasoning_context": result.get("final_response", ""),
        }
        return (idx, sm)

    except Exception as e:
        print(f"[Worker] Error processing sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        # Try to build a zero-metric dict so this sample still counts as IoU=0
        try:
            _mask_gt = sample.get("mask_gt")
            _gt_union = int(_mask_gt.sum()) if _mask_gt is not None else 1
        except Exception:
            _gt_union = 1
        return (idx, {
            "sample_id": sample_id,
            "sample_index": idx,
            "iou": 0.0,
            "intersection": 0,
            "union": _gt_union if _gt_union > 0 else 1,
            "precision_at_50": 0.0,
            "KLD": None,
            "SIM": 0.0,
            "NSS": 0.0,
            "task": sample.get("task", ""),
            "object_name": sample.get("object_name"),
            "aff_name": sample.get("aff_name", ""),
            "part_name": sample.get("part_name", ""),
            "tool_trajectory": [],
            "llm_reasoning": [],
            "strategy_reasoning": "",
            "strategy_reflection": "",
            "dynamic_params_used": {},
            "decision_trace": [],
            "image_path": image_path,
            "_memory_tool_calls": [],
            "_memory_tool_summaries": [],
            "_memory_llm_reasoning": [],
            "_memory_task_context": None,
            "_memory_mask_path": None,
            "_memory_reasoning_context": "",
            "detection_failed": True,
            "failure_reason": f"Exception: {e}",
        })


def _write_batch_to_memory(
    agent: AffordanceAgent,
    batch_results: List[Dict[str, Any]],
) -> int:
    """Replay add_entry for every successful sample in a finished batch.

    Returns the number of entries actually written.
    """
    if agent.memory_manager is None:
        return 0
    written = 0
    for sm in batch_results:
        try:
            agent.memory_manager.add_entry(
                sample_id=sm["sample_id"],
                image_path=sm["image_path"],
                task=sm["task"],
                object_name=sm.get("object_name"),
                tool_calls=sm.get("_memory_tool_calls"),
                tool_summaries=sm.get("_memory_tool_summaries"),
                task_context=sm.get("_memory_task_context"),
                mask_path=sm.get("_memory_mask_path"),
                llm_reasoning=sm.get("_memory_llm_reasoning"),
                reasoning_context=sm.get("_memory_reasoning_context", ""),
                strategy_reasoning=sm.get("strategy_reasoning", ""),
                strategy_reflection=sm.get("strategy_reflection", ""),
                dynamic_params_used=sm.get("dynamic_params_used"),
                decision_trace=sm.get("decision_trace"),
            )
            written += 1
        except Exception as e:
            print(f"[BatchSync] Failed to write memory for sample {sm.get('sample_id')}: {e}")
    return written


def evaluate_on_reasonaff(
    dataset_path: str,
    agent: AffordanceAgent,
    output_dir: str,
    max_samples: Optional[int] = None,
    save_predictions: bool = True,
    resume: bool = False,
    num_workers: int = 1,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 0,
) -> Dict[str, float]:
    """ ReasonAff 
    
    Args:
        dataset_path: 
        agent: AffordanceAgent 
        output_dir: 
        max_samples: （None ）
        save_predictions: 
        resume: 
        num_workers:  worker （1=）
        batch_size: （0=auto， num_workers）
        agent_kwargs: agent （ worker agent）
    
    Returns:
        
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    #  DatasetDict（ split）， test split
    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        available = list(dataset.keys())
        split = "test" if "test" in available else available[0]
        print(f"[Dataset] DatasetDict detected, using split '{split}' (available: {available})")
        dataset = dataset[split]

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    all_samples_metrics_file = os.path.join(output_dir, "all_samples_metrics.json")
    realtime_metrics_file = os.path.join(output_dir, "metrics_realtime.json")
    evaluation_results_file = os.path.join(output_dir, "evaluation_results.json")

    os.makedirs(predictions_dir, exist_ok=True)
    if resume:
        print(f"[Resume] Enabled. Reusing output directory: {output_dir}")

    #  P_{50-95}
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # （resume ）
    if resume:
        (
            processed_sample_ids,
            all_samples_metrics,
            ious,
            cumulative_intersection,
            cumulative_union,
            precisions_at_50,
            precisions_at_thresholds,
        ) = _load_resume_state_reasonaff(output_dir, predictions_dir, iou_thresholds)
        print(f"[Resume] Already processed: {len(processed_sample_ids)} samples")
    else:
        processed_sample_ids = set()
        all_samples_metrics = []
        ious = []
        cumulative_intersection = 0  # cIoU: 
        cumulative_union = 0         # cIoU: 
        precisions_at_50 = []
        precisions_at_thresholds = {thresh: [] for thresh in iou_thresholds}

    # KLD/SIM/NSS accumulators
    kld_all: List[float] = []
    sim_all: List[float] = []
    nss_all: List[float] = []

    failed_samples: List[str] = []

    # ---- Pre-build sample list (with GT pre-processing) ----
    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_id = str(sample.get("id", f"sample_{idx}"))
        if resume and sample_id in processed_sample_ids:
            continue

        image = sample.get("image")
        if image is None:
            failed_samples.append(sample_id)
            continue

        mask_gt_raw = np.array(sample.get("mask", []))
        if mask_gt_raw.size == 0:
            failed_samples.append(sample_id)
            continue

        if mask_gt_raw.ndim > 2:
            mask_gt_raw = mask_gt_raw.reshape(-1, mask_gt_raw.shape[-1])
        mask_gt = mask_gt_raw.astype(bool)

        image_width, image_height = image.size
        if mask_gt.shape != (image_height, image_width):
            from PIL import Image as PILImage
            mask_gt_pil = PILImage.fromarray(mask_gt.astype(np.uint8) * 255)
            mask_gt_pil = mask_gt_pil.resize((image_width, image_height), PILImage.NEAREST)
            mask_gt = np.array(mask_gt_pil) > 0

        problem = sample.get("problem", "")
        aff_name = sample.get("aff_name", "")
        part_name = sample.get("part_name", "")

        pending.append((idx, {
            "sample_id": sample_id,
            "image": image,
            "mask_gt": mask_gt,
            "image_height": image_height,
            "image_width": image_width,
            "task": problem,
            "object_name": part_name if part_name else None,
            "aff_name": aff_name,
            "part_name": part_name,
        }))

    total = len(dataset)
    print(f"Pending: {len(pending)} samples to evaluate (skipped {len(processed_sample_ids)} already done)")

    # ---- Accumulator & display closures ----
    def _accumulate_one(sm: Dict[str, Any]) -> None:
        nonlocal cumulative_intersection, cumulative_union

        iou_val = sm["iou"]
        inter = sm["intersection"]
        union_val = sm["union"]
        p50_val = sm["precision_at_50"]
        kld_val = sm.get("KLD")
        sim_val = sm["SIM"]
        nss_val = sm["NSS"]

        ious.append(iou_val)
        cumulative_intersection += inter
        cumulative_union += union_val
        precisions_at_50.append(p50_val)
        for t in iou_thresholds:
            precisions_at_thresholds[t].append(1.0 if iou_val >= t else 0.0)
        if kld_val is not None and np.isfinite(kld_val):
            kld_all.append(float(kld_val))
        sim_all.append(sim_val)
        nss_all.append(nss_val)

        # Feed metrics back to memory only for successful detections
        if not sm.get("detection_failed") and agent.enable_memory and agent.memory_manager:
            sid = sm.get("sample_id")
            if sid:
                eval_metrics = {
                    "iou": iou_val,
                    "precision_at_50": p50_val,
                }
                agent.memory_manager.update_entry_metrics(sid, eval_metrics)

    def _display_and_save(sm: Dict[str, Any], completed: int, total_count: int) -> None:
        sample_id = sm["sample_id"]
        iou_val = sm["iou"]
        kld_val = sm.get("KLD")
        sim_val = sm["SIM"]
        nss_val = sm["NSS"]
        is_failed = sm.get("detection_failed", False)

        cur_gIoU = np.mean(ious)
        cur_cIoU = float(cumulative_intersection) / float(cumulative_union) if cumulative_union > 0 else 0.0
        cur_P50 = np.mean(precisions_at_50)
        cur_P50_95 = float(np.mean([np.mean(precisions_at_thresholds[t]) for t in iou_thresholds])) if ious else 0.0
        finite_kld = [v for v in kld_all if np.isfinite(v)]
        cur_KLD = np.mean(finite_kld) if finite_kld else float('inf')
        cur_SIM = np.mean(sim_all)
        cur_NSS = np.mean(nss_all)

        valid_count = len(ious) - len(failed_samples)
        kld_s = f"{kld_val:.4f}" if kld_val is not None and np.isfinite(kld_val) else "inf"
        ckld_s = f"{cur_KLD:.4f}" if np.isfinite(cur_KLD) else "inf"
        status_prefix = "[FAILED] " if is_failed else ""
        print(f"\n[{completed}/{total_count}] {status_prefix}{sample_id}")
        if is_failed:
            print(f"  Detection failed: {sm.get('failure_reason', 'unknown')} -> IoU=0.0 counted")
        else:
            print(f"  IoU={iou_val:.4f}, KLD={kld_s}, SIM={sim_val:.4f}, NSS={nss_val:.4f}")
        print(f"  Cumul: gIoU={cur_gIoU:.4f}, cIoU={cur_cIoU:.4f}, P@50={cur_P50:.4f}, P@50-95={cur_P50_95:.4f}, KLD={ckld_s}, SIM={cur_SIM:.4f}, NSS={cur_NSS:.4f}")
        print(f"  Processed={len(ious)}, Valid={valid_count}, Failed={len(failed_samples)}")

        # Dedup and save
        existing_idx = next((i for i, m in enumerate(all_samples_metrics) if m.get("sample_id") == sample_id), None)
        if existing_idx is not None:
            all_samples_metrics[existing_idx] = sm
        else:
            all_samples_metrics.append(sm)
        processed_sample_ids.add(sample_id)

        if save_predictions:
            pred_path = os.path.join(predictions_dir, f"{sm['sample_id']}_prediction.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(sm, f, indent=2, ensure_ascii=False)

        with open(realtime_metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "current_sample": completed,
                "total_samples": total_count,
                "valid_samples": valid_count,
                "failed_samples": len(failed_samples),
                "total_processed": len(ious),
                "gIoU": float(cur_gIoU),
                "cIoU": float(cur_cIoU),
                "P_50": float(cur_P50),
                "P_50-95": float(cur_P50_95),
                "KLD": float(cur_KLD) if np.isfinite(cur_KLD) else None,
                "SIM": float(cur_SIM),
                "NSS": float(cur_NSS),
            }, f, indent=2, ensure_ascii=False)

        with open(all_samples_metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_samples_metrics, f, indent=2, ensure_ascii=False)

    # ---- Dispatch: batch-parallel or serial ----
    if num_workers > 1 and agent_kwargs is not None and pending:
        # ====== Batch-parallel evaluation ======
        effective_batch = batch_size if batch_size > 0 else num_workers
        n_batches = (len(pending) + effective_batch - 1) // effective_batch
        print(f"\n[BatchParallel] {num_workers} workers, batch_size={effective_batch}, "
              f"{len(pending)} pending samples -> {n_batches} batches")

        agent_pool: _queue.Queue[AffordanceAgent] = _queue.Queue()
        # Disable memory writes on ALL pool agents (including the main one)
        # so that no agent calls add_entry during detect_affordance.
        # Memory is written centrally via _write_batch_to_memory after each batch.
        agent.enable_memory = False
        agent_pool.put(agent)
        for _i in range(num_workers - 1):
            wk = agent_kwargs.copy()
            wk["enable_memory"] = False
            worker = AffordanceAgent(**wk)
            if agent.memory_manager is not None:
                worker.memory_manager = agent.memory_manager
                worker.skill_registry.set_memory_manager(agent.memory_manager)
            agent_pool.put(worker)

        import time as _time
        import random as _random

        def _worker(idx_sample: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[Dict[str, Any]]]:
            _idx, _sample = idx_sample
            a = agent_pool.get()
            try:
                result = _process_one_sample_reasonaff(_idx, _sample, a, predictions_dir)
                if result[1] is None:
                    wait = _random.uniform(3.0, 8.0)
                    print(f"  [Worker] Sample {_idx} failed, retrying after {wait:.1f}s ...")
                    _time.sleep(wait)
                    result = _process_one_sample_reasonaff(_idx, _sample, a, predictions_dir)
                return result
            finally:
                agent_pool.put(a)

        completed_count = len(processed_sample_ids)
        for batch_idx in range(n_batches):
            batch_start = batch_idx * effective_batch
            batch_end = min(batch_start + effective_batch, len(pending))
            batch_chunk = pending[batch_start:batch_end]
            batch_successful: List[Dict[str, Any]] = []

            print(f"\n--- Batch {batch_idx + 1}/{n_batches} "
                  f"(samples {batch_start}-{batch_end - 1}, size={len(batch_chunk)}) ---")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for i, (idx, s) in enumerate(batch_chunk):
                    futures[executor.submit(_worker, (idx, s))] = idx
                    if i < num_workers and i > 0:
                        _time.sleep(0.5)
                for future in as_completed(futures):
                    result_idx, sm = future.result()
                    completed_count += 1

                    if sm is None:
                        # Safety net: shouldn't happen since worker now returns zero-metric dicts
                        failed_samples.append(f"sample_{result_idx}")
                        print(f"  [{completed_count}/{total}] Sample {result_idx} FAILED (unrecoverable)")
                        continue

                    if sm.get("detection_failed"):
                        failed_samples.append(sm.get("sample_id", f"sample_{result_idx}"))

                    try:
                        _accumulate_one(sm)
                        _display_and_save(sm, completed_count, total)
                        if not sm.get("detection_failed"):
                            batch_successful.append(sm)
                    except Exception as e:
                        print(f"Error accumulating sample {result_idx}: {e}")
                        if not sm.get("detection_failed"):
                            failed_samples.append(sm.get("sample_id", f"sample_{result_idx}"))

            # ---- Inter-batch memory sync ----
            if batch_successful:
                written = _write_batch_to_memory(agent, batch_successful)
                print(f"  [BatchSync] Wrote {written}/{len(batch_successful)} "
                      f"entries to memory after batch {batch_idx + 1}")
                if agent.memory_manager and hasattr(agent.memory_manager, 'experience_pool'):
                    try:
                        agent.memory_manager.experience_pool._maybe_distill()
                    except Exception as e:
                        print(f"  [BatchSync] Distill warning: {e}")

        # Restore enable_memory on the main agent for post-evaluation use
        agent.enable_memory = True
    else:
        # ====== Serial evaluation ======
        completed_count = len(processed_sample_ids)
        for idx, s in tqdm(pending, desc="Evaluating"):
            result_idx, sm = _process_one_sample_reasonaff(idx, s, agent, predictions_dir)
            completed_count += 1

            if sm is None:
                # Safety net: shouldn't happen since worker now returns zero-metric dicts
                failed_samples.append(f"sample_{idx}")
                continue

            if sm.get("detection_failed"):
                failed_samples.append(sm.get("sample_id", f"sample_{idx}"))

            try:
                _accumulate_one(sm)
                _display_and_save(sm, completed_count, total)
            except Exception as e:
                print(f"Error accumulating sample {idx}: {e}")
                if not sm.get("detection_failed"):
                    failed_samples.append(s.get("sample_id", f"sample_{idx}"))

    # ---- Final metrics ----
    if len(ious) == 0:
        print("Warning: No valid predictions!")
        return {
            "gIoU": 0.0, "cIoU": 0.0, "P_50": 0.0, "P_50-95": 0.0,
            "KLD": 0.0, "SIM": 0.0, "NSS": 0.0,
            "num_samples": 0, "failed_samples": len(failed_samples),
        }

    gIoU = float(np.mean(ious))
    cIoU = float(cumulative_intersection) / float(cumulative_union) if cumulative_union > 0 else 0.0
    P_50 = float(np.mean(precisions_at_50))
    P_50_95 = float(np.mean([np.mean(precisions_at_thresholds[t]) for t in iou_thresholds]))
    finite_kld = [v for v in kld_all if np.isfinite(v)]
    final_KLD = float(np.mean(finite_kld)) if finite_kld else float('inf')
    final_SIM = float(np.mean(sim_all))
    final_NSS = float(np.mean(nss_all))

    metrics = {
        "gIoU": gIoU, "cIoU": cIoU, "P_50": P_50, "P_50-95": P_50_95,
        "KLD": final_KLD if np.isfinite(final_KLD) else None,
        "SIM": final_SIM, "NSS": final_NSS,
        "num_samples": len(ious), "failed_samples": len(failed_samples),
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    kld_display = f"{final_KLD:.4f}" if np.isfinite(final_KLD) else "inf"
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"gIoU:     {gIoU:.4f}")
    print(f"cIoU:     {cIoU:.4f}")
    print(f"P_{{50}}:    {P_50:.4f}")
    print(f"P_{{50-95}}: {P_50_95:.4f}")
    print(f"KLD:      {kld_display}")
    print(f"SIM:      {final_SIM:.4f}")
    print(f"NSS:      {final_NSS:.4f}")
    print(f"Valid samples: {len(ious)}")
    print(f"Failed samples: {len(failed_samples)}")
    print("=" * 60)
    
    #  memory 
    if agent.enable_memory and agent.memory_manager:
        try:
            vis_path = agent.memory_manager.export_visualization(
                os.path.join(output_dir, "memory_vis.html")
            )
            print(f"Memory visualization: {vis_path}")
            
            traj_path = agent.memory_manager.export_trajectory_log(
                os.path.join(output_dir, "trajectory_log.json")
            )
            print(f"Trajectory log: {traj_path}")
        except Exception as e:
            print(f"[Warning] Failed to export memory visualization: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate affordance detection on ReasonAff dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/reasonaff/test",
        help="Path to ReasonAff test dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/evaluation_reasonaff",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (overrides config/env)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (overrides config/env)",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=None,
        help="API base URL (overrides config/env)",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Do not save individual prediction results",
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Only expose 'detection' tool to the LLM (disable dreamer, zoom_in, web_search). For ablation experiments.",
    )
    parser.add_argument(
        "--detection_backend",
        type=str,
        default="qwen3vl_api",
        choices=["qwen3vl_api", "rex_omni"],
        help="Detection backend: qwen3vl_api (cloud API, default) or rex_omni (local model)",
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        default=None,
        help=(
            "Override the detection API model name (for qwen3vl_api backend). "
            "Default: qwen3-vl-235b-a22b-instruct. "
            "Example: qwen3-vl-8b-instruct, Qwen3.5-397B-A17B. "
            "Sets env var DETECTION_MODEL_NAME used by detection.py."
        ),
    )
    parser.add_argument(
        "--detection_api_key",
        type=str,
        default=None,
        help="Override detection API key (sets DETECTION_API_KEY). "
             "Use when detection model is on a different endpoint, e.g. PAI-EAS.",
    )
    parser.add_argument(
        "--detection_api_base_url",
        type=str,
        default=None,
        help="Override detection API base URL (sets DETECTION_API_BASE_URL). "
             "Example: http://xxx.pai-eas.aliyuncs.com/api/predict/qwen3_5",
    )
    parser.add_argument(
        "--sam_backend",
        type=str,
        default="sam2",
        choices=["sam2", "sam3"],
        help="Segmentation backend: sam2 (default) or sam3 (Meta SAM3 via transformers). Sets env var SAM_BACKEND.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output_dir and skip processed samples",
    )
    parser.add_argument(
        "--clear_memory",
        action="store_true",
        help="Clear existing memory before starting evaluation (start fresh)",
    )
    parser.add_argument(
        "--clear_experience",
        action="store_true",
        help="Also clear the experience pool when clearing memory",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of concurrent workers (1 = serial, >1 = parallel evaluation)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0,
        help="Batch size for batch-parallel evaluation (0 = auto, same as num_workers). "
             "Smaller batches sync memory more frequently but add overhead.",
    )
    
    parser.add_argument(
        "--commonsense_templates_dir", type=str, default=None,
        help="Directory containing commonsense_templates.json (overrides default)",
    )
    parser.add_argument(
        "--disable_commonsense",
        action="store_true",
        help="Disable commonsense template bank (ablation: no commonsense knowledge).",
    )
    parser.add_argument(
        "--disable_memory",
        action="store_true",
        help="Disable episodic memory module entirely (ablation: no historical memory).",
    )
    parser.add_argument(
        "--fixed_skill_chain",
        action="store_true",
        help=(
            "Baseline: force fixed skill execution order "
            "zoom_in → dreamer → web_search → detection, "
            "without LLM dynamic tool selection. "
            "Automatically implies --disable_memory and --disable_commonsense."
        ),
    )

    args = parser.parse_args()
    
    import importlib.util
    config_path = PROJECT_ROOT / "config.py"
    config: Dict[str, Any] = {}
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("aff_cfg", config_path)
        if spec and spec.loader:
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
            config = {
                "API_KEY": getattr(cfg, "API_KEY", None),
                "API_BASE_URL": getattr(cfg, "API_BASE_URL", None),
                "DEFAULT_MODEL": getattr(cfg, "DEFAULT_MODEL", None),
            }
    
    api_key = args.api_key or config.get("API_KEY") or os.getenv("API_KEY")
    api_url = args.api_url or config.get("API_BASE_URL") or os.getenv("API_BASE_URL")
    model = args.model or config.get("DEFAULT_MODEL") or os.getenv("DEFAULT_MODEL", "gpt-4o")
    
    if args.detection_model:
        os.environ["DETECTION_MODEL_NAME"] = args.detection_model
        print(f"[Detection Model Override] Using detection model: {args.detection_model}")
    if args.detection_api_key:
        os.environ["DETECTION_API_KEY"] = args.detection_api_key
        print(f"[Detection API Override] Using custom API key")
    if args.detection_api_base_url:
        os.environ["DETECTION_API_BASE_URL"] = args.detection_api_base_url
        print(f"[Detection API Override] Using API base URL: {args.detection_api_base_url}")

    if args.sam_backend != "sam2":
        os.environ["SAM_BACKEND"] = args.sam_backend
        print(f"[SAM Backend Override] Using SAM backend: {args.sam_backend}")

    #  skill  memory  commonsense
    if getattr(args, "fixed_skill_chain", False):
        args.disable_memory = True
        args.disable_commonsense = True
        print("⚠️  Fixed skill chain mode: zoom_in → dreamer → web_search → detection (no LLM tool selection).")

    #  agent
    tool_filter = {"detection"} if args.detection_only else None
    if tool_filter:
        print(f"⚠️  Detection-only mode: LLM can only call {tool_filter}")
    print(f"Detection backend: {args.detection_backend}")
    
    if args.disable_commonsense:
        print("⚠️  Commonsense-disabled mode: commonsense template bank is turned off.")
    if args.disable_memory:
        print("⚠️  Memory-disabled mode: episodic memory module is turned off.")

    agent_kwargs = dict(
        api_key=api_key,
        api_base_url=api_url,
        model_name=model,
        output_dir=args.output_dir,
        tool_filter=tool_filter,
        detection_backend=args.detection_backend,
        commonsense_templates_dir=args.commonsense_templates_dir,
        enable_memory=not args.disable_memory,
        enable_commonsense_templates=not args.disable_commonsense,
        fixed_skill_chain=getattr(args, "fixed_skill_chain", False),
    )

    #  run_config.json（）
    from datetime import datetime as _dt
    os.makedirs(args.output_dir, exist_ok=True)
    _detection_model_used = (
        args.detection_model
        or os.environ.get("DETECTION_MODEL_NAME")
        or "qwen3-vl-235b-a22b-instruct"
    )
    _run_config = {
        "dataset": "reasonaff",
        "dataset_path": args.dataset_path,
        "decision_model": model,
        "detection_backend": args.detection_backend,
        "detection_model": _detection_model_used,
        "num_workers": args.num_workers,
        "max_samples": getattr(args, "max_samples", None),
        "detection_only": getattr(args, "detection_only", False),
        "fixed_skill_chain": getattr(args, "fixed_skill_chain", False),
        "disable_commonsense": getattr(args, "disable_commonsense", False),
        "disable_memory": getattr(args, "disable_memory", False),
        "sam_backend": getattr(args, "sam_backend", "sam2"),
        "start_time": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as _f:
        json.dump(_run_config, _f, indent=2, ensure_ascii=False)
    print(f"[Config] decision_model={model}, detection_model={_detection_model_used}, sam_backend={_run_config['sam_backend']}")

    agent = AffordanceAgent(**agent_kwargs)
    
    # （ --clear_memory）
    if args.clear_memory:
        print("🗑️  Clearing existing memory before evaluation...")
        agent.clear_memory(clear_experience=args.clear_experience)
    
    if args.num_workers > 1:
        eff_bs = args.batch_size if args.batch_size > 0 else args.num_workers
        print(f"Batch-parallel mode: {args.num_workers} workers, batch_size={eff_bs}")

    metrics = evaluate_on_reasonaff(
        dataset_path=args.dataset_path,
        agent=agent,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        save_predictions=not args.no_save_predictions,
        resume=args.resume,
        num_workers=args.num_workers,
        agent_kwargs=agent_kwargs,
        batch_size=args.batch_size,
    )

    agent.flush_experience_pool()


if __name__ == "__main__":
    main()
