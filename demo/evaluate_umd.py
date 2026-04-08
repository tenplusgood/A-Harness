"""


:
  python evaluate_umd.py \
      --dataset_path datasets/UMD_preprocessed \
      --output_dir output/evaluation_umd \
      --api_url https://api.vveai.com/v1 \
      --model gpt-4o

  #  detection（）
  python evaluate_umd.py \
      --dataset_path datasets/UMD_preprocessed \
      --output_dir output/evaluation_umd_detection_only \
      --model gpt-4o \
      --detection_only

  python evaluate_umd.py \
      --dataset_path datasets/UMD_preprocessed \
      --output_dir output/evaluation_umd_test \
      --model gpt-4o \
      --max_samples 20
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import queue as _queue
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import AffordanceAgent


# Metrics ( evaluate_reasonaff.py ，)

def calculate_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    if mask_pred.shape != mask_gt.shape:
        mask_pred_pil = Image.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def calculate_intersection_and_union(
    mask_pred: np.ndarray, mask_gt: np.ndarray
) -> Tuple[int, int]:
    if mask_pred.shape != mask_gt.shape:
        mask_pred_pil = Image.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    intersection = int(np.logical_and(mask_pred, mask_gt).sum())
    union = int(np.logical_or(mask_pred, mask_gt).sum())
    return intersection, union


def _ensure_shape(mask_pred: np.ndarray, mask_gt: np.ndarray) -> np.ndarray:
    """Resize mask_pred to match mask_gt shape if needed."""
    if mask_pred.shape != mask_gt.shape:
        mask_pred_pil = Image.fromarray(mask_pred.astype(np.uint8) * 255)
        mask_pred_pil = mask_pred_pil.resize((mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
        mask_pred = np.array(mask_pred_pil) > 0
    return mask_pred


def calculate_kld(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """KLD (Kullback-Leibler Divergence). （0 = ）。"""
    mask_pred = _ensure_shape(mask_pred, mask_gt)
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
    mask_pred = _ensure_shape(mask_pred, mask_gt)
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
    mask_pred = _ensure_shape(mask_pred, mask_gt)
    pred = mask_pred.astype(np.float64)
    gt = mask_gt.astype(np.bool_)
    if not gt.any():
        return 0.0
    pred_std = pred.std()
    if pred_std == 0:
        return 0.0
    pred_normalized = (pred - pred.mean()) / pred_std
    return float(pred_normalized[gt].mean())


# UMD Dataset Loader ( A4-Agent umd_reader.py)

def load_umd_samples(dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
     UMD 。
    ：
    1. ： (tool_dir/file_name)
    2. JSON ： JSON  (e.g. small_dataset_umd.json)

     = (image_path, mask_path, affordance_type, category, tool_instance)
    """
    samples = []

    # Case 1: JSON file
    if os.path.isfile(dataset_path) and dataset_path.endswith(".json"):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Resolve relative paths if needed
                base_dir = os.path.dirname(dataset_path)
                for item in data:
                    if not all(k in item for k in ["image_path", "mask_path", "affordance_type"]):
                        continue

                    if not os.path.exists(item["image_path"]):
                        alt_img = os.path.join(base_dir, item["image_path"])
                        if os.path.exists(alt_img):
                            item["image_path"] = alt_img

                    if not os.path.exists(item["mask_path"]):
                        alt_mask = os.path.join(base_dir, item["mask_path"])
                        if os.path.exists(alt_mask):
                            item["mask_path"] = alt_mask

                    samples.append(item)
            print(f"Loaded {len(samples)} samples from JSON file: {dataset_path}")
        except Exception as e:
            print(f"Error loading UMD JSON: {e}")
            return []

    # Case 2: Directory
    elif os.path.isdir(dataset_path):
        for tool_dir in sorted(os.listdir(dataset_path)):
            tool_dir_path = os.path.join(dataset_path, tool_dir)
            if not os.path.isdir(tool_dir_path):
                continue

            category = tool_dir.split("_")[0]

            for file_name in sorted(os.listdir(tool_dir_path)):
                if not file_name.endswith("_gt_mask.png"):
                    continue

                mask_path = os.path.join(tool_dir_path, file_name)

                # Extract affordance type: e.g. bowl_01_00000003_contain_gt_mask.png → contain
                affordance_type = file_name.split("_gt_mask.png")[0].split("_")[-1]

                # Construct image path: e.g. bowl_01_00000003_rgb.jpg
                base_name = "_".join(file_name.split("_")[:-3])  # remove {aff}_gt_mask.png
                img_path = os.path.join(tool_dir_path, base_name + "_rgb.jpg")

                if not os.path.exists(img_path):
                    continue

                samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "affordance_type": affordance_type,
                    "category": category,
                    "tool_instance": tool_dir,
                    "sample_name": f"{tool_dir}_{affordance_type}",
                })
    else:
        print(f"Dataset path not found or invalid: {dataset_path}")
        return []

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


# Mask extraction (from agent result)

def extract_mask_from_agent_result(
    result: Dict[str, Any], image_height: int, image_width: int
) -> Optional[np.ndarray]:
    """ agent  mask（ evaluate_reasonaff.py ）"""

    tool_storage = result.get("_tool_results_storage", {})
    for tool_call_id, skill_result in tool_storage.items():
        if isinstance(skill_result, dict) and "mask_image_path" in skill_result:
            mask_path = skill_result.get("mask_image_path")
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask_img) > 0
                    if mask_arr.shape != (image_height, image_width):
                        mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
                        mask_arr = np.array(mask_img) > 0
                    return mask_arr
                except Exception:
                    pass

    # Fallback: messages
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
        if "mask_image_path" in payload:
            mask_path = payload.get("mask_image_path")
            if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_arr = np.array(mask_img) > 0
                    if mask_arr.shape != (image_height, image_width):
                        mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
                        mask_arr = np.array(mask_img) > 0
                    return mask_arr
                except Exception:
                    pass

    return None


def _load_resume_state_umd(
    output_dir: str,
    predictions_dir: str,
    iou_thresholds: np.ndarray,
) -> Tuple[
    set,                       # processed_sample_names
    List[Dict[str, Any]],      # all_sample_metrics
    List[float],               # ious_all
    int,                       # cum_inter_all
    int,                       # cum_union_all
    List[float],               # p50_all
    Dict[float, List[float]],  # p_thresh_all
    defaultdict,               # per_aff
    List[float],               # kld_all
    List[float],               # sim_all
    List[float],               # nss_all
]:
    """Load previous UMD sample metrics to support resume mode.

    Merges results from both all_samples_metrics.json AND individual
    prediction files in predictions/ to be as robust as possible.
    """
    all_samples_metrics_file = os.path.join(output_dir, "all_samples_metrics.json")
    loaded_metrics: List[Dict[str, Any]] = []
    _seen_names: set = set()  # track sample_name to dedup across sources

    # Source 1: all_samples_metrics.json (primary)
    if os.path.exists(all_samples_metrics_file):
        try:
            with open(all_samples_metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for m in data:
                    sn = m.get("sample_name")
                    if sn and sn not in _seen_names:
                        _seen_names.add(sn)
                        loaded_metrics.append(m)
                print(f"[Resume] Loaded {len(loaded_metrics)} samples from all_samples_metrics.json")
        except Exception as e:
            print(f"[Resume] Failed to load all_samples_metrics.json: {e}")

    # Source 2: individual prediction files (supplement — always scanned)
    extra_count = 0
    if os.path.exists(predictions_dir):
        for fn in sorted(os.listdir(predictions_dir)):
            if not fn.endswith("_prediction.json"):
                continue
            fp = os.path.join(predictions_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    sn = payload.get("sample_name")
                    if sn and sn not in _seen_names:
                        _seen_names.add(sn)
                        loaded_metrics.append(payload)
                        extra_count += 1
            except Exception:
                continue
    if extra_count:
        print(f"[Resume] Supplemented {extra_count} extra samples from predictions/")

    processed_sample_names: set = set()
    all_sample_metrics: List[Dict[str, Any]] = []

    ious_all: List[float] = []
    cum_inter_all = 0
    cum_union_all = 0
    p50_all: List[float] = []
    p_thresh_all = {t: [] for t in iou_thresholds}
    per_aff = defaultdict(
        lambda: {
            "ious": [],
            "cum_inter": 0,
            "cum_union": 0,
            "p50": [],
            "p_thresh": {t: [] for t in iou_thresholds},
            "kld": [],
            "sim": [],
            "nss": [],
        }
    )
    kld_all: List[float] = []
    sim_all: List[float] = []
    nss_all: List[float] = []

    for metric in loaded_metrics:
        sample_name = metric.get("sample_name")
        aff_type = metric.get("affordance_type", "unknown")
        iou = metric.get("iou")
        inter = metric.get("intersection", metric.get("sample_intersection"))
        union = metric.get("union", metric.get("sample_union"))
        p50 = metric.get("precision_at_50")

        if sample_name in (None, ""):
            continue
        if not isinstance(iou, (int, float)):
            continue
        if not isinstance(inter, (int, float)):
            continue
        if not isinstance(union, (int, float)):
            continue

        sname = str(sample_name)
        if sname in processed_sample_names:
            continue
        processed_sample_names.add(sname)
        all_sample_metrics.append(metric)

        iou_f = float(iou)
        inter_i = int(inter)
        union_i = int(union)
        p50_f = float(p50) if isinstance(p50, (int, float)) else (1.0 if iou_f >= 0.5 else 0.0)

        ious_all.append(iou_f)
        cum_inter_all += inter_i
        cum_union_all += union_i
        p50_all.append(p50_f)
        for t in iou_thresholds:
            v = 1.0 if iou_f >= t else 0.0
            p_thresh_all[t].append(v)

        aff = per_aff[aff_type]
        aff["ious"].append(iou_f)
        aff["cum_inter"] += inter_i
        aff["cum_union"] += union_i
        aff["p50"].append(p50_f)
        for t in iou_thresholds:
            aff["p_thresh"][t].append(1.0 if iou_f >= t else 0.0)

        # Restore KLD/SIM/NSS from saved metrics
        kld_v = metric.get("KLD")
        sim_v = metric.get("SIM")
        nss_v = metric.get("NSS")
        if isinstance(kld_v, (int, float)):
            kld_all.append(float(kld_v))
            aff["kld"].append(float(kld_v))
        if isinstance(sim_v, (int, float)):
            sim_all.append(float(sim_v))
            aff["sim"].append(float(sim_v))
        if isinstance(nss_v, (int, float)):
            nss_all.append(float(nss_v))
            aff["nss"].append(float(nss_v))

    return (
        processed_sample_names,
        all_sample_metrics,
        ious_all,
        cum_inter_all,
        cum_union_all,
        p50_all,
        p_thresh_all,
        per_aff,
        kld_all,
        sim_all,
        nss_all,
    )


# Compute metrics helper

def compute_metrics_from_lists(ious, cum_inter, cum_union, p50_list, p_thresh_dict, iou_thresholds,
                               kld_list=None, sim_list=None, nss_list=None):
    if len(ious) == 0:
        return {"gIoU": 0.0, "cIoU": 0.0, "P_50": 0.0, "P_50-95": 0.0,
                "KLD": 0.0, "SIM": 0.0, "NSS": 0.0, "num_samples": 0}
    gIoU = float(np.mean(ious))
    cIoU = float(cum_inter) / float(cum_union) if cum_union > 0 else 0.0
    P_50 = float(np.mean(p50_list))
    P_50_95 = float(np.mean([np.mean(p_thresh_dict[t]) for t in iou_thresholds]))
    result = {"gIoU": gIoU, "cIoU": cIoU, "P_50": P_50, "P_50-95": P_50_95, "num_samples": len(ious)}
    if kld_list:
        finite_kld = [v for v in kld_list if np.isfinite(v)]
        result["KLD"] = float(np.mean(finite_kld)) if finite_kld else float('inf')
    else:
        result["KLD"] = 0.0
    result["SIM"] = float(np.mean(sim_list)) if sim_list else 0.0
    result["NSS"] = float(np.mean(nss_list)) if nss_list else 0.0
    return result


# Per-sample worker (shared by serial & concurrent paths)

def _process_one_sample_umd(
    idx: int,
    sample: Dict[str, Any],
    agent: AffordanceAgent,
    predictions_dir: str,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Process a single UMD sample (thread-safe, no shared mutable state).

    Returns:
        (sample_index, metrics_dict) always; detection_failed=True in dict means IoU=0.
        Returns (sample_index, None) only for unrecoverable exceptions.
    """
    # Extract sample metadata upfront so the exception handler can build a zero-metric dict
    sample_name = sample.get("sample_name", f"sample_{idx}")
    affordance_type = sample.get("affordance_type", "unknown")
    category = sample.get("category", "unknown")
    pred_image_path = os.path.join(predictions_dir, f"{idx}_image.png")

    try:
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        affordance_type = sample["affordance_type"]
        category = sample["category"]
        sample_name = sample["sample_name"]

        # Load GT mask
        mask_gt = np.array(Image.open(mask_path).convert("L")) > 0
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        if mask_gt.shape != (image_height, image_width):
            mask_gt_pil = Image.fromarray(mask_gt.astype(np.uint8) * 255)
            mask_gt_pil = mask_gt_pil.resize((image_width, image_height), Image.NEAREST)
            mask_gt = np.array(mask_gt_pil) > 0

        # Generate instruction (align with A4-Agent)
        task = f"Find the part of the object in the center of the image that can {affordance_type}"

        # Copy image to predictions dir with unique name
        pred_image_path = os.path.join(predictions_dir, f"{idx}_image.png")
        image.save(pred_image_path)

        def _zero_metric(failure_reason: str) -> Dict[str, Any]:
            """Return an IoU=0 sample dict for a failed detection."""
            gt_union = int(mask_gt.sum())
            return {
                "sample_name": sample_name,
                "sample_index": idx,
                "category": category,
                "affordance_type": affordance_type,
                "iou": 0.0,
                "intersection": 0,
                "union": gt_union if gt_union > 0 else 1,
                "precision_at_50": 0.0,
                "KLD": None,
                "SIM": 0.0,
                "NSS": 0.0,
                "task": task,
                "tool_trajectory": [],
                "llm_reasoning": [],
                "strategy_reasoning": "",
                "strategy_reflection": "",
                "dynamic_params_used": {},
                "decision_trace": [],
                "image_path": pred_image_path,
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
            image_path=pred_image_path,
            task=task,
            object_name=None,
            save_conversation=False,
            sample_id=sample_name,
        )

        if not result.get("success"):
            reason = result.get('error', 'Detection failed')
            print(f"[Worker] FAILED {sample_name}: {reason}")
            return (idx, _zero_metric(reason))

        # Extract predicted mask
        mask_pred = extract_mask_from_agent_result(result, image_height, image_width)

        if mask_pred is None:
            reason = "No mask extracted from agent result (all detection calls returned no bbox/points)"
            print(f"[Worker] FAILED {sample_name}: {reason}")
            return (idx, _zero_metric(reason))

        # Resize if needed
        if mask_gt.shape != mask_pred.shape:
            mp = Image.fromarray(mask_pred.astype(np.uint8) * 255)
            mp = mp.resize((mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
            mask_pred = np.array(mp) > 0

        # Compute metrics
        iou = calculate_iou(mask_pred, mask_gt)
        inter, union_val = calculate_intersection_and_union(mask_pred, mask_gt)
        p50 = 1.0 if iou >= 0.5 else 0.0
        kld = calculate_kld(mask_pred, mask_gt)
        sim = calculate_sim(mask_pred, mask_gt)
        nss = calculate_nss(mask_pred, mask_gt)

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

        sample_metric = {
            "sample_name": sample_name,
            "sample_index": idx,
            "category": category,
            "affordance_type": affordance_type,
            "iou": float(iou),
            "intersection": int(inter),
            "union": int(union_val),
            "precision_at_50": p50,
            "KLD": float(kld) if np.isfinite(kld) else None,
            "SIM": float(sim),
            "NSS": float(nss),
            "task": task,
            "tool_trajectory": tool_trajectory,
            "llm_reasoning": llm_reasoning_texts,
            "strategy_reasoning": result.get("strategy_reasoning", ""),
            "strategy_reflection": result.get("strategy_reflection", ""),
            "dynamic_params_used": result.get("dynamic_params_used", {}),
            "decision_trace": result.get("decision_trace", []),
            "image_path": pred_image_path,
            "_memory_tool_calls": result.get("_memory_tool_calls", []),
            "_memory_tool_summaries": result.get("_memory_tool_summaries", []),
            "_memory_llm_reasoning": result.get("_memory_llm_reasoning", []),
            "_memory_task_context": result.get("_memory_task_context"),
            "_memory_mask_path": result.get("_memory_mask_path"),
            "_memory_reasoning_context": result.get("final_response", ""),
        }
        return (idx, sample_metric)

    except Exception as e:
        print(f"[Worker] Error processing sample {idx} ({sample.get('sample_name', 'unknown')}): {e}")
        import traceback
        traceback.print_exc()
        # Try to build a zero-metric dict so this sample still counts as IoU=0
        try:
            _mask_gt = np.array(Image.open(sample["mask_path"]).convert("L")) > 0
            _gt_union = int(_mask_gt.sum())
        except Exception:
            _gt_union = 1
        _task = f"Find the part of the object in the center of the image that can {affordance_type}"
        return (idx, {
            "sample_name": sample_name,
            "sample_index": idx,
            "category": category,
            "affordance_type": affordance_type,
            "iou": 0.0,
            "intersection": 0,
            "union": _gt_union if _gt_union > 0 else 1,
            "precision_at_50": 0.0,
            "KLD": None,
            "SIM": 0.0,
            "NSS": 0.0,
            "task": _task,
            "tool_trajectory": [],
            "llm_reasoning": [],
            "strategy_reasoning": "",
            "strategy_reflection": "",
            "dynamic_params_used": {},
            "decision_trace": [],
            "image_path": pred_image_path,
            "_memory_tool_calls": [],
            "_memory_tool_summaries": [],
            "_memory_llm_reasoning": [],
            "_memory_task_context": None,
            "_memory_mask_path": None,
            "_memory_reasoning_context": "",
            "detection_failed": True,
            "failure_reason": f"Exception: {e}",
        })


def _write_batch_to_memory_umd(
    agent: AffordanceAgent,
    batch_results: List[Dict[str, Any]],
) -> int:
    """Replay add_entry for every successful sample in a finished batch."""
    if agent.memory_manager is None:
        return 0
    written = 0
    for sm in batch_results:
        try:
            sid = sm.get("sample_name", sm.get("sample_id", ""))
            agent.memory_manager.add_entry(
                sample_id=sid,
                image_path=sm.get("image_path", ""),
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
            eval_metrics = {"iou": sm["iou"], "precision_at_50": sm["precision_at_50"]}
            agent.memory_manager.update_entry_metrics(sid, eval_metrics)
            written += 1
        except Exception as e:
            print(f"[BatchSync] Failed to write memory for sample {sm.get('sample_name')}: {e}")
    return written


# Main evaluation

def evaluate_on_umd(
    dataset_path: str,
    agent: AffordanceAgent,
    output_dir: str,
    max_samples: Optional[int] = None,
    save_predictions: bool = True,
    resume: bool = False,
    num_workers: int = 1,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 0,
) -> Dict[str, Any]:

    print(f"Loading UMD dataset from {dataset_path}...")
    samples = load_umd_samples(dataset_path, max_samples=max_samples)
    print(f"Evaluating on {len(samples)} samples ({len(set(s['image_path'] for s in samples))} unique images)...")

    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    all_samples_metrics_file = os.path.join(output_dir, "all_samples_metrics.json")
    realtime_metrics_file = os.path.join(output_dir, "metrics_realtime.json")
    evaluation_results_file = os.path.join(output_dir, "evaluation_results.json")

    # SAFE: never delete existing data. Always auto-detect & skip processed samples.
    os.makedirs(predictions_dir, exist_ok=True)

    # Global accumulators
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Always try to load existing results (auto-resume)
    (
        processed_sample_names,
        all_sample_metrics,
        ious_all,
        cum_inter_all,
        cum_union_all,
        p50_all,
        p_thresh_all,
        per_aff,
        kld_all,
        sim_all,
        nss_all,
    ) = _load_resume_state_umd(output_dir, predictions_dir, iou_thresholds)
    if processed_sample_names:
        print(f"[Auto-resume] Found {len(processed_sample_names)} already-processed samples, will skip them.")

    failed_samples: List[str] = []

    # Build list of pending samples (always skip already-processed)
    pending = []
    for idx, sample in enumerate(samples):
        # Skip by sample_name (from loaded metrics)
        if sample["sample_name"] in processed_sample_names:
            continue
        # Also skip if prediction file already exists on disk (robust index-based check)
        if os.path.exists(os.path.join(predictions_dir, f"{idx}_prediction.json")):
            continue
        pending.append((idx, sample))

    # -- helpers shared by serial / concurrent paths --

    def _accumulate_one(sm: Dict[str, Any]) -> None:
        """Accumulate one processed sample into global and per-affordance accumulators."""
        nonlocal cum_inter_all, cum_union_all

        iou_val = sm["iou"]
        inter = sm["intersection"]
        union_val = sm["union"]
        p50_val = sm["precision_at_50"]
        kld_val = sm.get("KLD")
        sim_val = sm["SIM"]
        nss_val = sm["NSS"]
        affordance_type = sm["affordance_type"]

        ious_all.append(iou_val)
        cum_inter_all += inter
        cum_union_all += union_val
        p50_all.append(p50_val)
        for t in iou_thresholds:
            p_thresh_all[t].append(1.0 if iou_val >= t else 0.0)
        if kld_val is not None:
            kld_all.append(kld_val)
        sim_all.append(sim_val)
        nss_all.append(nss_val)

        aff = per_aff[affordance_type]
        aff["ious"].append(iou_val)
        aff["cum_inter"] += inter
        aff["cum_union"] += union_val
        aff["p50"].append(p50_val)
        for t in iou_thresholds:
            aff["p_thresh"][t].append(1.0 if iou_val >= t else 0.0)
        if kld_val is not None:
            aff["kld"].append(kld_val)
        aff["sim"].append(sim_val)
        aff["nss"].append(nss_val)

        # Feed metrics back to memory only for successful detections
        if not sm.get("detection_failed") and agent.enable_memory and agent.memory_manager:
            sample_name = sm.get("sample_name")
            if sample_name:
                eval_metrics = {
                    "iou": iou_val,
                    "precision_at_50": p50_val,
                }
                agent.memory_manager.update_entry_metrics(sample_name, eval_metrics)

    def _display_and_save(sm: Dict[str, Any], completed: int, total: int) -> None:
        """Print progress and save incremental files after one sample is done."""
        sample_name = sm["sample_name"]
        iou_val = sm["iou"]
        p50_val = sm["precision_at_50"]
        kld_val = sm.get("KLD")
        sim_val = sm["SIM"]
        nss_val = sm["NSS"]
        is_failed = sm.get("detection_failed", False)

        cur_gIoU = np.mean(ious_all)
        cur_cIoU = float(cum_inter_all) / float(cum_union_all) if cum_union_all > 0 else 0.0
        cur_P50 = np.mean(p50_all)
        cur_P50_95 = float(np.mean([np.mean(p_thresh_all[t]) for t in iou_thresholds])) if len(ious_all) > 0 else 0.0
        finite_kld = [v for v in kld_all if np.isfinite(v)]
        cur_KLD = np.mean(finite_kld) if finite_kld else float('inf')
        cur_SIM = np.mean(sim_all)
        cur_NSS = np.mean(nss_all)

        valid_count = len(ious_all) - len(failed_samples)
        kld_s = f"{kld_val:.4f}" if kld_val is not None and np.isfinite(kld_val) else "inf"
        ckld_s = f"{cur_KLD:.4f}" if np.isfinite(cur_KLD) else "inf"
        status_prefix = "[FAILED] " if is_failed else ""
        print(f"\n[{completed}/{total}] {status_prefix}{sample_name}")
        if is_failed:
            print(f"  Detection failed: {sm.get('failure_reason', 'unknown')} -> IoU=0.0 counted")
        else:
            print(f"  IoU={iou_val:.4f}, P@50={p50_val:.0f}, KLD={kld_s}, SIM={sim_val:.4f}, NSS={nss_val:.4f}")
        print(f"  Cumul: gIoU={cur_gIoU:.4f}, cIoU={cur_cIoU:.4f}, P@50={cur_P50:.4f}, P@50-95={cur_P50_95:.4f}, KLD={ckld_s}, SIM={cur_SIM:.4f}, NSS={cur_NSS:.4f}")
        print(f"  Processed={len(ious_all)}, Valid={valid_count}, Failed={len(failed_samples)}")

        # Dedup and add to all_sample_metrics
        existing_idx = next((i for i, m in enumerate(all_sample_metrics) if m.get("sample_name") == sample_name), None)
        if existing_idx is not None:
            all_sample_metrics[existing_idx] = sm
        else:
            all_sample_metrics.append(sm)
        processed_sample_names.add(sample_name)

        if save_predictions:
            pred_path = os.path.join(predictions_dir, f"{sm['sample_index']}_prediction.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(sm, f, indent=2, ensure_ascii=False)

        with open(realtime_metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "current_sample": completed,
                "total_samples": total,
                "valid_samples": valid_count,
                "failed_samples": len(failed_samples),
                "total_processed": len(ious_all),
                "gIoU": float(cur_gIoU),
                "cIoU": float(cur_cIoU),
                "P_50": float(cur_P50),
                "P_50-95": float(cur_P50_95),
                "KLD": float(cur_KLD) if np.isfinite(cur_KLD) else None,
                "SIM": float(cur_SIM),
                "NSS": float(cur_NSS),
            }, f, indent=2, ensure_ascii=False)

        with open(all_samples_metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_sample_metrics, f, indent=2, ensure_ascii=False)

    # -- dispatch to serial or concurrent evaluation --

    if num_workers > 1 and agent_kwargs is not None and pending:
        # ====== Batch-parallel evaluation ======
        effective_batch = batch_size if batch_size > 0 else num_workers
        n_batches = (len(pending) + effective_batch - 1) // effective_batch
        print(f"\n[BatchParallel] {num_workers} workers, batch_size={effective_batch}, "
              f"{len(pending)} pending samples -> {n_batches} batches")

        agent_pool: _queue.Queue[AffordanceAgent] = _queue.Queue()
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
                result = _process_one_sample_umd(_idx, _sample, a, predictions_dir)
                if result[1] is None:
                    wait = _random.uniform(3.0, 8.0)
                    print(f"  [Worker] Sample {_idx} failed, retrying after {wait:.1f}s ...")
                    _time.sleep(wait)
                    result = _process_one_sample_umd(_idx, _sample, a, predictions_dir)
                return result
            finally:
                agent_pool.put(a)

        completed_count = len(processed_sample_names)
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
                        print(f"  [{completed_count}/{len(samples)}] Sample {result_idx} FAILED (unrecoverable)")
                        continue

                    # Track failed samples (detection_failed=True means IoU=0 but still counted)
                    if sm.get("detection_failed"):
                        failed_samples.append(sm.get("sample_name", f"sample_{result_idx}"))

                    try:
                        _accumulate_one(sm)
                        _display_and_save(sm, completed_count, len(samples))
                        # Only write successful detections to memory
                        if not sm.get("detection_failed"):
                            batch_successful.append(sm)
                    except Exception as e:
                        print(f"Error accumulating sample {result_idx}: {e}")
                        if not sm.get("detection_failed"):
                            failed_samples.append(sm.get("sample_name", f"sample_{result_idx}"))

            if batch_successful:
                written = _write_batch_to_memory_umd(agent, batch_successful)
                print(f"  [BatchSync] Wrote {written}/{len(batch_successful)} "
                      f"entries to memory after batch {batch_idx + 1}")
                if agent.memory_manager and hasattr(agent.memory_manager, 'experience_pool'):
                    try:
                        agent.memory_manager.experience_pool._maybe_distill()
                    except Exception as e:
                        print(f"  [BatchSync] Distill warning: {e}")

        agent.enable_memory = True
    else:
        # ====== Serial evaluation (original behaviour) ======
        completed_count = len(processed_sample_names)
        for idx, sample in tqdm(pending, desc="Evaluating"):
            result_idx, sm = _process_one_sample_umd(idx, sample, agent, predictions_dir)
            completed_count += 1

            if sm is None:
                # Safety net: shouldn't happen since worker now returns zero-metric dicts
                failed_samples.append(sample.get("sample_name", f"sample_{idx}"))
                continue

            if sm.get("detection_failed"):
                failed_samples.append(sm.get("sample_name", f"sample_{idx}"))

            try:
                _accumulate_one(sm)
                _display_and_save(sm, completed_count, len(samples))
            except Exception as e:
                print(f"Error accumulating sample {idx}: {e}")
                if not sm.get("detection_failed"):
                    failed_samples.append(sample.get("sample_name", f"sample_{idx}"))

    # ----- Final metrics -----
    overall = compute_metrics_from_lists(ious_all, cum_inter_all, cum_union_all, p50_all, p_thresh_all, iou_thresholds,
                                         kld_list=kld_all, sim_list=sim_all, nss_list=nss_all)
    overall["failed_samples"] = len(failed_samples)

    # Per-affordance metrics
    per_affordance_metrics = {}
    for aff_type, aff_data in sorted(per_aff.items()):
        per_affordance_metrics[aff_type] = compute_metrics_from_lists(
            aff_data["ious"], aff_data["cum_inter"], aff_data["cum_union"],
            aff_data["p50"], aff_data["p_thresh"], iou_thresholds,
            kld_list=aff_data["kld"], sim_list=aff_data["sim"], nss_list=aff_data["nss"],
        )

    final_results = {
        "overall": overall,
        "per_affordance": per_affordance_metrics,
    }

    # Save
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    samples_path = os.path.join(output_dir, "all_samples_metrics.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(all_sample_metrics, f, indent=2, ensure_ascii=False)

    # Print
    print("\n" + "=" * 70)
    print("UMD Part-Affordance Evaluation Results")
    print("=" * 70)
    kld_display = f"{overall['KLD']:.4f}" if np.isfinite(overall.get('KLD', 0)) else "inf"
    print(f"\nOverall ({overall['num_samples']} valid, {len(failed_samples)} failed):")
    print(f"  gIoU:     {overall['gIoU']:.4f}")
    print(f"  cIoU:     {overall['cIoU']:.4f}")
    print(f"  P@50:     {overall['P_50']:.4f}")
    print(f"  P@50:95:  {overall['P_50-95']:.4f}")
    print(f"  KLD:      {kld_display}")
    print(f"  SIM:      {overall['SIM']:.4f}")
    print(f"  NSS:      {overall['NSS']:.4f}")

    print(f"\nPer-Affordance:")
    print(f"  {'Affordance':<15} {'gIoU':>8} {'cIoU':>8} {'P@50':>8} {'P@50:95':>8} {'KLD':>8} {'SIM':>8} {'NSS':>8} {'N':>6}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for aff_type, m in per_affordance_metrics.items():
        akld = f"{m['KLD']:>8.4f}" if np.isfinite(m.get('KLD', 0)) else "     inf"
        print(f"  {aff_type:<15} {m['gIoU']:>8.4f} {m['cIoU']:>8.4f} {m['P_50']:>8.4f} {m['P_50-95']:>8.4f} {akld} {m['SIM']:>8.4f} {m['NSS']:>8.4f} {m['num_samples']:>6}")
    print("=" * 70)
    print(f"Results saved to: {results_path}")

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

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate affordance detection on UMD Part-Affordance dataset")
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/UMD_preprocessed",
                        help="Path to preprocessed UMD dataset")
    parser.add_argument("--output_dir", type=str,
                        default="output/evaluation_umd",
                        help="Output directory for evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (overrides config/env)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key")
    parser.add_argument("--api_url", type=str, default=None,
                        help="API base URL")
    parser.add_argument("--no_save_predictions", action="store_true",
                        help="Do not save individual prediction results")
    parser.add_argument("--detection_only", action="store_true",
                        help="Only expose 'detection' tool to LLM (ablation)")
    parser.add_argument("--detection_backend", type=str, default="qwen3vl_api",
                        choices=["qwen3vl_api", "rex_omni"],
                        help="Detection backend: qwen3vl_api (cloud API, default) or rex_omni (local model)")
    parser.add_argument("--detection_model", type=str, default=None,
                        help=(
                            "Override the detection API model name (for qwen3vl_api backend). "
                            "Default: qwen3-vl-235b-a22b-instruct. "
                            "Example: qwen3-vl-8b-instruct, Qwen3.5-397B-A17B. "
                            "Sets env var DETECTION_MODEL_NAME used by detection.py."
                        ))
    parser.add_argument("--detection_api_key", type=str, default=None,
                        help="Override detection API key (sets DETECTION_API_KEY). "
                             "Use when detection model is on a different endpoint, e.g. PAI-EAS.")
    parser.add_argument("--detection_api_base_url", type=str, default=None,
                        help="Override detection API base URL (sets DETECTION_API_BASE_URL). "
                             "Example: http://xxx.pai-eas.aliyuncs.com/api/predict/qwen3_5")
    parser.add_argument("--sam_backend", type=str, default="sam2",
                        choices=["sam2", "sam3"],
                        help="Segmentation backend: sam2 (default) or sam3 (Meta SAM3 via transformers). Sets env var SAM_BACKEND.")
    parser.add_argument("--resume", action="store_true",
                        help="(Deprecated: auto-resume is now always on) Resume from existing output_dir and skip processed samples")
    parser.add_argument("--clean", action="store_true",
                        help="Clean output_dir before starting. Creates a timestamped backup first!")
    parser.add_argument("--clear_memory", action="store_true",
                        help="Clear existing memory before starting evaluation (start fresh)")
    parser.add_argument("--clear_experience", action="store_true",
                        help="Also clear the experience pool when clearing memory (by default experience pool is preserved across evaluations)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of concurrent workers (1 = serial, >1 = parallel evaluation)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for batch-parallel evaluation (0 = auto, same as num_workers)")
    parser.add_argument("--commonsense_templates_dir", type=str, default=None,
                        help="Directory containing commonsense_templates.json (overrides default)")
    parser.add_argument("--disable_commonsense", action="store_true",
                        help="Disable commonsense template bank (ablation: no commonsense knowledge).")
    parser.add_argument("--disable_memory", action="store_true",
                        help="Disable episodic memory module entirely (ablation: no historical memory).")
    parser.add_argument("--fixed_skill_chain", action="store_true",
                        help=(
                            "Baseline: force fixed skill execution order "
                            "zoom_in → dreamer → web_search → detection, "
                            "without LLM dynamic tool selection. "
                            "Automatically implies --disable_memory and --disable_commonsense."
                        ))

    args = parser.parse_args()

    # Load config
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

    tool_filter = {"detection"} if args.detection_only else None
    if tool_filter:
        print(f"⚠️  Detection-only mode: LLM can only call {tool_filter}")
    print(f"Detection backend: {args.detection_backend}")

    # --clean: backup existing output, then remove it
    if args.clean and os.path.exists(args.output_dir):
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = args.output_dir.rstrip("/") + f"_backup_{ts}"
        print(f"[Clean] Backing up {args.output_dir} -> {backup_dir}")
        shutil.copytree(args.output_dir, backup_dir)
        shutil.rmtree(args.output_dir)
        print(f"[Clean] Removed {args.output_dir}. Backup at: {backup_dir}")

    if args.disable_commonsense:
        print("⚠️  Commonsense-disabled mode: commonsense template bank is turned off.")
    if args.disable_memory:
        print("⚠️  Memory-disabled mode: episodic memory module is turned off.")

    #  run_config.json（）
    from datetime import datetime as _dt
    import importlib.util as _ilu
    os.makedirs(args.output_dir, exist_ok=True)
    _detection_model_used = (
        args.detection_model
        or os.environ.get("DETECTION_MODEL_NAME")
        or "qwen3-vl-235b-a22b-instruct"
    )
    _run_config = {
        "dataset": "umd",
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

    agent = AffordanceAgent(
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

    # （ --clear_memory）
    if args.clear_memory:
        print("🗑️  Clearing existing memory before evaluation...")
        agent.clear_memory(clear_experience=args.clear_experience)

    # Agent creation kwargs (used to spawn worker agents in concurrent mode)
    agent_kwargs = dict(
        api_key=api_key, api_base_url=api_url, model_name=model,
        output_dir=args.output_dir, tool_filter=tool_filter,
        detection_backend=args.detection_backend,
        commonsense_templates_dir=args.commonsense_templates_dir,
        fixed_skill_chain=getattr(args, "fixed_skill_chain", False),
    )

    if args.num_workers > 1:
        eff_bs = args.batch_size if args.batch_size > 0 else args.num_workers
        print(f"Batch-parallel mode: {args.num_workers} workers, batch_size={eff_bs}")

    evaluate_on_umd(
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

    # （ pending observations ）
    agent.flush_experience_pool()


if __name__ == "__main__":
    main()
