"""
Precompute commonsense template bank before evaluation.

This script scans datasets/* and writes a compact representative template bank
to a dedicated folder. During evaluation, MemoryManager only reads this bank
and does not build on the fly.

After building templates and materializing assets, it also pre-computes image
embeddings for all template images using both SigLIP2 and DINOv2 so that
evaluation-time retrieval can combine multi-encoder similarity with text
similarity for more robust results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .manager import MemoryManager


def _build_image_embeddings(templates_dir: Path, templates: list, model_name: str) -> dict:
    """Pre-compute SigLIP image embeddings for all template images.

    Saves two files under *templates_dir*:
      - clip_embeddings.npy   – float32 array of shape (N, D)
      - clip_index.json       – ordered list of template_ids (row ↔ id mapping)

    Returns summary dict.
    """
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        from PIL import Image
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    print(f"[SigLIP] Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    index_ids: list = []
    embeddings: list = []
    skipped = 0

    for i, t in enumerate(templates):
        tid = t.get("template_id", "")
        rel_img = t.get("image_path", "")
        if not rel_img:
            skipped += 1
            continue
        img_path = Path(rel_img)
        if not img_path.is_absolute():
            img_path = templates_dir / img_path
        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().flatten())
            index_ids.append(tid)
        except Exception as exc:
            print(f"[SigLIP] Failed to embed {img_path.name}: {exc}")
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"[SigLIP] Embedded {i + 1}/{len(templates)} ...")

    if not embeddings:
        return {"success": False, "error": "no images could be embedded"}

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    emb_path = templates_dir / "clip_embeddings.npy"
    idx_path = templates_dir / "clip_index.json"

    np.save(str(emb_path), emb_array)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_ids, f)

    print(
        f"[SigLIP] Saved {emb_array.shape[0]} embeddings "
        f"(dim={emb_array.shape[1]}) to {emb_path}"
    )
    return {
        "success": True,
        "embedded": emb_array.shape[0],
        "dim": int(emb_array.shape[1]),
        "skipped": skipped,
        "emb_path": str(emb_path),
        "index_path": str(idx_path),
    }


def _build_dino_embeddings(templates_dir: Path, templates: list, model_name: str) -> dict:
    """Pre-compute DINOv2 image embeddings for all template images.

    Saves two files under *templates_dir*:
      - dino_embeddings.npy   – float32 array of shape (N, D)
      - dino_index.json       – ordered list of template_ids (row <-> id mapping)

    Returns summary dict.
    """
    try:
        import torch
        from transformers import AutoModel, AutoImageProcessor
        from PIL import Image
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    print(f"[DINOv2] Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)

    index_ids: list = []
    embeddings: list = []
    skipped = 0

    for i, t in enumerate(templates):
        tid = t.get("template_id", "")
        rel_img = t.get("image_path", "")
        if not rel_img:
            skipped += 1
            continue
        img_path = Path(rel_img)
        if not img_path.is_absolute():
            img_path = templates_dir / img_path
        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0]  # CLS token
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy().flatten())
            index_ids.append(tid)
        except Exception as exc:
            print(f"[DINOv2] Failed to embed {img_path.name}: {exc}")
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"[DINOv2] Embedded {i + 1}/{len(templates)} ...")

    if not embeddings:
        return {"success": False, "error": "no images could be embedded"}

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    emb_path = templates_dir / "dino_embeddings.npy"
    idx_path = templates_dir / "dino_index.json"

    np.save(str(emb_path), emb_array)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_ids, f)

    print(
        f"[DINOv2] Saved {emb_array.shape[0]} embeddings "
        f"(dim={emb_array.shape[1]}) to {emb_path}"
    )
    return {
        "success": True,
        "embedded": emb_array.shape[0],
        "dim": int(emb_array.shape[1]),
        "skipped": skipped,
        "emb_path": str(emb_path),
        "index_path": str(idx_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare commonsense template bank")
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=None,
        help="Datasets root directory (default: data/)",
    )
    parser.add_argument(
        "--templates_dir",
        type=str,
        default=None,
        help="Output folder for commonsense_templates.json (default: commonsense_templates/)",
    )
    parser.add_argument(
        "--max_per_pair",
        type=int,
        default=3,
        help="Max templates kept for each object-affordance_part pair",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Max total templates in bank (<=0 means unlimited)",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild even if template file already exists",
    )
    parser.add_argument(
        "--no_materialize_assets",
        action="store_true",
        help="Do not copy image/GT assets into templates_dir",
    )
    parser.add_argument(
        "--no_clip_embeddings",
        action="store_true",
        help="Skip image embedding pre-computation",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="google/siglip2-base-patch16-384",
        help="Vision encoder model name for image embedding (default: SigLIP2)",
    )
    parser.add_argument(
        "--no_dino_embeddings",
        action="store_true",
        help="Skip DINOv2 image embedding pre-computation",
    )
    parser.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov2-base",
        help="DINOv2 model name for image embedding (default: facebook/dinov2-base)",
    )
    args = parser.parse_args()

    mm = MemoryManager(
        enable_commonsense_templates=True,
        commonsense_max_per_pair=args.max_per_pair,
        commonsense_max_total=args.max_total,
        datasets_root=args.datasets_root,
        commonsense_templates_dir=args.templates_dir,
        commonsense_auto_build=True,  # preparation stage allows building
    )
    result = mm.prepare_commonsense_templates(force_rebuild=args.force_rebuild)
    if not args.no_materialize_assets:
        materialize_result = mm.materialize_commonsense_assets(rewrite_relative=True)
        result["materialize_assets"] = materialize_result

    # Pre-compute CLIP image embeddings for hybrid retrieval
    if not args.no_clip_embeddings:
        tdir = (
            Path(args.templates_dir)
            if args.templates_dir
            else (Path(__file__).resolve().parent / "commonsense_templates")
        )
        clip_result = _build_image_embeddings(tdir, mm._commonsense_templates, args.clip_model)
        result["clip_embeddings"] = clip_result
    else:
        print("[CLIP] Skipping embedding computation (--no_clip_embeddings)")

    # Pre-compute DINOv2 image embeddings for hybrid retrieval
    if not args.no_dino_embeddings:
        tdir = (
            Path(args.templates_dir)
            if args.templates_dir
            else (Path(__file__).resolve().parent / "commonsense_templates")
        )
        dino_result = _build_dino_embeddings(tdir, mm._commonsense_templates, args.dino_model)
        result["dino_embeddings"] = dino_result
    else:
        print("[DINOv2] Skipping embedding computation (--no_dino_embeddings)")

    print(json.dumps(result, indent=2, ensure_ascii=False))

    fp = mm._commonsense_file_path()
    if fp and Path(fp).exists():
        print(f"Template bank saved to: {fp}")


if __name__ == "__main__":
    main()

