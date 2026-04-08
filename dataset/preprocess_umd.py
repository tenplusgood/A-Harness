"""Preprocess UMD .mat labels into binary affordance masks."""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm


def preprocess_umd_dataset(dataset_path: str, output_path: str, sample_stride: int = 30, sample_offset: int = 2) -> None:
    """Convert UMD label_rank mats into per-affordance PNG masks."""
    affordance_map = {
        "knife": ["cut", "grasp"],
        "saw": ["cut", "grasp"],
        "scissors": ["cut", "grasp"],
        "shears": ["cut", "grasp"],
        "scoop": ["scoop", "grasp"],
        "spoon": ["scoop", "grasp"],
        "trowel": ["scoop", "grasp"],
        "bowl": ["contain", "grasp"],
        "cup": ["contain", "grasp"],
        "ladle": ["contain", "grasp"],
        "mug": ["contain", "grasp"],
        "pot": ["contain", "grasp"],
        "shovel": ["support", "grasp"],
        "turner": ["support", "grasp"],
        "hammer": ["pound", "grasp"],
        "mallet": ["pound", "grasp"],
        "tenderizer": ["pound", "grasp"],
    }
    affordance_to_index = {
        "grasp": 1,
        "cut": 2,
        "scoop": 3,
        "contain": 4,
        "pound": 5,
        "support": 6,
        "wrap-grasp": 7,
    }

    os.makedirs(output_path, exist_ok=True)
    tool_dirs = sorted(d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)))

    for tool_dir in tqdm(tool_dirs, desc="Processing tool categories"):
        category = tool_dir.split("_")[0]
        if category not in affordance_map:
            continue

        actions = affordance_map[category]
        tool_dir_path = os.path.join(dataset_path, tool_dir)
        output_tool_dir = os.path.join(output_path, tool_dir)
        os.makedirs(output_tool_dir, exist_ok=True)

        label_rank_files = sorted(f for f in os.listdir(tool_dir_path) if f.endswith("_label_rank.mat"))
        rgb_files = sorted(f for f in os.listdir(tool_dir_path) if f.endswith(".jpg"))
        if len(label_rank_files) != len(rgb_files):
            raise RuntimeError(f"Mismatched files in {tool_dir}: {len(label_rank_files)} mats vs {len(rgb_files)} images")

        for i, label_rank_file in enumerate(label_rank_files):
            if i % sample_stride != sample_offset:
                continue
            rgb_file = rgb_files[i]
            mat_data = scipy.io.loadmat(os.path.join(tool_dir_path, label_rank_file))
            gt_mat = mat_data["gt_label"]
            h, w = gt_mat.shape[:2]

            for action in actions:
                if action not in affordance_to_index:
                    continue
                action_index = affordance_to_index[action] - 1
                if gt_mat.shape[2] <= action_index:
                    continue

                rank_channel = gt_mat[:, :, action_index]
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                gt_mask[rank_channel == 1] = 255

                output_mask_name = label_rank_file.replace("_label_rank.mat", f"_{action}_gt_mask.png")
                Image.fromarray(gt_mask).save(os.path.join(output_tool_dir, output_mask_name))
                shutil.copy(os.path.join(tool_dir_path, rgb_file), os.path.join(output_tool_dir, rgb_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UMD Part-Affordance dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to raw UMD tools directory")
    parser.add_argument("--output_path", required=True, help="Output path for preprocessed dataset")
    parser.add_argument("--sample_stride", type=int, default=30, help="Keep every N-th frame")
    parser.add_argument("--sample_offset", type=int, default=2, help="Offset within stride")
    args = parser.parse_args()
    preprocess_umd_dataset(args.dataset_path, args.output_path, args.sample_stride, args.sample_offset)
    print("UMD preprocessing complete.")


if __name__ == "__main__":
    main()
