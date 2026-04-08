"""UMD Part-Affordance dataset reader."""

from __future__ import annotations

import os
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class UmdDataset(Dataset):
    """Read preprocessed UMD images and per-affordance masks."""

    dataset_type = "UMD"

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.mask_files: List[str] = []

        if not os.path.exists(root_dir):
            raise RuntimeError(f"Root directory not found: {root_dir}")

        for tool_dir in sorted(os.listdir(self.root_dir)):
            tool_dir_path = os.path.join(self.root_dir, tool_dir)
            if not os.path.isdir(tool_dir_path):
                continue
            for file_name in sorted(os.listdir(tool_dir_path)):
                if file_name.endswith("_gt_mask.png"):
                    self.mask_files.append(os.path.join(tool_dir_path, file_name))

    def __len__(self) -> int:
        return len(self.mask_files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if torch.is_tensor(idx):
            idx = int(idx.tolist())

        mask_path = self.mask_files[idx]
        mask_filename = os.path.basename(mask_path)
        affordance_type = mask_filename.split("_gt_mask.png")[0].split("_")[-1]

        base_filename = "_".join(mask_filename.split("_")[:-3])
        image_path = os.path.join(os.path.dirname(mask_path), base_filename + "_rgb.jpg")
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "mask_path": mask_path,
            "image_path": image_path,
            "affordance_type": affordance_type,
        }
