"""
ReasonAff Dataset Reader.

Loads the ReasonAff test dataset (HuggingFace datasets format saved via save_to_disk).
Each sample contains:
  - id: sample id
  - image: PIL Image
  - mask: ground truth mask (numpy array)
  - problem: task description / question
  - aff_name: affordance name
  - part_name: object part name
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ReasonAffDataset(Dataset):
    """PyTorch Dataset for reading the ReasonAff dataset."""

    dataset_type = "ReasonAff"

    def __init__(self, base_dir: str, max_samples: int = None):
        """
        Args:
            base_dir: Path to the dataset directory (load_from_disk compatible).
            max_samples: Optional max number of samples to load.
        """
        self.base_dir = base_dir

        try:
            from datasets import load_from_disk
        except ImportError:
            raise ImportError("Please install HuggingFace datasets: pip install datasets")

        print(f"Loading ReasonAff dataset from {base_dir}...")
        self.hf_dataset = load_from_disk(base_dir)

        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(
                range(min(max_samples, len(self.hf_dataset)))
            )

        print(f"Loaded {len(self.hf_dataset)} samples from ReasonAff dataset")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.hf_dataset[idx]

        image = sample.get("image")
        if image is None:
            raise ValueError(f"Sample {idx} has no image")

        mask = np.array(sample.get("mask", []))
        if mask.size == 0:
            mask = np.zeros((image.height, image.width), dtype=bool)
        if mask.ndim > 2:
            mask = mask.reshape(-1, mask.shape[-1])
        mask = mask.astype(bool)

        image_width, image_height = image.size
        if mask.shape != (image_height, image_width):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((image_width, image_height), Image.NEAREST)
            mask = np.array(mask_pil) > 0

        sample_id = sample.get("id", f"sample_{idx}")
        problem = sample.get("problem", "")
        aff_name = sample.get("aff_name", "")
        part_name = sample.get("part_name", "")

        return {
            "image": image,
            "mask": mask,
            "image_path": None,
            "mask_path": None,
            "sample_id": sample_id,
            "question": problem,
            "aff_name": aff_name,
            "part_name": part_name,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    ds = ReasonAffDataset(args.dataset_path, max_samples=args.max_samples)
    print(f"\nDataset size: {len(ds)}")

    for i in range(min(3, len(ds))):
        s = ds[i]
        print(f"\n--- Sample {i} ---")
        print(f"  ID: {s['sample_id']}")
        print(f"  Question: {s['question']}")
        print(f"  Aff name: {s['aff_name']}, Part name: {s['part_name']}")
        print(f"  Image size: {s['image'].size}")
        print(f"  Mask shape: {s['mask'].shape}")
