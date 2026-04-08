---
name: detection
description: An integrated visual detection and segmentation tool that combines open-vocabulary object detection with prompt-driven precise segmentation capabilities, implementing a complete recognition pipeline from textual descriptions to pixel-level masks. 
characteristics:
  - mandatory final step for every sample
  - supports multi-target detection and segmentation
  - runs on the original image and outputs mask files
when_to_use:
  - always call this tool before finishing
  - call directly when target object/part is already clear
  - call after helper tools when additional evidence is needed
when_not_to_use:
  - do not skip detection
  - do not use zoomed or generated images as detection input
---
Inputs:
- image_path (string, required): absolute path to the RGB scene image.
- task (string, required): affordance task description (e.g., "Find the part of the bench that can be sat on").
- object_name (string, optional): object category name if known (e.g., "bench", "mug").

All coordinates should be normalized to \[0,1] in image space (origin at top-left, x rightward, y downward).

Outputs (JSON object):
- mask_image_path (string): path to the saved binary mask image.
- visualization_path (string): path to the saved visualization image (mask overlay on the input image).
- mask_shape (array[2] of int, optional): \[height, width] of the mask (if provided by the implementation).
