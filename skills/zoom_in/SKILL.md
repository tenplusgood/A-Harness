---
name: zoom-in
description: refine ambiguous recognition by zooming into a selected region for closer analysis
characteristics:
  - very fast local visual inspection tool
  - produces cropped reference image for reasoning only
  - helps resolve tiny/occluded target ambiguity
when_to_use:
  - target is small, far, blurry, or partially occluded
  - model needs closer inspection before choosing object part
when_not_to_use:
  - object and part are already clearly visible
  - do not pass zoomed image as detection input
---
Inputs:
- image_path (string, required): absolute path to the RGB scene image.
- object_name (string, required): The name of the object to be identified.

Behavior (for the model to follow, no scripts needed):
- When the current recognition result is uncertain, you can select a local area, crop and zoom in on it, and reanalyze only that zoomed-in area.
- Use the zoomed-in view to refine which object/part it is and update your reasoning before calling other skills (e.g. rex_omni, sam2).

Outputs (conceptual, no actual file I/O required):
- refined_hypothesis (string): updated, more confident description of the object or part in the zoomed region.