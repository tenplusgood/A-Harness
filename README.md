# A-Harness: An Agentic Framework for Affordance Detection

[![Paper](#)](#) [![Project Page](#)](#)

<div class="authors">
  <a href="http://www.wonghougin.me/">Haojian Huang</a><sup>1,2*</sup>, 
  <a href="https://tenplusgood.github.io/">Jiahao Shi</a><sup>2,3*</sup>, 
  Yinchuan Li<sup>1,2</sup>, 
  Yingcong Chen<sup>1,2†</sup>
</div>

<div class="affiliations">
  <sup>1</sup>HKUST(GZ), <sup>2</sup>Knowin, <sup>3</sup>Harbin Engineering University<br>
  <sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author
</div>

<!-- Teaser image placeholder -->
<div class="teaser-box">
  <img src="{{ '/Figures/comp.png' | relative_url }}" alt="A-Harness Teaser">
  <p style="margin-top: 1rem; font-style: italic;">A-Harness: A verification-gated runtime that unifies heterogeneous skills with adaptive routing and episodic memory.</p>
</div>
<!-- ![A-Harness](assets/teaser.png) -->

A-Harness is an agentic framework for affordance detection with hierarchical memory.
It combines tool-use planning, visual grounding, segmentation, and memory-augmented reasoning.

## 📰 News

* **[TBD]** Paper and project page links will be released.
* **[TBD]** Public code release.

## Contents

* [Installation](#installation)
* [Run on Custom Images](#run-on-custom-images)
* [Run on Benchmark Datasets](#run-on-benchmark-datasets)
* [Project Structure](#project-structure)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

## Installation

### 1. Create environment

```bash
conda create -n a-harness python=3.11
conda activate a-harness
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Use `.env` (recommended):

```bash
cp .env.example .env
# Fill API_BASE_URL and API_KEY in .env
```

Optional fallback:

```bash
cp config.example.py config.py
```

## Run on Custom Images

```bash
python demo/run.py \
  --image_path /path/to/image.jpg \
  --task "What part should be pressed?"
```

You can also check command templates:

```bash
bash scripts/commands.sh
```

## Run on Benchmark Datasets

### ReasonAff

```bash
python demo/evaluate_reasonaff.py \
  --dataset_path dataset/reasonaff/test \
  --output_dir output/eval_reasonaff
```

### UMD

```bash
python demo/evaluate_umd.py \
  --dataset_path dataset/UMD_preprocessed \
  --output_dir output/eval_umd
```

## Project Structure

```text
A-Harness/
├── agent.py
├── demo/
│   ├── run.py
│   ├── evaluate_reasonaff.py
│   └── evaluate_umd.py
├── .env.example
├── config.example.py
├── requirements.txt
├── api/
├── prompts/
├── skills/
├── memory/
├── dataset/
└── scripts/
```

## Acknowledgement

* [RAGNet](https://github.com/wudongming97/AffordanceNet)
* [Affordance-R1](https://github.com/hq-King/Affordance-R1)

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{tbd,
  title={TBD},
  author={TBD},
  journal={TBD},
  year={TBD}
}
```
