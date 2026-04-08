#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   bash scripts/commands.sh

echo "A-Harness command reference"
echo

echo "1) Single-image demo"
echo 'python demo/run.py --image_path /path/to/image.jpg --task "What part should be pressed?"'
echo

echo "2) ReasonAff demo evaluation"
echo 'python demo/evaluate_reasonaff.py --dataset_path dataset/reasonaff/test --output_dir output/eval_reasonaff'
echo

echo "3) UMD demo evaluation"
echo 'python demo/evaluate_umd.py --dataset_path dataset/UMD_preprocessed --output_dir output/eval_umd'
echo

echo "4) Build commonsense template bank"
echo 'python -m memory.prepare_templates --datasets_root dataset --templates_dir memory/commonsense_templates'
