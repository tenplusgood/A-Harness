"""
CLI entrypoint for the affordance agent.

The agent:
- calls an LLM via API_KEY (OpenAI-compatible chat/completions style)
- receives text + image inputs
- orchestrates skills in `skills/*` (rex_omni → sam2)
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import AffordanceAgent


def _load_config() -> Dict[str, Any]:
    """Load config.py if present, otherwise return env defaults."""
    cfg_path = PROJECT_ROOT / "config.py"
    config: Dict[str, Any] = {}
    if cfg_path.exists():
        spec = importlib.util.spec_from_file_location("aff_cfg", cfg_path)
        if spec and spec.loader:
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)  # type: ignore
            config = {
                "API_KEY": getattr(cfg, "API_KEY", None),
                "API_BASE_URL": getattr(cfg, "API_BASE_URL", None),
                "DEFAULT_MODEL": getattr(cfg, "DEFAULT_MODEL", None),
                "DEFAULT_OUTPUT_DIR": getattr(cfg, "DEFAULT_OUTPUT_DIR", None),
            }
    return config


def _resolve_arg(value: Any, *candidates) -> Any:
    """Return the first non-empty candidate."""
    for candidate in (value, *candidates):
        if candidate not in (None, ""):
            return candidate
    return None


def main():
    config = _load_config()

    parser = argparse.ArgumentParser(description="Run affordance agent with skills")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--task", required=True, help="Affordance task description")
    parser.add_argument("--object_name", default=None, help="Optional object name")
    parser.add_argument("--model", default=None, help="LLM model name (overrides config/env)")
    parser.add_argument("--api_key", default=None, help="API key (overrides config/env)")
    parser.add_argument("--api_url", default=None, help="API base URL (overrides config/env)")
    parser.add_argument("--output_dir", default=None, help="Output directory for logs/results")
    parser.add_argument("--no_save", action="store_true", help="Do not save conversation log")
    parser.add_argument("--disable_memory", action="store_true", help="Disable memory module")
    parser.add_argument("--memory_max_size", type=int, default=80, help="Memory window size")
    parser.add_argument("--memory_eviction_model", type=str, default=None, help="Model used by memory dedupe/eviction; default follows --model")
    parser.add_argument("--memory_retrieval_top_k", type=int, default=20, help="TopK memories to retrieve")
    parser.add_argument("--memory_dedupe_top_k", type=int, default=5, help="TopK candidates for enqueue dedupe")
    parser.add_argument("--memory_duplicate_threshold", type=float, default=0.6, help="Similarity threshold for duplicate replacement")
    parser.add_argument("--memory_eviction_strategy", type=str, default="model_decision", choices=["fifo", "similarity", "model_decision"], help="Memory eviction strategy")
    parser.add_argument("--clear_memory", action="store_true", help="Clear existing memory before running (start fresh)")
    parser.add_argument("--clear_experience", action="store_true", help="Also clear the experience pool when clearing memory")
    args = parser.parse_args()

    api_key = _resolve_arg(args.api_key, config.get("API_KEY"), os.getenv("API_KEY"))
    api_url = _resolve_arg(args.api_url, config.get("API_BASE_URL"), os.getenv("API_BASE_URL"))
    model = _resolve_arg(args.model, config.get("DEFAULT_MODEL"), os.getenv("DEFAULT_MODEL"), "gpt-4o")
    output_dir = _resolve_arg(args.output_dir, config.get("DEFAULT_OUTPUT_DIR"), "output")

    agent = AffordanceAgent(
        api_key=api_key,
        api_base_url=api_url,
        model_name=model,
        output_dir=output_dir,
        enable_memory=not args.disable_memory,
        memory_max_size=args.memory_max_size,
        memory_eviction_strategy=args.memory_eviction_strategy,
        memory_eviction_model=args.memory_eviction_model,
        memory_retrieval_top_k=args.memory_retrieval_top_k,
        memory_dedupe_top_k=args.memory_dedupe_top_k,
        memory_duplicate_threshold=args.memory_duplicate_threshold,
    )

    # （ --clear_memory）
    if args.clear_memory:
        print("🗑️  Clearing existing memory before running...")
        agent.clear_memory(clear_experience=args.clear_experience)

    print("=" * 60)
    print("Affordance Agent")
    print("=" * 60)
    print(f"Model      : {model}")
    print(f"Image path : {args.image_path}")
    print(f"Task       : {args.task}")
    if args.object_name:
        print(f"Object name: {args.object_name}")
    print("=" * 60)

    result = agent.detect_affordance(
        image_path=args.image_path,
        task=args.task,
        object_name=args.object_name,
        save_conversation=not args.no_save,
    )

    print("\n" + "=" * 60)
    if result.get("success"):
        print("✓ Detection completed")
        print("\nFinal Response:\n")
        print(result.get("final_response", ""))
        if "conversation_path" in result:
            print(f"\nConversation saved to: {result['conversation_path']}")
    else:
        print("✗ Detection failed")
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("=" * 60)

    # Always emit raw JSON for downstream pipelines
    print("\nRaw result JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

