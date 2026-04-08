"""
Configuration file for A-Harness.

Copy this file to config.py and fill in your API credentials:
    cp config.example.py config.py

All values can also be overridden via environment variables.
Priority: CLI args > environment variables > config.py
"""

# ── LLM API (OpenAI-compatible) ──────────────────────────────────────────
API_KEY = ""                          # e.g. "sk-xxxx"
API_BASE_URL = ""                     # e.g. "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"             # Decision model

# ── Qwen3.5 PAI-EAS (optional, auto-detected by model name) ─────────────
QWEN35_API_BASE_URL = ""
QWEN35_API_KEY = ""
QWEN35_MODEL_NAME = "Qwen3.5-397B-A17B"

# ── Google Search API (Serper, optional) ─────────────────────────────────
SERPER_API_KEY = ""                   # https://serper.dev (free tier: 2500/month)

# ── Local Model Paths (optional) ─────────────────────────────────────────
# Set these to local directories if you have pre-downloaded model weights.
# Otherwise leave as None to download from HuggingFace/ModelScope.
LOCAL_MODELS_DIR = "models"
HF_HOME = None                        # e.g. "/path/to/hf_cache"
REX_OMNI_MODEL_PATH = None            # e.g. "/path/to/IDEA-Research_Rex-Omni"
SAM2_MODEL_PATH = None                # e.g. "/path/to/facebook_sam2.1-hiera-large"
SAM3_MODEL_PATH = "facebook/sam3"     # Supports: HF repo id, "modelscope::repo", or local path

# ── Output ───────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "output"
