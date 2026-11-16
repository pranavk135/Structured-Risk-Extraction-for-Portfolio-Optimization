"""RACE-Fin configuration: model names, API settings, experimental constants."""
import os
from pathlib import Path

# ── Model identifiers (change these strings to switch models) ──────────────
EXTRACTOR_MODEL_A = "ollama/qwen2.5:7b-instruct"       # Alibaba — Extractor #1
EXTRACTOR_MODEL_B = "ollama/gemma2:9b"                  # Google  — Extractor #2
CRITIC_MODEL      = "ollama/llama3.1:8b"                # Meta   — Critic

# ── Ollama server ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── API keys (set if using an API model — not needed for Ollama-only) ──
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Fixed experimental settings (report all of these) ─────────────────────
TEMPERATURE = 0.0
PROMPT_VERSION = "v1.0"
SEED = 42
SELF_CONSISTENCY_RUNS = 5  # for API models without logprobs
SELF_CONSISTENCY_TEMP = 0.7  # sampling temperature for self-consistency

# ── Defect conditions ─────────────────────────────────────────────────────
DEFECT_NAMES = ["S_staleness", "C_conflict", "M_missingness", "A_low_authority"]
NUM_CONDITIONS = 16   # 2^4 full factorial
PILOT_CONDITIONS = 8  # main effects only

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
