# RACE-Fin Model Card

## Fixed Experimental Settings

| Setting | Value |
|---|---|
| Temperature | 0.0 (deterministic extraction) |
| Prompt version | v1.0 (frozen, versioned in each agent module) |
| Output format | JSON validated against Pydantic schemas |
| Confidence — open models | Token log-probabilities via `logprobs=True` |
| Confidence — API models | 5 sampled runs at temperature 0.7 (self-consistency) |
| Quantisation | 4-bit Q4_K_M GGUF |
| Defect conditions (full) | 16 (2⁴ full factorial) |
| Defect conditions (pilot) | 8 (clean + 4 single + 3 strongest pairs) |

## Models

| Role | Model | Provider | Parameters | Access |
|---|---|---|---|---|
| Risk Extractor (open) | Qwen2.5-7B-Instruct | Ollama | ~7B (4-bit ≈ 4.5 GB) | Free, local |
| Risk Extractor (API) | Claude 3.5 Haiku | Anthropic | Proprietary | API key |
| Critic | Llama-3.1-8B-Instruct | Ollama | ~8B (4-bit ≈ 5 GB) | Free, local |
| Evidence Agent | — | Python rules | — | No LLM |
| Aggregator | — | Python code | — | No LLM |

## Why These Models

- **Qwen and Haiku**: one open, one API — shows findings aren't tied to one vendor.
- **Llama for critic**: different family from the extractor so it catches different mistakes.
- **3 models total**: keeps comparison clean; "models are interchangeable by design."

## Hardware Requirements

- 7–8B at 4-bit: fits a free Colab / Kaggle T4 (16 GB VRAM).
- 14B at 4-bit (~9 GB): also fits a T4. Use only if 7B is too weak.
- Serve via Ollama for simplicity, or vLLM if batch speed matters.

## Prompt Versioning

All prompts are frozen in source code with `PROMPT_VERSION = "v1.0"`.
Changes require a version bump and re-run of affected experiments.
