# RACE-Fin | Evidence-Reliability Financial Intelligence Platform

Evidence-Reliability-Aware Financial Risk Extraction & Downstream Prioritisation under Compound Disclosure Defects.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-6%20Passing-34D399.svg)](tests/)

---

## ◈ System Architecture

```
Disclosures ──→ [Evidence Agent] ──→ [Risk Extractor] ──→ [Defect-Aware Critic] ──→ [Aggregator] ──→ Reliability-Aware
(SEC 10-K /       (Rules & Dates)      (Qwen / Gemma)        (Llama 3.1-8B)            (Code)           Risk Ranking
 Synthetic)                                                                                                  │
                                                                                                             ▼
                                                                                                  [Portfolio Optimiser]
                                                                                                     (HRP & PCA Allocation)
```

| Component | Technology / Model | Family / Method | Role |
|---|---|---|---|
| **Evidence Agent** | Deterministic Python Rules | Regex & Date Matching | Extracts source tier and flags temporal staleness |
| **Risk Extractor** | `qwen2.5:7b-instruct` or `gemma2:9b` | Alibaba Qwen / Google Gemma | Extracts structured claims, categories, severity, confidence |
| **Defect-Aware Critic** | `llama3.1:8b` | Meta Llama | Flags Staleness, Conflict, Missingness, Low Authority |
| **Aggregator** | Deterministic Math | Additive vs. Interaction-Aware | Computes non-linear reliability compounding and triage route |
| **Portfolio Engine** | HRP & PCA Eigen-Risk | Lopez de Prado (2016) | 3-Policy backtest proving decision impact on holdout data |

---

## ⚡ Quick Start

```bash
cd RACE-Fin_Disclosure_Demo_Expanded
source .venv/bin/activate
pip install -r requirements.txt

# Run the unified flagship platform
streamlit run app.py
```
App opens at **`http://localhost:8501`** in dark theme (`#0B0F14` / `#34D399` Emerald).

---

## 📂 Datasets Included

1. **S&P 500 Market Price Universe (`data/market/SP500Data.csv`)**:
   - 498 real US equities with 448 aligned daily price observations (2018–2019).
   - Used for portfolio covariance estimation, clustering, and holdout backtesting.
2. **Real SEC 10-K & XBRL Filings (`data/sec_10k/`)**:
   - 50+ actual SEC filings across AAPL, NVDA, MSFT, AMZN, BA, CAT, CVX, JNJ, JPM, etc.
   - Cleanly extracted Item 1A (Risk Factors) and Item 7 (MD&A) text sections.
   - 2,000+ structured XBRL financial facts (`xbrl_key_facts.csv`) for numerical reconciliation.
3. **Controlled Synthetic Defect Benchmark (`data/racefin_demo_disclosures.csv`)**:
   - 30 standardized scenarios across 5 companies with known ground-truth $S \times C \times M \times A$ flags.

---

## 🧪 Running Experiments

Run the full factorial (16 conditions) or pilot (8 conditions) evaluation suite:

```bash
# Pilot: 8 conditions × 30 records (open model)
python -m pipeline.experiment_runner

# Multi-Model Comparison: Qwen 2.5-7B + Gemma 2 9B
python -m pipeline.experiment_runner --both

# Ablation Study: Pipeline without Critic Agent
python -m pipeline.experiment_runner --no-critic

# Full Factorial Evaluation: 16 conditions (2⁴)
python -m pipeline.experiment_runner --full --both
```

Results are stored as JSONL in `data/results/` and visualized interactively in the **Calibration & Experiments** tab.

---

## 🤖 Running Live Local Models with Ollama

All models are **100% free and open-source**:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the three models (different families)
ollama pull qwen2.5:7b-instruct   # Extractor 1 (Alibaba)
ollama pull gemma2:9b              # Extractor 2 (Google)
ollama pull llama3.1:8b            # Critic (Meta)
```

In the Streamlit sidebar, flip the **"Enable Live LLMs"** toggle ON to run real-time inference.

---

## 🧪 Running Unit Tests

```bash
pytest tests/
```
All 6 test suites validate portfolio mathematical properties, defect compounding amplification, and evidence rules.
