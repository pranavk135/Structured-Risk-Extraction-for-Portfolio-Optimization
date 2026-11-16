"""Experiment Runner: batch execution across models × conditions × items.

Manages the full experimental matrix, saves results as JSONL, and
provides a simple progress interface for Streamlit or CLI use.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

import litellm
litellm.drop_params = True

from config import (
    CRITIC_MODEL,
    EXTRACTOR_MODEL_A,
    EXTRACTOR_MODEL_B,
    RESULTS_DIR,
)
from pipeline.defect_injector import create_experiment_matrix
from pipeline.orchestrator import run_single
from schemas import ExperimentResult


ProgressCallback = Callable[[int, int], None] | None


def run_experiment(
    df: pd.DataFrame,
    extractor_models: list[str] | None = None,
    critic_model: str = CRITIC_MODEL,
    pilot: bool = True,
    use_critic: bool = True,
    save: bool = True,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Run the full experiment.

    Args:
        df:                Source dataset (clean records).
        extractor_models:  List of extractor model IDs to test.
        critic_model:      Critic model ID.
        pilot:             Use 8 pilot conditions (True) or full 16 (False).
        use_critic:        Include critic agent (True) or ablation (False).
        save:              Persist results to disk.
        progress_callback: Optional (current, total) callback.

    Returns:
        DataFrame of result rows (one per record × condition × model).
    """
    if extractor_models is None:
        extractor_models = [EXTRACTOR_MODEL_A, EXTRACTOR_MODEL_B]

    # Build the (record × condition) matrix
    matrix = create_experiment_matrix(df, pilot=pilot)
    total = len(matrix) * len(extractor_models)
    current = 0

    results: list[dict] = []

    for model in extractor_models:
        use_logprobs = False  # Ollama does not support logprobs via litellm

        for _, row in matrix.iterrows():
            row_dict = row.to_dict()
            current += 1

            try:
                output = run_single(
                    row_dict,
                    extractor_model=model,
                    critic_model=critic_model,
                    use_critic=use_critic,
                    logprobs=use_logprobs,
                )

                result = ExperimentResult(
                    record_id=row_dict["record_id"],
                    condition_id=int(row_dict.get("condition_id", 0)),
                    condition_label=str(row_dict.get("condition_label", "clean")),
                    extractor_model=model,
                    critic_model=critic_model if use_critic else "none",
                    extraction=output["extraction"],
                    critic_flags=output["critic_flags"],
                    aggregated=output["aggregated"],
                    ground_truth_label=row_dict.get("illustrative_reference_label"),
                )
                results.append(result.model_dump())

            except Exception:
                results.append({
                    "record_id": row_dict["record_id"],
                    "condition_id": int(row_dict.get("condition_id", 0)),
                    "condition_label": str(row_dict.get("condition_label", "clean")),
                    "extractor_model": model,
                    "error": traceback.format_exc(),
                })

            if progress_callback:
                progress_callback(current, total)

    results_df = pd.DataFrame(results)

    if save and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "pilot" if pilot else "full"
        critic_tag = "with_critic" if use_critic else "no_critic"
        filename = f"experiment_{mode_tag}_{critic_tag}_{timestamp}.jsonl"
        filepath = RESULTS_DIR / filename

        with open(filepath, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")

        print(f"✓ Results saved → {filepath}")

    return results_df


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = Path(__file__).parent.parent / "data" / "racefin_demo_disclosures.csv"
    df = pd.read_csv(data_path)

    pilot_flag = "--full" not in sys.argv
    no_critic = "--no-critic" in sys.argv

    models = [EXTRACTOR_MODEL_A]
    if "--both" in sys.argv:
        models.append(EXTRACTOR_MODEL_B)

    def _progress(cur: int, tot: int) -> None:
        print(f"\r  [{cur}/{tot}]", end="", flush=True)

    print(f"Running {'pilot' if pilot_flag else 'full'} experiment …")
    print(f"  Extractor models: {models}")
    print(f"  Critic: {'on' if not no_critic else 'off (ablation)'}")

    result_df = run_experiment(
        df,
        extractor_models=models,
        pilot=pilot_flag,
        use_critic=not no_critic,
        progress_callback=_progress,
    )

    print(f"\n✓ Done — {len(result_df)} result rows.")
