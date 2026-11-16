"""Defect-Aware Critic Agent — LLM-powered evidence quality assessment.

Uses a DIFFERENT model family from the Risk Extractor (Llama vs Qwen)
so the two don't share blind spots.

Settings to report: temperature 0, fixed prompt v1.0, JSON schema output.
"""
from __future__ import annotations

import json
import re

import litellm

from config import CRITIC_MODEL, TEMPERATURE, OLLAMA_BASE_URL
from schemas import CriticFlag

PROMPT_VERSION = "v1.0"

# ── System prompt (frozen — bump PROMPT_VERSION if you edit this) ─────────

SYSTEM_PROMPT = """\
You are a financial disclosure critic. Your job is to assess the quality and
reliability of evidence supporting a risk claim.

Check for FOUR specific defects:

1. Staleness (S)
   Is the evidence outdated? Look for old fiscal years, archived reports,
   or data that is not from the current reporting period (FY2025-26).

2. Conflict (C)
   Does the evidence contradict other sources? Look for different numbers
   for the same metric, conflicting statements, or unreconciled values.

3. Missingness (M)
   Is key evidence missing? Look for claims without supporting data,
   omitted tables, absent metrics, or incomplete disclosures.

4. Low Authority (A)
   Is the source unreliable? Look for blogs, social media, forums,
   anonymous posts, or unverified third-party sources.

Output a JSON object with exactly this schema:
{
    "staleness_flag": true/false,
    "staleness_reasoning": "<why or why not>",
    "conflict_flag": true/false,
    "conflict_reasoning": "<why or why not>",
    "missingness_flag": true/false,
    "missingness_reasoning": "<why or why not>",
    "low_authority_flag": true/false,
    "low_authority_reasoning": "<why or why not>",
    "overall_reliability": <0.0 to 1.0>
}

Rules:
- Be conservative: flag a defect only if there is clear evidence of it.
- overall_reliability starts at 1.0; subtract roughly 0.15–0.25 per defect.
- Output ONLY valid JSON. No markdown fences, no commentary outside the JSON.
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# ── Public API ─────────────────────────────────────────────────────────────

def critique(
    record_id: str,
    passage: str,
    evidence_context: str,
    source_type: str,
    extracted_risks: list[dict],
    model: str = CRITIC_MODEL,
    temperature: float = TEMPERATURE,
) -> CriticFlag:
    """Critique one record's evidence quality.

    Args:
        record_id:        Unique record identifier.
        passage:          The disclosure claim text.
        evidence_context: Supporting context / provenance description.
        source_type:      Source label.
        extracted_risks:  Risks produced by the extractor (passed as dicts).
        model:            LiteLLM model string for the critic.
        temperature:      Sampling temperature.

    Returns:
        A validated CriticFlag object.
    """
    risks_text = (
        json.dumps(extracted_risks, indent=2) if extracted_risks
        else "No risks were extracted by the extractor."
    )

    user_prompt = (
        f'Risk claim:\n"{passage}"\n\n'
        f'Evidence context:\n"{evidence_context}"\n\n'
        f"Source type: {source_type}\n\n"
        f"Extracted risks (from another model):\n{risks_text}\n\n"
        f"Assess the evidence quality. Output JSON only."
    )

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = OLLAMA_BASE_URL

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    parsed = _parse_json(content)

    return CriticFlag(
        record_id=record_id,
        model_used=model,
        staleness_flag=bool(parsed.get("staleness_flag", False)),
        staleness_reasoning=str(parsed.get("staleness_reasoning", "")),
        conflict_flag=bool(parsed.get("conflict_flag", False)),
        conflict_reasoning=str(parsed.get("conflict_reasoning", "")),
        missingness_flag=bool(parsed.get("missingness_flag", False)),
        missingness_reasoning=str(parsed.get("missingness_reasoning", "")),
        low_authority_flag=bool(parsed.get("low_authority_flag", False)),
        low_authority_reasoning=str(parsed.get("low_authority_reasoning", "")),
        overall_reliability=float(parsed.get("overall_reliability", 1.0)),
    )
