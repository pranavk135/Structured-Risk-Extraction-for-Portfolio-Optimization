"""Risk Extractor Agent — LLM-powered structured risk extraction.

Uses litellm for model-agnostic inference.
Supports: Ollama (Qwen2.5-7B) and Anthropic API (Claude Haiku 3.5).

Settings to report: temperature 0, fixed prompt v1.0, JSON schema output.
"""
from __future__ import annotations

import json
import re
from collections import Counter

import litellm

litellm.drop_params = True

from config import TEMPERATURE, OLLAMA_BASE_URL, SELF_CONSISTENCY_RUNS, SELF_CONSISTENCY_TEMP
from schemas import RiskExtraction, RiskItem, Severity

PROMPT_VERSION = "v1.0"

# ── System prompt (frozen — bump PROMPT_VERSION if you edit this) ─────────

SYSTEM_PROMPT = """\
You are a financial risk extraction agent. Given a disclosure passage and its
evidence context, extract structured risk information.

Output a JSON object with exactly this schema:
{
    "risks": [
        {
            "risk_type": "<Supply-chain | Environmental | Liquidity | Financial | Cybersecurity | Operational>",
            "severity": "<low | medium | high | critical>",
            "claim": "<the specific risk claim, quoted from the passage>",
            "supporting_evidence_ids": ["<record_id>"],
            "confidence": <0.0 to 1.0>,
            "reasoning": "<brief explanation>"
        }
    ],
    "raw_confidence": <0.0 to 1.0>
}

Severity guide:
  low      — minor operational issue
  medium   — could affect quarterly results
  high     — material to annual results
  critical — threatens going concern

Confidence guide:
  1.0 — clear, well-supported claim from a current audited source
  0.7 — reasonable claim but some evidence gaps
  0.5 — ambiguous or partially supported
  0.3 — speculative or from a weak source
  0.0 — no supporting evidence

If the evidence is stale, conflicting, or from a low-authority source,
lower your confidence accordingly.

Output ONLY valid JSON. No markdown fences, no commentary outside the JSON.
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Best-effort JSON parse with fallback regex extraction."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"risks": [], "raw_confidence": 0.0}


def _to_severity(val: str) -> Severity:
    """Coerce a string into the Severity enum, defaulting to MEDIUM."""
    try:
        return Severity(val.lower())
    except (ValueError, AttributeError):
        return Severity.MEDIUM


# ── Public API ─────────────────────────────────────────────────────────────

def extract(
    record_id: str,
    passage: str,
    evidence_context: str,
    model: str,
    source_type: str = "",
    temperature: float = TEMPERATURE,
    logprobs: bool = False,
) -> RiskExtraction:
    """Extract risks from a single disclosure record.

    Args:
        record_id:        Unique record identifier.
        passage:          The disclosure claim text.
        evidence_context: Supporting context / provenance description.
        model:            LiteLLM model string (e.g. "ollama/qwen2.5:7b-instruct-q4_K_M").
        source_type:      Source label (e.g. "Annual report").
        temperature:      Sampling temperature (0 for deterministic).
        logprobs:         Request token log-probabilities (open models only).

    Returns:
        A validated RiskExtraction object.
    """
    user_prompt = (
        f'Disclosure passage:\n"{passage}"\n\n'
        f'Evidence context:\n"{evidence_context}"\n\n'
        f"Source type: {source_type}\n"
        f"Record ID: {record_id}\n\n"
        f"Extract all risks as JSON."
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
    elif logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    parsed = _parse_json(content)

    # Extract token log-probs if the model returned them
    token_logprobs = None
    choice = response.choices[0]
    if hasattr(choice, "logprobs") and choice.logprobs:
        lp_content = getattr(choice.logprobs, "content", None)
        if lp_content:
            token_logprobs = [lp.logprob for lp in lp_content]

    # Build validated risk items
    risks: list[RiskItem] = []
    for r in parsed.get("risks", []):
        risks.append(
            RiskItem(
                risk_type=r.get("risk_type", "Unknown"),
                severity=_to_severity(r.get("severity", "medium")),
                claim=r.get("claim", ""),
                supporting_evidence_ids=r.get("supporting_evidence_ids", [record_id]),
                confidence=float(r.get("confidence", 0.5)),
                reasoning=r.get("reasoning", ""),
            )
        )

    return RiskExtraction(
        record_id=record_id,
        model_used=model,
        risks=risks,
        raw_confidence=float(parsed.get("raw_confidence", 0.5)),
        logprobs=token_logprobs,
    )


def extract_self_consistency(
    record_id: str,
    passage: str,
    evidence_context: str,
    model: str,
    source_type: str = "",
    n_runs: int = SELF_CONSISTENCY_RUNS,
    temperature: float = SELF_CONSISTENCY_TEMP,
) -> tuple[RiskExtraction, float]:
    """Run multiple extractions and return (majority result, consistency score).

    Use for API models that don't expose log-probabilities.
    """
    results: list[RiskExtraction] = []
    for _ in range(n_runs):
        r = extract(
            record_id=record_id,
            passage=passage,
            evidence_context=evidence_context,
            model=model,
            source_type=source_type,
            temperature=temperature,
        )
        results.append(r)

    if not results:
        return RiskExtraction(record_id=record_id, model_used=model), 0.0

    # Signature = sorted (risk_type, severity) tuples
    sigs = [
        tuple(sorted((ri.risk_type, ri.severity.value) for ri in r.risks))
        for r in results
    ]
    majority_sig, count = Counter(sigs).most_common(1)[0]
    consistency = count / len(results)

    # Return the first result matching the majority
    for r, sig in zip(results, sigs):
        if sig == majority_sig:
            r.raw_confidence = consistency
            return r, consistency

    results[0].raw_confidence = consistency
    return results[0], consistency
