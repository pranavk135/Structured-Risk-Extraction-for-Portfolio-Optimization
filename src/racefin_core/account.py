"""User profile + simulated (paper) trading account, persisted to data/user/*.json (git-ignored)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import ROOT

USER_DIR = ROOT / "data" / "user"
PROFILE_JSON, ACCOUNT_JSON = USER_DIR / "profile.json", USER_DIR / "account.json"
GOALS = ["Preserve capital", "Steady growth", "Maximise growth"]
LEVELS = ["Beginner", "Intermediate", "Experienced"]


@dataclass
class Profile:
    name: str = ""
    age: int = 25
    monthly_income: float = 50_000.0
    capital: float = 250_000.0
    horizon_years: int = 5
    risk_tolerance: int = 3  # 1..5
    goal: str = GOALS[1]
    experience: str = LEVELS[0]

    def bucket(self) -> str:
        lvl = 0 if self.risk_tolerance <= 2 else (1 if self.risk_tolerance == 3 else 2)
        if self.horizon_years < 3 or self.age >= 60:
            lvl -= 1
        if self.goal == GOALS[0]:
            lvl -= 1
        elif self.goal == GOALS[2]:
            lvl += 1
        return ["Low risk", "Balanced", "High return"][max(0, min(2, lvl))]


def _load(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save(path, obj):
    USER_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_profile() -> Profile:
    d = _load(PROFILE_JSON, {})
    return Profile(**{k: v for k, v in d.items() if k in Profile.__dataclass_fields__})


def save_profile(p: Profile):
    _save(PROFILE_JSON, asdict(p))


def new_account(cash: float = 0.0) -> dict:
    return {"cash": cash, "positions": {}, "trades": [], "created": datetime.now().isoformat(timespec="seconds")}


def load_account() -> dict:
    return _load(ACCOUNT_JSON, new_account())


def save_account(a: dict):
    _save(ACCOUNT_JSON, a)


def execute_plan(acct: dict, weights: dict, prices: dict, capital: float, algo: str) -> dict:
    """Rebalance paper account towards `weights` using `capital` of available cash. Fractional shares."""
    capital = min(capital, acct["cash"])
    for t, w in weights.items():
        amt = capital * w
        px = float(prices[t])
        sh = amt / px
        pos = acct["positions"].setdefault(t, {"shares": 0.0, "avg": 0.0})
        tot = pos["shares"] + sh
        pos["avg"] = (pos["shares"] * pos["avg"] + sh * px) / tot if tot else 0.0
        pos["shares"] = tot
        acct["trades"].append({"time": datetime.now().isoformat(timespec="seconds"), "side": "BUY", "ticker": t,
                               "shares": round(sh, 4), "price": round(px, 2), "algo": algo})
    acct["cash"] -= capital
    return acct
