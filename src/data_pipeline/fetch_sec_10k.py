#!/usr/bin/env python3
"""
SEC EDGAR 10-K + XBRL Dataset Fetcher for RACE-Fin (PW26_PSJ_05).

Downloads official 10-K filings (primary HTML), full XBRL companyfacts JSON,
extracts clean plain text + Item 1A (Risk Factors) and Item 7 (MD&A),
builds normalized XBRL key-facts CSV, and logs full provenance in filings_manifest.csv.

Includes a disk-space safety guard (--min-free-gb) because fetching all 10,405
SEC issuers across all years requires hundreds of GBs.
"""

import argparse
import csv
import json
import os
import re
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
from bs4 import BeautifulSoup

USER_AGENT = "PESUniversityCapstone RACEFin/1.0 (pw26_psj_05@pes.edu)"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

DEFAULT_COHORT = {
    # Tech & Cyber
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "ORCL": "Technology",
    # Industrials & Supply Chain
    "CAT": "Industrials/Supply-Chain",
    "HON": "Industrials/Supply-Chain",
    "BA": "Aerospace/Supply-Chain",
    "F": "Automotive/Supply-Chain",
    # Energy, Utilities & Materials (Environmental / Climate)
    "XOM": "Energy",
    "CVX": "Energy",
    "NEE": "Utilities/Clean-Energy",
    "FCX": "Materials/Mining",
    # Financials & Payments (Liquidity / Credit)
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "V": "Financials/Payments",
    # Healthcare & Retail (Regulatory / Operational)
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "WMT": "Retail/Operational",
    "AMZN": "Retail/Cloud",
}

XBRL_TAGS = {
    "Revenues": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "NetIncomeLoss": ["NetIncomeLoss", "ProfitLoss"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "StockholdersEquity": ["StockholdersEquity"],
    "LongTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "CashAndEquivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def sec_get(url: str, delay: float = 0.18) -> bytes:
    time.sleep(delay)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
        return resp.read()


def load_ticker_map(out_dir: Path) -> dict:
    map_path = out_dir / "company_tickers.json"
    if not map_path.exists():
        raw = sec_get("https://www.sec.gov/files/company_tickers.json")
        map_path.write_bytes(raw)
    data = json.loads(map_path.read_text(encoding="utf-8"))
    by_ticker = {}
    for entry in data.values():
        t = str(entry["ticker"]).upper().strip()
        by_ticker[t] = {
            "cik": str(entry["cik_str"]).zfill(10),
            "cik_int": int(entry["cik_str"]),
            "title": entry["title"],
        }
    return by_ticker


def html_to_clean_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_item_section(full_text: str, start_pat: str, end_pats: list[str], max_chars: int = 250_000) -> str:
    """Locate real body section (picking the longest reasonable match after TOC)."""
    matches = list(re.finditer(start_pat, full_text, flags=re.IGNORECASE | re.MULTILINE))
    if not matches:
        return ""
    # Skip short table-of-contents hits by picking match with substantial content before next item
    best = ""
    for m in matches:
        start_idx = m.start()
        sub = full_text[start_idx : start_idx + max_chars]
        end_idx = len(sub)
        for ep in end_pats:
            em = re.search(ep, sub[200:], flags=re.IGNORECASE | re.MULTILINE)
            if em:
                end_idx = min(end_idx, 200 + em.start())
        candidate = sub[:end_idx].strip()
        if len(candidate) > len(best):
            best = candidate
    # Avoid returning tiny TOC snippet (< 500 chars) if no body found
    return best if len(best) >= 500 else ""


def extract_xbrl_key_facts(ticker: str, cik: str, cf_json: dict) -> list[dict]:
    rows = []
    us_gaap = cf_json.get("facts", {}).get("us-gaap", {})
    for metric, candidates in XBRL_TAGS.items():
        for tag in candidates:
            if tag not in us_gaap:
                continue
            units = us_gaap[tag].get("units", {})
            unit_key = "USD" if "USD" in units else (next(iter(units), None))
            if not unit_key:
                continue
            for item in units[unit_key]:
                if item.get("form") != "10-K":
                    continue
                fy = item.get("fy")
                if not fy or int(fy) < 2021:
                    continue
                rows.append({
                    "ticker": ticker,
                    "cik": cik,
                    "metric": metric,
                    "xbrl_tag": tag,
                    "unit": unit_key,
                    "fy": fy,
                    "fp": item.get("fp", ""),
                    "start": item.get("start", ""),
                    "end": item.get("end", ""),
                    "filed": item.get("filed", ""),
                    "val": item.get("val", ""),
                    "accn": item.get("accn", ""),
                })
            break  # use highest-priority matching tag available
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch SEC 10-K filings and XBRL facts from EDGAR.")
    parser.add_argument("--out-dir", default="/Users/thedemigod/Documents/Capstone/datasets/sec_10k")
    parser.add_argument("--years", type=int, default=3, help="Number of most recent 10-K filings per company")
    parser.add_argument("--tickers", nargs="*", help="Custom tickers, or 'SP500' / 'ALL'")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="Stop downloading if free disk drops below this")
    parser.add_argument("--skip-html", action="store_true", help="Delete raw HTML after text extraction to save space")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ("raw_html", "extracted_text", "extracted_sections", "xbrl_json"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    ticker_map = load_ticker_map(out_dir)

    if not args.tickers:
        targets = list(DEFAULT_COHORT.items())
    elif args.tickers == ["ALL"]:
        targets = [(t, "SEC-All") for t in sorted(ticker_map.keys())]
    elif args.tickers == ["SP500"]:
        sp_csv = Path("/Users/thedemigod/Documents/Capstone/Demo-related-to-Capstone/RACE-Fin_Review2_portfolio_ver/RACE-Fin_Review_Demo/data/SP500Data.csv")
        header = sp_csv.read_text(encoding="utf-8").splitlines()[0].split(",")[1:]
        targets = [(t.strip().upper(), "S&P500") for t in header if t.strip().upper() in ticker_map]
    else:
        targets = [(t.upper(), DEFAULT_COHORT.get(t.upper(), "Custom")) for t in args.tickers]

    manifest_rows = []
    xbrl_rows = []
    downloaded_filings = 0

    for idx, (ticker, sector) in enumerate(targets, 1):
        avail = free_gb(out_dir)
        if avail < args.min_free_gb:
            print(f"[STOP] Free disk space {avail:.2f} GiB < {args.min_free_gb:.2f} GiB safety floor. Halting cleanly.")
            break

        info = ticker_map.get(ticker)
        if not info:
            print(f"[SKIP] {ticker}: not found in SEC company_tickers.json")
            continue

        cik = info["cik"]
        cik_int = info["cik_int"]
        print(f"[{idx}/{len(targets)}] {ticker} (CIK {cik}, {info['title']}) | free={avail:.2f} GiB")

        # 1. Submissions metadata
        try:
            sub_bytes = sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
            sub_json = json.loads(sub_bytes.decode("utf-8"))
        except Exception as e:
            print(f"  [ERR] submissions failed for {ticker}: {e}")
            continue

        # 2. XBRL companyfacts
        cf_path = out_dir / "xbrl_json" / f"{ticker}_companyfacts.json"
        if not cf_path.exists():
            try:
                cf_bytes = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
                cf_path.write_bytes(cf_bytes)
            except Exception as e:
                print(f"  [WARN] companyfacts unavailable for {ticker}: {e}")
                cf_bytes = None
        else:
            cf_bytes = cf_path.read_bytes()

        if cf_bytes:
            try:
                xbrl_rows.extend(extract_xbrl_key_facts(ticker, cik, json.loads(cf_bytes.decode("utf-8"))))
            except Exception as e:
                print(f"  [WARN] XBRL parse failed for {ticker}: {e}")

        # 3. Latest N 10-K filings
        recent = sub_json.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accs = recent.get("accessionNumber", [])
        f_dates = recent.get("filingDate", [])
        r_dates = recent.get("reportDate", [])
        docs = recent.get("primaryDocument", [])

        picked = 0
        for form, acc, fdate, rdate, doc in zip(forms, accs, f_dates, r_dates, docs):
            if form != "10-K":
                continue
            if picked >= args.years:
                break
            if not doc or not doc.lower().endswith((".htm", ".html")):
                continue

            acc_nodash = acc.replace("-", "")
            edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
            fy_tag = (rdate or fdate)[:4]
            stem = f"{ticker}_{fy_tag}_{acc_nodash}"
            html_path = out_dir / "raw_html" / f"{stem}.htm"
            txt_path = out_dir / "extracted_text" / f"{stem}.txt"
            i1a_path = out_dir / "extracted_sections" / f"{stem}_item1a.txt"
            i7_path = out_dir / "extracted_sections" / f"{stem}_item7.txt"

            if not txt_path.exists():
                try:
                    if not html_path.exists():
                        raw_html = sec_get(edgar_url)
                        html_path.write_bytes(raw_html)
                    else:
                        raw_html = html_path.read_bytes()
                    clean_txt = html_to_clean_text(raw_html)
                    txt_path.write_text(clean_txt, encoding="utf-8")
                    if args.skip_html and html_path.exists():
                        html_path.unlink()
                except Exception as e:
                    print(f"  [ERR] 10-K {stem} failed: {e}")
                    continue
            else:
                clean_txt = txt_path.read_text(encoding="utf-8")

            if not i1a_path.exists():
                i1a = extract_item_section(
                    clean_txt,
                    r"(?:^|\n)\s*Item\s+1A\.?\s*[-–—:]?\s*Risk\s+Factors",
                    [r"\n\s*Item\s+1B\b", r"\n\s*Item\s+1C\b", r"\n\s*Item\s+2\b"],
                )
                i1a_path.write_text(i1a, encoding="utf-8")
            else:
                i1a = i1a_path.read_text(encoding="utf-8")

            if not i7_path.exists():
                i7 = extract_item_section(
                    clean_txt,
                    r"(?:^|\n)\s*Item\s+7\.?\s*[-–—:]?\s*Management",
                    [r"\n\s*Item\s+7A\b", r"\n\s*Item\s+8\b"],
                )
                i7_path.write_text(i7, encoding="utf-8")
            else:
                i7 = i7_path.read_text(encoding="utf-8")

            manifest_rows.append({
                "ticker": ticker,
                "cik": cik,
                "company_name": info["title"],
                "sector": sector,
                "form": form,
                "filing_date": fdate,
                "report_date": rdate,
                "accession_number": acc,
                "edgar_url": edgar_url,
                "raw_html_file": str(html_path.relative_to(out_dir)) if html_path.exists() else "",
                "text_file": str(txt_path.relative_to(out_dir)),
                "item1a_file": str(i1a_path.relative_to(out_dir)),
                "item7_file": str(i7_path.relative_to(out_dir)),
                "full_text_chars": len(clean_txt),
                "item1a_chars": len(i1a),
                "item7_chars": len(i7),
            })
            picked += 1
            downloaded_filings += 1
            print(f"  + {stem} (filed {fdate}): full={len(clean_txt):,} chars | Item1A={len(i1a):,} | Item7={len(i7):,}")

    # Write manifest & XBRL CSVs
    manifest_csv = out_dir / "filings_manifest.csv"
    if manifest_rows:
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    xbrl_csv = out_dir / "xbrl_key_facts.csv"
    if xbrl_rows:
        with open(xbrl_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(xbrl_rows[0].keys()))
            writer.writeheader()
            writer.writerows(xbrl_rows)

    print(f"\nDONE: {downloaded_filings} 10-K filings | {len(xbrl_rows)} XBRL fact rows | free disk={free_gb(out_dir):.2f} GiB")


if __name__ == "__main__":
    main()
