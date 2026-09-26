#!/usr/bin/env python3
"""Submit one RunPod Serverless job per filtered experiment_id.

Filters cells from the manifest CSV and writes a run manifest YAML
alongside the submission so re-runs can be tracked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import concurrent.futures
import threading

import requests

REPO = Path(__file__).resolve().parents[2]
RUNPOD_API_BASE = "https://api.runpod.ai/v2"


def _split_csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _filter_experiment_ids(args: argparse.Namespace) -> list[int]:
    """Read the ablation matrix CSV and filter rows by the run's axes.

    Returns the matching ``experiment_id`` column as a sorted list.
    """
    import csv
    csv_path = REPO / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"ablation matrix CSV not found: {csv_path}")

    want_datasets = set(_split_csv(args.datasets))
    want_targets = set(_split_csv(args.targets))
    want_imputers = set(_split_csv(args.imputers))
    want_classifiers = set(_split_csv(args.classifiers))
    want_regimes = set(_split_csv(args.regimes))
    want_rates = {int(r) for r in _split_csv(args.rates)}
    want_seeds = {int(s) for s in _split_csv(args.seeds)}

    eids: list[int] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if want_datasets and row.get("dataset") not in want_datasets:
                continue
            if want_targets and row.get("target") not in want_targets:
                continue
            if want_imputers and row.get("imputer") not in want_imputers:
                continue
            if want_classifiers and row.get("classifier") not in want_classifiers:
                continue
            if want_regimes and row.get("regime") not in want_regimes:
                continue
            try:
                if want_rates and int(row.get("rate", -1)) not in want_rates:
                    continue
                if want_seeds and int(row.get("seed", -1)) not in want_seeds:
                    continue
            except (TypeError, ValueError):
                continue
            try:
                eids.append(int(row["experiment_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(set(eids))


def _write_run_manifest(args: argparse.Namespace, n_cells: int, runs_dir: Path) -> Path:
    """Emit a minimal YAML record describing the run for later auditing."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runs_dir / f"{args.name}.yaml"
    lines = [
        f"name: {args.name}",
        f"description: {args.description or ''!r}",
        f"datasets: {args.datasets or 'D1,D2,D3-G1-2,D3-G3,D4'}",
        f"imputers: {args.imputers}",
        f"classifiers: {args.classifiers}",
        f"regimes: {args.regimes}",
        f"rates: {args.rates}",
        f"seeds: {args.seeds}",
        f"split: {args.split}",
        f"n_cells_total: {n_cells}",
        f"created_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    ]
    out_yaml.write_text("\n".join(lines) + "\n")
    return out_yaml


def _submit_one(endpoint: str, api_key: str, eid: int, run_name: str,
                csv: str, time_budget_sec: int,
                max_attempts: int = 5, retry_base_sleep: float = 1.0,
                train_rate: int | None = None,
                maskmlp_preset: str | None = None) -> str:
    url = f"{RUNPOD_API_BASE}/{endpoint}/run"
    body_input: dict = {
        "experiment_id": eid,
        "run_name": run_name,
        "csv": csv,
        "time_budget_sec": time_budget_sec,
    }
    if train_rate is not None:
        body_input["train_rate"] = int(train_rate)
    if maskmlp_preset is not None:
        body_input["maskmlp_preset"] = str(maskmlp_preset)
    body = {"input": body_input}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = requests.HTTPError(f"http {resp.status_code}")
                if not is_last:
                    time.sleep(retry_base_sleep * (2 ** attempt))
                continue
            resp.raise_for_status()
            body_doc = resp.json()
            if "id" not in body_doc:
                last_err = RuntimeError(f"malformed response (no 'id'): {body_doc!r}")
                if not is_last:
                    time.sleep(retry_base_sleep * (2 ** attempt))
                continue
            return body_doc["id"]
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if not is_last:
                time.sleep(retry_base_sleep * (2 ** attempt))
    raise RuntimeError(f"submit failed for experiment_id={eid} after "
                       f"{max_attempts} attempts: {last_err}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--csv", default="docs/ablation_matrix.csv")
    p.add_argument("--datasets", default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--imputers", required=True)
    p.add_argument("--classifiers", required=True)
    p.add_argument("--regimes", required=True)
    p.add_argument("--rates", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--split", default="student", choices=["student", "school"])
    p.add_argument("--logs-dir", default="logs/runpod")
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--time-budget-sec", type=int, default=5400)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--retry-base-sleep", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results-dir", default="results",
                   help="Where collect_results.py writes per-cell JSONs. The submitter "
                        "skips experiment_ids that already have a complete cell file here, "
                        "so re-running submit on the same --name only POSTs missing cells.")
    p.add_argument("--train-rate", type=int, default=None,
                   help="Optional override: missingness rate for the *training* mask, "
                        "decoupled from --rates (= test rate). Default None means "
                        "symmetric (use --rates for both train and test). Set to 0 for "
                        "'train-on-clean, test-with-missingness' sweeps.")
    p.add_argument("--maskmlp-preset", choices=["paper", "notebook"], default=None,
                   help="MaskMLP training preset. 'paper' = wrapper recipe defaults, "
                        "'notebook' = alternative high-epoch recipe. Default None "
                        "leaves the cell-level default ('paper') in place.")
    args = p.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not args.dry_run and not (api_key and endpoint):
        print("error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set", file=sys.stderr)
        return 2

    eids = _filter_experiment_ids(args)
    print(f"[submit_runpod] filtered to {len(eids)} cells")
    if not eids:
        return 0

    runs_dir = Path(args.runs_dir)
    _write_run_manifest(args, len(eids), runs_dir)

    # Skip experiment_ids that already have a complete cell file on disk.
    out_dir = Path(args.results_dir) / args.name
    if out_dir.exists():
        done = set()
        for p in out_dir.rglob("*.json"):
            name = p.name
            if not (name.startswith("cell_") or name.startswith("seed")):
                continue
            try:
                doc = json.loads(p.read_text())
            except (json.JSONDecodeError, ValueError, OSError):
                continue
            if isinstance(doc, dict) and doc.get("status") is not None:
                try:
                    done.add(int(doc["experiment_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
        before = len(eids)
        eids = [e for e in eids if e not in done]
        skipped = before - len(eids)
        if skipped:
            print(f"[submit_runpod] skipping {skipped} cells already completed; {len(eids)} to submit")
        if not eids:
            print("[submit_runpod] nothing to do — all cells already on disk")
            return 0

    logs_dir = Path(args.logs_dir) / args.name
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = logs_dir / "jobs.jsonl"

    if args.dry_run:
        print(f"[submit_runpod] dry-run: would submit {len(eids)} cells")
        return 0

    write_lock = threading.Lock()
    n_done = 0

    def _submit_and_record(eid: int) -> None:
        nonlocal n_done
        job_id = _submit_one(endpoint, api_key, eid, args.name,
                             args.csv, args.time_budget_sec,
                             retry_base_sleep=args.retry_base_sleep,
                             train_rate=args.train_rate,
                             maskmlp_preset=args.maskmlp_preset)
        with write_lock, jobs_path.open("a") as f:
            f.write(json.dumps({
                "job_id": job_id,
                "experiment_id": eid,
                "submitted_at": time.time(),
            }) + "\n")
            n_done += 1
            if n_done % 100 == 0:
                print(f"[submit_runpod] submitted {n_done}/{len(eids)}")

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(_submit_and_record, e): e for e in eids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors.append((futures[fut], e))

    print(f"[submit_runpod] done: {n_done}/{len(eids)} submitted, "
          f"{len(errors)} failures, jobs at {jobs_path}")
    if errors:
        for eid, e in errors[:5]:
            print(f"  experiment_id={eid}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
