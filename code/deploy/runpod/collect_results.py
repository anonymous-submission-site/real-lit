#!/usr/bin/env python3
"""Poll RunPod /status for each submitted job and rsync cell JSONs locally.

Architecture: workers write the full cell JSON to the shared Network Volume
(/runpod-volume on workers, mounted at e.g. /workspace on a CPU "puller" pod).
This collector polls /status purely for terminal-state detection and rsyncs
results from the puller pod into a local results directory. The handler's
HTTP response carries no cell data — it never hits RunPod's ~10 MB /job-done
payload limit, regardless of dataset size or subgroup_metrics depth.

Idempotent: if results/<run_name>/ already contains a JSON for an experiment_id,
that job is skipped. Safe to Ctrl-C and re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}


def _existing_eids(out_dir: Path) -> set[int]:
    eids: set[int] = set()
    for p in out_dir.rglob("*.json"):
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(doc, dict) and "experiment_id" in doc:
            try:
                eids.add(int(doc["experiment_id"]))
            except (TypeError, ValueError):
                continue
    return eids


def _rsync_volume(rsync_source: str | None, rsh: str | None,
                  run_name: str, local_dir: Path) -> int:
    """Rsync the run's results from the volume puller pod to local.

    Returns the rsync subprocess returncode. 0 = success; logs but does not
    raise on non-zero (transient SSH issues should not stop the polling loop).
    """
    if not rsync_source:
        return 0
    src = f"{rsync_source.rstrip('/')}/{run_name}/"
    dst = str(local_dir) + "/"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az", "--partial"]
    if rsh:
        cmd += ["-e", rsh]
    cmd += [src, dst]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # Suppress rc=23 ("source dir doesn't exist yet") before any worker writes.
            err = result.stderr.strip()
            if result.returncode == 23 and "No such file or directory" in err:
                return result.returncode
            print(f"[rsync] returncode={result.returncode}: {err[:200]}",
                  file=sys.stderr)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("[rsync] timed out (>5 min); will retry next poll", file=sys.stderr)
        return -1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--logs-dir", default="logs/runpod")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--poll-interval", type=float, default=15.0)
    p.add_argument("--timeout-sec", type=float, default=24 * 3600)
    p.add_argument("--volume-rsync-source", default=os.environ.get("N26_VOLUME_RSYNC_SOURCE"),
                   help="rsync source for the network-volume puller pod, "
                        "e.g. 'root@213.173.111.105:/workspace/results'. Can also "
                        "be set via N26_VOLUME_RSYNC_SOURCE env var.")
    p.add_argument("--volume-rsync-rsh",
                   default=os.environ.get("N26_VOLUME_RSYNC_RSH",
                                          "ssh -o StrictHostKeyChecking=no"),
                   help="SSH command for rsync, e.g. "
                        "'ssh -p 25838 -i ~/.ssh/id_ed25519'. Can also be set via "
                        "N26_VOLUME_RSYNC_RSH env var.")
    args = p.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not (api_key and endpoint):
        print("error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set", file=sys.stderr)
        return 2

    jobs_path = Path(args.logs_dir) / args.name / "jobs.jsonl"
    if not jobs_path.exists():
        print(f"error: no jobs file at {jobs_path}", file=sys.stderr)
        return 1
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]

    out_dir = Path(args.results_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _rsync_volume(args.volume_rsync_source, args.volume_rsync_rsh,
                  args.name, out_dir)
    skip = _existing_eids(out_dir)
    pending = [j for j in jobs if int(j["experiment_id"]) not in skip]
    print(f"[collect_results] {len(pending)} pending / {len(jobs)} total "
          f"({len(jobs) - len(pending)} already on disk)")
    if not pending:
        return 0

    deadline = time.time() + args.timeout_sec
    headers = {"Authorization": f"Bearer {api_key}"}
    n_ok = 0
    n_fail = 0
    fail_log = jobs_path.parent / "failures.jsonl"

    while pending and time.time() < deadline:
        _rsync_volume(args.volume_rsync_source, args.volume_rsync_rsh,
                      args.name, out_dir)
        eids_on_disk = _existing_eids(out_dir)

        still: list[dict] = []
        for job in pending:
            eid = int(job["experiment_id"])
            url = f"{RUNPOD_API_BASE}/{endpoint}/status/{job['job_id']}"
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 404:
                    with fail_log.open("a") as fl:
                        fl.write(json.dumps({"job": job, "status": "ORPHAN_404",
                                             "url": url}) + "\n")
                    n_fail += 1
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                still.append(job)
                continue
            body = resp.json()
            status = body.get("status", "UNKNOWN")
            if status in TERMINAL_OK:
                # Trust volume rsync to deliver the JSON; retry next cycle if absent.
                if eid in eids_on_disk:
                    n_ok += 1
                else:
                    still.append(job)
            elif status in TERMINAL_FAIL:
                with fail_log.open("a") as fl:
                    fl.write(json.dumps({"job": job, "status": status,
                                         "body": body}) + "\n")
                n_fail += 1
            else:
                still.append(job)
        if still:
            print(f"[collect_results] {n_ok} ok, {n_fail} fail, {len(still)} pending — "
                  f"sleeping {args.poll_interval}s")
            time.sleep(args.poll_interval)
        pending = still

    _rsync_volume(args.volume_rsync_source, args.volume_rsync_rsh,
                  args.name, out_dir)
    print(f"[collect_results] done: ok={n_ok} fail={n_fail} pending={len(pending)} "
          f"out_dir={out_dir}")
    return 0 if not pending else 1


if __name__ == "__main__":
    sys.exit(main())
