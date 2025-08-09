# 03_event_schema_and_logging.py

import json
import math
import glob
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw_events"
DERIVED = DATA / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

# time helper


def now_iso():
    # utc timestamp in iso-8601 with Z suffix
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# schema notes
# minimal common fields: ts, event_id, kind, run_id
# optional fields: span_id, parent_span_id, name, attrs, status, error, duration_s, step, action, ok, attempts, data, task, reason

# loader utilities


def read_jsonl(path):
    # read a jsonl file into a list of dicts
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines but keep going
                continue
    return out


def load_all_raw(glob_pattern="*.jsonl"):
    # load all jsonl files under RAW matching a pattern
    paths = sorted(RAW.glob(glob_pattern))
    records = []
    for p in paths:
        records.extend(read_jsonl(p))
    return records


def to_dataframe(records):
    # convert list of dicts to a normalized dataframe
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # ensure expected columns exist
    cols = [
        "ts", "event_id", "kind", "run_id", "span_id", "parent_span_id", "name", "attrs",
        "status", "error", "duration_s", "step", "action", "ok", "attempts", "data", "task", "reason"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # coerce types
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce", downcast="integer")
    df["attempts"] = pd.to_numeric(
        df["attempts"], errors="coerce", downcast="integer")
    df["ok"] = df["ok"].astype("boolean")

    # sort for stable downstream operations
    df = df.sort_values(
        ["ts", "run_id"], kind="mergesort").reset_index(drop=True)

    # derive helpful columns
    df["date"] = df["ts"].dt.date
    df["minute"] = df["ts"].dt.floor("min")
    df["hour"] = df["ts"].dt.floor("h")

    # event class shortcuts
    df["is_span_start"] = df["kind"].eq("span_start")
    df["is_span_end"] = df["kind"].eq("span_end")
    df["is_run_start"] = df["kind"].eq("run_start")
    df["is_run_ok"] = df["kind"].eq("run_ok")
    df["is_run_fail"] = df["kind"].eq("run_fail")
    df["is_action_result"] = df["kind"].eq("action_result")
    df["is_retry"] = df["kind"].eq("retry")

    # numeric duration helpers
    df["duration_ms"] = (df["duration_s"] * 1000.0).round(3)

    #
    df["data_json"] = df["data"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(
        x, (dict, list)) else (None if pd.isna(x) else str(x)))
    df["attrs_json"] = df["attrs"].apply(lambda x: json.dumps(
        x, ensure_ascii=False) if isinstance(x, (dict, list)) else (None if pd.isna(x) else str(x)))

    return df


def persist(df, parquet_path=DERIVED / "events.parquet", csv_sample_path=DERIVED / "events_sample.csv", sample_n=5000):
    if df.empty:
        print("no events to persist")
        return
    cols_for_parquet = [c for c in df.columns if c not in [
        "data", "attrs"]]  # exclude raw object columns
    df_parquet = df[cols_for_parquet].copy()
    df_parquet.to_parquet(parquet_path, index=False)
    sample = df_parquet.sample(
        n=min(sample_n, len(df_parquet)), random_state=42)
    sample.to_csv(csv_sample_path, index=False)
    print("wrote:", parquet_path)
    print("wrote sample:", csv_sample_path)


# sanity checks and first metrics
def check_event_counts(df):
    # print basic distribution to verify ingestion
    by_kind = df["kind"].value_counts(dropna=False).sort_index()
    print("event counts by kind:")
    print(by_kind.to_string())


def basic_run_outcomes(df):
    # success and failure counts by run
    run_ok = df.loc[df["is_run_ok"], ["run_id"]].assign(ok=True)
    run_fail = df.loc[df["is_run_fail"], ["run_id"]].assign(ok=False)
    runs = pd.concat([run_ok, run_fail],
                     ignore_index=True).drop_duplicates("run_id")
    if runs.empty:
        print("no run summaries found")
        return runs
    summary = runs["ok"].value_counts(normalize=False).rename(
        index={True: "ok", False: "fail"})
    rate = runs["ok"].mean() if not runs.empty else float("nan")
    print("run outcome counts:")
    print(summary.to_string())
    print("success_rate:", round(rate, 3))
    return runs


def latency_overview(df):
    # look at span_end durations by name
    spans = df.loc[df["is_span_end"] & df["duration_s"].notna(), [
        "name", "duration_s"]].copy()
    if spans.empty:
        print("no span_end durations found")
        return
    stats = spans.groupby("name")["duration_s"].agg(
        ["count", "mean", "median", "max", "quantile"])
    # pandas quantile default needs a parameter, compute specific percentiles instead
    q = spans.groupby("name")["duration_s"].quantile(
        [0.5, 0.9, 0.95, 0.99]).unstack(level=-1)
    stats = spans.groupby("name")["duration_s"].agg(
        ["count", "mean", "median", "max"])
    stats["p90"] = q[0.9]
    stats["p95"] = q[0.95]
    stats["p99"] = q[0.99]
    print("span latency summary (seconds):")
    print(stats.round(4).to_string())


def retries_by_action(df):
    # distribution of attempts for action_result events
    ar = df.loc[df["is_action_result"], ["action", "attempts", "ok"]].dropna(subset=[
                                                                             "action"])
    if ar.empty:
        print("no action_result events")
        return
    pivot = ar.pivot_table(index="action", columns="attempts",
                           values="ok", aggfunc="count", fill_value=0)
    print("attempt count distribution by action:")
    print(pivot.to_string())


def runs_per_minute(df):
    # quick rate view
    starts = df.loc[df["is_run_start"], ["minute"]]
    if starts.empty:
        print("no run_start events")
        return
    rpm = starts.groupby("minute").size().rename("runs_started")
    print("runs per minute:")
    print(rpm.to_string())


if __name__ == "__main__":
    # load raw events from previous sections
    records = load_all_raw("*.jsonl")

    # convert to dataframe
    df = to_dataframe(records)

    # persist to disk for downstream sections
    persist(df)

    # quick sanity checks
    check_event_counts(df)
    basic_run_outcomes(df)
    latency_overview(df)
    retries_by_action(df)
    runs_per_minute(df)
