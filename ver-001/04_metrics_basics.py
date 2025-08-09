# 04_metrics_basics.py

import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# paths
BASE = Path(__file__).resolve().parent
DERIVED = (BASE / "data" / "derived")
DERIVED.mkdir(parents=True, exist_ok=True)
PARQUET = DERIVED / "events.parquet"


# load
def load_events():
    # reads the normalized parquet created in section 3
    if not PARQUET.exists():
        raise FileNotFoundError(
            "derived parquet not found. run 03_event_schema_and_logging.py first")
    df = pd.read_parquet(PARQUET)
    return df


# helpers
def save_txt(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for ln in lines:
                f.write(str(ln) + "\n")


def print_head(title, df, n=5):
    print(title)
    print(df.head(n).to_string())
    print("rows:", len(df))
    print("-" * 40)


def ensure_numeric_frame(df_like):
    # returns a numeric-only view, or None if empty
    if df_like is None or len(df_like) == 0:
        return None
    num = df_like.select_dtypes(include=[np.number])
    return num if not num.empty else None


# example 1: basic run rates
def metric_runs_per_minute(df):
    starts = df.loc[df["is_run_start"], ["minute"]]
    if starts.empty:
        return None
    rpm = starts.groupby("minute").size().rename("runs_started")
    fig = plt.figure()
    rpm.plot(kind="bar")
    plt.title("runs started per minute")
    plt.xlabel("minute")
    plt.ylabel("count")
    out = DERIVED / "metric_runs_per_minute.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return rpm


# example 2: overall outcome rates
def metric_outcomes(df):
    run_ok = df.loc[df["is_run_ok"], ["run_id"]].assign(ok=True)
    run_fail = df.loc[df["is_run_fail"], ["run_id"]].assign(ok=False)
    runs = pd.concat([run_ok, run_fail],
                     ignore_index=True).drop_duplicates("run_id")
    if runs.empty:
        return {"total": 0, "success_rate": np.nan}
    total = len(runs)
    success_rate = runs["ok"].mean()
    return {"total": total, "success_rate": round(success_rate, 3)}


# example 3: span latency distributions
def plot_latency_by_span(df):
    spans = df.loc[df["is_span_end"] & df["duration_s"].notna(), [
        "name", "duration_s"]].copy()
    if spans.empty:
        return None
    g = spans.groupby("name")["duration_s"]
    stats = g.agg(["count", "mean", "median", "max"]).round(4)
    q = g.quantile([0.9, 0.95, 0.99]).unstack(level=-1)
    stats["p90"] = q[0.9].round(4)
    stats["p95"] = q[0.95].round(4)
    stats["p99"] = q[0.99].round(4)

    fig = plt.figure()
    stats[["mean", "median", "p90", "p95", "p99"]].plot(kind="bar")
    plt.title("span latency summary (seconds)")
    plt.xlabel("span name")
    plt.ylabel("seconds")
    out = DERIVED / "latency_by_span.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return stats


# example 4: attempts distribution by action
def plot_retries_by_action(df):
    # collect only action_result rows with non-null attempts
    ar = df.loc[df["is_action_result"].fillna(False) & df["action"].notna(), [
        "action", "attempts"]].copy()
    if ar.empty:
        print("plot_retries_by_action: no action_result rows to plot")
        return None

    # force attempts to numeric and drop rows where it isn't
    ar["attempts"] = pd.to_numeric(ar["attempts"], errors="coerce")
    ar = ar.dropna(subset=["attempts"])
    if ar.empty:
        print("plot_retries_by_action: attempts are all NaN after coercion")
        return None

    # compute counts with groupby/size to guarantee numeric values
    counts = ar.groupby(["action", "attempts"]).size().unstack(
        fill_value=0).sort_index(axis=1)

    # double-check we actually have numeric data
    numeric = counts.select_dtypes(include=[np.number])
    if numeric.empty:
        print("plot_retries_by_action: no numeric data after grouping")
        return None

    fig = plt.figure()
    numeric.plot(kind="bar", stacked=True)
    plt.title("attempt counts by action")
    plt.xlabel("action")
    plt.ylabel("events")
    out = DERIVED / "retries_by_action.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return numeric


# example 5: error rate by action
def plot_error_rates_by_action(df):
    ar = df.loc[df["is_action_result"] &
                df["action"].notna(), ["action", "ok"]].copy()
    if ar.empty:
        return None
    err = ar.assign(err=lambda d: ~d["ok"].fillna(False))
    rate = err.groupby("action")["err"].mean().rename("error_rate").round(3)
    fig = plt.figure()
    rate.plot(kind="bar")
    plt.title("error rate by action")
    plt.xlabel("action")
    plt.ylabel("error rate")
    out = DERIVED / "error_rates_by_action.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return rate


# example 6: tail latency by action spans
def plot_tail_latency_for_actions(df):
    spans = df.loc[df["is_span_end"] & df["name"].str.startswith(
        "act:"), ["name", "duration_s"]].dropna()
    if spans.empty:
        return None
    g = spans.groupby("name")["duration_s"]
    tl = g.quantile([0.9, 0.95, 0.99]).unstack(
        level=-1).rename(columns={0.9: "p90", 0.95: "p95", 0.99: "p99"}).round(4)
    fig = plt.figure()
    tl.plot(kind="bar")
    plt.title("tail latency for act:* spans")
    plt.xlabel("span name")
    plt.ylabel("seconds")
    out = DERIVED / "tail_latency_by_action.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return tl


if __name__ == "__main__":
    df = load_events()

    head = df.head(3)
    print_head("sample rows", head, n=3)

    rpm = metric_runs_per_minute(df)
    outcomes = metric_outcomes(df)
    stats = plot_latency_by_span(df)
    retries = plot_retries_by_action(df)
    err_rates = plot_error_rates_by_action(df)
    tails = plot_tail_latency_for_actions(df)

    lines = []
    lines.append(f"total_runs: {outcomes['total']}")
    lines.append(f"success_rate: {outcomes['success_rate']}")
    if rpm is not None:
        lines.append(f"minutes_observed: {len(rpm)}")
        lines.append(f"max_runs_in_a_minute: {int(rpm.max())}")
    if stats is not None:
        lines.append(f"span_count: {int(stats['count'].sum())}")
    if err_rates is not None:
        lines.append(f"actions_observed: {len(err_rates)}")
    save_txt(DERIVED / "metrics_overview.txt", lines)

    print("wrote plots to", DERIVED)
