# 06_combined_dashboard.py

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DERIVED = DATA / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

EVENTS_PARQUET = DERIVED / "events.parquet"
SEQS_PARQUET = DERIVED / "sequences.parquet"


# io helpers
def load_events():
    # load events with defensive coercions
    if not EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            "events.parquet not found. run 03_event_schema_and_logging.py first")
    df = pd.read_parquet(EVENTS_PARQUET)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    for col in ["duration_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["step", "attempts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce", downcast="integer")
    if "ok" in df.columns:
        df["ok"] = df["ok"].astype("boolean")
    # ensure minute floor exists
    if "minute" not in df.columns:
        df["minute"] = df["ts"].dt.floor("min")
    return df


def load_sequences():
    # load per-run sequences, if missing return empty frame
    if not SEQS_PARQUET.exists():
        return pd.DataFrame()
    seqs = pd.read_parquet(SEQS_PARQUET)
    return seqs


# safe numeric check for plotting
def numeric_only(df):
    if df is None or len(df) == 0:
        return None
    num = df.select_dtypes(include=[np.number])
    return num if not num.empty else None


# overview metrics
def compute_overview(events):
    # run outcomes
    run_ok = events.loc[events.get("is_run_ok", False), [
        "run_id"]].assign(ok=True)
    run_fail = events.loc[events.get("is_run_fail", False), [
        "run_id"]].assign(ok=False)
    runs = pd.concat([run_ok, run_fail],
                     ignore_index=True).drop_duplicates("run_id")
    total_runs = len(runs)
    success_rate = float(runs["ok"].mean()) if total_runs > 0 else np.nan

    # rates
    starts = events.loc[events.get("is_run_start", False), ["minute"]]
    rpm = starts.groupby("minute").size().rename(
        "runs_started") if not starts.empty else pd.Series(dtype=int)
    max_rpm = int(rpm.max()) if not rpm.empty else 0

    # span latency summary
    spans = events.loc[events.get(
        "is_span_end", False) & events["duration_s"].notna(), ["name", "duration_s"]]
    lat = None
    if not spans.empty:
        g = spans.groupby("name")["duration_s"]
        lat = g.agg(["count", "mean", "median", "max"]).round(4)
        q = g.quantile([0.9, 0.95, 0.99]).unstack(level=-1)
        lat["p90"] = q[0.9].round(4)
        lat["p95"] = q[0.95].round(4)
        lat["p99"] = q[0.99].round(4)

    return {
        "total_runs": total_runs,
        "success_rate": success_rate,
        "max_rpm": max_rpm,
        "rpm_series": rpm,
        "latency_table": lat,
    }


# spike detection
def detect_rpm_spikes(rpm, z_thresh=2.0, min_count=5):
    # returns a dataframe of minutes with z-scores above threshold
    if rpm is None or rpm.empty:
        return pd.DataFrame()
    s = rpm.astype(float)
    mu = s.mean()
    sigma = s.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.DataFrame()
    z = (s - mu) / sigma
    out = pd.DataFrame(
        {"minute": z.index, "runs_started": s.values, "z": z.values})
    out = out[(out["z"] >= z_thresh) & (out["runs_started"] >=
                                        min_count)].sort_values("z", ascending=False)
    return out


# drilldown for a given minute bucket
def runs_in_minute(events, minute_ts):
    # collect run_ids that started in the specified minute
    mask = events.get("is_run_start", False) & events["minute"].eq(minute_ts)
    starts = events.loc[mask, ["run_id"]]
    if starts.empty:
        return []
    return starts["run_id"].unique().tolist()


def slice_events_for_runs(events, run_ids):
    if not run_ids:
        return pd.DataFrame()
    sl = events[events["run_id"].isin(run_ids)].copy()
    return sl


# sequence helpers
def slice_seqs_for_runs(seqs, run_ids):
    if seqs is None or seqs.empty or not run_ids:
        return pd.DataFrame()
    return seqs[seqs["run_id"].isin(run_ids)].copy()


def ngrams_from_actions(actions, n=2):
    if not isinstance(actions, list) or len(actions) < n:
        return []
    return [tuple(actions[i:i+n]) for i in range(len(actions)-n+1)]


def frequent_ngrams(seqs_slice, n=2, top_k=15):
    if seqs_slice is None or seqs_slice.empty:
        return pd.DataFrame()
    rows = []
    for _, r in seqs_slice.iterrows():
        for g in ngrams_from_actions(r["actions"], n=n):
            rows.append({"ngram": g})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.value_counts("ngram").rename("count").reset_index()
    return agg.head(top_k)


def transition_counts_from_events(events_slice):
    # build transitions from action_result ordering
    ar = events_slice.loc[events_slice.get("is_action_result", False), [
        "run_id", "ts", "step", "attempts", "action"]].dropna(subset=["run_id", "ts", "action"]).copy()
    if ar.empty:
        return pd.DataFrame()
    ar["step"] = pd.to_numeric(ar["step"], errors="coerce")
    ar["attempts"] = pd.to_numeric(ar["attempts"], errors="coerce")
    ar = ar.sort_values(["run_id", "ts", "step", "attempts"], kind="mergesort")
    edges = []
    for run_id, grp in ar.groupby("run_id"):
        xs = grp["action"].astype(str).tolist()
        for a, b in zip(xs, xs[1:]):
            edges.append((a, b))
    if not edges:
        return pd.DataFrame()
    edf = pd.DataFrame(edges, columns=["src", "dst"])
    counts = edf.groupby(["src", "dst"]).size().rename(
        "count").reset_index().sort_values("count", ascending=False)
    return counts


def self_loop_share(actions):
    if not isinstance(actions, list) or len(actions) < 2:
        return 0.0
    total = 0
    sl = 0
    for a, b in zip(actions, actions[1:]):
        total += 1
        if a == b:
            sl += 1
    return sl / total if total else 0.0


# plotting helpers
def plot_series_bar(series, title, out_path, xlabel, ylabel):
    if series is None or series.empty:
        return None
    fig = plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_counts_bar(df_counts, x_col, y_col, title, out_path, rotate=0):
    if df_counts is None or df_counts.empty:
        return None
    # ensure numeric y
    df_counts = df_counts.copy()
    df_counts[y_col] = pd.to_numeric(df_counts[y_col], errors="coerce")
    df_counts = df_counts.dropna(subset=[y_col])
    if df_counts.empty:
        return None
    fig = plt.figure()
    plt.bar(df_counts[x_col].astype(str).values, df_counts[y_col].values)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# dashboard main
def build_dashboard(z_thresh=2.0, top_k=15, minute_override=None):
    events = load_events()
    seqs = load_sequences()

    overview = compute_overview(events)

    # write overview text
    overview_txt = DERIVED / "dashboard_overview.txt"
    lines = []
    lines.append(f"total_runs: {overview['total_runs']}")
    lines.append(f"success_rate: {overview['success_rate']:.3f}" if not np.isnan(
        overview["success_rate"]) else "success_rate: nan")
    lines.append(f"max_runs_per_minute: {overview['max_rpm']}")
    with open(overview_txt, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    # plot rpm
    rpm_png = None
    if "rpm_series" in overview and isinstance(overview["rpm_series"], pd.Series) and not overview["rpm_series"].empty:
        rpm_png = plot_series_bar(
            overview["rpm_series"],
            "runs started per minute",
            DERIVED / "dash_runs_per_minute.png",
            "minute",
            "count",
        )

    # latency summary table to csv
    if isinstance(overview["latency_table"], pd.DataFrame) and not overview["latency_table"].empty:
        overview["latency_table"].to_csv(DERIVED / "dash_latency_summary.csv")

    # choose minute to drill down
    if minute_override is not None:
        minute_choice = pd.to_datetime(minute_override, utc=True)
        spikes = pd.DataFrame(
            {"minute": [minute_choice], "runs_started": [np.nan], "z": [np.nan]})
    else:
        spikes = detect_rpm_spikes(
            overview["rpm_series"], z_thresh=z_thresh, min_count=3)
        minute_choice = spikes["minute"].iloc[0] if not spikes.empty else None

    spike_csv = DERIVED / "dash_spikes.csv"
    if not spikes.empty:
        spikes.to_csv(spike_csv, index=False)

    # if no spike detected, pick the busiest minute if available
    if minute_choice is None and isinstance(overview["rpm_series"], pd.Series) and not overview["rpm_series"].empty:
        minute_choice = overview["rpm_series"].idxmax()

    # drilldown only if we have a minute
    if minute_choice is not None:
        run_ids = runs_in_minute(events, minute_choice)
        runs_csv = DERIVED / "dash_runs_in_minute.csv"
        pd.DataFrame({"run_id": run_ids}).to_csv(runs_csv, index=False)

        ev_slice = slice_events_for_runs(events, run_ids)
        seq_slice = slice_seqs_for_runs(seqs, run_ids)

        # top ngrams
        fg2 = frequent_ngrams(seq_slice, n=2, top_k=top_k)
        fg3 = frequent_ngrams(seq_slice, n=3, top_k=top_k)
        if not fg2.empty:
            fg2.to_csv(DERIVED / "dash_top_bigrams.csv", index=False)
            plot_counts_bar(fg2.assign(label=fg2["ngram"].astype(
                str)), "label", "count", "top bigrams in spike minute", DERIVED / "dash_top_bigrams.png", rotate=60)
        if not fg3.empty:
            fg3.to_csv(DERIVED / "dash_top_trigrams.csv", index=False)
            plot_counts_bar(fg3.assign(label=fg3["ngram"].astype(
                str)), "label", "count", "top trigrams in spike minute", DERIVED / "dash_top_trigrams.png", rotate=60)

        # transition counts
        trans = transition_counts_from_events(ev_slice)
        if not trans.empty:
            trans.to_csv(DERIVED / "dash_transitions.csv", index=False)
            top_trans = trans.head(top_k).copy()
            plot_counts_bar(top_trans.assign(label=top_trans["src"] + "→" + top_trans["dst"]), "label",
                            "count", "top transitions in spike minute", DERIVED / "dash_top_transitions.png", rotate=60)

        # loopiness snapshot
        loopiness = pd.DataFrame()
        if not seq_slice.empty and "loopiness_score" in seq_slice.columns:
            loopiness = seq_slice.sort_values("loopiness_score", ascending=False)[
                ["run_id", "count", "loopiness_score", "self_loop_share", "loopiness_bigram"]].head(top_k)
            loopiness.to_csv(DERIVED / "dash_loopiness_top.csv", index=False)

    # return a small dict with pointers for convenience
    return {
        "overview_txt": str(overview_txt),
        "rpm_png": str(rpm_png) if rpm_png else None,
        "spike_csv": str(spike_csv) if not spikes.empty else None,
        "minute_drilled": str(minute_choice) if minute_choice is not None else None,
    }


# cli
def parse_args():
    p = argparse.ArgumentParser(
        description="mini dashboard for agentic AI performance")
    p.add_argument("--z", type=float, default=2.0,
                   help="z-score threshold for spike detection")
    p.add_argument("--top", type=int, default=15,
                   help="top-k items to display in per-minute drilldown")
    p.add_argument("--minute", type=str, default=None,
                   help="explicit minute to drill down, e.g., 2025-08-09T12:34:00Z")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # accept minute override in either Z suffix or +00:00 form
    minute_override = None
    if args.minute:
        s = args.minute
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        minute_override = s
    out = build_dashboard(z_thresh=args.z, top_k=args.top,
                          minute_override=minute_override)
    print("dashboard outputs")
    for k, v in out.items():
        print(k, "->", v)


# dashboard_overview.txt with total runs, success rate, max rpm
# dash_runs_per_minute.png if there’s any rate data
# dash_latency_summary.csv with span latency stats
# dash_spikes.csv if spikes were detected
# dash_runs_in_minute.csv listing run_ids at the drilled minute
# dash_top_bigrams.csv/.png and dash_top_trigrams.csv/.png if sequences exist
# dash_transitions.csv and dash_top_transitions.png for the minute
# dash_loopiness_top.csv with the loopy runs in that minute

# then run the dashboard
# python 06_combined_dashboard.py

# or drill into a specific minute
# python 06_combined_dashboard.py --minute 2025-08-09T10:12:00Z

# or make spike detection stricter and limit top lists
# python 06_combined_dashboard.py --z 2.5 --top 10
