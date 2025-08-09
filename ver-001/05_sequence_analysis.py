# 05_sequence_analysis.py

import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# paths
BASE = Path(__file__).resolve().parent
DERIVED = BASE / "data" / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)
PARQUET = DERIVED / "events.parquet"

# io helpers


def load_events():
    # load normalized events from section 3
    if not PARQUET.exists():
        raise FileNotFoundError(
            "events.parquet not found. run 03_event_schema_and_logging.py first")
    df = pd.read_parquet(PARQUET)
    # basic guards and coercions
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    if "step" in df.columns:
        df["step"] = pd.to_numeric(
            df["step"], errors="coerce", downcast="integer")
    if "attempts" in df.columns:
        df["attempts"] = pd.to_numeric(
            df["attempts"], errors="coerce", downcast="integer")
    if "ok" in df.columns:
        df["ok"] = df["ok"].astype("boolean")
    return df

# sequence reconstruction


def build_action_sequences(df):
    # select action_result events with ordering
    cols = ["run_id", "ts", "step", "action", "ok", "attempts"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("build_action_sequences: missing columns", missing)
        return pd.DataFrame(), []
    ar = df.loc[df["is_action_result"].fillna(False)].copy()
    if ar.empty:
        print("build_action_sequences: no action_result events")
        return pd.DataFrame(), []
    # coerce types
    ar["step"] = pd.to_numeric(ar["step"], errors="coerce")
    ar["attempts"] = pd.to_numeric(ar["attempts"], errors="coerce")
    ar = ar.dropna(subset=["run_id", "ts", "step", "action"])
    if ar.empty:
        print("build_action_sequences: empty after dropna")
        return pd.DataFrame(), []
    # stable order: by run, then time, then step, then attempts
    ar = ar.sort_values(["run_id", "ts", "step", "attempts"], kind="mergesort")
    # normalize action to string for safety
    ar["action"] = ar["action"].astype(str)
    # aggregate into ordered sequences per run
    seqs = ar.groupby("run_id").agg(
        actions=("action", list),
        oks=("ok", lambda s: [bool(x) if pd.notna(
            x) else None for x in s.tolist()]),
        steps=("step", list),
        attempts=("attempts", list),
        count=("action", "size"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
    ).reset_index()
    seqs["duration_s"] = (
        seqs["last_ts"] - seqs["first_ts"]).dt.total_seconds()
    # compute loopiness heuristics
    seqs["unique_actions"] = seqs["actions"].apply(
        lambda xs: len(set(xs)) if isinstance(xs, list) else np.nan)
    seqs["loopiness_ua"] = seqs.apply(
        lambda r: 1.0 - (r["unique_actions"] / r["count"]) if r["count"] else np.nan, axis=1)
    # repeated bigram rate

    def rep_bigram_rate(xs):
        if not isinstance(xs, list) or len(xs) < 2:
            return 0.0
        bigrams = [tuple(xs[i:i+2]) for i in range(len(xs)-1)]
        c = Counter(bigrams)
        repeats = sum(v for v in c.values() if v > 1)
        return repeats / max(1, len(bigrams))
    seqs["loopiness_bigram"] = seqs["actions"].apply(rep_bigram_rate)
    # simple self-loop share

    def self_loop_share(xs):
        if not isinstance(xs, list) or len(xs) < 2:
            return 0.0
        total = 0
        self_loops = 0
        for a, b in zip(xs, xs[1:]):
            total += 1
            if a == b:
                self_loops += 1
        return self_loops / total if total else 0.0
    seqs["self_loop_share"] = seqs["actions"].apply(self_loop_share)
    # combined loopiness score
    seqs["loopiness_score"] = seqs[["loopiness_ua",
                                    "loopiness_bigram", "self_loop_share"]].mean(axis=1)
    return ar, seqs

# n-gram mining


def mine_ngrams(seqs, n_vals=(1, 2, 3), top_k=30):
    rows = []
    for _, row in seqs.iterrows():
        xs = row["actions"]
        run_id = row["run_id"]
        if not isinstance(xs, list) or len(xs) == 0:
            continue
        for n in n_vals:
            if len(xs) < n:
                continue
            grams = [tuple(xs[i:i+n]) for i in range(len(xs)-n+1)]
            for g in grams:
                rows.append({"run_id": run_id, "n": n, "ngram": g})
    if not rows:
        print("mine_ngrams: no ngrams")
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby(["n", "ngram"]).size().rename(
        "count").reset_index().sort_values(["n", "count"], ascending=[True, False])
    top = agg.groupby("n").head(top_k).reset_index(drop=True)
    # save
    top_path = DERIVED / "ngrams_top.csv"
    top.to_csv(top_path, index=False)
    print("wrote:", top_path)
    return df, top

# transition matrix and simple cycles


def build_transition_matrix(seqs):
    edges = []
    for _, row in seqs.iterrows():
        xs = row["actions"]
        if not isinstance(xs, list) or len(xs) < 2:
            continue
        for a, b in zip(xs, xs[1:]):
            edges.append((a, b))
    if not edges:
        print("build_transition_matrix: no edges")
        return pd.DataFrame(), pd.DataFrame()
    edge_df = pd.DataFrame(edges, columns=["src", "dst"])
    counts = edge_df.groupby(["src", "dst"]).size().rename(
        "count").reset_index()
    # normalize by row to get probabilities
    row_sums = counts.groupby("src")["count"].transform("sum")
    counts["prob"] = counts["count"] / row_sums
    # pivot to matrix
    matrix = counts.pivot_table(
        index="src", columns="dst", values="prob", fill_value=0.0)
    matrix_path = DERIVED / "transition_matrix.csv"
    matrix.to_csv(matrix_path)
    print("wrote:", matrix_path)
    return counts, matrix


def plot_transition_graph(counts, max_nodes=12):
    if counts.empty:
        print("plot_transition_graph: no counts")
        return None
    # limit to top nodes by total degree for readability
    top_nodes = (counts.groupby("src")["count"].sum(
    ) + counts.groupby("dst")["count"].sum()).sort_values(ascending=False)
    keep = set(top_nodes.head(max_nodes).index)
    sub = counts[counts["src"].isin(keep) & counts["dst"].isin(keep)].copy()
    if sub.empty:
        print("plot_transition_graph: nothing after node limit")
        return None
    G = nx.DiGraph()
    for _, r in sub.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(
            r["count"]), prob=float(r["prob"]))
    pos = nx.spring_layout(G, seed=7)
    fig = plt.figure()
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=9)
    widths = [1.0 + 3.0 * (G[u][v]["prob"]) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle="-|>")
    out = DERIVED / "transition_graph.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote:", out)
    return out

# cycle detection per run


def detect_cycles_per_run(seqs, max_cycle_len=6, max_cycles_per_run=50):
    records = []
    for _, row in seqs.iterrows():
        xs = row["actions"]
        run_id = row["run_id"]
        if not isinstance(xs, list) or len(xs) < 2:
            continue
        # build directed graph for this run
        G = nx.DiGraph()
        for a, b in zip(xs, xs[1:]):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
        # find simple cycles with a guard
        cycles = []
        try:
            for cyc in nx.simple_cycles(G):
                if 2 <= len(cyc) <= max_cycle_len:
                    cycles.append(tuple(cyc))
                if len(cycles) >= max_cycles_per_run:
                    break
        except nx.NetworkXNoCycle:
            cycles = []
        # record results
        ccounts = Counter(cycles)
        for c, k in ccounts.items():
            records.append({"run_id": run_id, "cycle": c, "occurrences": k})
    if not records:
        print("detect_cycles_per_run: no cycles found")
        return pd.DataFrame()
    cyc_df = pd.DataFrame(records).sort_values(
        ["run_id", "occurrences"], ascending=[True, False]).reset_index(drop=True)
    out = DERIVED / "run_cycles.csv"
    cyc_df.to_csv(out, index=False)
    print("wrote:", out)
    return cyc_df

# summarize red flags


def summarize_flags(seqs, counts, matrix):
    lines = []
    if not seqs.empty:
        # top loopiness runs
        top_loopy = seqs.nlargest(5, "loopiness_score")[
            ["run_id", "count", "loopiness_score", "self_loop_share", "loopiness_bigram"]]
        lines.append("top loopy runs:")
        for _, r in top_loopy.iterrows():
            lines.append(
                f"- run {r['run_id'][:8]} count={int(r['count'])} score={r['loopiness_score']:.3f} self_loop={r['self_loop_share']:.3f} bigram={r['loopiness_bigram']:.3f}")
    if not counts.empty:
        # high self-loop actions
        self_loops = counts[counts["src"] == counts["dst"]
                            ].sort_values("prob", ascending=False)
        if not self_loops.empty:
            lines.append("high self-loop actions:")
            for _, r in self_loops.head(5).iterrows():
                lines.append(
                    f"- {r['src']} -> itself prob={r['prob']:.2f} count={int(r['count'])}")
        # risky transitions by probability and volume
        heavy = counts.sort_values(["prob", "count"], ascending=[
                                   False, False]).head(5)
        lines.append("dominant transitions:")
        for _, r in heavy.iterrows():
            lines.append(
                f"- {r['src']} -> {r['dst']} prob={r['prob']:.2f} count={int(r['count'])}")
    report = DERIVED / "sequence_summary.txt"
    with open(report, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print("wrote:", report)
    return lines


if __name__ == "__main__":
    df = load_events()

    ar, seqs = build_action_sequences(df)
    if seqs.empty:
        print("no sequences built, exiting early")
    else:
        seqs_path = DERIVED / "sequences.parquet"
        seqs.to_parquet(seqs_path, index=False)
        print("wrote:", seqs_path)

        # n-grams
        _, top = mine_ngrams(seqs, n_vals=(1, 2, 3), top_k=30)

        # transitions
        counts, matrix = build_transition_matrix(seqs)
        plot_transition_graph(counts)

        # cycles per run
        detect_cycles_per_run(seqs)

        # summary
        summarize_flags(seqs, counts, matrix)

        # print a tiny console preview
        print("preview sequences")
        print(seqs.head(3)[["run_id", "actions", "count",
              "loopiness_score"]].to_string(index=False))
