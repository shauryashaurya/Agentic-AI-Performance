# 00_setup_and_smoke_test.py

import os
import time
import random
import json
import uuid
from datetime import datetime
from pathlib import Path

# ensure data directories exist
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw_events"
DERIVED = DATA / "derived"
for p in [DATA, RAW, DERIVED]:
    p.mkdir(parents=True, exist_ok=True)

# simple monotonic timer wrapper


class Timer:
    # lightweight utility to measure elapsed seconds
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0

# simple id helpers


def new_run_id():
    # unique identifier for an agent run
    return str(uuid.uuid4())


def now_iso():
    # iso-8601 timestamp with utc suffix
    return datetime.utcnow().isoformat() + "Z"

# entropy check


def random_seed(seed=42):
    # set pseudo-random seed for reproducibility
    random.seed(seed)


def jitter(p=0.5):
    # return True with probability p to simulate nondeterminism
    return random.random() < p


def write_jsonl(path, records):
    # append records to a jsonl file
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def smoke_test():
    # basic check: timing, randomness, file write
    random_seed(123)
    t = Timer()
    rid = new_run_id()
    samples = []
    for i in range(10):
        ok = jitter(0.7)
        samples.append({"i": i, "ok": ok})
        time.sleep(0.01)
    elapsed = t.elapsed()
    out = {
        "run_id": rid,
        "ts": now_iso(),
        "elapsed_s": elapsed,
        "samples": samples,
    }
    write_jsonl(RAW / "00_smoke.jsonl", [out])
    return out


if __name__ == "__main__":
    result = smoke_test()
    print("wrote 1 record to data/raw_events/00_smoke.jsonl")
    print("elapsed_s:", round(result["elapsed_s"], 4))
    print("ok_rate:", sum(
        1 for s in result["samples"] if s["ok"]) / len(result["samples"]))
