# 08_gpu_optional.py

import os
import io
import json
import time
import uuid
import math
import random
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# optional torch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw_events"
RAW.mkdir(parents=True, exist_ok=True)
SINK_PATH = RAW / "08_gpu.jsonl"

# time and ids


def now_iso():
    # utc iso-8601 with Z suffix
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def eid():
    # random uuid4
    return str(uuid.uuid4())

# lightweight jsonl writer


class JsonlSink:
    # append-only jsonl with small buffer
    def __init__(self, path, buffer_size=8192):
        self.path = Path(path)
        self.buf = io.StringIO()
        self.buffer_size = buffer_size

    def write(self, rec):
        self.buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.buf.tell() >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buf.tell() == 0:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(self.buf.getvalue())
        self.buf = io.StringIO()

# minimal emitter


class Emitter:
    # emits structured events
    def __init__(self, sink_path):
        self.sink = JsonlSink(sink_path)

    def emit(self, **kwargs):
        evt = {"ts": now_iso(), "event_id": eid()}
        evt.update(kwargs)
        self.sink.write(evt)

    def flush(self):
        self.sink.flush()

# gpu detection helpers


def _nvidia_smi_query():
    # returns parsed info dict if nvidia-smi is available
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=3,
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    gpus = []
    for ln in lines:
        try:
            name, mem = [x.strip() for x in ln.split(",")]
            gpus.append({"name": name, "total_mem_mb": int(mem)})
        except Exception:
            continue
    return {"gpus": gpus} if gpus else None


def detect_gpu():
    # unify torch and nvidia-smi signals
    info = {
        "available": False,
        "via": None,
        "device_count": 0,
        "devices": [],
    }
    if TORCH_AVAILABLE and torch.cuda.is_available():
        cnt = torch.cuda.device_count()
        info["available"] = True
        info["via"] = "torch.cuda"
        info["device_count"] = cnt
        devs = []
        for i in range(cnt):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            try:
                total = torch.cuda.get_device_properties(i).total_memory
                total_gb = round(total / (1024**3), 2)
            except Exception:
                total_gb = None
            devs.append({"index": i, "name": name,
                        "capability": cap, "total_mem_gb": total_gb})
        info["devices"] = devs
        return info
    smi = _nvidia_smi_query()
    if smi:
        info["available"] = True
        info["via"] = "nvidia-smi"
        info["device_count"] = len(smi["gpus"])
        info["devices"] = [{"index": i, "name": g["name"], "total_mem_gb": round(
            g["total_mem_mb"] / 1024, 2)} for i, g in enumerate(smi["gpus"])]
        return info
    return info

# token-timing emitter


def emit_generation_timings(emitter, run_id, n_tokens, mean_ms=18.0, jitter_ms=8.0):
    # emits per-token timings under act:generate spans
    for i in range(n_tokens):
        t0 = time.perf_counter()
        # cpu fallback uses sleep to simulate token latency
        sleep_ms = max(1.0, random.gauss(mean_ms, jitter_ms))
        time.sleep(sleep_ms / 1000.0)
        dur = time.perf_counter() - t0
        emitter.emit(kind="span_start", run_id=run_id, span_id=eid(),
                     name="act:generate", attrs={"token_index": i})
        emitter.emit(kind="span_end", run_id=run_id, span_id=eid(),
                     name="act:generate", status="ok", duration_s=dur)
        emitter.emit(kind="metric", run_id=run_id,
                     name="token_latency_s", duration_s=dur)

# gpu-backed tiny benchmark if torch is available


def torch_matmul_benchmark(device, sizes=((512, 512, 512), (1024, 1024, 1024)), warmup=5, iters=20):
    # runs a few matmuls to estimate throughput and latency
    results = []
    if not TORCH_AVAILABLE:
        return results
    for m, k, n in sizes:
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        # warmup
        for _ in range(warmup):
            c = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg = elapsed / iters
        flops = 2.0 * m * k * n
        gflops = (flops / avg) / 1e9
        results.append(
            {"m": m, "k": k, "n": n, "avg_s": avg, "gflops": gflops})
    return results

# main demo


def main():
    emitter = Emitter(SINK_PATH)
    run_id = eid()

    gpu = detect_gpu()
    emitter.emit(kind="gpu_info", run_id=run_id, info=gpu)

    if TORCH_AVAILABLE and gpu["available"] and torch.cuda.is_available():
        device = torch.device("cuda:0")
        emitter.emit(kind="run_start", run_id=run_id, task="gpu_benchmark")
        bench = torch_matmul_benchmark(device, sizes=(
            (512, 512, 512), (1024, 1024, 1024)), warmup=5, iters=30)
        for row in bench:
            emitter.emit(kind="metric", run_id=run_id, name="matmul_avg_s", attrs={
                         "m": row["m"], "k": row["k"], "n": row["n"]}, duration_s=row["avg_s"])
            emitter.emit(kind="metric", run_id=run_id, name="matmul_gflops", attrs={
                         "m": row["m"], "k": row["k"], "n": row["n"]}, duration_s=row["gflops"])
        emit_generation_timings(
            emitter, run_id, n_tokens=50, mean_ms=12.0, jitter_ms=5.0)
        emitter.emit(kind="run_ok", run_id=run_id, steps=1)
    else:
        emitter.emit(kind="run_start", run_id=run_id, task="cpu_baseline")
        emit_generation_timings(
            emitter, run_id, n_tokens=40, mean_ms=20.0, jitter_ms=8.0)
        emitter.emit(kind="run_ok", run_id=run_id, steps=1)

    emitter.flush()
    print("wrote events to", SINK_PATH)


if __name__ == "__main__":
    random.seed(11)
    main()
