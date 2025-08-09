# 01_agent_basics.py

import json
import uuid
import time
import random
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw_events"
RAW.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def eid():
    return str(uuid.uuid4())


def write_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Emitter:
    # minimal structured event emitter
    def __init__(self, sink_path):
        self.sink_path = sink_path

    def emit(self, **kwargs):
        # every event gets a timestamp and event_id
        evt = {"ts": now_iso(), "event_id": eid()}
        evt.update(kwargs)
        write_jsonl(self.sink_path, [evt])

# minimal agent skeleton with a single step and optional retry


class ToyAgent:
    # the agent processes a task by calling a "tool"
    def __init__(self, emitter, max_retries=2):
        self.emitter = emitter
        self.max_retries = max_retries

    def tool_call(self, payload):
        # simulate variable latency and nondeterministic failure
        t0 = time.perf_counter()
        time.sleep(random.uniform(0.02, 0.12))
        fail = random.random() < 0.25
        latency = time.perf_counter() - t0
        return {"ok": not fail, "latency_s": latency}

    def run(self, task):
        run_id = eid()
        self.emitter.emit(kind="run_start", run_id=run_id, task=task)
        attempts = 0
        while True:
            attempts += 1
            self.emitter.emit(kind="step_start",
                              run_id=run_id, attempt=attempts)
            result = self.tool_call({"task": task})
            if result["ok"]:
                self.emitter.emit(kind="step_ok", run_id=run_id,
                                  attempt=attempts, latency_s=result["latency_s"])
                self.emitter.emit(
                    kind="run_ok", run_id=run_id, attempts=attempts)
                return {"run_id": run_id, "ok": True, "attempts": attempts}
            else:
                self.emitter.emit(kind="step_fail", run_id=run_id,
                                  attempt=attempts, latency_s=result["latency_s"])
                if attempts >= self.max_retries + 1:
                    self.emitter.emit(
                        kind="run_fail", run_id=run_id, attempts=attempts)
                    return {"run_id": run_id, "ok": False, "attempts": attempts}
                else:
                    self.emitter.emit(
                        kind="retry", run_id=run_id, next_attempt=attempts + 1)


if __name__ == "__main__":
    random.seed(7)
    emitter = Emitter(RAW / "01_agent_basics.jsonl")
    agent = ToyAgent(emitter, max_retries=2)
    results = []
    for i in range(15):
        task = f"task_{i}"
        results.append(agent.run(task))
    print("ran", len(results), "tasks")
    print("success_rate:", sum(1 for r in results if r["ok"]) / len(results))
