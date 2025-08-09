# 02_agentic_101_sim.py

import json
import uuid
import time
import random
from pathlib import Path
from datetime import datetime, timezone

# paths
BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw_events"
RAW.mkdir(parents=True, exist_ok=True)


# time and ids
def now_iso():
    # returns ISO 8601 timestamp in UTC with Z suffix
    return datetime.now(timezone.utc).isoformat()


def eid():
    return str(uuid.uuid4())


# writer
def write_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# emitter with minimal schema
class Emitter:
    # events are append-only json lines
    def __init__(self, sink_path):
        self.sink_path = sink_path

    def emit(self, **kwargs):
        evt = {"ts": now_iso(), "event_id": eid()}
        evt.update(kwargs)
        write_jsonl(self.sink_path, [evt])


# span context manager for start/end pairs
class Span:
    # emits span_start and span_end with duration and status
    def __init__(self, emitter, run_id, span_id, name, parent_span_id=None, attrs=None):
        self.emitter = emitter
        self.run_id = run_id
        self.span_id = span_id
        self.name = name
        self.parent_span_id = parent_span_id
        self.attrs = attrs or {}
        self.t0 = None
        self.status = "ok"
        self.err = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.emitter.emit(
            kind="span_start",
            run_id=self.run_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            attrs=self.attrs,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.status = "error"
            self.err = repr(exc)
        dur = time.perf_counter() - self.t0
        self.emitter.emit(
            kind="span_end",
            run_id=self.run_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            status=self.status,
            error=self.err,
            duration_s=dur,
        )
        return False


# simulated tools
class Toolset:
    # a small set of tools with nondeterminism and latency
    def __init__(self, emitter, run_id):
        self.emitter = emitter
        self.run_id = run_id

    # make search flakier to create more retries
    # increase fail_p when name == "search"
    # fail_p=... higher failure means more retries, more events, and more variance that’s interesting to analyze.
    # latency_range=(...) doesn’t change count but slows runtime; widen carefully if you want longer durations for latency metrics.
    def _call(self, name, latency_range=(0.02, 0.15), fail_p=0.1, attrs=None):
        span_id = eid()
        with Span(self.emitter, self.run_id, span_id, f"tool:{name}", attrs=attrs):
            time.sleep(random.uniform(*latency_range))
            if random.random() < fail_p:
                return {"ok": False, "name": name, "data": None}
            # return synthetic data with slight randomness
            data = {"value": random.randint(1, 9), "conf": round(
                random.uniform(0.5, 0.99), 2)}
            return {"ok": True, "name": name, "data": data}

    def calc(self, x, y):
        # sometimes slow, rarely fails
        return self._call("calc", latency_range=(0.03, 0.2), fail_p=0.05, attrs={"x": x, "y": y})

    def search(self, query):
        # moderate latency, occasional ambiguity
        return self._call("search", latency_range=(0.05, 0.25), fail_p=0.12, attrs={"q": query})

    def memory_get(self, key):
        # very fast, may miss
        miss = random.random() < 0.3
        if miss:
            return {"ok": True, "name": "memory_get", "data": {"hit": False, "key": key}}
        return {"ok": True, "name": "memory_get", "data": {"hit": True, "key": key, "value": random.choice(["A", "B", "C"])}}


# planner
class Planner:
    # decides next action based on task and partial results
    def __init__(self, emitter, run_id):
        self.emitter = emitter
        self.run_id = run_id

    # random.random() < 0.3 branch controls how often we take less certain paths, increasing steps and sequences.
    def plan(self, task, context):
        span_id = eid()
        with Span(self.emitter, self.run_id, span_id, "planner:next_action", attrs={"task": task, "context_keys": list(context.keys())}):
            if "sum" in task:
                return {"action": "calc", "args": {"x": random.randint(1, 5), "y": random.randint(1, 5)}}
            if "lookup" in task:
                return {"action": "search", "args": {"query": f"{task} site:example.com"}}
            if "remember" in task:
                return {"action": "memory_get", "args": {"key": "user_pref"}}
            # ambiguous case to provoke loops or retries
            if random.random() < 0.3:
                return {"action": "search", "args": {"query": f"{task} general"}}
            return {"action": "calc", "args": {"x": random.randint(1, 3), "y": random.randint(1, 3)}}


# agent with retries and loop guard
class Agent:
    # executes planner decisions and collects spans
    def __init__(self, emitter, max_steps=6, max_retries=2):
        self.emitter = emitter
        self.max_steps = max_steps
        self.max_retries = max_retries

    def run(self, task):
        run_id = eid()
        self.emitter.emit(kind="run_start", run_id=run_id, task=task)
        tools = Toolset(self.emitter, run_id)
        planner = Planner(self.emitter, run_id)
        context = {}
        history = []
        step = 0

        try:
            # loop guard policy
            # loosen the guard or raise its trigger probability to capture more loops
            # disable it to let runs hit max_steps_exceeded, generating even more spans
            while step < self.max_steps:
                step += 1
                self.emitter.emit(kind="step_start", run_id=run_id,
                                  step=step, context_keys=list(context.keys()))
                decision = planner.plan(task, context)
                action = decision["action"]
                args = decision["args"]

                attempts = 0
                while attempts <= self.max_retries:
                    attempts += 1
                    span_id = eid()
                    with Span(self.emitter, run_id, span_id, f"act:{action}", parent_span_id=None, attrs={"attempt": attempts, "args": args}):
                        if action == "calc":
                            res = tools.calc(**args)
                        elif action == "search":
                            res = tools.search(**args)
                        elif action == "memory_get":
                            res = tools.memory_get(**args)
                        else:
                            res = {"ok": False, "name": action, "data": None}

                    self.emitter.emit(kind="action_result", run_id=run_id, step=step,
                                      action=action, ok=res["ok"], attempts=attempts, data=res["data"])

                    # adjust the confidence threshold (conf >= 0.85) and continuation probability to encourage longer traces before success.
                    # increase loop risk by weakening the stop condition
                    # lower conf threshold or increase random continuation chance
                    # currently: conf >= 0.85
                    if res["ok"]:
                        history.append((action, res["data"]))
                        # stochastic decision to either stop or continue based on confidence
                        conf = res["data"]["conf"] if isinstance(
                            res["data"], dict) and "conf" in res["data"] else 0.8
                        context[f"{action}_{step}"] = res["data"]
                        if conf >= 0.85 and random.random() < 0.6:
                            self.emitter.emit(
                                kind="run_ok", run_id=run_id, steps=step, attempts=sum(1 for _ in history))
                            return {"run_id": run_id, "ok": True, "steps": step, "history": history}
                        break
                    else:
                        if attempts > self.max_retries:
                            self.emitter.emit(
                                kind="partial_fail", run_id=run_id, step=step, action=action)
                            # let planner decide another path
                            break

                # simple loop guard: if actions repeat without new context, bail
                if len(history) >= 2 and history[-1][0] == history[-2][0] and random.random() < 0.5:
                    self.emitter.emit(
                        kind="loop_guard_triggered", run_id=run_id, step=step, last_action=history[-1][0])
                    self.emitter.emit(
                        kind="run_fail", run_id=run_id, reason="loop_guard")
                    return {"run_id": run_id, "ok": False, "steps": step, "history": history, "reason": "loop_guard"}

            self.emitter.emit(kind="run_fail", run_id=run_id,
                              reason="max_steps_exceeded")
            return {"run_id": run_id, "ok": False, "steps": self.max_steps, "history": history, "reason": "max_steps"}
        except Exception as e:
            self.emitter.emit(kind="run_error", run_id=run_id, error=repr(e))
            return {"run_id": run_id, "ok": False, "error": repr(e)}


# harness to produce diverse behaviors
def generate_tasks():
    tasks = []
    # simple sums
    for i in range(6):
        tasks.append(f"sum small_{i}")
    # lookups
    for i in range(6):
        tasks.append(f"lookup thing_{i}")
    # memory probes
    for i in range(4):
        tasks.append(f"remember preference_{i}")
    # ambiguous tasks to provoke loops or retries
    for i in range(8):
        tasks.append(f"analyze unstructured_{i}")
    random.shuffle(tasks)
    return tasks


if __name__ == "__main__":
    random.seed(17)
    emitter = Emitter(RAW / "02_agentic_101.jsonl")
    # Agent(max_steps=...) increases per-run span count because each step creates planner + action + tool spans.
    # Agent(max_retries=...) adds extra action attempts and more action_result + span pairs per step.
    agent = Agent(emitter, max_steps=6, max_retries=2)

    results = []
    tasks = generate_tasks()
    for t in tasks:
        results.append(agent.run(t))

    total = len(results)
    ok = sum(1 for r in results if r.get("ok"))
    loops = sum(1 for r in results if r.get("reason") == "loop_guard")
    print("runs:", total)
    print("success_rate:", round(ok / total, 3))
    print("loop_guard_trips:", loops)


# set a different random seed for a different trace distribution
# random.seed(2025)
#
# make search flakier to create more retries
# in Toolset._call: increase fail_p when name == "search"
#
# increase loop risk by weakening the stop condition
# in Agent.run: lower conf threshold or increase random continuation chance
