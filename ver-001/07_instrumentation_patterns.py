# 07_instrumentation_patterns.py

import os
import io
import json
import time
import uuid
import math
import inspect
import random
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ContextDecorator
from functools import wraps, partial
try:
    import asyncio
except ImportError:
    asyncio = None

# paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw_events"
RAW.mkdir(parents=True, exist_ok=True)


# time and ids
def now_iso():
    # utc timestamp as iso-8601 with Z
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def eid():
    # random uuid4 string
    return str(uuid.uuid4())


def _call_hook(fn, args, kwargs, default=None):
    # safely call a user hook with flexible signatures
    if not callable(fn):
        return default
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn(**kwargs)
        except TypeError:
            return default


def _resolve_run_id(run_id_fn, args, kwargs):
    # prefer explicit run_id first, then hook, then new id
    if isinstance(kwargs, dict) and kwargs.get("run_id"):
        return kwargs["run_id"]
    rid = _call_hook(run_id_fn, args, kwargs, default=None)
    return rid if rid else eid()


# jsonl sink with rotation and buffering
class JsonlSink:
    # append-only jsonl with size-based rotation
    def __init__(self, path, rotate_mb=20, buffer_size=8192):
        self.path = Path(path)
        self.rotate_bytes = int(rotate_mb * 1024 * 1024)
        self.buffer = io.StringIO()
        self.buffer_size = buffer_size

    def _maybe_rotate(self):
        try:
            sz = self.path.stat().st_size
        except FileNotFoundError:
            return
        if sz >= self.rotate_bytes:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            rotated = self.path.with_suffix(f".{ts}.jsonl")
            self.path.rename(rotated)

    def write_record(self, rec):
        s = json.dumps(rec, ensure_ascii=False)
        self.buffer.write(s + "\n")
        if self.buffer.tell() >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer.tell() == 0:
            return
        self._maybe_rotate()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(self.buffer.getvalue())
        self.buffer = io.StringIO()


# redaction helper
def redact_obj(obj, redact_keys):
    # returns a shallowly redacted copy for dicts and lists
    if obj is None:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in redact_keys:
                out[k] = "[redacted]"
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [redact_obj(x, redact_keys) for x in obj]
    return obj


# config
class TelemetryConfig:
    # config that can be controlled via env vars
    def __init__(self,
                 sink_path=RAW / "07_instrumented.jsonl",
                 enable=os.environ.get("TRACE_ENABLE", "1") == "1",
                 sample=float(os.environ.get("TRACE_SAMPLE", "1.0")),
                 rotate_mb=float(os.environ.get("TRACE_ROTATE_MB", "20")),
                 include_args=os.environ.get("TRACE_INCLUDE_ARGS", "1") == "1",
                 redact_keys=None):
        self.enable = enable
        self.sample = max(0.0, min(1.0, sample))
        self.sink = JsonlSink(sink_path, rotate_mb=rotate_mb)
        self.include_args = include_args
        self.redact_keys = set(redact_keys or [])


# emitter with sampling
class Emitter:
    # minimal structured event emitter with sampling and redaction
    def __init__(self, cfg: TelemetryConfig):
        self.cfg = cfg

    def _should_sample(self):
        if not self.cfg.enable:
            return False
        if self.cfg.sample >= 1.0:
            return True
        return random.random() < self.cfg.sample

    def emit(self, **kwargs):
        if not self._should_sample():
            return
        evt = {"ts": now_iso(), "event_id": eid()}
        evt.update(kwargs)
        self.cfg.sink.write_record(evt)

    def flush(self):
        self.cfg.sink.flush()


# span context
class Span(ContextDecorator):
    # start/end span with duration and status; safe if disabled
    def __init__(self, emitter: Emitter, run_id, name, parent_span_id=None, attrs=None):
        self.emitter = emitter
        self.run_id = run_id
        self.name = name
        self.parent_span_id = parent_span_id
        self.attrs = attrs or {}
        self.span_id = eid()
        self.t0 = None
        self.status = "ok"
        self.err = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.emitter.emit(kind="span_start",
                          run_id=self.run_id,
                          span_id=self.span_id,
                          parent_span_id=self.parent_span_id,
                          name=self.name,
                          attrs=self.attrs)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.status = "error"
            self.err = repr(exc)
        dur = time.perf_counter() - self.t0 if self.t0 is not None else None
        self.emitter.emit(kind="span_end",
                          run_id=self.run_id,
                          span_id=self.span_id,
                          parent_span_id=self.parent_span_id,
                          name=self.name,
                          status=self.status,
                          error=self.err,
                          duration_s=dur)
        return False


# decorator: trace function calls
def trace_span(emitter: Emitter, name=None, run_id_fn=None, attrs_fn=None, include_args=True):
    # wraps a function and emits span_start/span_end; works for sync functions
    def deco(fn):
        span_name = name or f"func:{fn.__name__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # rid = run_id_fn(
            #     *args, **kwargs) if callable(run_id_fn) else kwargs.get("run_id") or eid()
            rid = _resolve_run_id(run_id_fn, args, kwargs)
            # attrs = attrs_fn(*args, **kwargs) if callable(attrs_fn) else {}
            attrs = _call_hook(attrs_fn, args, kwargs,
                               default={}) if callable(attrs_fn) else {}
            if include_args and emitter.cfg.include_args:
                try:
                    sig = inspect.signature(fn)
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    argmap = {k: v for k, v in bound.arguments.items()}
                except Exception:
                    argmap = {"args_len": len(
                        args), "kwargs_keys": list(kwargs.keys())}
                argmap = redact_obj(argmap, emitter.cfg.redact_keys)
                attrs = {**attrs, "args": argmap}
            with Span(emitter, rid, span_name, attrs=attrs):
                return fn(*args, **kwargs)

        return wrapper
    return deco


# decorator: async trace
def trace_span_async(emitter: Emitter, name=None, run_id_fn=None, attrs_fn=None, include_args=True):
    # wraps an async function and emits span events; no-op if asyncio not present
    def deco(fn):
        if not asyncio or not inspect.iscoroutinefunction(fn):
            return trace_span(emitter, name=name, run_id_fn=run_id_fn, attrs_fn=attrs_fn, include_args=include_args)(fn)

        span_name = name or f"afunc:{fn.__name__}"

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            # rid = run_id_fn(
            #     *args, **kwargs) if callable(run_id_fn) else kwargs.get("run_id") or eid()
            rid = _resolve_run_id(run_id_fn, args, kwargs)
            # attrs = attrs_fn(*args, **kwargs) if callable(attrs_fn) else {}
            attrs = _call_hook(attrs_fn, args, kwargs,
                               default={}) if callable(attrs_fn) else {}
            if include_args and emitter.cfg.include_args:
                try:
                    sig = inspect.signature(fn)
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    argmap = {k: v for k, v in bound.arguments.items()}
                except Exception:
                    argmap = {"args_len": len(
                        args), "kwargs_keys": list(kwargs.keys())}
                argmap = redact_obj(argmap, emitter.cfg.redact_keys)
                attrs = {**attrs, "args": argmap}
            with Span(emitter, rid, span_name, attrs=attrs):
                return await fn(*args, **kwargs)

        return wrapper
    return deco


# counter for exceptions by type
def count_exceptions(emitter: Emitter, name=None, run_id_fn=None):
    # increments a counter-like event on exception
    def deco(fn):
        ename = name or f"exc:{fn.__name__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            rid = run_id_fn(
                *args, **kwargs) if callable(run_id_fn) else kwargs.get("run_id") or eid()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                emitter.emit(kind="counter", run_id=rid,
                             name=ename, exc_type=type(e).__name__)
                raise

        return wrapper
    return deco


# measured block context for ad-hoc timings
class MeasuredBlock(ContextDecorator):
    # emit a generic metric with duration
    def __init__(self, emitter: Emitter, run_id, name, attrs=None):
        self.emitter = emitter
        self.run_id = run_id
        self.name = name
        self.attrs = attrs or {}
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self.t0 if self.t0 is not None else None
        self.emitter.emit(kind="metric", run_id=self.run_id,
                          name=self.name, duration_s=dur)
        return False


# example tools using decorators
def example_setup():
    cfg = TelemetryConfig(
        sink_path=RAW / "07_instrumented.jsonl",
        sample=1.0,
        include_args=True,
        redact_keys={"secret", "password"},
    )
    return Emitter(cfg)


def flaky_add(a, b, fail_p=0.1):
    # simple function that sometimes fails
    if random.random() < fail_p:
        raise RuntimeError("simulated failure")
    time.sleep(random.uniform(0.01, 0.05))
    return a + b


def run_examples():
    emitter = example_setup()

    @trace_span(emitter, name="tool:flaky_add", run_id_fn=lambda **kw: kw.get("run_id"))
    @count_exceptions(emitter, name="exceptions:flaky_add", run_id_fn=lambda **kw: kw.get("run_id"))
    def wrapped_add(a, b, run_id=None):
        return flaky_add(a, b)

    rid = eid()
    results = []
    for i in range(12):
        try:
            results.append(wrapped_add(i, i + 1, run_id=rid))
        except Exception:
            results.append(None)
    emitter.flush()
    print("wrapped_add calls:", len(results), "successes:",
          sum(1 for x in results if x is not None))

    # ad-hoc timing
    with MeasuredBlock(emitter, rid, "block:batch_add", attrs={"n": 100}):
        s = 0
        for i in range(100):
            try:
                s += wrapped_add(i, i + 2, run_id=rid)
            except Exception:
                pass
    emitter.flush()
    print("batch sum computed")


# async example if asyncio available
async def run_async_example():
    emitter = example_setup()

    @trace_span_async(emitter, name="tool:async_sleep", run_id_fn=lambda **kw: kw.get("run_id"))
    async def async_sleep_ms(ms, run_id=None):
        await asyncio.sleep(ms / 1000.0)
        return ms

    rid = eid()
    vals = [10, 30, 20, 5]
    outs = []
    for v in vals:
        outs.append(await async_sleep_ms(v, run_id=rid))
    emitter.flush()
    print("async sleeps:", outs)


# overhead sanity check
def overhead_check(n=1000):
    emitter = example_setup()

    @trace_span(emitter, name="tool:no_op", run_id_fn=lambda **kw: kw.get("run_id"))
    def no_op(x, run_id=None):
        return x

    rid = eid()
    t0 = time.perf_counter()
    for i in range(n):
        no_op(i, run_id=rid)
    emitter.flush()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for i in range(n):
        pass
    t3 = time.perf_counter()

    traced = t1 - t0
    bare = t3 - t2
    overhead_us = (traced - bare) * 1e6 / max(1, n)
    print("estimated per-call overhead_us:", round(overhead_us, 2))


# integration with section-2 agent: minimal changes
def integrate_with_agent_example():
    emitter = example_setup()

    class Toolset:
        def __init__(self):
            pass

        @trace_span(emitter, name="tool:calc", run_id_fn=lambda **kw: kw.get("run_id"), attrs_fn=lambda **kw: {"x": kw.get("x"), "y": kw.get("y")})
        def calc(self, x, y, run_id=None):
            time.sleep(random.uniform(0.02, 0.08))
            if random.random() < 0.05:
                raise RuntimeError("calc failed")
            return {"value": x + y, "conf": round(random.uniform(0.6, 0.99), 2)}

        @trace_span(emitter, name="tool:search", run_id_fn=lambda **kw: kw.get("run_id"), attrs_fn=lambda **kw: {"q": kw.get("query")})
        def search(self, query, run_id=None):
            time.sleep(random.uniform(0.04, 0.12))
            if random.random() < 0.12:
                raise RuntimeError("search failed")
            return {"value": random.randint(1, 9), "conf": round(random.uniform(0.5, 0.95), 2)}

    rid = eid()
    tools = Toolset()

    # planner span using context manager
    with Span(emitter, rid, "planner:next_action", attrs={"task": "sum small"}):
        action = "calc"
        args = {"x": 2, "y": 3}

    ok = True
    try:
        if action == "calc":
            res = tools.calc(**args, run_id=rid)
        else:
            res = tools.search(query="fallback", run_id=rid)
    except Exception as e:
        ok = False
        res = None
        emitter.emit(kind="action_result", run_id=rid,
                     action=action, ok=False, attempts=1, error=repr(e))
    else:
        emitter.emit(kind="action_result", run_id=rid, action=action,
                     ok=True, attempts=1, data={"conf": res.get("conf")})
    finally:
        emitter.flush()
    print("agent integration ok:", ok)


if __name__ == "__main__":
    random.seed(7)
    run_examples()
    if asyncio:
        try:
            asyncio.run(run_async_example())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_async_example())
    overhead_check(n=2000)
    integrate_with_agent_example()
