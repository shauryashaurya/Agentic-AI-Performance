# 09_ollama_local_llm.py

# prereqs
#     install ollama and run the server in the background
#     pull a small model once, for example: ollama pull phi3:mini or ollama pull qwen2.5:0.5b or ollama pull llama3.2:1b
#     default config here uses phi3:mini, change MODEL_NAME below if you prefer another tiny model
#
# ensure sections 1–8 files exist if you want to reuse the same data folder
# run this sibling script directly
#     python 09_ollama_local_llm.py

# then fold events into the pipeline
#     python 03_event_schema_and_logging.py
#     python 04_metrics_basics.py
#     python 05_sequence_analysis.py
#     python 06_combined_dashboard.py


import json
import time
import uuid
import io
import sys
import socket
import random
from pathlib import Path
from datetime import datetime, timezone
from http.client import HTTPConnection

# configuration
HOST = "127.0.0.1"
PORT = 11434
MODEL_NAME = "phi3:mini"
TEMPERATURE = 0.2
TIMEOUT_S = 30
PROMPT = "Write three bullet points summarizing why instrumentation helps agent reliability."
SYSTEM = "You are a concise assistant. Keep answers short."


# paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw_events"
RAW.mkdir(parents=True, exist_ok=True)
SINK_PATH = RAW / "09_ollama.jsonl"


# time and ids
def now_iso():
    # utc iso-8601 with Z suffix
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def eid():
    # random uuid4
    return str(uuid.uuid4())


# simple jsonl sink
class JsonlSink:
    # append-only jsonl writer with small buffer
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


# emitter
class Emitter:
    # structured event emitter
    def __init__(self, sink_path):
        self.sink = JsonlSink(sink_path)

    def emit(self, **kwargs):
        evt = {"ts": now_iso(), "event_id": eid()}
        evt.update(kwargs)
        self.sink.write(evt)

    def flush(self):
        self.sink.flush()


# span helper
class Span:
    # emits span_start and span_end with duration and status
    def __init__(self, emitter, run_id, name, parent_span_id=None, attrs=None):
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


# connectivity checks
def is_port_open(host, port, timeout=2.0):
    # returns True if tcp port is accepting connections
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def ollama_tags(host=HOST, port=PORT, timeout=TIMEOUT_S):
    # query the model list to verify server health
    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", "/api/tags")
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            return {"ok": False, "status": resp.status, "body": data.decode("utf-8", errors="ignore")}
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        return {"ok": True, "models": obj.get("models", [])}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


# streaming generate
def ollama_stream_generate(model, prompt, system=None, temperature=0.2, host=HOST, port=PORT, timeout=TIMEOUT_S):
    # streams json lines from ollama /api/generate
    # yields dicts per chunk with at least keys: response, done, and timing fields near the end
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": {"temperature": float(temperature)}}
    if system:
        payload["system"] = system
    body = json.dumps(payload).encode("utf-8")

    conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", "/api/generate", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            text = resp.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ollama generate http {resp.status}: {text}")
        buff = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buff += chunk
            while b"\n" in buff:
                line, buff = buff.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                yield obj
    finally:
        try:
            conn.close()
        except Exception:
            pass


# per-token metrics during streaming
def stream_with_metrics(emitter, run_id, model, prompt, system=None, temperature=0.2):
    # returns the full text and emits per-token timing metrics
    pieces = []
    token_ix = 0
    last_ts = None
    first_token_latency_s = None
    start = time.perf_counter()

    with Span(emitter, run_id, "act:generate", attrs={"model": model, "temperature": temperature}):
        for obj in ollama_stream_generate(model=model, prompt=prompt, system=system, temperature=temperature):
            if "response" in obj:
                now = time.perf_counter()
                if last_ts is None:
                    first_token_latency_s = now - start
                else:
                    token_latency_s = now - last_ts
                    emitter.emit(kind="metric", run_id=run_id, name="token_latency_s",
                                 duration_s=token_latency_s, attrs={"token_index": token_ix})
                last_ts = now
                pieces.append(obj["response"])
                token_ix += 1
            if obj.get("done"):
                break

    total_s = time.perf_counter() - start
    text = "".join(pieces)
    tokens = max(1, token_ix)
    tps = tokens / total_s if total_s > 0 else 0.0
    emitter.emit(kind="metric", run_id=run_id, name="first_token_latency_s",
                 duration_s=(first_token_latency_s or 0.0))
    emitter.emit(kind="metric", run_id=run_id,
                 name="throughput_tokens_per_s", duration_s=tps)
    emitter.emit(kind="action_result", run_id=run_id, action="generate",
                 ok=True, attempts=1, data={"tokens": tokens})
    return text, {"tokens": tokens, "total_s": total_s, "tps": tps, "ftl_s": first_token_latency_s or 0.0}


# main
def main():
    emitter = Emitter(SINK_PATH)
    run_id = eid()

    if not is_port_open(HOST, PORT):
        emitter.emit(kind="run_start", run_id=run_id, task="ollama_generate")
        emitter.emit(kind="run_fail", run_id=run_id,
                     reason="ollama_port_closed")
        emitter.flush()
        print("ollama server not reachable on", f"{HOST}:{PORT}")
        print("start it and pull a tiny model first, e.g., 'ollama pull phi3:mini'")
        sys.exit(1)

    tag_info = ollama_tags()
    emitter.emit(kind="ollama_tags", run_id=run_id, info=tag_info)

    found = any(m.get("model") == MODEL_NAME for m in tag_info.get(
        "models", [])) if tag_info.get("ok") else False
    if not found:
        print("warning: model", MODEL_NAME,
              "not found in /api/tags. ollama should still auto-pull on first use if allowed.")

    emitter.emit(kind="run_start", run_id=run_id,
                 task="ollama_generate", attrs={"model": MODEL_NAME})
    try:
        text, stats = stream_with_metrics(
            emitter, run_id, model=MODEL_NAME, prompt=PROMPT, system=SYSTEM, temperature=TEMPERATURE)
        emitter.emit(kind="span_start", run_id=run_id, span_id=eid(),
                     name="postprocess", attrs={"chars": len(text)})
        # trivial postprocess
        summary_len = len(text.split())
        time.sleep(0.02)
        emitter.emit(kind="span_end", run_id=run_id, span_id=eid(),
                     name="postprocess", status="ok", duration_s=0.02)
        emitter.emit(kind="run_ok", run_id=run_id, steps=1, attempts=1)
        print("model:", MODEL_NAME)
        print("tokens:", stats["tokens"], "tps:", round(
            stats["tps"], 2), "ftl_s:", round(stats["ftl_s"], 3))
        print("output:")
        print(text.strip())
    except Exception as e:
        emitter.emit(kind="run_fail", run_id=run_id,
                     reason="generate_error", error=repr(e))
        print("generation failed:", repr(e))
    finally:
        emitter.flush()


if __name__ == "__main__":
    random.seed(23)
    main()
