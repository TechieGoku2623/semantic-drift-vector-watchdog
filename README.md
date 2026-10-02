# Semantic Drift Vector Watchdog

A high-throughput, low-latency asynchronous engine engineered to resolve silent centroid drift in 32-dimensional embedding windows by scoring cosine distance and L2 distance with the standard math library and alerting when a window leaves the centroid.

## 🏗️ Systems Architecture & Event Topology

`SemanticDriftWatchdog` holds one centroid, a plain list of 32 floats, guarded by an `asyncio.Lock`. `score_window` is the coroutine that accepts a batch of vectors. The in-process `asyncio.Queue` stands in for the Kafka topic `genai.embeddings.window`. `drain_headers` pulls the packed alert headers a producer would publish. `stats` reports the running distances.

A Redis structure would replace the list when several workers must share a centroid. This process does not open that connection. `logging.basicConfig` timestamps every line. A zero vector, a non-finite component, or a vector whose dimension is not 32 raises `EngineKernelException` and does not move the centroid.

Cosine and L2 are implemented with `math` only. There is no native BLAS call on the hot path.

## 📊 Core Visual Walkthrough & Engine Pipeline Flow

```
embedding window (N vectors, dim 32)
    |
    v
finite and dimension gate
    |
    +-- zero vector -----> isolate, refuse centroid update, warning
    |
    v
cosine distance to centroid
L2 distance to centroid
    |
    +-- distance > threshold (default 0.35) --> alert, header on the queue
    |
    v
centroid update under asyncio.Lock
    |
    v
topic genai.embeddings.window
```

Insert the structural terminal walkthrough recording at docs/assets/terminal-walkthrough.gif before publishing the release notes.

## ⚡ Low-Level OS Mechanics & Network Physics

Cosine similarity is the dot product divided by the product of Euclidean norms. Both norms are `math.hypot` accumulated across the 32 lanes, which stays accurate when a component is near zero. Cosine distance is one minus that similarity, clamped so a round-off of 1.0000001 does not go negative. L2 is the hypot of the component-wise difference. An orthogonal planted vector has cosine distance 1 against a centroid on a different axis, which is the alert the harness asserts.

The centroid write is the only critical section. Scoring a window allocates the distance pair and nothing else that grows with vocabulary size. `statistics` summarizes the window's distance series for the log line. The queue bound keeps a slow consumer from pinning the process; `drain_headers` is how the scenario observes that the alert was actually queued.

## ⚖️ Architecture Trade-offs & Pragmatic Decisions

A learned drift detector would need a reference corpus and a threshold fit on that corpus. The watchdog uses a fixed cosine-distance threshold (default 0.35) and a fixed window (default 8). The threshold is the contract. Operators who need a different band pass it to the constructor; the harness does not sweep it.

The centroid is a running mean, not a reservoir sample. A reservoir would resist a slow poison and would also hide a real topic shift until the reservoir turned over. The running mean moves immediately, and the alert fires on the same window that moved it. Zero vectors are excluded from that mean so a padding row cannot drag every lane toward the origin.

## 🚀 Local Installation & Benchmarking

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
python src/test_harness.py
```

```python
import asyncio

from src.main import SemanticDriftWatchdog


async def demo() -> None:
    watchdog = SemanticDriftWatchdog(cosine_distance_threshold=0.35, window=8)
    baseline = [0.0] * 32
    baseline[0] = 1.0
    orthogonal = [0.0] * 32
    orthogonal[1] = 1.0
    await watchdog.score_window([baseline, baseline, orthogonal])


asyncio.run(demo())
```

The runtime is the Python 3.12 standard library. `pip install -r requirements.txt` succeeds without a dependency index fetch.

## 🖥️ Terminal Diagnostic Output Preview

```
INFO semantic.drift.watchdog centroid cache=memory topic=genai.embeddings.window
INFO semantic.drift.watchdog isolated 1 zero vector(s) from a mixed window
WARNING semantic.drift.watchdog cosine distance 1.000 crossed 0.350 on genai.embeddings.window
WARNING semantic.drift.watchdog zero vector isolated; refusing centroid update
WARNING semantic.drift.watchdog non-finite component rejected; centroid unchanged
INFO semantic.drift.watchdog scenario complete alerts=1 drained=5
```

`python src/main.py` exits 0. The stdout dict reports `max_cosine_distance` of 1.0 and `zero_rejected` true.

## 📊 Empirical Benchmarking Performance Report

Measured by `python src/test_harness.py` with a deterministic seed, 5000 iterations, `time.perf_counter_ns` latency in microseconds, and `tracemalloc` peak.

| Metric | Measured |
| --- | ---: |
| Status | PASS |
| Iterations | 5000 |
| Average latency | 489.07 µs |
| Empirical P99 | 834.85 µs |
| tracemalloc peak | 235444 bytes |
| Edge: zero vector | PASS |
| Edge: orthogonal shift | PASS |

## 🛡️ Edge-Case Resilience & SOC2/Regulatory Compliance

A zero vector is isolated, logged, and refused as a centroid update. A planted orthogonal vector produces cosine distance 1.0 and crosses the 0.35 threshold, which raises the alert path. A non-finite component raises `EngineKernelException` and leaves the centroid unchanged.

The vectors are caller-supplied floats. The module does not fetch embeddings and does not log prompt text. SOC 2 processing integrity is the control: an alert header is queued, a rejected vector does not move the centroid, and `stats` remains consistent with the last accepted window.
