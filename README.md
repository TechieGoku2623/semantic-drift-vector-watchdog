# Semantic Drift Vector Watchdog

A high-throughput, low-latency asynchronous engine engineered to resolve silent centroid drift in 32-dimensional embedding windows by maintaining a Welford reference mean and declaring drift with a two-sided CUSUM on cosine distance.

## 🏗️ Systems Architecture & Event Topology

`SemanticDriftVectorWatchdog` keeps a 32-wide reference centroid under an `asyncio.Lock`. `run(records)` validates a batch, scores each vector, updates a two-sided CUSUM, and returns a JSON-serializable dict: `cosine_distance`, `l2`, `cusum_pos`, `cusum_neg`, `drift`, and `centroid_updates`.

The in-process `asyncio.Queue` is the stand-in for Kafka topic `genai.embeddings.window`. Each accepted batch is packed with a little-endian header, `struct` format `<HHI`: dimension, count, and flags. A `<ffff` score tail carries cosine distance, L2, and both CUSUM accumulators. The queue is bounded; a full queue drops the oldest header before the new one is published.

`EngineKernelException` is the kernel fault. It is raised for an all-zero vector, a non-finite component, or a dimension other than 32, and the centroid is left unchanged.

## 📊 Core Visual Walkthrough & Engine Pipeline Flow

```
records (N vectors, dim 32)
        |
        v
finite + dimension + zero gate ---- all-zero --> EngineKernelException
        |
        v
cold start? -- yes --> Welford seed, distance 0
        |
        no
        v
cosine distance and L2 versus the Welford mean
        |
        +--> two-sided CUSUM (allowance k, threshold h)
        |
        +--> in-control vector --> fold into the mean
        |
        +--> out-of-control vector --> withhold from the mean
        |
        v
header on genai.embeddings.window
        |
        v
{cosine_distance, l2, cusum_pos, cusum_neg, drift, centroid_updates}
```

Insert the structural terminal walkthrough recording at docs/assets/terminal-walkthrough.gif before publishing the release notes.

## ⚡ Low-Level OS Mechanics & Network Physics

Cosine similarity is the dot product divided by the product of Euclidean norms. Both norms and the L2 residual use `math.sqrt` on summed squares. Similarity is clamped to `[-1, 1]` so a round-off past 1.0 cannot produce a negative distance. Cosine distance is `1 - similarity`.

The reference mean is Welford's online update, component by component:

```
n = n + 1
delta = x - mean
mean = mean + delta / n
m2 = m2 + delta * (x - mean)
```

The second moment stays available for a variance term on the drift log line. No BLAS call and no extra process are on the hot path. The only shared-state critical section is the `asyncio.Lock` around the centroid, the CUSUM pair, and the queue.

The tabular CUSUM, with target `mu` defaulting to 0, allowance `k` defaulting to 0.10, and threshold `h` defaulting to 1.50, is:

```
C+ = max(0, C+ + (distance - mu) - k)
C- = max(0, C- - (distance - mu) - k)
drift = C+ > h or C- > h
```

A vector whose cosine distance is above `mu + k` is withheld from the mean. A near-orthogonal burst therefore keeps a stable reference, and the positive accumulator crosses `h` inside the configured window (default 8). `statistics.fmean` and `statistics.pstdev` summarize that window when drift latches.

## ⚖️ Architecture Trade-offs & Pragmatic Decisions

The detector uses a fixed allowance and a fixed threshold. Operators who need a different band pass `allowance_k`, `threshold_h`, `window`, and `target_mu` to the constructor. The harness does not sweep them.

Folding every vector, including the orthogonal ones, would drag the centroid toward the fault and shrink the cosine distance before the CUSUM could accumulate. Withholding out-of-control samples keeps the reference as the in-control Welford mean. A slow in-band wander still updates the mean, because those distances sit under the allowance.

The first vector in the life of the engine seeds the centroid and reports distance 0. Drift is meaningful only after that seed exists. The negative CUSUM side is updated on every scored sample so a distance below the target is visible; with a target of 0 and non-negative cosine distance it stays at 0 while the positive side carries the alarm.

## 🚀 Local Installation & Benchmarking

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python -m semantic_drift_vector_watchdog
python -m semantic_drift_vector_watchdog.harness
```

```python
import asyncio

from semantic_drift_vector_watchdog import SemanticDriftVectorWatchdog


async def demo() -> None:
    engine = SemanticDriftVectorWatchdog()
    baseline = [0.0] * 32
    baseline[0] = 1.0
    result = await engine.run([baseline])
    print(result)


asyncio.run(demo())
```

The runtime is the Python 3.12 standard library. `pip install -r requirements.txt` succeeds with comments only. Install the package with `pip install .`.

## 🖥️ Terminal Diagnostic Output Preview

```
2026-10-02T02:54:16+0000 INFO [semantic.drift.watchdog] centroid topic=genai.embeddings.window cache=memory
2026-10-02T02:54:16+0000 INFO [semantic.drift.watchdog] withheld out-of-control vector topic=genai.embeddings.window cosine=0.960032
2026-10-02T02:54:16+0000 WARNING [semantic.drift.watchdog] CUSUM drift topic=genai.embeddings.window cosine=0.960032 window_mean=0.240012 spread=0.415704 var=0.000000 cusum_pos=1.720064 cusum_neg=0.000000
2026-10-02T02:54:16+0000 INFO [semantic.drift.watchdog] scenario complete drift=True cusum_pos=6.880256 updates=20
{'cosine_distance': 0.9600319803881706, 'l2': 1.385659965368828, 'cusum_pos': 6.880255843105366, 'cusum_neg': 0.0, 'drift': True, 'centroid_updates': 20}
```

`python -m semantic_drift_vector_watchdog` exits 0. Logging is configured only in `__main__`, with format `%(asctime)s %(levelname)s [%(name)s] %(message)s` and date format `%Y-%m-%dT%H:%M:%S%z`.

## 📊 Empirical Benchmarking Performance Report

Measured by `python -m semantic_drift_vector_watchdog.harness` with seed `20261002`, 5000 iterations, `perf_counter_ns` latency in microseconds, and `tracemalloc` peak. The harness prints this status dict and exits 0 only when every edge passes:

```
{'status': 'ok', 'failures': 0, 'latency_us': 71.155, 'memory_peak_bytes': 188088, 'benchmark_iterations': 5000, 'benchmark_avg_us': 71.155, 'benchmark_p99_us': 110.891}
```

| Metric | Measured |
| --- | ---: |
| Status | ok |
| Failures | 0 |
| Iterations | 5000 |
| Average latency | 71.155 µs |
| Empirical P99 | 110.891 µs |
| tracemalloc peak | 188088 bytes |
| Edge: all-zero vector | pass |
| Edge: near-orthogonal CUSUM | pass |

## 🛡️ Edge-Case Resilience & SOC2/Regulatory Compliance

An all-zero vector raises `EngineKernelException` before any Welford update. A following in-control vector increments `centroid_updates` by one, which is the evidence that the zero vector did not move the reference.

A planted near-orthogonal vector, a unit vector with a 0.04 residual on the reference axis and its mass on the next axis, produces cosine distance about 0.96. The positive CUSUM crosses `h` on the second presentation, which is inside the window of 8. Those samples are withheld from the centroid.

The vectors are caller-supplied floats. The module scores them and does not fetch prompts. SOC 2 processing integrity is the control: a rejected vector does not move the centroid, a drift header is queued on `genai.embeddings.window`, and `run` returns a JSON-serializable state consistent with the last accepted batch.
