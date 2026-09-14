# vLLM Benchmark Analysis: GuideLLM Evaluation

## 1. Overview of the Benchmark Setup

The benchmark evaluates the serving performance of a local **vLLM** deployment serving `Qwen3-0.6B` using **GuideLLM**.

* **Target URL:** `http://localhost:8000`
* **Model:** `Qwen/Qwen3-0.6B`
* **Benchmark Strategy:** `--profile synchronous`
* **Request Count:** 10 requests (`--max-requests 10`)
* **Workload Dimensions:**
* Prompt tokens: 32
* Output tokens: 16
* Samples: 32



---

## 2. Observed Latency & Throughput Metrics

| Metric | Mean | p50 (Median) | p95 | p99 |
| --- | --- | --- | --- | --- |
| **TTFT (Time to First Token)** | 68.40 ms | 58.19 ms | 132.13 ms | 132.13 ms |
| **ITL (Inter-Token Latency)** | 44.77 ms | 44.40 ms | 47.01 ms | 47.01 ms |
| **E2E Latency** | 0.74 s | 0.72 s | 0.84 s | 0.84 s |
| **Output Token Count** | 16.00 tokens | 16.00 tokens | 16.00 tokens | 16.00 tokens |

* **Request Throughput:** `1.35 req/s`
* **Token Throughput:** `21.9 output tokens/s`

---

## 3. Explaining the 1.35 req/s Throughput

### Mathematical Derivation

In a synchronous evaluation profile, concurrency is fixed at 1. Each request is dispatched only after the preceding request finishes:

$$\text{Throughput (req/s)} \approx \frac{1}{\text{Mean E2E Latency}} = \frac{1}{0.74\text{ s}} \approx 1.35\text{ req/s}$$

### Output Token Rate Connection

With each request fixed to return 16 output tokens:

$$\text{Token Rate} = 1.35\text{ req/s} \times 16\text{ tokens/req} \approx 21.6\text{ to } 21.9\text{ output tokens/s}$$

### Key Architectural Takeaways

* **Single-Stream Serialization:** Because requests run serially, this number represents the inverse of single-stream latency, not maximum hardware or engine capacity.
* **Continuous Batching Underutilization:** vLLM's core strengths—continuous batching, dynamic memory paging (PagedAttention), and concurrent execution—are dormant at a concurrency factor of 1.
* **Production Contrast:** To evaluate true system saturation and peak request handling, workloads must be benchmarked using concurrent profiles (e.g., constant arrival rates, varying concurrency steps, or larger request counts).
