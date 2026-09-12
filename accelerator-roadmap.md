# From Hugging Face to custom silicon — a 15-week plan

**Start:** week of 14 Sept 2026 · **Target:** interview-ready by 21 Dec 2026
**Goal order:** deep understanding → portfolio project → open-source contribution

---

## Ground rules

1. **One project runs through all 15 weeks.** The "hardware portability harness" (see below). Every week adds a column, a backend, or a measurement to it. Don't start side projects.
2. **Always measure accuracy alongside speed.** A 3x speedup that quietly costs 4 points of F1 is a bug, not a win. This habit is the single thing that will separate you from other candidates.
3. **Write as you go.** A short note per week — what you measured, what surprised you. Two of these become blog posts. Blog posts become interview material.
4. **Rent, don't buy.** Colab free tier for months 1–2. RunPod/Vast.ai by the hour in month 3 when you need a real GPU for Triton benchmarking.

---

## The project: hardware portability harness

A repeatable benchmark suite that runs the *same fine-tuned model* across every execution path you can reach, and reports latency, throughput, memory, and accuracy drift in one table.

Backends to support by the end:

| Path | Encoder | Decoder |
|---|---|---|
| PyTorch eager, CPU | ✓ | ✓ |
| PyTorch eager, MPS (M1) | ✓ | ✓ |
| PyTorch eager, CUDA | ✓ | ✓ |
| `torch.compile` (Inductor) | ✓ | ✓ |
| ONNX Runtime FP32 / INT8 | ✓ | — |
| llama.cpp / GGUF quantized | — | ✓ |
| vLLM | — | ✓ |

Plus a CI job that flags regressions. This is the deliverable you walk into interviews with.

---

## Month 1 — Encoder, end to end (weeks 1–4)

### Week 1 — PyTorch mechanics and honest measurement

- The ATen dispatcher: dispatch keys, kernel registration, why `.to(device)` is the only thing that moves work.
- Async execution: why `time.time()` around a CUDA call lies, and what `torch.cuda.synchronize()` is for.
- `torch.profiler` — read a trace, find where time actually goes.
- Read: Edward Yang's "PyTorch internals" blog post. Twice.

**Deliverable:** `bench.py` that times DistilBERT inference correctly on CPU and MPS, with warmup, synchronization, and p50/p95 latency.

### Week 2 — Fine-tune and build the eval

- Pick one dataset and commit: **Financial PhraseBank** (fast, clean, 4.8k sentences) or **Hallmarks of Cancer / HoC** (matches your medical interest, harder).
- Model: start `distilbert-base-uncased`. Stretch: `deberta-v3-base` or ModernBERT.
- LoRA with PEFT, then **merge the adapter** — `merge_and_unload()`. Understand why an unmerged adapter hurts downstream compilation.
- Build a held-out eval set with a single command that prints accuracy and macro-F1.

**Deliverable:** fine-tuned merged model + `eval.py`. This eval script is now the referee for everything that follows.

### Week 3 — Export and quantize

- Optimum: `ORTModelForSequenceClassification`, `optimum-cli export onnx`.
- Look at the exported graph in Netron. Find the ops. See what got fused.
- ONNX Runtime dynamic INT8 quantization, then static (calibration-based). Compare.
- Learn the difference: post-training quantization vs quantization-aware training; per-tensor vs per-channel scales.

**Deliverable:** harness v1 — a table with eager / ONNX FP32 / ONNX INT8 rows and an accuracy column.

### Week 4 — `torch.compile` on the encoder

- `torch.compile(model)` on CPU first. `TORCH_COMPILE_DEBUG=1` and actually read the generated C++.
- `torch._dynamo.explain()` — find your graph breaks. Fix at least one.
- Move to Colab, compile on CUDA, read the generated **Triton** kernels. This is your first contact with Triton.
- Understand compile modes: `default`, `reduce-overhead` (CUDA graphs), `max-autotune`.
- Understand recompilation: dynamic shapes, guards, why batch size 1 and 8 may compile twice.

**Deliverable:** harness v2 with compile rows. **Write-up #1:** "What actually happens when you `torch.compile` a BERT."

---

## Month 2 — Decoder and serving (weeks 5–9)

### Week 5 — Decoder mechanics from scratch

- Write your own generation loop by hand: no `.generate()`. Greedy decoding, then a KV cache you built yourself.
- Feel the difference between prefill (compute-bound, big matmuls) and decode (memory-bandwidth-bound, one token at a time).
- Roofline thinking: arithmetic intensity, why decode is bandwidth-limited, why batching helps decode but not prefill.
- Model: `granite-3.x-2b` or `Llama-3.2-1B`. Small enough for Colab and your M1.

**Deliverable:** a hand-rolled `generate()` with KV cache, benchmarked against HF's.

### Week 6 — LLM quantization

- Weight-only quantization: GPTQ, AWQ, bitsandbytes NF4. Why weight-only works so well for decode.
- GGUF and llama.cpp on your M1 — Q4_K_M, Q8_0. Measure tokens/sec and memory.
- Build an LLM eval that isn't vibes: perplexity on a held-out set, plus a small task-specific accuracy check.

**Deliverable:** quantization comparison table for the decoder, with quality numbers.

### Week 7 — vLLM

- Continuous batching vs static batching. PagedAttention and why KV cache fragmentation matters.
- Run vLLM on Colab. Sweep concurrency, plot the latency-vs-throughput curve. Find the knee.
- Understand TTFT vs ITL (time to first token vs inter-token latency) — the two SLAs customers actually care about.
- Read IBM's `vllm-spyre` plugin structure. You now have the context to understand it.

**Deliverable:** a latency/throughput curve chart. This chart is interview gold.

### Week 8 — Compilation for decoders

- Why static shapes matter to accelerators, and how LLM serving fakes them (bucketing, padding, CUDA graphs).
- `torch.compile` + CUDA graphs on a decoder. Measure the CPU-overhead win at small batch sizes.
- Read how vLLM integrates `torch.compile`, and why IBM chose `torch.compile` as Spyre's frontend rather than ONNX.

**Deliverable:** harness v3 — decoders included, all backends.

### Week 9 — Consolidate

- Clean up the repo. README with the results table. Reproducible setup.
- **Write-up #2:** "Encoder vs decoder: why the same optimization playbook doesn't work twice."

---

## Month 3 — Kernels, backends, and contribution (weeks 10–15)

### Weeks 10–11 — Triton

Work the official Triton tutorials in order, benchmarking each against the PyTorch native op:

1. Vector add — the programming model: program IDs, blocks, masks.
2. Fused softmax — why fusion wins, and row-wise reduction.
3. Matmul — tiling, accumulators, `tl.dot`, autotuning configs.
4. Fused attention — the FlashAttention idea, made concrete.

Rent a 4090 or A100 by the hour for benchmarking. Consumer GPU is fine; you're learning, not setting records.

**Deliverable:** a repo of Triton kernels with benchmark plots vs `torch.*` equivalents.

### Week 12 — Your own `torch.compile` backend

- Write a backend that just prints the FX graph it receives. Understand what a "graph" actually contains.
- Then one that counts ops, or rewrites one op into another.
- Then one that routes the graph to your Triton kernels.
- Understand `aot_autograd`, decompositions, and the "core ATen opset" — the ~180-op surface a new backend has to cover.

**Deliverable:** a working custom backend, in pure Python, with a blog-post-length explanation.

### Week 13 — Fake accelerator, and reading torch-spyre

- Register a `PrivateUse1` device in Python with `torch.library`. Implement a handful of ops backed by CPU code. Move a tensor to `"myaccel"` and run something.
- Read PyTorch's OpenReg reference implementation — IBM has stated it plans to upstream OpenReg primitives so out-of-tree device testing becomes first-class.
- Read the `torch-spyre` Python frontend. Map what you see onto what you just built.
- Skim KTIR (IBM's MLIR-based tile IR) conceptually. Don't try to master MLIR — just understand what a tile-level IR is *for*.

**Deliverable:** a toy accelerator backend + a written comparison to how torch-spyre does it.

### Week 14 — Contribute something real

Targets, easiest first:

- Documentation gaps in `torch-spyre`, `vllm-spyre`, or IBM's `foundation-model-stack`.
- A benchmark or test contribution — IBM is explicitly building an out-of-tree CI test pyramid and says they want patterns other accelerator teams can adopt.
- A Triton kernel benchmark comparison.
- A well-researched issue with a reproduction. This counts more than people think.

Ask your mentor to point you at something the team actually wants done. That conversation is worth more than a month of guessing.

**Deliverable:** one merged PR or one substantive issue. Plus the conversation with your mentor.

### Week 15 — Interview preparation

Rehearse three formats out loud:

1. **Deep dive on your own work.** Twenty minutes on the harness. Every claim backed by a number you measured.
2. **System design, FDE-flavoured.** "A hospital customer has a clinical classifier, an on-prem accelerator, a 50ms p99 SLA, and can't send data to the cloud. Design the deployment." Practice the tradeoff reasoning out loud.
3. **Fundamentals.** Explain the dispatcher, explain a graph break, explain why decode is bandwidth-bound, explain what quantization costs you.

---

## Explicitly out of scope

Skipping these is a decision, not an oversight:

- **C++.** Not needed for any of the above. Revisit only if you join a hardware team.
- **MLIR internals.** Understand what an IR is for; don't learn the dialect system.
- **Raw CUDA C++.** Triton covers the concepts at a fraction of the cost.
- **Pretraining / large-scale distributed training.** Different job family.
- **TPU/XLA and JAX.** Interesting, but it's a second ecosystem. Finish this one first.

---

## Core resources

- **PyTorch internals** — Edward Yang's blog post (the dispatcher explanation).
- **`torch.compile` docs** — the "torch.compile, the missing manual" doc, and the Dynamo deep-dive tutorials.
- **Triton tutorials** — official docs, work them in order.
- **vLLM docs** — the PagedAttention paper, then the architecture docs.
- **IBM Research blog** — "Building PyTorch-native support for the IBM Spyre Accelerator" and the PyTorch Conference posts.
- **Repos** — `IBM/torch-spyre`, `vllm-project/vllm-spyre`, `foundation-model-stack`, `pytorch/pytorch` (`torch/_dynamo`, `torch/_inductor`).

---

## Checkpoints

| Date | You should have |
|---|---|
| Mid Oct | Fine-tuned encoder, ONNX + INT8 + compiled, harness v1, write-up #1 |
| Mid Nov | Decoder path, vLLM benchmarks, harness v3, write-up #2 |
| Mid Dec | Triton kernels, custom backend, one contribution, rehearsed answers |
