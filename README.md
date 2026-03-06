# GPU-Accelerated LLM Inference Engine

**Author:** Shiva Kumar Vanam
**GitHub:** [VANAM-SHIVA-KUMAR](https://github.com/VANAM-SHIVA-KUMAR) | **LinkedIn:** [shiva-kumar-vanam](https://linkedin.com/in/shiva-kumar-vanam) | **Portfolio:** [vanamshivakumar.vercel.app](https://vanamshivakumar.vercel.app)

> *I'm Shiva Kumar Vanam — an AI/ML Engineer from Hyderabad, India. I build systems where AI actually ships to production — fast, efficient, and reliable. This project was built to understand and demonstrate every optimization layer that makes LLM inference viable in real-world deployments, from quantization to Flash Attention to containerized APIs.*

---

## What This Project Does

Running a 7-billion parameter language model in production is expensive. A naive implementation uses 14GB of VRAM and generates tokens slowly. This project applies **3 optimization layers** to Mistral-7B to make it 3.2× faster and fit on consumer hardware.

### The 3 Optimizations

**1. INT8 Quantization (bitsandbytes)**
Model weights are normally stored as 16-bit floats (2 bytes per value). INT8 quantization converts them to 8-bit integers (1 byte). This halves VRAM usage — 14GB → 7GB — with minimal quality loss. The key insight is that during the actual matrix multiplication ("compute"), we temporarily upcast back to float16, so accuracy is preserved.

**2. Flash Attention 2**
Standard attention computes the full N×N attention matrix between all tokens. For a sequence of 1,000 tokens, that's 1,000,000 values — most of which waste memory. Flash Attention never materializes the full matrix. It processes attention in tiles that fit in GPU SRAM (L1 cache), making attention O(N) in memory instead of O(N²). Result: 3-4× faster attention, much less memory.

**3. KV-Cache Management**
When generating token #50, a naive model recomputes the keys and values for tokens 1-49 again. KV-cache stores those computations and reuses them. With proper cache management, each new token only requires computing attention for itself — O(1) per step instead of O(N²).

---

## Benchmark Results

| Configuration                   | Throughput   | TTFT    |
|---------------------------------|--------------|---------|
| Baseline (fp16, HuggingFace)    | ~18 tok/s    | ~480ms  |
| + INT8 Quantization             | ~32 tok/s    | ~280ms  |
| + Flash Attention 2             | ~48 tok/s    | ~180ms  |
| + KV-Cache (this project)       | **~58 tok/s**| **<150ms** |

**3.2× throughput improvement** over baseline. Tested on single A100 80GB, batch size 8.

---

## Architecture

```
HTTP Request  ──▶  FastAPI (async)
                        │
            ┌───────────┼───────────────┐
            │           │               │
       /generate  /generate/stream   /batch
       (standard)  (SSE real-time)  (up to 8)
            │           │               │
            └───────────┼───────────────┘
                        │
               LLMInferenceEngine
                        │
                ┌───────┴────────┐
                │  Mistral-7B    │
                │  ─────────── │
                │  INT8 Quant   │  ← 7GB VRAM (vs 14GB fp16)
                │  Flash Attn 2 │  ← O(N) memory attention
                │  KV-Cache     │  ← no recomputation
                └───────────────┘
```

---

## Quickstart

### Docker (Recommended)
```bash
# Set HuggingFace token (needed to download Mistral)
export HF_TOKEN=hf_...

docker-compose up --build

# Test it
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "[INST] What is Flash Attention? [/INST]", "max_new_tokens": 100}'
```

### Local (requires CUDA GPU)
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # needs CUDA headers

python server.py
# API docs: http://localhost:8080/docs
```

---

## API Reference

### POST /generate — Standard response
```json
{
  "prompt": "[INST] Your question [/INST]",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```
Set `"temperature": 0.0` for deterministic greedy decoding.

### POST /generate/stream — Token-by-token streaming (SSE)
Returns Server-Sent Events. Each event contains one token as it's generated.
```javascript
const es = new EventSource('/generate/stream');
es.onmessage = (e) => {
  if (e.data === '[DONE]') return es.close();
  process.stdout.write(e.data);  // print each token live
};
```

### POST /batch — Multiple prompts at once (up to 8)
```json
{ "prompts": ["Question 1", "Question 2", "..."], "max_new_tokens": 128 }
```

### GET /metrics — Throughput stats
```json
{ "total_requests": 42, "total_tokens": 8400, "mean_tokens_per_sec": 54.3 }
```

---

## Project Structure

```
llm-inference-engine/
├── inference.py        # Core optimizations — quantization, Flash Attn, KV-cache, streaming
├── server.py           # FastAPI server — standard, streaming, batch endpoints
├── Dockerfile          # CUDA 12.1 container
├── docker-compose.yml  # GPU-enabled deployment
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

| Setup           | Min VRAM | Works On                     |
|-----------------|----------|------------------------------|
| fp16 (baseline) | 16 GB    | A100, H100                   |
| INT8 (this)     | **8 GB** | RTX 3090, RTX 4090, A100, H100 |

---

## Bugs Fixed (Code Review Notes)

These were real bugs found and fixed during a senior engineering review:

| # | File | Bug | Fix Applied |
|---|------|-----|-------------|
| 15 | `inference.py` | **Critical**: `temperature = temperature or self.config.temperature` — Python's `or` treats `0.0` as falsy, so `0.0 or 0.7 → 0.7`. Greedy decoding (`temperature=0`) was **completely broken** — it silently defaulted to sampling | Changed to `if temperature is None: temperature = self.config.temperature` |
| 15b | `inference.py` | `do_sample=temperature > 0` only works correctly after the above fix — was always True before | Correct now that temperature=0.0 is preserved |
| 16 | `inference.py` | `StoppingCriteria`, `StoppingCriteriaList` imported but never used. `t_first_token = None` set but never used | Removed dead imports and dead variable |
| 17 | `server.py` | `asyncio.get_event_loop()` is **deprecated in Python 3.10+** inside async functions — raises DeprecationWarning, fails in 3.12+ | Changed to `asyncio.get_running_loop()` throughout |
| 17b | `server.py` | In the SSE streaming generator, `loop.run_in_executor(None, _stream_worker)` was not awaited — the worker was scheduled but the coroutine was silently dropped | Wrapped with `asyncio.ensure_future()` so it's properly scheduled |

---

## Tech Stack

- **Mistral-7B-Instruct-v0.2** — base model
- **bitsandbytes** — INT8 quantization
- **flash-attn** — Flash Attention 2
- **HuggingFace Transformers** — model loading + generation pipeline
- **FastAPI + uvicorn** — async REST API
- **Docker + NVIDIA Container Toolkit** — GPU-enabled containerization
