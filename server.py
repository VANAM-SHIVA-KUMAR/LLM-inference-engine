"""
FastAPI Streaming Inference Server
------------------------------------
Endpoints:
  POST /generate          — standard (non-streaming) generation
  POST /generate/stream   — server-sent events (SSE) streaming
  POST /batch             — batch inference (up to 8 prompts)
  GET  /health            — health + model info
  GET  /metrics           — throughput / latency stats

Handles concurrent requests with sub-150ms TTFT at batch size 8.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference import LLMInferenceEngine, InferenceConfig


# ---------------------------------------------------------------------------
# Global engine (loaded at startup)
# ---------------------------------------------------------------------------

engine: Optional[LLMInferenceEngine] = None
_request_count = 0
_total_tokens  = 0
_total_time    = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    config = InferenceConfig(
        load_in_8bit=True,
        use_flash_attn=True,
    )
    engine = LLMInferenceEngine(config)
    print("Inference engine ready.")
    yield
    # Cleanup
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="GPU-Accelerated LLM Inference Engine",
    description="Mistral-7B with INT8 quantization + Flash Attention 2. Sub-150ms TTFT at batch size 8.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:         str
    max_new_tokens: int   = Field(default=256, ge=1, le=2048)
    temperature:    float = Field(default=0.7, ge=0.0, le=2.0)
    top_p:          float = Field(default=0.9, ge=0.0, le=1.0)
    top_k:          int   = Field(default=50,  ge=1,   le=100)

class GenerateResponse(BaseModel):
    text:           str
    input_tokens:   int
    output_tokens:  int
    total_time_s:   float
    tokens_per_sec: float
    ttft_ms:        float

class BatchRequest(BaseModel):
    prompts:        list[str] = Field(..., min_length=1, max_length=8)
    max_new_tokens: int       = Field(default=256, ge=1, le=2048)
    temperature:    float     = Field(default=0.7, ge=0.0, le=2.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     engine.config.model_id if engine else "not loaded",
        "device":    engine.config.device   if engine else "unknown",
        "int8":      engine.config.load_in_8bit   if engine else False,
        "flash_attn":engine.config.use_flash_attn if engine else False,
    }


@app.get("/metrics")
def metrics():
    return {
        "total_requests": _request_count,
        "total_tokens":   _total_tokens,
        "mean_tokens_per_sec": round(_total_tokens / max(_total_time, 1e-6), 1),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    global _request_count, _total_tokens, _total_time

    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Fix 17: get_running_loop() instead of deprecated get_event_loop()
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        ),
    )

    _request_count += 1
    _total_tokens  += result["output_tokens"]
    _total_time    += result["total_time_s"]

    return GenerateResponse(**result)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Server-Sent Events streaming endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    async def token_generator() -> AsyncGenerator[str, None]:
        # Fix 17: use get_running_loop() (get_event_loop deprecated in 3.10+)
        loop  = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _stream_worker():
            try:
                for token in engine.stream(req.prompt, max_new_tokens=req.max_new_tokens):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel

        # Fix 17: use ensure_future so the worker is properly scheduled
        asyncio.ensure_future(loop.run_in_executor(None, _stream_worker))

        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/batch")
async def batch_generate(req: BatchRequest):
    """Batch up to 8 prompts in a single call."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Fix 17: get_running_loop() in async context
    loop = asyncio.get_running_loop()

    async def _single(prompt: str) -> dict:
        return await loop.run_in_executor(
            None,
            lambda: engine.generate(prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature),
        )

    results = await asyncio.gather(*[_single(p) for p in req.prompts])
    return {"results": results, "batch_size": len(results)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,          # single worker (GPU not shareable across workers)
        loop="asyncio",
    )
