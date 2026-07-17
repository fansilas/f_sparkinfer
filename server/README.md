# sparkinfer-server

OpenAI-compatible HTTP API for local GGUF inference — backend for sparkinfer.com.

Enable with `-DBUILD_SERVER=ON` when building this repo (`dev` branch).

## Build

Requires **Rust/cargo** (build-time only) for HuggingFace `tokenizer.json` via [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build -j$(nproc) --target sparkinfer_server
```

Or reuse the bench harness build root:

```bash
bench/scripts/_common.sh  # optional: sets ARCH
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build --target sparkinfer_server
```

## Run

```bash
export SPARKINFER_ROOT="$(pwd)"
# Native C++ tokenizer (tokenizers-cpp). Requires rustc/cargo at build time only.

# download model on first bench run, or:
# bench/scripts/bench.sh --download

# Default: unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_M (~22 GB)
# https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
./server/run.sh --download
# or:
./build/server/sparkinfer_server -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8080
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/models` | OpenAI model list |
| `GET /v1/info` | Model limits (`max_context`, `max_output_tokens`) |
| `POST /v1/tokenize` | Token count for a chat request body |
| `POST /v1/chat/completions` | Chat (JSON `messages`, optional `stream`, `enable_thinking`). Responses include OpenAI `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`). Streaming sends a final chunk with `choices:[]` + `usage` before `[DONE]`. |

### RTX PRO 6000 deploy (32k / 4k)

See [`changelog-pro6000.md`](../changelog-pro6000.md) for the full 5090→PRO 6000 migration
notes and benchmark table.

```bash
export CTX=36864          # 32k prompt + 4k completion KV pool
export HOST=0.0.0.0
./server/run.sh --download
curl -s http://127.0.0.1:8080/v1/info
# {"model":"qwen3.6-35b-a3b","max_context":32768,"max_output_tokens":4096}
```

On RTX 5090 (32 GB) use a smaller `--ctx` (8k–16k) or `CTX=0` for GGUF defaults — the
same binary, different memory budget.

### Example

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sparkinfer","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

With API key (optional):

```bash
./build/server/sparkinfer_server -m model.gguf --api-key secret
curl ... -H 'Authorization: Bearer secret'
```

## Request isolation

Each `/v1/chat/completions` call runs through `Qwen35Model::generate()`:

- Fresh KV allocation per request (`kv->allocate` / `kv->free` inside `generate()`)
- Hybrid Gated-DeltaNet recurrent state reset at position 0 on cold prompts
- Correct prefill path (interior prompt tokens skip LM head unless `SPARKINFER_PREFILL_LEGACY=1`)
- Optional **shared prefix cache**: set `SPARKINFER_SERVER_PREFIX_TOKEN_FILE` (JSON array of token ids)
  or `SPARKINFER_SERVER_PREFIX_TOKEN_IDS` (comma-separated). When the chat prompt starts with those
  tokens, the server calls `cache_prefix()` (batched prefill) before `generate()`, which only
  token-loops the suffix.

Prior requests cannot leak decode context into later ones (KV is freed after each `generate()`).

## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_ROOT` | `.` | Repo root (tokenizer script path) |
| `CTX` | `36864` (PRO 6000) / `0` (5090) | KV pool size passed as `--ctx` |
| `SPARKINFER_KV_INT8` | model-dependent | Same as `qwen3_gguf_generate` |
| `SPARKINFER_TOKENIZER_URL` | Qwen3-30B tokenizer | Override tokenizer download |
| `SPARKINFER_SERVER_PREFIX_TOKEN_FILE` | — | JSON `[id,...]` warmed via `cache_prefix` each request |
| `SPARKINFER_SERVER_PREFIX_TOKEN_IDS` | — | Comma-separated token ids (same as above) |
| `SPARKINFER_PREFILL_BATCHED` | `1` | Batched prefill in `cache_prefix` / cold `generate` |
