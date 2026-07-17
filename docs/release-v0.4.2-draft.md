sparkinfer now **beats llama.cpp on Qwythos prefill at every tracked context** — **2.18× faster at 64k**
(17,772 vs 8,154 pp tok/s) after a **55×** climb since v0.4.1. The first public demo is live at
**[sparkinfer.com](https://sparkinfer.com/)** with an OpenAI-compatible API at
**[api.sparkinfer.com](https://api.sparkinfer.com/)**. Qwen3.6 decode frontier holds at **473 tok/s (+71%)**.
Attested **Linux + Windows** binaries attached.

## Qwythos prefill — 2.18× llama.cpp at 64k

Qwythos-9B · Q4_K_M · RTX 5090 · same pinned llama.cpp commit (`6f4f53f`) · Polaris-attested eval.

| context | sparkinfer | llama.cpp | vs llama |
|---|---:|---:|---:|
| **4k prefill** | **16,083 pp/s** | 11,105 pp/s | **+45%** |
| **32k prefill** | **17,631 pp/s** | 9,772 pp/s | **+80%** |
| **64k prefill** | **17,772 pp/s** | 8,154 pp/s | **+118% · 2.18×** |

Prefill frontier: **290 → 16,083 pp/s @ 4k** since v0.4.1. Long-context prefill is where agents feel latency first — this release closes that gap.

## Live demo

| | |
|---|---|
| **Website** | [sparkinfer.com](https://sparkinfer.com/) — chat UI, benchmarks, SN74 competition |
| **Demo API** | [api.sparkinfer.com](https://api.sparkinfer.com/) — OpenAI-compatible (`/v1/models`, `/v1/chat/completions`, streaming) |
| **Self-host** | `sparkinfer-server` — [`server/README.md`](https://github.com/gittensor-ai-lab/sparkinfer/blob/main/server/README.md) · continuous batching + prefix cache |

Try the API:

```bash
curl -s https://api.sparkinfer.com/v1/models
curl -s https://api.sparkinfer.com/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwythos-9b","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Qwen3.6 decode — frontier held

| context | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128 decode** | **473 tok/s** | 276 tok/s | **+71%** |
| **32k decode** | **428 tok/s** | 280 tok/s | **+53%** |

## Attested binaries

| Platform | Asset |
|---|---|
| Linux (sm_120) | `sparkinfer-v0.4.2-linux-x86_64-cuda13-sm120.tar.gz` |
| Windows (sm_120) | `sparkinfer-v0.4.2-windows-amd64-cuda13-sm120.zip` |

Each bundle: `sparkinfer-bin/{bin,lib}`, `BUILD_MANIFEST.json`, `SHA256SUMS`. Verified with GitHub Artifact Attestations.

```bash
gh attestation verify sparkinfer-bin/bin/qwen3_gguf_bench -R gittensor-ai-lab/sparkinfer
```

Bench scripts auto-fetch prebuilt tarballs: `bench/scripts/bench.sh --download`.

## Roadmap

**Now** — fastest MoE/LLM inference on every Blackwell edge GPU (RTX Spark → PRO 6000); desktop app, RAG, memory.

**Next** — trustable AI on confidential compute: TDX + NVIDIA CC attestation, source-verified binaries in enclave, SparkDistill domain models, licensed on-prem.

## Landed since v0.4.1

- **#398** (`eval:XL`) — batched prompt prefill — 14× jump at 4k
- **#422 / #465 / #474** (`eval:XL`) — int8 tensor-core prefill GEMM + attention + native `mma.sync` GEMM
- **#455 / #464** (`eval:XL`) — windowed prefill attention + fused dequant/lane-parallel attn
- **#475** — OpenAI-compatible `sparkinfer-server`
- **#520** — continuous batching with per-request KV sessions
- **#472 / #521** — attested Linux + Windows CI binaries

**Verified:** RTX 5090 · Qwythos **17,772 pp/s @ 64k (2.18× llama)** · Qwen3.6 **473 tok/s (+71%)**.

## Contributors

- **@James-CUDA** — #387, #464
- **@fansilas** — #398, #465
- **@inference2026** — #422, #463
- **@Paral1995** — #455, #379
- **@blinkeye-lcm** — #474, #355
- **@ai-hpc** — #475, #476, #481, #490
- **@reyanthony062001-ops** — #389, #393
- **@skyrocket2026** — #506, #520, attested binaries, eval + dashboard

Full notes: [CHANGELOG.md](https://github.com/gittensor-ai-lab/sparkinfer/blob/v0.4.2/CHANGELOG.md)
