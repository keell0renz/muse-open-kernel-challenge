# muse-open-kernel-challenge

Team **keell0renz**'s best fair kernels from the OKC Muse Spark Open Kernel Challenge (H200 SXM, `sm_90a`). One file per
board, each the exact file the judge timed (or a judge-checked successor of it). No cheats, no luck tricks in the code.

**[LESSONS.md](LESSONS.md)**: what worked, what failed and why, judge facts, H200 numbers, method notes.

## Results

Leaderboard snapshot of 2026-10-31 17:14 CET (speedup = geomean over the board's scored shapes vs the SGLang baseline):

| Board | Our speedup | Rank | Teams ranked | #1 | Gap to #1 |
|---|---|---|---|---|---|
| **decode layer** (= overall rank) | **2.2659** | **#38** | 1085 | 2.4178 (mpeng19) | 6.7% |
| attention decode | 2.7168 | #35 | 1166 | 3.2221 (tjiang-fair) | 18.6% |
| bf16 projections | 1.5876 | #26 | 1516 | 1.6523 (arashattar) | 4.1% |
| prefill layer | 1.9134 | #29 | 729 | 2.0325 (zzz0906) | 6.2% |
| MoE W4A16 (NVFP4) | 2.4203 | #58 | 1219 | 2.7578 (dolanzhao) | 13.9% |

**Open-source standing.** As of 2026-11-01 this is the highest-ranked publicly available OKC code we know of on all five
boards. The other public submissions we tracked rank lower (best: alygg77, decode #132, MoE #79, attention #80, bf16
#162, prefill #57; then vjeux, chenwainuo, NathenFernandes, Madhavan113). ZhimaoL (decode #29) published their branches in
a public sglang fork but deleted them on 2026-11-01. Repos we did not find are not counted.

A board row keeps a team's best submission, and the judge's run-to-run noise is about +-1.5% (1 sigma ~0.63%), so some rows are
high draws of these files. The judge's own numbers for the files in this repo:

| File | Judge runs of this exact code |
|---|---|
| `kernels/muse_decode_layer.py` | check 2.2641, submit 2.2532 (the 2.2659 row is the same kernels minus the last +0.27% change) |
| `kernels/muse_moe_w4a16.py` | submits 2.3884 / 2.3893 (the 2.4203 row is a high draw of its predecessor) |
| `kernels/muse_attention_decode.py` | row 2.7168 (a high draw; same-box screens 2.63-2.66) |
| `kernels/muse_bf16_proj.py` | check 1.5742, submit 1.5690 (row 1.5876: its pre-fix twin) |
| `kernels/muse_prefill_layer.py` | submit 1.9141 |

About 460 logged experiments went into these over the challenge. Most were measured dead; the wins are small and stacked.

## What is in each file

- **decode layer**: the whole layer as a chain of ~9 kernels linked by programmatic dependent launch (PDL). The projections are
  `sk2` TMA-ring GEMVs that read BF16 weights through a lossless 12-bit re-encoding with exact escape corrections; pooler and
  router in one launch; Triton routing (split-K sum, sigmoid + bias top-8, pre-expert norm); a persistent NVFP4 W4A16 MoE
  (TMA weight ring, warp-specialized issue, per-batch-size builds; for 17+ tokens a 4-warpgroup wgmma stream-K build with
  lossless 6-bit block scales and warp-parallel stage ranges); unpooler; cluster-based residual / norm boundary.
- **MoE W4A16**: the same MoE kernel standalone, with a dedicated build per scored batch size (bs 8 gets the ring-issuer
  warp variant).
- **attention decode**: one fused gated-attention kernel per call over the paged FP8 KV cache, split-KV with a TMA
  bulk-reduction merge; ZhimaoL's verify-attention kernels for q_len 8.
- **bf16 projections**: a per-shape dispatcher (picked from judge profiles) over public kernels plus our 12-bit lossless
  weight format and pinned cuBLASLt plans for the large-m cells.
- **prefill layer**: cuBLASLt GEMMs (tuned and pinned), ZhimaoL's FMHA with the N72 PV layout, a persistent MoE.

## Running

Each file implements the challenge's problem interface: `prepare(weights) -> state` (untimed) and
`kernel_fn(data, state) -> outputs`, with the first line `# okc: problem=<name>`. The problem definitions, reference
implementations and judge belong to the challenge and are not included. Needs an H100 / H200 (`sm_90a`), CUDA 12.8+,
PyTorch 2.11, Triton; every CUDA extension is JIT-built by `torch.utils.cpp_extension.load_inline` on first use.

Known issue: `muse_prefill_layer.py` mis-routes on the "turn 1, full" shape when its state is first used from a fresh OS
thread on the default stream (root cause not found).

## License and credits

Apache License 2.0 (`LICENSE`), the challenge's submission license. These files build on other teams' public Apache-2.0
and MIT code; every file keeps its original attribution lines, and `NOTICE` lists them all.
