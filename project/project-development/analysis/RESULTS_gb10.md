# Analyse-Testing — cuTile dtype-Machbarkeit auf GB10

**Datum:** 30.06.2026 · **Maschine:** NVIDIA GB10 (Grace-Blackwell, sm_121), aarch64, Treiber 580.159.03
**Skript:** [`dtype_analyse.py`](./dtype_analyse.py) — Batched-GEMM `(B,M,K)×(B,K,N)→(B,M,N)`, B=4, M=K=N=512, Tile (128,128,64), je wenige Iterationen, Korrektheit gegen fp32-`torch.bmm`-Referenz.
**Umgebung:** torch 2.11.0+cu130, CUDA-Runtime 13.0, triton 3.6.0, cupy 14.0.1, `cuda.tile` vorhanden.

## Ergebnis — alle Kandidaten compilen, laufen, sind numerisch korrekt

| dtype (compute→acc) | compiles | runs | korrekt vs Ref | max_abs_err | grobe TFLOP/s* | acc |
|---|---|---|---|---|---|---|
| fp16 → fp32 (Anker) | ✅ | ✅ | PASS | 1.7e-4 | 16.0 | fp32 |
| bf16 → fp32 | ✅ | ✅ | PASS | 1.1e-4 | 15.4 | fp32 |
| tf32 → fp32 | ✅ | ✅ | PASS | 3.4e-2 | 6.2 | fp32 |
| fp8 e4m3 → fp32 | ✅ | ✅ | PASS | 1.5e-5 | 18.7 | fp32 |
| fp8 e4m3 → fp16 | ✅ | ✅ | PASS | 0.16 | 28.0 | fp16 |
| fp8 e5m2 → fp32 | ✅ | ✅ | PASS | 1.5e-5 | 18.9 | fp32 |
| fp32 → fp32 (plain) | ✅ | ✅ | PASS | 1.1e-4 | 0.2 | fp32 |
| fp64 → fp64 | ✅ | ✅ | PASS | 2.1e-13 | 0.1 | fp64 |

\* Absolutwerte niedrig, weil das Testproblem absichtlich winzig/single-tile-per-block ist → **relative** Indikatoren, nicht Peak.

**fp4 / int4:** in diesem cuTile-Build **nicht darstellbar** (keine `float4_e2m1fn`/`int4`-Symbole; `ct.mma`-dtype-Tabelle endet bei 8 Bit). → exkludiert.

## cuTile-API-Details (bake into the tool)

- **Akkumulator** = `ct.full((tm,tn), 0, dtype=<acc>)` (`ct.float32` / `ct.float16` / `ct.float64`).
- **fp16/bf16/fp32/fp64:** native torch-dtypes; load→reshape→mma direkt. bf16 & tf32 brauchen **fp32**-Akku.
- **tf32:** **kein mma-Flag** — `ct.mma`-Signatur ist exakt `(x, y, acc)`. tf32 ist ein **dtype**: fp32-Tiles laden, dann `ct.astype(tile, ct.tfloat32)` vor `ct.mma`, fp32-Akku. (Plain-fp32 ohne Cast nutzt **keine** Tensor-Cores → 0.2 vs 6.2 TFLOP/s.)
- **fp8 e4m3/e5m2:** host-seitig `torch.randn(...).to(torch.float8_e4m3fn / e5m2)` (randn kann fp8 nicht direkt). Im Kernel direkt load/reshape/mma; Akku fp16 (schneller) oder fp32. e4m3 vs e5m2 verifiziert genuin verschieden (kein stilles Upcast).
- **Output:** `ct.store(C, index=..., tile=ct.astype(acc, C.dtype))`.
- **Orientierung:** Plain-GEMM-Output `m,n` braucht **keine** Permute (`ct.mma(a2d=(tm,tk), b2d=(tk,tn), acc)`). A06s `permute`+Operanden-Swap ist nur wegen dessen `yx`-Output-Layout nötig.

## Hardware / Roofline-Inputs (GB10)

- Blackwell, 48 SM, 6144 CUDA-Cores, 192 5th-gen Tensor-Cores, ~2.42 GHz (max 3.0), TDP SoC 140 W.
- **Unified Memory: 128 GB LPDDR5x, 273 GB/s** (theoretisch; real ~70–85 % ⇒ ~190–230 GB/s, da keine publizierte STREAM-Zahl).
- **Peak-Compute (dense, gemessen):** FP4 ≈ 427 (nur via `mma_scaled`, hier n/a) · FP16/BF16 ≈ 213 · FP8 ≈ 214 · INT8 ≈ 215 TOPS · TF32 ≈ 53 · FP64 ≈ vernachlässigbar (kein FP64-Tensor-Core auf sm_12x).
- **Ridge-Points** (= Peak/273e9) hoch: BF16 ≈ 780 FLOP/Byte ⇒ **stark memory-bound** für übliche GEMM/einsum-Shapes.

## Quellen

- nvidia-smi lokal (GB10, sm_121, CUDA 13.0, Treiber 580.159.03)
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993 (mmapeak)
- https://docs.nvidia.com/cuda/cutile-python/data.html · .../generated/cuda.tile.matmul.html · .../generated/cuda.tile.mma_scaled.html
- https://github.com/nvidia/cutile-python (Toolchain: Blackwell + Ampere/Ada, Treiber r580+, CUDA 13.1+)
- https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/ · https://arxiv.org/html/2507.10789v2 (FP64-Pfad)

**Unsicherheiten:** Kein offizielles GB10-Whitepaper mit Theorie-Peaks (Peaks aus Forum-Microbenchmark, gemessen/glaubwürdig). Reale Bandbreite vs. 273 GB/s ohne publizierte STREAM-Zahl. cuTile brandneu — `matmul`-Akkumulator-dtype und ein eigenständiges `ct.mma`-Doc nicht vollständig dokumentiert; empirisch funktioniert `ct.mma(x,y,acc)` wie hier getestet.
