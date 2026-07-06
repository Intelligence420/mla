# ==========================================================================
# Auto-generiert vom cuTile Performance Lab (Codegen C1).
# Aus einer RunConfig erzeugt.
# Ausdruck : ik,kj->ij
# Format   : tf32 -> fp32 (Akku)
# Tile     : TM=256 TN=256 TK=128 | swizzle=False
# ==========================================================================
"""Generierter cuTile-GEMM (Codegen C1) — Kontraktion ik,kj->ij.

Input-dtype: tf32 -> im Kernel via ct.astype auf ct.tfloat32 gecastet (VOR ct.mma; ohne Cast liefe es auf CUDA-Cores).
Akkumulator: fp32 (ct.float32).
Tile-Literale: TM=256, TN=256, TK=128 (fest in den Quelltext gebacken).

Bewiesene Orientierung: a=(TM,TK), b=(TK,TN), ct.mma(a,b,acc)->(TM,TN),
KEIN Operanden-Swap, KEIN Permute. i=bid(0)=M-Kachel, j=bid(1)=N-Kachel.
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = 256
TN = 256
TK = 128


@ct.kernel
def gemm(A, B, C,
         M: ct.Constant[int],
         N: ct.Constant[int],
         K: ct.Constant[int]):
    """Berechne eine (TM, TN)-Ausgabekachel von C = A @ B."""
    # 2D-Grid: i laeuft ueber M-Kacheln, j ueber N-Kacheln.
    i = ct.bid(0)
    j = ct.bid(1)

    # Akkumulator unabhaengig vom Input-dtype (Standardmuster aus cuTile).
    acc = ct.full((TM, TN), 0, dtype=ct.float32)

    # K-Schleife: ceil(K / TK) K-Kacheln; Padding-Zeros am Rand sind fuer den
    # MAC neutral (0 * x + acc == acc), daher kein explizites Masking noetig.
    for kk in range(ct.cdiv(K, TK)):
        a = ct.load(A, index=(i, kk), shape=(TM, TK),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(kk, j), shape=(TK, TN),
                    padding_mode=ct.PaddingMode.ZERO)
        a = ct.astype(a, ct.tfloat32)
        b = ct.astype(b, ct.tfloat32)
        acc = ct.mma(a, b, acc)

    # ct.store schneidet out-of-bounds Elemente am Rand automatisch ab.
    ct.store(C, index=(i, j), tile=ct.astype(acc, C.dtype))


def launch(A, B, C):
    """Starte den GEMM-Kernel: C = A @ B (C ist vorab alloziert).

    A=(M,K), B=(K,N), C=(M,N). Grid = (cdiv(M,TM), cdiv(N,TN)).
    M/N/K sind ct.Constant[int]-Launch-Args; TM/TN/TK sind Quelltext-Literale.
    """
    M, K = A.shape
    _, N = B.shape
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN))
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, gemm, (A, B, C, M, N, K))
    return C
