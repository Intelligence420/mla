# ==========================================================================
# Auto-generiert vom cuTile Performance Lab (Codegen C1).
# Aus einer RunConfig erzeugt.
# Ausdruck : ik,kj->ij
# Format   : fp8e5m2 -> fp32 (Akku)
# Tile     : TM=128 TN=128 TK=64 | swizzle=False
# ==========================================================================
"""Generierter cuTile-GEMM (Codegen C1) — kanonisches Batched-GEMM (B,M,K)x(B,K,N)->(B,M,N).

Jede 2-Operanden-Kontraktion wird host-seitig (B1-Reshape) auf diese Form
gebracht; dieser Kernel emittiert die EINE bewiesene Struktur (B=1 = Plain-GEMM).
Input-dtype: fp8e5m2 (Laufzeit-torch-dtype, steht NICHT im Kernel-Koerper).
Akkumulator: fp32 (ct.float32).
Tile-Literale: TM=128, TN=128, TK=64 (fest in den Quelltext gebacken).

Bewiesene Orientierung: a=(TM,TK), b=(TK,TN), ct.mma(a,b,acc)->(TM,TN),
KEIN Operanden-Swap, KEIN Permute. i=bid(0)=M-Kachel, j=bid(1)=N-Kachel,
bb=bid(2)=Batch.
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = 128
TN = 128
TK = 64


@ct.kernel
def gemm(A, B, C,
         M: ct.Constant[int],
         N: ct.Constant[int],
         K: ct.Constant[int]):
    """Berechne eine (TM, TN)-Ausgabekachel von C[bb] = A[bb] @ B[bb]."""
    # 3D-Grid: i ueber M-Kacheln, j ueber N-Kacheln, bb ueber den Batch.
    i = ct.bid(0)
    j = ct.bid(1)
    bb = ct.bid(2)

    # Akkumulator unabhaengig vom Input-dtype (Standardmuster aus cuTile).
    acc = ct.full((TM, TN), 0, dtype=ct.float32)

    # K-Schleife: ceil(K / TK) K-Kacheln; Padding-Zeros am Rand sind fuer den
    # MAC neutral (0 * x + acc == acc), daher kein explizites Masking noetig.
    # Batch-Offset steckt im fuehrenden Index-Slot (bb) des 3D-Tensors; das
    # (1, TM, TK)-Tile selektiert die Batch-Scheibe bb und wird auf 2D reshaped.
    for kk in range(ct.cdiv(K, TK)):
        a = ct.load(A, index=(bb, i, kk), shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO)
        a = ct.reshape(a, (TM, TK))
        b = ct.load(B, index=(bb, kk, j), shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.reshape(b, (TK, TN))
        acc = ct.mma(a, b, acc)

    # (TM, TN)-Ergebnis in die (1, TM, TN)-Batch-Scheibe zurueckformen; ct.store
    # schneidet out-of-bounds Elemente am M/N-Rand automatisch ab.
    ct.store(C, index=(bb, i, j),
             tile=ct.reshape(ct.astype(acc, C.dtype), (1, TM, TN)))


def launch(A, B, C):
    """Starte den batched GEMM-Kernel: C[bb] = A[bb] @ B[bb] (C vorab alloziert).

    A=(B,M,K), B=(B,K,N), C=(B,M,N). Grid = (cdiv(M,TM), cdiv(N,TN), B).
    M/N/K sind ct.Constant[int]-Launch-Args; TM/TN/TK sind Quelltext-Literale;
    die Batch-Groesse B kommt ueber die dritte Grid-Achse (bb=bid(2)).
    """
    Bb, M, K = A.shape
    _, _, N = B.shape
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), Bb)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, gemm, (A, B, C, M, N, K))
    return C
