# ==========================================================================
# Auto-generiert vom cuTile Performance Lab (Codegen C1).
# Aus einer RunConfig erzeugt.
# Ausdruck : ik,kj->ij
# Format   : fp16 -> fp32 (Akku)
# Tile     : TM=128 TN=128 TK=64 | swizzle=True
# ==========================================================================
"""Generierter cuTile-GEMM (Codegen C1) — Kontraktion ik,kj->ij.

Input-dtype: fp16 (Laufzeit-torch-dtype, steht NICHT im Kernel-Koerper).
Akkumulator: fp32 (ct.float32).
Tile-Literale: TM=128, TN=128, TK=64 (fest in den Quelltext gebacken).

Bewiesene Orientierung: a=(TM,TK), b=(TK,TN), ct.mma(a,b,acc)->(TM,TN),
KEIN Operanden-Swap, KEIN Permute. i=bid(0)=M-Kachel, j=bid(1)=N-Kachel. L2-Swizzle EIN (grouped-M-Rasterung, GROUP_M=8): i/j werden bijektiv umgeordnet (dieselbe Kachelmenge, L2-freundlichere Reihenfolge).
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = 128
TN = 128
TK = 64
GROUP_M = 8


@ct.kernel
def gemm(A, B, C,
         M: ct.Constant[int],
         N: ct.Constant[int],
         K: ct.Constant[int]):
    """Berechne eine (TM, TN)-Ausgabekachel von C = A @ B."""
    # L2-Swizzle: grouped-M-Rasterung — dieselben (i, j) wie ohne Swizzle,
    # nur in L2-freundlicherer Reihenfolge (Bloecke einer Gruppe teilen sich
    # B-Spalten). Bijektiv -> Ergebnis unveraendert; Orientierung/mma unberuehrt.
    num_pid_m = ct.cdiv(M, TM)
    num_pid_n = ct.cdiv(N, TN)
    pid = ct.bid(0) * num_pid_n + ct.bid(1)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    local = pid % num_pid_in_group
    i = first_pid_m + (local % group_size_m)
    j = local // group_size_m

    # Akkumulator unabhaengig vom Input-dtype (Standardmuster aus cuTile).
    acc = ct.full((TM, TN), 0, dtype=ct.float32)

    # K-Schleife: ceil(K / TK) K-Kacheln; Padding-Zeros am Rand sind fuer den
    # MAC neutral (0 * x + acc == acc), daher kein explizites Masking noetig.
    for kk in range(ct.cdiv(K, TK)):
        a = ct.load(A, index=(i, kk), shape=(TM, TK),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(kk, j), shape=(TK, TN),
                    padding_mode=ct.PaddingMode.ZERO)
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
