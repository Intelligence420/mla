"""tool_pipeline.intermediate_representation.reshape — B1: Kontraktion → kanonisches (Batched-)GEMM.

**Ziel (Endausbau, TZ 6):** *jede* 2-Operanden-Kontraktion host-seitig per
zero-copy-View auf die kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)` reshapen, damit
der Codegen nur **eine** bewiesene Struktur emittieren muss (Risiko ④: die
View-/Stride-Mathematik muss korrekt sein).

**TZ 1 (hier): reiner Passthrough, Batch=1.** Der Plain-GEMM `ik,kj->ij` liegt
bereits in kanonischer 2D-Form (`A=(M,K)`, `B=(K,N)`, `out=(M,N)`) — es ist
**keine** Transformation nötig. Wir prüfen das via `ir.is_canonical_gemm()` und
geben nur die kanonischen Größen zurück. Jeder Ausdruck, der eine echte
Umformung (Permute/Fuse/Split) bräuchte, wird mit klarem Verweis auf TZ 6
abgelehnt — der echte B1-Reshape wird hier **nicht** vorgebaut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .parse import ContractionIR


@dataclass
class Canonical:
    """Kanonische (Batched-)GEMM-Beschreibung, die Codegen/Measure konsumieren.

    Felder sind die fusionierten Größen. TZ 6 ergänzt hier **additiv** die
    Operanden-View-Spezifikationen (Permute/Reshape je Operand); der TZ-1-
    Passthrough braucht keine — `transform_needed=False`.
    """

    M: int
    N: int
    K: int
    B: int = 1
    transform_needed: bool = False
    ir: Optional[ContractionIR] = None


def to_canonical(ir: ContractionIR) -> Canonical:
    """`ContractionIR` → `Canonical`. TZ 1: nur Plain-GEMM-Passthrough.

    Raises:
        NotImplementedError: wenn der Ausdruck eine echte Umformung bräuchte
            (Batch, Permute, mehrdimensionale M/N/K) — das ist TZ 6.
    """
    if not ir.is_canonical_gemm():
        raise NotImplementedError(
            f"TZ 1 unterstützt nur den kanonischen Plain-GEMM (z. B. 'ik,kj->ij') "
            f"als Passthrough; '{ir.expr}' bräuchte den echten B1-Reshape "
            f"(Permute/Fuse/Batch) — das ist TZ 6."
        )
    return Canonical(M=ir.M, N=ir.N, K=ir.K, B=1, transform_needed=False, ir=ir)
