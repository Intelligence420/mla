"""tool_pipeline.measure.fusion — fused-vs-sequentiell-Vergleich (TZ 9).

Der fusionierte Kontraktions-Epilog (``codegen/templates/contraction.py``,
``epilog=bias/relu``) wird gegen den **sequentiellen** Zwei-Kernel-Pfad gemessen:
erst die Plain-Kontraktion (schreibt den Zwischentensor nach DRAM), dann ein
separater Elementwise-Lauf, der ihn wieder liest und den Epilog anwendet. Genau
diesen DRAM-Umweg des Zwischentensors (``2·out·B·M·N`` Bytes) spart die Fusion —
das ist die honest-Story: bei kleiner/memory-bound Kontraktion mit relativ teurem
Epilog gewinnt Fusion, bei compute-dominierter Kontraktion ist sie neutral bis
leicht negativ (A04: 0,984×).

Analog zu ``measure/baselines.py`` eine **Zweitmessung** INNERHALB des fused-``run()``
(gleiche ``bench.benchmark``-Schleife, cold-L2 → direkt vergleichbare ms), optional
und **graceful**: schlägt sie fehl, trägt der Rückgabe-dict ``available=False`` + Grund,
ohne den bereits verifizierten+gemessenen fused-Lauf zu kippen. Geht **nicht** in den
Slug ein (der Kernel-Quelltext ändert sich dadurch nicht).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch

from ..codegen.compile import load_kernel
from ..hardware import dtype_bytes
from ..intermediate_representation.reshape import from_canonical_output
from ..schema import RunConfig
from .bench import benchmark
from .metrics import gemm_bytes, gemm_flops
from .verify import verify

# Epilog → sequentieller Elementwise-Zwilling (Op im Elementwise-Template):
# bias = C_int + D (binäres add), relu = max(C_int, 0) (unäres relu).
_EPILOG_TO_EW_OP = {"bias": "add", "relu": "relu"}


def measure_sequential(config: RunConfig, A_c, B_c, C_c, D_c,
                       canonical, ref_operands: list, fused_ms: float,
                       warmup: int = 5, iters: int = 20) -> dict[str, Any]:
    """Miss den sequentiellen Zwei-Kernel-Pfad und vergleiche ihn mit dem fused-Kernel.

    :param config:       die fused ``RunConfig`` (``epilog`` gesetzt).
    :param A_c, B_c, C_c: kanonische Operanden ``(B,M,K)``/``(B,K,N)``/``(B,M,N)`` des
                          fused-Laufs (C_c wird NICHT überschrieben — eigene Buffer).
    :param D_c:          Bias-Operand ``(B,M,N)`` (Compute-dtype) bei ``epilog=bias``,
                          sonst ``None``.
    :param canonical:    ``Canonical`` (Shapes/Rück-View).
    :param ref_operands: ``[A_nat, B_nat, (D_nat)]`` für die verify-Referenz des
                          sequentiellen Outputs (verify-before-trust auch hier).
    :param fused_ms:     der bereits gemessene fused-Median (für den Speedup).
    :returns:            dict für ``metrics["fusion"]`` — bei Erfolg mit
                          ``sequential_ms``/``speedup``/``*_bytes``/``*_ai``, sonst
                          ``available=False`` + ``note`` (graceful).
    """
    ep = config.epilog
    ew_op = _EPILOG_TO_EW_OP.get(ep)
    if ew_op is None:
        return {"available": False, "note": f"kein sequentieller Zwilling für epilog={ep!r}"}
    try:
        B, M, N = canonical.c_shape
        K = canonical.K

        # 1) Plain-Kontraktion (epilog=None) — schreibt den Zwischentensor C_int nach DRAM.
        plain_cfg = replace(config, epilog=None)
        contract_launch = load_kernel(plain_cfg).launch

        # 2) Elementwise-Zwilling des Epilogs (add bei bias, relu bei relu). Die 2D-Sicht
        #    (rows=B·M, cols=N) ist ein freier View der kanonischen (B,M,N)-Tensoren.
        binary = (ew_op == "add")
        ew_expr = "ij,ij->ij" if binary else "ij->ij"
        ew_cfg = RunConfig(
            family="elementwise", op=ew_op, expr=ew_expr,
            dim_sizes={"i": B * M, "j": N},
            dtype=config.acc_dtype, acc_dtype=config.acc_dtype,
            tile=dict(config.tile),
        )
        ew_launch = load_kernel(ew_cfg).launch

        rows, cols = B * M, N
        C_int = torch.empty_like(C_c)                 # Zwischentensor (acc_dtype)
        C_seq = torch.empty_like(C_c)                 # sequentieller Output (eigener Buffer)
        C_int_2d = C_int.reshape(rows, cols)
        C_seq_2d = C_seq.reshape(rows, cols)

        if binary:
            D_2d = D_c.reshape(rows, cols)            # D bei Compute-dtype (fair vs. fused)

            def _sequential(*_ignore):
                contract_launch(A_c, B_c, C_int)      # C_int = A@B (nach DRAM)
                ew_launch(C_int_2d, D_2d, C_seq_2d)   # C_seq = C_int + D (liest C_int zurück)
                return C_seq
        else:  # relu

            def _sequential(*_ignore):
                contract_launch(A_c, B_c, C_int)
                ew_launch(C_int_2d, C_seq_2d)         # C_seq = max(C_int, 0)
                return C_seq

        # 3) verify-before-trust: der sequentielle Output muss dieselbe fp32-Referenz
        #    (einsum GEFOLGT vom Epilog) treffen wie der fused-Kernel.
        _sequential()
        acc = verify(from_canonical_output(canonical, C_seq), ref_operands, config)
        if not acc["passed"]:
            return {"available": False,
                    "note": f"sequentieller Pfad verify_failed "
                            f"(max_abs_err={acc['max_abs_err']:.4g})"}

        # 4) warme Messung (dieselbe bench-Schleife wie der fused-Kernel → vergleichbar).
        b = benchmark(_sequential, C_seq, warmup=warmup, iters=iters)
        seq_ms = b["run_ms"]

        # 5) analytische Bytes/AI (algorithmischer Mindest-Traffic, wie gemm_bytes):
        #    fused liest A+B(+D) und schreibt C; sequentiell zusätzlich den Zwischen-
        #    tensor-Roundtrip (2·out·B·M·N). D wird in BEIDEN Pfaden bei Compute-dtype
        #    gelesen (fair) ⇒ die Ersparnis ist exakt der Roundtrip.
        out_b = dtype_bytes(config.acc_dtype)
        in_b = dtype_bytes(config.dtype)
        mn = B * M * N
        d_bytes = in_b * mn if ep == "bias" else 0
        contract_bytes = gemm_bytes(M, N, K, config.dtype, config.acc_dtype, B)
        fused_bytes = contract_bytes + d_bytes
        seq_bytes = contract_bytes + 2 * out_b * mn + d_bytes  # + Zwischentensor-Roundtrip
        flops = gemm_flops(M, N, K, B)

        return {
            "available": True,
            "epilog": ep,
            "ew_op": ew_op,
            "fused_ms": round(fused_ms, 5),
            "sequential_ms": round(seq_ms, 5),
            # speedup > 1 ⇒ Fusion gewinnt; ~1 neutral; < 1 leicht negativ (A04).
            "speedup": round(seq_ms / fused_ms, 3) if fused_ms > 0 else None,
            "fused_bytes": fused_bytes,
            "sequential_bytes": seq_bytes,
            "saved_bytes": seq_bytes - fused_bytes,   # = 2·out·B·M·N (Zwischentensor)
            "fused_ai": round(flops / fused_bytes, 2),
            "sequential_ai": round(flops / seq_bytes, 2),
        }
    except Exception as e:  # noqa: BLE001 — graceful: der fused-Lauf bleibt gültig
        return {"available": False, "note": f"{type(e).__name__}: {str(e)[:200]}"}
