"""Korrektheitsnetz für den Codegen (TZ 1): generierter GEMM vs. fp32.

Der Codegen ist die Hauptquelle **stiller** Falschergebnisse (v. a. die
mma-Orientierung, Risiko ①). Diese Tests fahren den **echten** Codegen-Pfad
(`emit` → `compile.load_kernel` → `launch`) auf der GPU und prüfen gegen eine
fp32-`torch.einsum`-Referenz — plus einen expliziten **Orientierungs-Wächter**,
der `A@B` von seinen transponierten Doppelgängern unterscheidet.

Lauffähig standalone (`python tests/test_codegen.py`, aus `project/`) **und**
via pytest. Braucht GPU + cuTile (fährt echte Kernel).
"""

from __future__ import annotations

import os
import sys

import torch

# project/ auf den Pfad, damit `tool_pipeline` importierbar ist (standalone-Lauf).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.codegen.compile import clear_cache, load_kernel  # noqa: E402
from tool_pipeline.codegen.emit import emit  # noqa: E402
from tool_pipeline.codegen.templates.contraction import build_gemm_module  # noqa: E402
from tool_pipeline.codegen.templates.elementwise import build_elementwise_module  # noqa: E402
from tool_pipeline.codegen.templates.reduction import build_reduction_module  # noqa: E402
from tool_pipeline.measure.verify import _TOLERANCES  # noqa: E402
from tool_pipeline.schema import RunConfig, RunResult  # noqa: E402

# Output ist fp32 (= acc_dtype) → STRAFFE Toleranz: der reale fp32-Akku-Fehler
# liegt ~1e-4, eine fp16-Akku/-Output-Regression (~3e-2) MUSS damit failen.
# (Die lockeren fp16-Output-Toleranzen 2e-1/2e-2 gehören erst in TZ 3.)
ATOL, RTOL = 1e-2, 1e-3

# Compute-/Akku-dtype-Label → torch-dtype für die Test-Operanden. tf32-Eingaben
# sind normale float32-Tensoren (der tfloat32-Cast passiert im Kernel), genau
# wie in run._build_natural_operands.
_TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16,
          "tf32": torch.float32, "fp32": torch.float32,
          "fp8e4m3": torch.float8_e4m3fn, "fp8e5m2": torch.float8_e5m2}

# TZ-4-Tile-Matrix: verschiedene TM/TN/TK inkl. des winzigen 16³ (= Tile der
# naive-cuTile-Baseline → muss ebenfalls verifizieren, sonst wären dessen
# Baseline-TFLOP/s nicht vertrauenswürdig) und eines großen 256er-Tiles.
_TZ4_TILES = [
    {"TM": 16, "TN": 16, "TK": 16},     # = naive-Baseline-Tile
    {"TM": 64, "TN": 64, "TK": 32},
    {"TM": 256, "TN": 128, "TK": 64},
]
# glatt (Tile-Vielfache) + ragged (nicht-teilbar → ZERO-Padding-Pfad je Tile).
_TZ4_SIZES = [(512, 512, 512), (130, 100, 70), (129, 127, 65)]


def _rand_operand(rows: int, cols: int, dtype: str):
    """Zufalls-Operand im Compute-`dtype`. fp8 via fp16→.to() (randn kann kein
    fp8), sonst direkt — spiegelt run._build_natural_operands."""
    t = _TORCH[dtype]
    if dtype.startswith("fp8"):
        return torch.randn(rows, cols, dtype=torch.float16, device="cuda").to(t)
    return torch.randn(rows, cols, dtype=t, device="cuda")


def _run_gemm(M: int, N: int, K: int, dtype: str = "fp16", acc: str = "fp32",
              tile: dict | None = None, swizzle: bool = False):
    """Echten Codegen-Pfad fahren: emit → load → launch. Gibt (A, B, C).

    `dtype`/`acc` steuern Compute- bzw. Akku-/Output-dtype (Default fp16→fp32 =
    der TZ-1-Anker). `tile`/`swizzle` steuern die TZ-4-Kachel/Swizzle-Achse
    (Default = die TZ-1-Kachel 128/128/64 ohne Swizzle).
    """
    kwargs = {"dim_sizes": {"i": M, "k": K, "j": N}, "dtype": dtype,
              "acc_dtype": acc, "swizzle": swizzle}
    if tile is not None:
        kwargs["tile"] = tile
    cfg = RunConfig(**kwargs)
    launch = load_kernel(cfg, emit(cfg)).launch
    torch.manual_seed(0)
    # Das Template ist kanonisch batched (B,M,K); B=1 = Plain-GEMM. Operanden als
    # (1,·,·) bauen (Seed-Reihenfolge unverändert → byte-gleiche Daten), aber 2D-
    # Views zurückgeben — die Assertions unten arbeiten mit ik,kj->ij (2D-View
    # teilt den Speicher mit dem 3D-Tensor, der Launch füllt beide).
    A = _rand_operand(M, K, dtype).reshape(1, M, K)
    B = _rand_operand(K, N, dtype).reshape(1, K, N)
    C = torch.empty(1, M, N, dtype=_TORCH[acc], device="cuda")   # Output = acc_dtype
    launch(A, B, C)
    torch.cuda.synchronize()
    return A.reshape(M, K), B.reshape(K, N), C.reshape(M, N)


def _rand_operand3(b: int, rows: int, cols: int, dtype: str):
    """Wie _rand_operand, aber 3D (b, rows, cols) für den batched Codegen-Test."""
    t = _TORCH[dtype]
    if dtype.startswith("fp8"):
        return torch.randn(b, rows, cols, dtype=torch.float16, device="cuda").to(t)
    return torch.randn(b, rows, cols, dtype=t, device="cuda")


def _run_gemm_batched(Bb: int, M: int, N: int, K: int, dtype: str = "fp16",
                      acc: str = "fp32", tile: dict | None = None, swizzle: bool = False):
    """Echten Codegen-Pfad fahren mit einem batched Ausdruck (B>1) → gibt die
    3D-Tensoren (Bb,M,K)/(Bb,K,N)/(Bb,M,N) zurück. Der Kernel-Körper ist
    ausdrucks-unabhängig; der Batch kommt allein aus der Operanden-Form."""
    kwargs = {"expr": "bik,bkj->bij", "dim_sizes": {"b": Bb, "i": M, "k": K, "j": N},
              "dtype": dtype, "acc_dtype": acc, "swizzle": swizzle}
    if tile is not None:
        kwargs["tile"] = tile
    cfg = RunConfig(**kwargs)
    launch = load_kernel(cfg, emit(cfg)).launch
    torch.manual_seed(0)
    A = _rand_operand3(Bb, M, K, dtype)
    B = _rand_operand3(Bb, K, N, dtype)
    C = torch.empty(Bb, M, N, dtype=_TORCH[acc], device="cuda")
    launch(A, B, C)
    torch.cuda.synchronize()
    return A, B, C


def _assert_matches_fp32(M: int, N: int, K: int, dtype: str, acc: str,
                         tile: dict | None = None, swizzle: bool = False):
    """Generierter Kernel stimmt gegen die fp32-Referenz — mit der
    **Produktions-Toleranz** des Formats (Quelle: measure.verify._TOLERANCES,
    keine Duplikat-Werte). Optional über Tile/Swizzle parametrisiert (TZ 4)."""
    atol, rtol = _TOLERANCES[(dtype, acc)]
    A, B, C = _run_gemm(M, N, K, dtype, acc, tile=tile, swizzle=swizzle)
    ref = torch.einsum("ik,kj->ij", A.float(), B.float())
    assert C.shape == (M, N), f"{dtype}->{acc} {(M, N, K)}: Shape {tuple(C.shape)}"
    assert C.dtype == _TORCH[acc], f"{dtype}->{acc}: Output-dtype {C.dtype} != {acc}"
    err = (C.float() - ref).abs().max().item()
    assert torch.allclose(C.float(), ref, atol=atol, rtol=rtol), \
        f"{dtype}->{acc} {(M, N, K)} weicht ab: max_abs_err={err:.3e} (atol={atol}, rtol={rtol})"


def _assert_orientation(dtype: str, acc: str,
                        tile: dict | None = None, swizzle: bool = False):
    """Orientierungs-Wächter je Format: exakt A@B, kein transponierter Doppelgänger.

    Quadratische Inputs → alle Doppelgänger sind shape-legal, nur die Zahlen
    unterscheiden sie. err_AB < 1.0 passt für alle in-scope-Formate (auch die
    fp16-/fp8-Akku-Pfade mit ~0.2 realem Fehler). Optional über Tile/Swizzle
    parametrisiert (TZ 4: neue Tiles/Swizzle = neue Orientierungs-Fehlerquelle)."""
    M = N = K = 256
    A, B, C = _run_gemm(M, N, K, dtype, acc, tile=tile, swizzle=swizzle)
    Af, Bf, Cf = A.float(), B.float(), C.float()
    err_AB = (Cf - Af @ Bf).abs().max().item()
    imposters = {
        "A@B^T": (Cf - Af @ Bf.T).abs().max().item(),
        "B@A": (Cf - Bf @ Af).abs().max().item(),
        "(A@B)^T": (Cf - (Af @ Bf).T).abs().max().item(),
    }
    assert err_AB < 1.0, f"{dtype}->{acc}: A@B sollte passen, err={err_AB:.3e}"
    assert min(imposters.values()) > 10.0, \
        f"{dtype}->{acc}: ein Doppelgänger liegt verdächtig nah: {imposters}"


def test_gemm_correct_across_sizes():
    """Generierter GEMM stimmt gegen fp32 — glatte UND ragged (Padding-)Größen.

    Enthält bewusst nicht-tile-teilbare Größen, damit der Rand-Pfad geprüft wird
    (ZERO-Padding im K-Loop ist im MAC neutral, ct.store clippt M/N-Ränder) —
    sonst bliebe genau dieser Pfad zu 0 % abgedeckt.
    """
    sizes = [
        (512, 512, 512), (256, 384, 128), (128, 128, 64),   # glatt (Tile-Vielfache)
        (130, 100, 70), (129, 127, 65), (1, 1, 1),          # ragged: M%TM,N%TN,K%TK != 0
    ]
    for (M, N, K) in sizes:
        A, B, C = _run_gemm(M, N, K)
        ref = torch.einsum("ik,kj->ij", A.float(), B.float())
        assert C.shape == (M, N), f"falsche Output-Shape bei {(M, N, K)}: {tuple(C.shape)}"
        # Output-dtype muss fp32 bleiben (= acc_dtype); eine fp16-Akku/-Output-
        # Regression würde hier UND an der straffen Toleranz scheitern.
        assert C.dtype == torch.float32, f"Output-dtype {C.dtype} != fp32 bei {(M, N, K)}"
        err = (C - ref).abs().max().item()
        assert torch.allclose(C, ref, atol=ATOL, rtol=RTOL), \
            f"GEMM {(M, N, K)} weicht ab: max_abs_err={err:.3e}"


def test_gemm_computes_AB_not_transpose():
    """Orientierungs-Wächter: der Kernel rechnet exakt A@B, keinen Doppelgänger.

    Auf quadratischen Inputs sind A@B, A@B^T, B@A und (A@B)^T alle shape-legal —
    nur die Zahlen unterscheiden sie. Genau hier würde ein stiller
    mma-Orientierungsfehler durchschlüpfen.
    """
    M = N = K = 256
    A, B, C = _run_gemm(M, N, K)
    Af, Bf = A.float(), B.float()
    err_AB = (C - Af @ Bf).abs().max().item()
    imposters = {
        "A@B^T": (C - Af @ Bf.T).abs().max().item(),
        "B@A": (C - Bf @ Af).abs().max().item(),
        "(A@B)^T": (C - (Af @ Bf).T).abs().max().item(),
    }
    assert err_AB < 1.0, f"A@B sollte passen, err={err_AB:.3e}"
    assert min(imposters.values()) > 10.0, \
        f"ein Doppelgänger liegt verdächtig nah: {imposters}"


def test_gemm_batched_verifies_and_orientation():
    """Batched GEMM (B>1): jede Batch-Scheibe stimmt gegen fp32 (einsum) UND die
    Orientierung ist A[b]@B[b] — kein transponierter Doppelgänger, kein Batch-Mix."""
    Bb, M, N, K = 3, 128, 96, 64
    A, B, C = _run_gemm_batched(Bb, M, N, K)          # (Bb,M,K),(Bb,K,N),(Bb,M,N)
    ref = torch.einsum("bik,bkj->bij", A.float(), B.float())
    assert tuple(C.shape) == (Bb, M, N), tuple(C.shape)
    err = (C.float() - ref).abs().max().item()
    assert torch.allclose(C.float(), ref, atol=1e-2, rtol=1e-3), f"batched weicht ab: {err:.3e}"

    # Orientierungs-Wächter je Batch (quadratisch → alle Doppelgänger shape-legal).
    A2, B2, C2 = _run_gemm_batched(3, 128, 128, 128)
    A2f, B2f, C2f = A2.float(), B2.float(), C2.float()
    err_AB = (C2f - torch.bmm(A2f, B2f)).abs().max().item()
    imp_T = (C2f - torch.bmm(A2f, B2f.transpose(1, 2))).abs().max().item()
    imp_rev = (C2f - torch.bmm(B2f, A2f)).abs().max().item()
    imp_mix = (C2f[0] - (A2f[1] @ B2f[1])).abs().max().item()   # Batch-Verwechslung
    assert err_AB < 1.0, f"A[b]@B[b] sollte passen: {err_AB:.3e}"
    assert min(imp_T, imp_rev, imp_mix) > 10.0, \
        f"Doppelgänger/Batch-Mix zu nah: T={imp_T:.3e} rev={imp_rev:.3e} mix={imp_mix:.3e}"


def test_swizzle_emit_structure():
    """Headless: swizzle=False ist strukturell die TZ-1-Variante (kein GROUP_M,
    plain bid); swizzle=True bringt die grouped-M-Rasterung — die mma-Orientierung
    ist in BEIDEN identisch (Swizzle fasst NUR die Block→Kachel-Zuordnung an)."""
    tile = {"TM": 128, "TN": 128, "TK": 64}
    plain = build_gemm_module(tile, "fp16", "fp32", swizzle=False)
    swz = build_gemm_module(tile, "fp16", "fp32", swizzle=True)
    # Nicht-Swizzle: unverändert.
    assert "GROUP_M" not in plain and "num_pid_n" not in plain, plain
    assert "i = ct.bid(0)" in plain and "j = ct.bid(1)" in plain, plain
    # Swizzle: grouped-M-Rasterung da.
    assert "GROUP_M = 8" in swz and "num_pid_n = ct.cdiv(N, TN)" in swz, swz
    assert "group_size_m = min(" in swz, swz
    # Orientierung in BEIDEN gleich (der eine Beweis-Invariant, Risiko ①).
    assert "acc = ct.mma(a, b, acc)" in plain and "acc = ct.mma(a, b, acc)" in swz


def test_reduction_emit_structure():
    """Headless: das Reduktions-Template ist memory-bound — ct.sum-Idiom, KEIN
    ct.mma/kein GEMM-K-Loop-Reshape; beide Pfade (single-shot + markierter
    Fallback) da; launch-Arity 2 (launch(A, C)); TK steuert LOOP_TILE."""
    src = build_reduction_module({"TM": 128, "TN": 128, "TK": 512}, "fp16", "fp32")
    # memory-bound: ct.sum da, KEIN Tensor-Core.
    assert "ct.sum(" in src and "ct.mma(" not in src, src
    # Beide Pfade vorhanden.
    assert "def row_sum_single(" in src and "def row_sum_loop(" in src, src
    assert "single-shot" in src and "FALLBACK" in src, src
    # launch-Arity 2 (1 Operand + Output) — NICHT die GEMM-Arity 3.
    assert "def launch(A, C):" in src, src
    # TK steuert die Fallback-Chunk-Breite (steht im Slug).
    assert "LOOP_TILE = 512" in src, src


def test_reduction_emit_rejects_bad_acc():
    """Unbekannter acc_dtype ⇒ ValueError (loud-fail, wie beim GEMM-Template)."""
    try:
        build_reduction_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "bf16")
    except ValueError:
        return
    raise AssertionError("erwartete ValueError für acc_dtype='bf16'")


def test_elementwise_emit_structure():
    """Headless: das Elementwise-Template ist memory-bound — cdiv-2D-Grid, KEIN
    ct.mma/kein Akku; binaere Ops laden A+B (Arity 3), copy nur A (Arity 2); das
    Op-Fragment wird korrekt substituiert (a+b / a*b / a)."""
    tile = {"TM": 64, "TN": 128, "TK": 64}
    add = build_elementwise_module(tile, "fp16", "fp32", "add")
    mul = build_elementwise_module(tile, "fp16", "fp32", "mul")
    cpy = build_elementwise_module(tile, "fp16", "fp32", "copy")
    # memory-bound: kein Tensor-Core, kein Akku, cdiv-Grid.
    for src in (add, mul, cpy):
        assert "ct.mma(" not in src and "ct.full(" not in src, src
        assert "ct.cdiv(M, TM)" in src and "ct.cdiv(N, TN)" in src, src
        assert "TM = 64" in src and "TN = 128" in src, src
    # Op-Fragment korrekt substituiert.
    assert "ct.astype(a + b, C.dtype)" in add, add
    assert "ct.astype(a * b, C.dtype)" in mul, mul
    assert "ct.astype(a, C.dtype)" in cpy, cpy
    # Arity: binaer laedt A+B und hat launch(A, B, C); copy nur A und launch(A, C).
    assert "def launch(A, B, C):" in add and "ct.load(B," in add, add
    assert "def launch(A, C):" in cpy and "ct.load(B," not in cpy, cpy


def test_elementwise_emit_rejects_bad_op():
    """Unbekannte Op ⇒ ValueError (loud-fail)."""
    try:
        build_elementwise_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "fp32", "relu")
    except ValueError:
        return
    raise AssertionError("erwartete ValueError für op='relu'")


def test_gemm_swizzle_correct_across_sizes():
    """verify-before-trust für den Swizzle-Kernel: stimmt gegen fp32 — glatt UND
    ragged (der Swizzle darf keine Kachel doppelt/gar nicht berechnen)."""
    for (M, N, K) in [(512, 512, 512), (256, 384, 128),
                      (130, 100, 70), (129, 127, 65), (1, 1, 1)]:
        _assert_matches_fp32(M, N, K, "fp16", "fp32", swizzle=True)


def test_gemm_swizzle_orientation():
    """Orientierungs-Wächter MIT Swizzle: der Kernel rechnet weiter exakt A@B."""
    _assert_orientation("fp16", "fp32", swizzle=True)


def test_swizzle_equals_noswizzle():
    """Der Swizzle ist eine reine Block-Umordnung → bei gleichen Eingaben
    numerisch identisch zum Nicht-Swizzle. Größe 1152×256×256 triggert bewusst
    eine partielle letzte Gruppe (num_pid_m=9 > GROUP_M=8 → min-Zweig)."""
    M, N, K = 1152, 256, 256
    _, _, Cn = _run_gemm(M, N, K, swizzle=False)
    _, _, Cs = _run_gemm(M, N, K, swizzle=True)
    d = (Cs.float() - Cn.float()).abs().max().item()
    assert d < 1e-4, f"Swizzle weicht vom Nicht-Swizzle ab: max_diff={d:.3e}"


def test_tile_matrix_correct_across_sizes_and_swizzle():
    """verify-before-trust je Tile: jede TM/TN/TK-Kombi (inkl. 16³ + 256er) stimmt
    gegen fp32 — über glatte UND ragged Größen, mit UND ohne Swizzle. Das ist die
    zentrale Silent-wrong-answer-Abwehr der TZ-4-Kachel/Swizzle-Achse."""
    for tile in _TZ4_TILES:
        for sw in (False, True):
            for (M, N, K) in _TZ4_SIZES:
                _assert_matches_fp32(M, N, K, "fp16", "fp32", tile=tile, swizzle=sw)


def test_tile_matrix_orientation():
    """Orientierungs-Wächter je Tile × Swizzle: jede Kombi rechnet exakt A@B,
    keinen transponierten Doppelgänger (Risiko ① je neuer Tile/Swizzle-Quelle)."""
    for tile in _TZ4_TILES:
        for sw in (False, True):
            _assert_orientation("fp16", "fp32", tile=tile, swizzle=sw)


def test_naive_baseline_tile_verifies():
    """Explizit: das 16³-Tile der naive-cuTile-Baseline stimmt gegen fp32 + korrekte
    Orientierung — sonst wären die als Untergrenze gezeigten TFLOP/s nicht ehrlich."""
    _assert_matches_fp32(512, 512, 512, "fp16", "fp32", tile={"TM": 16, "TN": 16, "TK": 16})
    _assert_orientation("fp16", "fp32", tile={"TM": 16, "TN": 16, "TK": 16})


def test_tile_dtype_cross_check():
    """Nicht-Default-Tile (64³) über mehrere dtypes × Swizzle: die Kachel-Achse ist
    im Codegen dtype-unabhängig — hier einmal real über bf16/tf32/fp8 belegt."""
    tile = {"TM": 64, "TN": 64, "TK": 32}
    for dtype, acc in [("bf16", "fp32"), ("tf32", "fp32"), ("fp8e4m3", "fp16")]:
        _assert_matches_fp32(256, 256, 256, dtype, acc, tile=tile, swizzle=True)


def test_gemm_bf16_across_sizes():
    """bf16→fp32 (Akku fp32 = Pflicht): stimmt gegen fp32 — glatt UND ragged.

    bf16 ist nativ (kein Kernel-Cast, Akku fp32) → derselbe erzeugte Kernel wie
    fp16, nur mit bf16-Operanden; prüft, dass der Codegen dtype-agnostisch bleibt.
    """
    for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
        _assert_matches_fp32(M, N, K, "bf16", "fp32")


def test_gemm_bf16_orientation():
    """bf16: Orientierungs-Wächter (rechnet A@B, keinen Doppelgänger)."""
    _assert_orientation("bf16", "fp32")


def test_gemm_tf32_across_sizes():
    """tf32→fp32: fp32-Eingaben, im Kernel auf tfloat32 gecastet; stimmt gegen fp32."""
    for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
        _assert_matches_fp32(M, N, K, "tf32", "fp32")


def test_gemm_tf32_orientation():
    """tf32: Orientierungs-Wächter."""
    _assert_orientation("tf32", "fp32")


def test_emit_tf32_has_astype_cast():
    """Performance-Wächter: der tf32-Kernel MUSS ct.astype(.., ct.tfloat32) VOR
    ct.mma enthalten — ohne Cast liefe fp32 still auf CUDA-Cores (rechnerisch
    korrekt, ~30x langsamer), was ein reiner Korrektheitstest NICHT fängt.
    fp16/bf16 dürfen umgekehrt KEINEN solchen Input-Cast haben.
    """
    tile = {"TM": 128, "TN": 128, "TK": 64}
    src = build_gemm_module(tile, "tf32", "fp32")
    assert "ct.astype(a, ct.tfloat32)" in src and "ct.astype(b, ct.tfloat32)" in src, \
        "tf32-Kernel fehlt der tfloat32-Cast (Tensor-Core-Pfad)"
    assert "ct.tfloat32" not in build_gemm_module(tile, "fp16", "fp32"), \
        "fp16-Kernel darf keinen tfloat32-Cast enthalten"


def test_build_gemm_module_rejects_unknown_input_dtype():
    """Unbekannter Input-dtype → ValueError (statt still einen Kernel ohne den
    ggf. nötigen Cast zu emittieren)."""
    try:
        build_gemm_module({"TM": 128, "TN": 128, "TK": 64}, "fp4", "fp32")
    except ValueError:
        return
    raise AssertionError("build_gemm_module hätte bei dtype='fp4' ValueError werfen müssen")


def test_gemm_fp16_fp16_acc():
    """fp16→fp16 (fp16-Akku + fp16-Output): der neue Akku-Pfad auch ohne fp8."""
    for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
        _assert_matches_fp32(M, N, K, "fp16", "fp16")


def test_gemm_fp8e4m3_verify():
    """fp8 e4m3 × Akku fp32 UND fp16 (der schnellste Kandidat): stimmt gegen fp32.

    fp8 braucht KEINEN Kernel-Cast (Host-seitig gecastet); der Kernel rechnet die
    fp8-Tiles direkt. Geprüft mit glatten UND ragged Größen.
    """
    for acc in ("fp32", "fp16"):
        for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
            _assert_matches_fp32(M, N, K, "fp8e4m3", acc)


def test_gemm_fp8e5m2_verify():
    """fp8 e5m2 × Akku fp32 UND fp16: stimmt gegen fp32 (glatt + ragged)."""
    for acc in ("fp32", "fp16"):
        for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
            _assert_matches_fp32(M, N, K, "fp8e5m2", acc)


def test_gemm_fp8_orientation():
    """fp8: Orientierungs-Wächter für BEIDE Akku-Kernel (fp32- und fp16-Akku)."""
    _assert_orientation("fp8e4m3", "fp32")
    _assert_orientation("fp8e4m3", "fp16")


def test_gemm_fp32_plain_verify():
    """fp32-plain (Anker/Diagnose, kein Tensor-Core): baubar UND gegen fp32
    verifiziert (schließt die 'erlaubt aber nicht baubar'-Lücke)."""
    for (M, N, K) in [(256, 256, 256), (130, 100, 70)]:
        _assert_matches_fp32(M, N, K, "fp32", "fp32")


def test_fp8_variants_differ():
    """e4m3 und e5m2 liefern echt verschiedene Ergebnisse (kein stiller Upcast
    auf ein gemeinsames Format) — beide fp8-Pfade sind real unterschiedlich."""
    _, _, C4 = _run_gemm(256, 256, 256, "fp8e4m3", "fp32")
    _, _, C5 = _run_gemm(256, 256, 256, "fp8e5m2", "fp32")
    diff = (C4.float() - C5.float()).abs().max().item()
    assert diff > 1e-2, f"e4m3 vs e5m2 verdächtig gleich: max_diff={diff:.3e}"


def test_emit_deterministic():
    """Gleiche Config → byte-identischer Quelltext (Cache-/Reproduzierbarkeit)."""
    assert emit(RunConfig()) == emit(RunConfig())


def test_build_gemm_module_rejects_unknown_acc():
    """Unbekannter Akkumulator-dtype → ValueError (statt still falschem Kernel)."""
    try:
        build_gemm_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "int4")
    except ValueError:
        return
    raise AssertionError("build_gemm_module hätte bei acc_dtype='int4' ValueError werfen müssen")


def test_run_end_to_end_ok():
    """Integration: run() über die ganze Pipeline → status ok, verifiziert."""
    from tool_pipeline.store import store as st
    import tool_pipeline.run as R

    orig = st.append_result
    st.append_result = lambda r, path=None: None   # Store isolieren (kein echter Append)
    try:
        res = R.run(RunConfig())
    finally:
        st.append_result = orig
    assert res.status == "ok", f"status={res.status} error={res.error}"
    assert res.accuracy["passed"] and res.metrics["tflops"] > 0


def test_run_returns_result_on_compile_error():
    """Nicht baubare Configs → RunResult mit compile_error, NIE eine Exception."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    orig = st.append_result
    st.append_result = lambda r, path=None: None
    try:
        for cfg in [
            RunConfig(expr="ij,jk,kl->il"),               # n-är (>2 Operanden) → parse lehnt ab
            RunConfig(family="elementwise"),              # falsche Familie → parse lehnt ab
            RunConfig(dtype="bf16", acc_dtype="fp16"),    # unzulässige Acc-Kombi → check_dtype_combo
        ]:
            res = R.run(cfg)
            assert isinstance(res, RunResult), "run() darf nicht werfen"
            assert res.status == "compile_error", f"{cfg.expr}/{cfg.family}/{cfg.dtype}: {res.status}"
            assert res.error
    finally:
        st.append_result = orig


def test_run_verify_failed_status():
    """Falsche Kernel-Zahlen → verify_failed (verify gemonkeypatcht); keine Metriken."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    orig_store, orig_verify = st.append_result, R.verify
    st.append_result = lambda r, path=None: None
    R.verify = lambda C, A, B, cfg: {"max_abs_err": 999.0, "passed": False, "atol": 0.2, "rtol": 0.02}
    try:
        res = R.run(RunConfig())
        assert res.status == "verify_failed", f"status={res.status}"
        assert res.accuracy["passed"] is False and res.metrics == {}
    finally:
        st.append_result, R.verify = orig_store, orig_verify


def test_run_error_on_launch_failure():
    """Laufzeit-Crash in der Messung → run_error (kein Raise)."""
    import tool_pipeline.run as R
    from tool_pipeline.store import store as st

    def _boom(*a, **k):
        raise RuntimeError("simulierter Launch-Crash")

    orig_store, orig_bench = st.append_result, R.benchmark
    st.append_result = lambda r, path=None: None
    R.benchmark = _boom
    try:
        res = R.run(RunConfig())
        assert res.status == "run_error", f"status={res.status}"
        assert res.error
    finally:
        st.append_result, R.benchmark = orig_store, orig_bench


def _main() -> int:
    clear_cache()
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
