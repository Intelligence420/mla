"""GEMM-Template des Codegen (C1).

Erzeugt vollstaendigen, self-contained cuTile-Modul-Quelltext fuer genau eine
Kontraktion ``ik,kj->ij`` (Plain-GEMM). Der emittierte Modul enthaelt einen
``@ct.kernel gemm(...)`` und eine Top-Level-Funktion ``launch(A, B, C)``.

Vorlage/Orientierung gespiegelt aus:
  * ``assignments/03_assignment/src/task_02.py`` (saubere Plain-GEMM-Vorlage)
  * ``project/project-development/analysis/dtype_analyse.py`` (Batched-GEMM,
    bewiesene Orientierung).

Bewiesene Orientierung (drei unabhaengige GPU-Verifikationen, GB10 sm_121):
pro k-Kachel ``a = load(A, (i, kk), (TM, TK))``, ``b = load(B, (kk, j),
(TK, TN))``, ``acc = ct.mma(a, b, acc)`` -> ``(TM, TN)``. KEIN Operanden-Swap,
KEIN Permute.
"""

# ---------------------------------------------------------------------------
# acc-dtype-Mapping: erweiterbar fuer TZ3 (bf16/tf32/fp8). In TZ1 nur fp32.
# Der String kommt aus der IR/Config; hier wird er auf den ct-dtype-Ausdruck
# abgebildet, der als Literal in den Kernel-Quelltext substituiert wird.
# ---------------------------------------------------------------------------
_ACC_DTYPE_MAP = {
    "fp32": "ct.float32",
    # TZ3 (noch NICHT implementiert):
    # "bf16": "ct.bfloat16",
    # "tf32": "ct.tfloat32",
    # "fp16": "ct.float16",
}


def build_gemm_module(tile: dict, dtype: str, acc_dtype: str) -> str:
    """Baue den cuTile-Modul-Quelltext fuer ein Plain-GEMM ``ik,kj->ij``.

    :param tile:      Tile-Literale ``{"TM": .., "TN": .., "TK": ..}``. Werden
                      als Zahlen-Literale fest in den Quelltext gebacken.
    :param dtype:     Input-dtype-Label (z.B. ``"fp16"``). Steht NICHT im
                      Kernel-Koerper (Inputs sind Laufzeit-torch-dtype), sondern
                      nur zur Doku im Docstring des generierten Moduls.
    :param acc_dtype: Akkumulator-dtype-Label; ueber ``_ACC_DTYPE_MAP`` auf den
                      ct-dtype-Ausdruck abgebildet.
    :returns:         Vollstaendiger, ausfuehrbarer Modul-Quelltext als String.
                      Konvention fuer den compile.py-Consumer: der Modul
                      definiert eine Funktion ``launch(A, B, C)`` (C ist vorab
                      alloziert, der Aufrufer kontrolliert den Output-dtype).
    """
    if acc_dtype not in _ACC_DTYPE_MAP:
        raise ValueError(
            f"acc_dtype {acc_dtype!r} nicht unterstuetzt "
            f"(verfuegbar: {sorted(_ACC_DTYPE_MAP)})"
        )
    acc_ct = _ACC_DTYPE_MAP[acc_dtype]

    try:
        tm = int(tile["TM"])
        tn = int(tile["TN"])
        tk = int(tile["TK"])
    except KeyError as e:
        raise ValueError(f"tile-dict fehlt Schluessel {e}") from e

    return f'''"""Generierter cuTile-GEMM (Codegen C1) — Kontraktion ik,kj->ij.

Input-dtype: {dtype} (Laufzeit-torch-dtype, steht NICHT im Kernel).
Akkumulator: {acc_dtype} ({acc_ct}).
Tile-Literale: TM={tm}, TN={tn}, TK={tk} (fest in den Quelltext gebacken).

Bewiesene Orientierung: a=(TM,TK), b=(TK,TN), ct.mma(a,b,acc)->(TM,TN),
KEIN Operanden-Swap, KEIN Permute. i=bid(0)=M-Kachel, j=bid(1)=N-Kachel.
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = {tm}
TN = {tn}
TK = {tk}


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
    acc = ct.full((TM, TN), 0, dtype={acc_ct})

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
'''


# ---------------------------------------------------------------------------
# Selbsttest: emittierten Text ausfuehren und EINEN GPU-Verifikationslauf
# (512^3, fp16 -> fp32) gegen torch.einsum fahren. Beweist, dass der generierte
# Quelltext wirklich laeuft und rechnerisch stimmt.
#
# WICHTIG: cuTile liest den Kernel-Quelltext via ``inspect.getsourcelines`` und
# braucht dafuer eine ECHTE Datei auf der Platte -- ein reines ``exec`` aus einem
# String scheitert mit ``OSError: could not get source code``. Der spaetere
# compile.py-Consumer schreibt den emittierten Text ohnehin nach
# ``results/kernels/<slug>.py`` und importiert von dort; dieser Selbsttest
# spiegelt genau diesen Datei-basierten Ladepfad.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib.util
    import tempfile
    import os

    import torch

    src = build_gemm_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "fp32")

    # Quelltext in eine echte Datei schreiben und als Modul laden (Datei-Pfad,
    # damit cuTile die Kernel-Source per inspect finden kann).
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = os.path.join(tmp, "generated_gemm.py")
        with open(mod_path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("generated_gemm", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launch = mod.launch

        M = N = K = 512
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        # Konvention: Output-dtype = acc_dtype (hier fp32). So bleibt die
        # fp32-Akku-Praezision erhalten (ehrliches Ergebnis, max_err ~1e-4).
        # Ein fp16-Output wuerde beim Store auf fp16 runden (~3e-2) und die
        # Akku-Genauigkeit verschenken.
        C = torch.empty(M, N, dtype=torch.float32, device="cuda")

        launch(A, B, C)
        torch.cuda.synchronize()

        ref = torch.einsum("ik,kj->ij", A.float(), B.float())
        max_abs_err = (C.float() - ref).abs().max().item()
        ok = torch.allclose(C.float(), ref, atol=2e-1, rtol=2e-2)
        print(f"512^3 fp16->fp32: max_abs_err={max_abs_err:.3e} allclose={ok}")
        assert ok, "generierter GEMM-Modul stimmt nicht gegen torch.einsum"
        print("OK: generierter Modul laeuft und stimmt.")
