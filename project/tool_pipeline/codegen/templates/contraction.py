"""GEMM-Template des Codegen (C1).

Erzeugt vollstaendigen, self-contained cuTile-Modul-Quelltext fuer das
**kanonische Batched-GEMM** ``(B,M,K) x (B,K,N) -> (B,M,N)``. Jede
2-Operanden-Kontraktion wird host-seitig (B1-Reshape, ``reshape.to_canonical``)
auf genau diese Form gebracht -> der Codegen emittiert die EINE bewiesene
Struktur. Der emittierte Modul enthaelt ``@ct.kernel gemm(...)`` + ``launch(A, B,
C)``. ``B=1`` (Plain-GEMM) ist der Sonderfall Grid-Z=1 derselben Struktur.

Vorlage/Orientierung gespiegelt aus:
  * ``assignments/05_assignment/src/kernel.py`` (kanonisches Batched-GEMM,
    no-swap-Blaupause) + ``project/project-development/analysis/dtype_analyse.py``.

Bewiesene Orientierung (drei unabhaengige GPU-Verifikationen, GB10 sm_121):
je Batch ``bb`` + k-Kachel ``a = load(A, (bb, i, kk), (1, TM, TK))`` -> ``(TM,
TK)``, ``b = load(B, (bb, kk, j), (1, TK, TN))`` -> ``(TK, TN)``, ``acc =
ct.mma(a, b, acc)`` -> ``(TM, TN)``. KEIN Operanden-Swap, KEIN Permute.
"""

# ---------------------------------------------------------------------------
# acc-dtype-Mapping: Akkumulator-Label -> ct-dtype-Ausdruck, als Literal in den
# Kernel substituiert. fp32 und fp16 sind nutzbar (fp16-/fp8-Compute duerfen in
# fp16 akkumulieren — schneller). bf16/tf32 sind reine COMPUTE-Formate und NIE
# Akkumulatoren (sie akkumulieren zwingend in fp32) — daher hier bewusst nicht.
# ---------------------------------------------------------------------------
_ACC_DTYPE_MAP = {
    "fp32": "ct.float32",
    "fp16": "ct.float16",
}


# ---------------------------------------------------------------------------
# Input-dtype-Mapping: braucht das Format einen Cast VOR ct.mma?
# tf32 hat auf diesem cuTile-Build KEIN mma-Flag — fp32-Daten muessen im Kernel
# per ct.astype(.., ct.tfloat32) auf den Tensor-Core-Pfad gecastet werden (ohne
# Cast liefe es still auf CUDA-Cores, ~0.2 statt ~6 TFLOP/s: rechnerisch korrekt,
# aber falsches Tempo). fp16/bf16 sind nativ, fp8 wird host-seitig gecastet ->
# alle brauchen KEINEN Kernel-Cast (Wert None). Zugleich die Validierungs-
# Whitelist der zulaessigen Input-dtypes.
# ---------------------------------------------------------------------------
_INPUT_CAST = {
    "fp16": None,
    "bf16": None,
    "tf32": "ct.tfloat32",
    "fp8e4m3": None,
    "fp8e5m2": None,
    "fp32": None,
}


# GROUP_M der L2-Swizzle-Rasterung (grouped-M): wie viele M-Kachel-Zeilen zu einer
# L2-lokalen Gruppe zusammengefasst werden. Fester Wert — der Swizzle ist ein
# Ein/Aus-Schalter (Control), GROUP_M die bewaehrte Standard-Gruppe (wie Triton).
_SWIZZLE_GROUP_M = 8


def build_gemm_module(tile: dict, dtype: str, acc_dtype: str,
                      swizzle: bool = False) -> str:
    """Baue den cuTile-Modul-Quelltext fuer ein kanonisches Batched-GEMM
    ``(B,M,K) x (B,K,N) -> (B,M,N)`` (B=1 = Plain-GEMM als Grid-Z=1).

    :param tile:      Tile-Literale ``{"TM": .., "TN": .., "TK": ..}``. Werden
                      als Zahlen-Literale fest in den Quelltext gebacken.
    :param dtype:     Input-dtype-Label (z.B. ``"fp16"``). Steht NICHT im
                      Kernel-Koerper (Inputs sind Laufzeit-torch-dtype), sondern
                      nur zur Doku im Docstring des generierten Moduls.
    :param acc_dtype: Akkumulator-dtype-Label; ueber ``_ACC_DTYPE_MAP`` auf den
                      ct-dtype-Ausdruck abgebildet.
    :param swizzle:   L2-Swizzle (grouped-M-Rasterung) an/aus. Bei ``False`` ist
                      der erzeugte Quelltext **byte-identisch** zu TZ 1-3; bei
                      ``True`` wird nur die Block->Kachel-Zuordnung bijektiv
                      umgeordnet (Orientierung/mma bleiben unberuehrt).
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

    if dtype not in _INPUT_CAST:
        raise ValueError(
            f"input-dtype {dtype!r} nicht unterstuetzt "
            f"(verfuegbar: {sorted(_INPUT_CAST)})"
        )
    cast_ct = _INPUT_CAST[dtype]

    try:
        tm = int(tile["TM"])
        tn = int(tile["TN"])
        tk = int(tile["TK"])
    except KeyError as e:
        raise ValueError(f"tile-dict fehlt Schluessel {e}") from e

    # Optionaler Input-Cast VOR ct.mma (nur tf32) — als Quelltext-Bloecke, damit
    # der emittierte Kernel byte-stabil und selbstdokumentierend bleibt.
    if cast_ct is not None:
        dtype_doc = (f"Input-dtype: {dtype} -> im Kernel via ct.astype auf {cast_ct} "
                     f"gecastet (VOR ct.mma; ohne Cast liefe es auf CUDA-Cores).")
        cast_block = (f"        a = ct.astype(a, {cast_ct})\n"
                      f"        b = ct.astype(b, {cast_ct})\n")
    else:
        dtype_doc = f"Input-dtype: {dtype} (Laufzeit-torch-dtype, steht NICHT im Kernel-Koerper)."
        cast_block = ""

    # Swizzle-Bloecke: bei swizzle=False EXAKT die TZ-1-Zeilen (byte-identisch);
    # bei swizzle=True eine bijektive grouped-M-Rasterung der Block->Kachel-Zuordnung.
    if swizzle:
        group_m = _SWIZZLE_GROUP_M
        swizzle_doc = (f" L2-Swizzle EIN (grouped-M-Rasterung, GROUP_M={group_m}): i/j "
                       f"werden bijektiv umgeordnet (dieselbe Kachelmenge, L2-freundlichere Reihenfolge).")
        group_const = f"GROUP_M = {group_m}\n"
        bid_block = (
            "    # L2-Swizzle: grouped-M-Rasterung — dieselben (i, j) wie ohne Swizzle,\n"
            "    # nur in L2-freundlicherer Reihenfolge (Bloecke einer Gruppe teilen sich\n"
            "    # B-Spalten). Bijektiv -> Ergebnis unveraendert; Orientierung/mma unberuehrt.\n"
            "    num_pid_m = ct.cdiv(M, TM)\n"
            "    num_pid_n = ct.cdiv(N, TN)\n"
            "    pid = ct.bid(0) * num_pid_n + ct.bid(1)\n"
            "    num_pid_in_group = GROUP_M * num_pid_n\n"
            "    group_id = pid // num_pid_in_group\n"
            "    first_pid_m = group_id * GROUP_M\n"
            "    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)\n"
            "    local = pid % num_pid_in_group\n"
            "    i = first_pid_m + (local % group_size_m)\n"
            "    j = local // group_size_m\n"
            "    bb = ct.bid(2)   # Batch-Achse (Swizzle ordnet nur i/j um)\n"
        )
    else:
        swizzle_doc = ""
        group_const = ""
        bid_block = (
            "    # 3D-Grid: i ueber M-Kacheln, j ueber N-Kacheln, bb ueber den Batch.\n"
            "    i = ct.bid(0)\n"
            "    j = ct.bid(1)\n"
            "    bb = ct.bid(2)\n"
        )

    return f'''"""Generierter cuTile-GEMM (Codegen C1) — kanonisches Batched-GEMM (B,M,K)x(B,K,N)->(B,M,N).

Jede 2-Operanden-Kontraktion wird host-seitig (B1-Reshape) auf diese Form
gebracht; dieser Kernel emittiert die EINE bewiesene Struktur (B=1 = Plain-GEMM).
{dtype_doc}
Akkumulator: {acc_dtype} ({acc_ct}).
Tile-Literale: TM={tm}, TN={tn}, TK={tk} (fest in den Quelltext gebacken).

Bewiesene Orientierung: a=(TM,TK), b=(TK,TN), ct.mma(a,b,acc)->(TM,TN),
KEIN Operanden-Swap, KEIN Permute. i=bid(0)=M-Kachel, j=bid(1)=N-Kachel,
bb=bid(2)=Batch.{swizzle_doc}
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = {tm}
TN = {tn}
TK = {tk}
{group_const}

@ct.kernel
def gemm(A, B, C,
         M: ct.Constant[int],
         N: ct.Constant[int],
         K: ct.Constant[int]):
    """Berechne eine (TM, TN)-Ausgabekachel von C[bb] = A[bb] @ B[bb]."""
{bid_block}
    # Akkumulator unabhaengig vom Input-dtype (Standardmuster aus cuTile).
    acc = ct.full((TM, TN), 0, dtype={acc_ct})

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
{cast_block}        acc = ct.mma(a, b, acc)

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

        # Batched (B=2), um die neue Batch-Achse zu ueben; B=1 ist der Sonderfall.
        Bb, M, N, K = 2, 512, 512, 512
        torch.manual_seed(0)
        A = torch.randn(Bb, M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(Bb, K, N, dtype=torch.float16, device="cuda")
        # Konvention: Output-dtype = acc_dtype (hier fp32). So bleibt die
        # fp32-Akku-Praezision erhalten (ehrliches Ergebnis, max_err ~1e-4).
        C = torch.empty(Bb, M, N, dtype=torch.float32, device="cuda")

        launch(A, B, C)
        torch.cuda.synchronize()

        ref = torch.einsum("bik,bkj->bij", A.float(), B.float())
        max_abs_err = (C.float() - ref).abs().max().item()
        ok = torch.allclose(C.float(), ref, atol=2e-1, rtol=2e-2)
        print(f"batched {Bb}x512^3 fp16->fp32: max_abs_err={max_abs_err:.3e} allclose={ok}")
        assert ok, "generierter batched GEMM-Modul stimmt nicht gegen torch.einsum"
        print("OK: generierter Modul laeuft und stimmt.")
