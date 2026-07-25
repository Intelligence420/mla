"""Reduktions-Template des Codegen (C1) — memory-bound.

Erzeugt vollstaendigen, self-contained cuTile-Modul-Quelltext fuer eine
**zeilenweise Summe** ``(rows, K) -> (rows,)`` ueber die reduzierte Achse. Der
Host permutiert+faltet den Operanden vorab auf diese 2D-Form ``(kept_size,
reduced_size)`` (``run.py``, TZ 7) — der Kernel sieht immer eine 2D-Matrix und
summiert axis=1. **Kein ``ct.mma``, kein FP32-Akku-Loop im GEMM-Sinn, kein
B1-Reshape** — das ist der Kern der memory-bound-Familie (niedrige arithmetische
Intensitaet ⇒ Roofline weit links).

Vorlage/Orientierung: ``assignments/02_assignment/src/task_02.py`` (row_sum,
``ct.sum(tile, axis=1)``, ``grid=(M,1,1)``, ``TILE_K`` = next-pow2(K), ZERO-Padding
neutral fuer die Summe). Struktur (Signatur/Doc/``launch``/``__main__``-Selbsttest)
gespiegelt aus ``contraction.build_gemm_module`` — nur ohne Tensor-Core.

**Zwei Pfade im erzeugten Modul (verify-before-trust!):**
  * **single-shot** (A02-bewiesen): passt die reduzierte Achse in EINE Kachel
    (``TILE_K`` = next-pow2(K), als Launch-Constant → Quelltext groessen-unabhaengig
    ⇒ slug-sicher), dann ein einziges ``ct.sum(...)`` — auf dem in den Akku-dtype
    gecasteten Tile, damit ``acc_dtype`` hier genauso wirkt wie im Loop-Pfad
    (siehe unten: **beide** Pfade summieren in ``acc_dtype``).
  * **K-Loop-Fallback** (NICHT in A02 bewiesen, klar markiert): reduzierte Achse
    groesser als eine Kachel → GEMM-artiger Akku-Loop ``acc += ct.sum(chunk)`` mit
    festem Chunk ``LOOP_TILE`` (= ``tile["TK"]``, steht im Slug). ``launch`` waehlt
    den Pfad zur Laufzeit anhand der Groesse.
"""

# ---------------------------------------------------------------------------
# acc-dtype-Mapping fuer den (Fallback-)Akkumulator. Bewusst NICHT ueber das
# GEMM-``_ACC_DTYPE_MAP`` erzwungen — fuer die Reduktion ist ``acc_dtype`` schlicht
# die Ausgabe-/Summen-Praezision (kein Tensor-Core-Akku). fp32 ist Default/genau.
# ---------------------------------------------------------------------------
_ACC_DTYPE_MAP = {
    "fp32": "ct.float32",
    "fp16": "ct.float16",
}

# Bis zu dieser Kachelbreite laeuft der single-shot-Pfad (eine (1, TILE_K)-Kachel).
# Konservativ gewaehlt; darueber greift der markierte K-Loop-Fallback. Als Literal
# in den erzeugten Quelltext gebacken (groessen-unabhaengig ⇒ slug-sicher).
_MAX_SINGLE_SHOT = 16384


def build_reduction_module(tile: dict, dtype: str, acc_dtype: str) -> str:
    """Baue den cuTile-Modul-Quelltext fuer eine zeilenweise Summen-Reduktion.

    :param tile:      Tile-Literale; genutzt wird ``tile["TK"]`` als Chunk-Breite
                      ``LOOP_TILE`` des K-Loop-Fallbacks (steht im Slug). TM/TN sind
                      fuer die Reduktion bedeutungslos.
    :param dtype:     Input-dtype-Label (nur Doku; Inputs sind Laufzeit-torch-dtype).
    :param acc_dtype: Ausgabe-/Summen-Praezision (Fallback-Akku-dtype). Ueber
                      ``_ACC_DTYPE_MAP`` auf den ct-dtype-Ausdruck abgebildet.
    :returns:         Vollstaendiger, ausfuehrbarer Modul-Quelltext als String.
                      Consumer-Konvention (compile.py): das Modul definiert
                      ``launch(A, C)`` (C vorab alloziert, Aufrufer kontrolliert
                      den Output-dtype). **Arity 2** (1 Operand) — vgl. GEMM (3).
    """
    if acc_dtype not in _ACC_DTYPE_MAP:
        raise ValueError(
            f"acc_dtype {acc_dtype!r} nicht unterstuetzt "
            f"(verfuegbar: {sorted(_ACC_DTYPE_MAP)})"
        )
    acc_ct = _ACC_DTYPE_MAP[acc_dtype]

    try:
        loop_tile = int(tile["TK"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"tile-dict fehlt/ungueltiger Schluessel 'TK': {e}") from e

    return f'''"""Generierter cuTile-Reduktions-Kernel (Codegen C1) — memory-bound, zeilenweise Summe.

Reduziert eine 2D-Matrix ``(rows, K)`` entlang axis=1 zu ``(rows,)`` (der Host
permutiert+faltet beliebige Reduktions-Achsen vorab auf diese Form). KEIN ct.mma,
KEIN B1-Reshape — reine Load->ct.sum->Store-Struktur (Vorlage A02 task_02).
Input-dtype: {dtype} (Laufzeit-torch-dtype). Ausgabe-/Akku-Praezision: {acc_dtype} ({acc_ct}).

Zwei Pfade (launch waehlt nach Groesse): single-shot (A02-bewiesen, TILE_K =
next-pow2(K) als Launch-Constant) bzw. K-Loop-Fallback (NICHT in A02 bewiesen,
Chunk LOOP_TILE={loop_tile}).
"""

import cuda.tile as ct
import torch

# Bis zu dieser Kachelbreite: single-shot (eine (1, TILE_K)-Kachel). Darueber der
# markierte K-Loop-Fallback.
MAX_SINGLE_SHOT = {_MAX_SINGLE_SHOT}
# Fester Chunk des Fallback-Loops (aus der Config, steht im Slug).
LOOP_TILE = {loop_tile}


@ct.kernel
def row_sum_single(mat, out, TILE_K: ct.Constant[int]):
    """single-shot (A02 task_02): ganze reduzierte Achse in EINE (1, TILE_K)-Kachel.
    ZERO-Padding ist fuer die Summe neutral (addiert 0). Summiert in {acc_ct}
    (Cast VOR ct.sum, wie im Loop-Pfad) — sonst liefe die Summe im Eingabeformat
    und {acc_dtype} waere auf diesem Pfad wirkungslos."""
    pid = ct.bid(0)   # ein Block je (behaltener) Zeile
    tile = ct.load(mat, index=(pid, 0), shape=(1, TILE_K),
                   padding_mode=ct.PaddingMode.ZERO)
    acc = ct.sum(ct.astype(tile, {acc_ct}), axis=1)
    ct.store(out, index=(pid,), tile=ct.astype(acc, out.dtype))


@ct.kernel
def row_sum_loop(mat, out, K: ct.Constant[int]):
    """FALLBACK (NICHT in A02 bewiesen): reduzierte Achse > eine Kachel → Akku-Loop
    im GEMM-K-Loop-Muster. Chunk-weise laden, in {acc_ct} akkumulieren
    (acc += ct.sum(chunk)), ZERO-Padding am Rand neutral."""
    pid = ct.bid(0)
    acc = ct.full((1,), 0, dtype={acc_ct})
    for kk in range(ct.cdiv(K, LOOP_TILE)):
        tile = ct.load(mat, index=(pid, kk), shape=(1, LOOP_TILE),
                       padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.sum(ct.astype(tile, {acc_ct}), axis=1)
    ct.store(out, index=(pid,), tile=ct.astype(acc, out.dtype))


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def launch(A, C):
    """Reduziere A=(rows, K) zeilenweise → C=(rows,) (C vorab alloziert).

    Grid = (rows, 1, 1), ein Block je Zeile. single-shot, wenn die reduzierte
    Achse (aufgerundet auf next-pow2) in eine Kachel passt; sonst der markierte
    K-Loop-Fallback.
    """
    rows, K = A.shape
    grid = (rows, 1, 1)
    tile_k = _next_pow2(K)
    if tile_k <= MAX_SINGLE_SHOT:
        ct.launch(torch.cuda.current_stream().cuda_stream,
                  grid, row_sum_single, (A, C, tile_k))
    else:
        ct.launch(torch.cuda.current_stream().cuda_stream,
                  grid, row_sum_loop, (A, C, K))
    return C
'''


# ---------------------------------------------------------------------------
# Selbsttest: emittierten Text ausfuehren und GPU-Verifikationslaeufe gegen
# torch.sum fahren — single-shot UND den K-Loop-Fallback. Datei-basierter
# Ladepfad (cuTile liest die Kernel-Source per inspect → echte Datei noetig),
# gespiegelt aus contraction.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib.util
    import os
    import tempfile

    import torch

    src = build_reduction_module({"TM": 128, "TN": 128, "TK": 1024}, "fp16", "fp32")

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = os.path.join(tmp, "generated_reduction.py")
        with open(mod_path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("generated_reduction", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launch = mod.launch

        torch.manual_seed(0)

        def _check(rows, K, label):
            A = torch.randn(rows, K, dtype=torch.float16, device="cuda")
            C = torch.empty(rows, dtype=torch.float32, device="cuda")
            launch(A, C)
            torch.cuda.synchronize()
            ref = torch.sum(A.float(), dim=1)
            err = (C.float() - ref).abs().max().item()
            ok = torch.allclose(C.float(), ref, atol=1e-1, rtol=1e-2)
            print(f"  {label}: rows={rows} K={K} max_abs_err={err:.3e} allclose={ok}")
            assert ok, f"Reduktion stimmt nicht gegen torch.sum ({label})"

        # single-shot (K klein, next-pow2 <= MAX_SINGLE_SHOT)
        _check(64, 128, "single-shot")
        _check(128, 4096, "single-shot (breit)")
        # K-Loop-Fallback erzwingen: next-pow2(20000)=32768 > MAX_SINGLE_SHOT
        _check(48, 20000, "K-Loop-Fallback")
        print("OK: generierter Reduktions-Modul laeuft und stimmt (single-shot + Fallback).")
