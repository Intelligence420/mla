"""Elementwise-Template des Codegen (C1) — memory-bound.

Erzeugt vollstaendigen, self-contained cuTile-Modul-Quelltext fuer eine
**elementweise Abbildung** ``C = A (op) B`` (binaer: add/mul) bzw. ``C = A``
(unaer: copy). Der Host faltet den/die Operanden vorab auf eine 2D-Sicht
``(rows, cols)`` mit ``cols`` = letzter (kontiguierter) Achse (``run.py``, TZ 7) —
der Kernel kachelt sie mit einem echten ``cdiv``-2D-Grid. **Kein ``ct.mma``, kein
FP32-Akku, kein B1-Reshape** — reine Load->Op->Store-Struktur (niedrige
arithmetische Intensitaet ⇒ Roofline weit links; ``copy`` = reine Bandbreite).

Vorlage/Orientierung: ``assignments/02_assignment/src/task_04.py`` (Copy, echtes
cdiv-2D-Grid) als Skelett + ``task_03.py`` (binaere Add) fuer die Op. **Bewiesen
(task_03-Benchmark): die kontiguierten inneren Achsen kacheln (~3,5x schneller)** —
daher tiled der Host die letzte Achse als ``cols``. Struktur (Signatur/Doc/
``launch``/``__main__``-Selbsttest) gespiegelt aus ``contraction.build_gemm_module``.

Die Op wird — analog ``cast_block``/``bid_block`` beim GEMM — als f-String-Fragment
substituiert; ``op`` bestimmt zugleich die **Arity** (binaer ⇒ ``launch(A, B, C)``,
unaer ⇒ ``launch(A, C)``).
"""

# ---------------------------------------------------------------------------
# Op-Katalog: Label -> (Arity, Quelltext-Fragment, Doku). Zugleich die
# Validierungs-Whitelist der zulaessigen Elementwise-Ops.
# ---------------------------------------------------------------------------
_OPS = {
    "add":  {"arity": 2, "frag": "a + b", "doc": "elementweise Summe A + B (binaer)"},
    "mul":  {"arity": 2, "frag": "a * b", "doc": "elementweise Produkt A * B (binaer)"},
    "copy": {"arity": 1, "frag": "a",     "doc": "reine Kopie A — nur Bandbreite (unaer)"},
    # ReLU (TZ 9): unaere elementweise Op max(A, 0). Dient zugleich als sequentieller
    # Zwilling der Kontraktions-Epilog-Fusion ``epilog="relu"`` (Plain-Kontraktion +
    # separater ReLU-Lauf) fuer den fused-vs-sequentiell-Vergleich.
    "relu": {"arity": 1, "frag": "ct.maximum(a, 0)", "doc": "elementweise ReLU max(A, 0) (unaer)"},
}


def build_elementwise_module(tile: dict, dtype: str, acc_dtype: str, op: str) -> str:
    """Baue den cuTile-Modul-Quelltext fuer eine elementweise Abbildung.

    :param tile:      Tile-Literale ``{"TM": .., "TN": ..}`` (TK ist fuer
                      Elementwise bedeutungslos). Werden als Zahlen-Literale fest
                      in den Quelltext gebacken.
    :param dtype:     Input-dtype-Label (nur Doku; Inputs sind Laufzeit-torch-dtype).
    :param acc_dtype: Ausgabe-dtype-Label (nur Doku; der Output-dtype wird ueber
                      ``C.dtype`` beim Store gesetzt — kein Akku bei Elementwise).
    :param op:        ``"add"``/``"mul"`` (binaer) oder ``"copy"`` (unaer). Bestimmt
                      Op-Fragment **und** Arity.
    :returns:         Vollstaendiger, ausfuehrbarer Modul-Quelltext als String.
                      Consumer-Konvention (compile.py): ``launch(A, B, C)`` (binaer)
                      bzw. ``launch(A, C)`` (unaer), C vorab alloziert.
    """
    if op not in _OPS:
        raise ValueError(
            f"Elementwise-Op {op!r} nicht unterstuetzt (verfuegbar: {sorted(_OPS)})."
        )
    spec = _OPS[op]
    arity, frag, op_doc = spec["arity"], spec["frag"], spec["doc"]

    try:
        tm = int(tile["TM"])
        tn = int(tile["TN"])
    except KeyError as e:
        raise ValueError(f"tile-dict fehlt Schluessel {e}") from e

    # Arity-abhaengige Bloecke (binaer laedt A+B, unaer nur A) — analog den
    # bid_block/cast_block-Substitutionen des GEMM-Templates.
    if arity == 2:
        kern_sig = "def elementwise(A, B, C):"
        load_block = (
            "    a = ct.load(A, index=(i, j), shape=(TM, TN), padding_mode=ct.PaddingMode.ZERO)\n"
            "    b = ct.load(B, index=(i, j), shape=(TM, TN), padding_mode=ct.PaddingMode.ZERO)\n"
        )
        launch_sig = "def launch(A, B, C):"
        launch_args = "(A, B, C)"
    else:
        kern_sig = "def elementwise(A, C):"
        load_block = (
            "    a = ct.load(A, index=(i, j), shape=(TM, TN), padding_mode=ct.PaddingMode.ZERO)\n"
        )
        launch_sig = "def launch(A, C):"
        launch_args = "(A, C)"

    return f'''"""Generierter cuTile-Elementwise-Kernel (Codegen C1) — memory-bound, {op}.

{op_doc}. Kachelt die 2D-Sicht (rows, cols) mit echtem cdiv-2D-Grid (Vorlage A02
task_03/04). KEIN ct.mma, kein Akku, kein B1-Reshape.
Input-dtype: {dtype} (Laufzeit-torch-dtype). Ausgabe-dtype: {acc_dtype} (via C.dtype).
Tile-Literale: TM={tm}, TN={tn} (fest in den Quelltext gebacken; TK fuer Elementwise
bedeutungslos). Arity: {arity} ({"launch(A, B, C)" if arity == 2 else "launch(A, C)"}).
"""

import cuda.tile as ct
import torch

# Tile-Literale (aus der Config in den Quelltext substituiert)
TM = {tm}
TN = {tn}


@ct.kernel
{kern_sig}
    """Verarbeite eine (TM, TN)-Kachel: i=bid(0) ueber die Zeilen, j=bid(1) ueber
    die (kontiguierten) Spalten. ZERO-Padding am Rand; ct.store schneidet
    out-of-bounds Elemente automatisch ab."""
    i = ct.bid(0)
    j = ct.bid(1)
{load_block}    ct.store(C, index=(i, j), tile=ct.astype({frag}, C.dtype))


{launch_sig}
    """Starte den Elementwise-Kernel auf der 2D-Sicht (rows, cols) (C vorab
    alloziert). Grid = (cdiv(rows, TM), cdiv(cols, TN), 1)."""
    M, N = A.shape
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid, elementwise, {launch_args})
    return C
'''


# ---------------------------------------------------------------------------
# Selbsttest: emittierten Text ausfuehren und GPU-Verifikationslaeufe gegen torch
# fahren — add/mul (binaer) in fp16/bf16/fp32, copy (unaer) zusaetzlich in fp8
# (nur load->store ⇒ auch in 8 Bit erlaubt; Arithmetik ist es NICHT, s. Pre-flight).
# Datei-basierter Ladepfad (cuTile liest die Source per inspect), wie contraction.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib.util
    import os
    import tempfile

    import torch

    def _load(op):
        src = build_elementwise_module({"TM": 128, "TN": 128, "TK": 64}, "fp16", "fp32", op)
        tmp = tempfile.mkdtemp()
        mod_path = os.path.join(tmp, f"generated_elementwise_{op}.py")
        with open(mod_path, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location(f"generated_elementwise_{op}", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch

    torch.manual_seed(0)
    _TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
              "fp8e4m3": torch.float8_e4m3fn, "fp8e5m2": torch.float8_e5m2}

    def _rand(shape, dt):
        if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            return torch.randn(*shape, dtype=torch.float16, device="cuda").to(dt)
        return torch.randn(*shape, dtype=dt, device="cuda")

    # binaere Ops (add/mul) in fp16/bf16/fp32, inkl. ragged (128x100).
    for op, ref in [("add", lambda a, b: a + b), ("mul", lambda a, b: a * b)]:
        launch = _load(op)
        for label in ("fp16", "bf16", "fp32"):
            dt = _TORCH[label]
            for (M, N) in [(256, 128), (128, 100)]:
                a = _rand((M, N), dt)
                b = _rand((M, N), dt)
                C = torch.empty(M, N, dtype=torch.float32, device="cuda")
                launch(a, b, C)
                torch.cuda.synchronize()
                exp = ref(a.float(), b.float())
                err = (C.float() - exp).abs().max().item()
                ok = torch.allclose(C.float(), exp, atol=1e-1, rtol=1e-2)
                print(f"  {op} {label} ({M},{N}): max_abs_err={err:.3e} allclose={ok}")
                assert ok, f"Elementwise {op} {label} stimmt nicht"

    # unaere copy in allen dtypes (auch fp8 — nur Bandbreite), exakt.
    launch = _load("copy")
    for label in ("fp16", "bf16", "fp32", "fp8e4m3", "fp8e5m2"):
        dt = _TORCH[label]
        a = _rand((256, 100), dt)
        C = torch.empty(256, 100, dtype=dt, device="cuda")
        launch(a, C)
        torch.cuda.synchronize()
        ok = torch.equal(C.float(), a.float())
        print(f"  copy {label} (256,100): exakt={ok}")
        assert ok, f"Elementwise copy {label} stimmt nicht"

    # unaere relu (arithmetisch) in fp16/bf16/fp32, inkl. ragged (128x100).
    launch = _load("relu")
    for label in ("fp16", "bf16", "fp32"):
        dt = _TORCH[label]
        for (M, N) in [(256, 128), (128, 100)]:
            a = _rand((M, N), dt)
            C = torch.empty(M, N, dtype=torch.float32, device="cuda")
            launch(a, C)
            torch.cuda.synchronize()
            exp = a.float().clamp(min=0)
            err = (C.float() - exp).abs().max().item()
            ok = torch.allclose(C.float(), exp, atol=1e-1, rtol=1e-2)
            print(f"  relu {label} ({M},{N}): max_abs_err={err:.3e} allclose={ok}")
            assert ok, f"Elementwise relu {label} stimmt nicht"

    print("OK: generierter Elementwise-Modul laeuft und stimmt (add/mul/copy/relu).")
