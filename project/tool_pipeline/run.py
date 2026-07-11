"""tool_pipeline.run — der Orchestrator UND der Vertrag Core↔GUI.

`run(config) -> RunResult` ist die **einzige Naht**: die Dash-App (TZ 2) ruft
ausschließlich diese Funktion in ihrem Hintergrund-Job. Ablauf (Plan §3):

    parse → reshape(B1) → emit(C1) → compile(+Cache) → Kalt-Lauf(=compile_ms)
          → verify(fp32) → bench(=run_ms) → metrics → Store

`run()` gibt **immer** ein `RunResult` zurück (nie eine Exception nach außen) —
Fehler werden nach `status`/`error` kategorisiert, damit die GUI sie anzeigen
kann statt abzustürzen:

* `compile_error`  — Ausdruck/Config nicht baubar oder cuTile-JIT scheitert.
* `verify_failed`  — Kernel läuft, weicht aber von der fp32-Referenz ab.
* `run_error`      — Kernel crasht zur Laufzeit (Launch/Bench).
* `ok`             — compiliert, verifiziert, gemessen.

TZ 1: nur `ik,kj->ij`, fp16→fp32, feste Tile. Spätere Teil-Ziele erweitern
einzelne Stufen — der Ablauf hier bleibt.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cuda.tile as ct
import torch

from .codegen.compile import load_kernel
from .intermediate_representation.parse import (
    ElementwiseIR,
    ReductionIR,
    parse,
)
from .intermediate_representation.reshape import (
    from_canonical_output,
    to_canonical,
    to_canonical_operands,
)
from .measure.baselines import measure_baselines
from .measure.bench import benchmark, time_first_launch
from .measure.metrics import (
    compute_metrics,
    compute_metrics_elementwise,
    compute_metrics_reduction,
)
from .measure.provenance import gpu_state
from .measure.verify import verify
from .schema import (
    STATUS_COMPILE_ERROR,
    STATUS_OK,
    STATUS_RUN_ERROR,
    STATUS_VERIFY_FAILED,
    RunConfig,
    RunResult,
    check_dtype_combo,
)
from .store import store

# dtype-Label → torch-dtype: reine Auflösungs-Tabelle (Compute-Input + Akku/
# Output), KEIN Zulässigkeits-Gate. Die Acc-Regeln erzwingt
# schema.check_dtype_combo; welche Input-dtypes tatsächlich baubar sind,
# entscheidet _build_operands. (tf32-Input ist torch.float32 + Kernel-Cast, fp8
# wird gecastet — beides in _build_operands; hier nur die direkt via
# torch.randn nutzbaren.)
_TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _build_operand(dtype: str, shape: tuple):
    """Ein Operand in beliebiger Rang-Shape im Compute-`dtype` (deterministisch;
    Seed außen).

    fp16/bf16/fp32 sind native `torch.randn`-dtypes; tf32 ist torch.float32 (+
    Kernel-Cast, nur Kontraktion); fp8 wird host-seitig gecastet (fp16→`.to(fp8)`,
    wie in analysis/dtype_analyse.py bewiesen).
    """
    if dtype in ("fp16", "bf16", "fp32"):
        return torch.randn(*shape, dtype=_TORCH_DTYPE[dtype], device="cuda")
    if dtype == "tf32":
        return torch.randn(*shape, dtype=torch.float32, device="cuda")
    if dtype in ("fp8e4m3", "fp8e5m2"):
        fp8 = torch.float8_e4m3fn if dtype == "fp8e4m3" else torch.float8_e5m2
        return torch.randn(*shape, dtype=torch.float16, device="cuda").to(fp8)
    raise NotImplementedError(f"input-dtype {dtype!r} noch nicht implementiert.")


def _build_natural_operands(dtype: str, shape_a: tuple, shape_b: tuple):
    """Baue A, B in **natürlicher einsum-Shape** (aus dem Ausdruck) im Compute-
    `dtype`. Der B1-View bringt sie danach auf die kanonische (B,M,K)/(B,K,N)-Form.
    (Dünner Wrapper um `_build_operand` — die Kontraktion braucht genau zwei.)
    """
    return _build_operand(dtype, shape_a), _build_operand(dtype, shape_b)


def _build_inputs(config: RunConfig, canonical):
    """Natürliche Operanden A_nat, B_nat + kanonische Kernel-Tensoren
    A_c=(B,M,K), B_c=(B,K,N), C_c=(B,M,N).

    Output-dtype = `acc_dtype` (ehrliches Ergebnis, bewahrt Akku-Präzision).
    Die Acc-Regeln werden HIER hart erzwungen (Stufe-2-Frühprüfung, erste
    Verteidigungslinie gegen still falsche Format-Kombis) — `measure.verify`
    prüft dieselben Kombis später ein zweites Mal über seine Toleranztabelle.
    Der B1-View (`to_canonical_operands`) bringt die natürlichen Operanden auf
    die kanonische Batched-GEMM-Form; `A_nat`/`B_nat` bleiben für die
    `torch.einsum`-Verifikation erhalten.
    """
    err = check_dtype_combo(config.dtype, config.acc_dtype)
    if err:
        raise NotImplementedError(err)
    if config.acc_dtype not in _TORCH_DTYPE:  # Sicherheitsnetz (Regeln decken das ab)
        raise NotImplementedError(f"acc-dtype {config.acc_dtype!r} nicht nach torch auflösbar.")
    torch.manual_seed(0)
    A_nat, B_nat = _build_natural_operands(config.dtype, canonical.a_natural_shape,
                                           canonical.b_natural_shape)
    A_c, B_c = to_canonical_operands(canonical, A_nat, B_nat)
    out_dt = _TORCH_DTYPE[config.acc_dtype]
    C_c = torch.empty(canonical.c_shape, dtype=out_dt, device="cuda")  # (B, M, N)
    return A_nat, B_nat, A_c, B_c, C_c


# ---------------------------------------------------------------------------
# Memory-bound-Familien (TZ 7): eigener, additiver Pfad — KEIN B1-Reshape, keine
# Kanonisierung, family-abhängige Operanden/Metriken. Die Kontraktion bleibt
# davon unberührt (eigener Zweig in `run`).
# ---------------------------------------------------------------------------
def _memory_bound_sizes(ir) -> dict:
    """`provenance["sizes"]` family-geformt (Anzeige/CLI; kein M/N/K)."""
    if isinstance(ir, ElementwiseIR):
        return {"shape": list(ir.shape), "elements": ir.num_elements, "arity": ir.arity}
    if isinstance(ir, ReductionIR):
        return {"in_shape": list(ir.in_shape), "kept": ir.kept_size,
                "reduced": ir.reduced_size, "out_shape": list(ir.out_shape)}
    return {}


def _build_memory_bound_inputs(config: RunConfig, ir):
    """Baue die Operanden für Elementwise/Reduktion — **ohne** B1/Kanonisierung.

    :returns: ``(operands, ref_operands, out_reshaper)`` mit
              * ``operands``     = Launch-Argumente (letzter = Output), 2D-geformt.
              * ``ref_operands`` = natürliche Operanden für die verify-Referenz.
              * ``out_reshaper`` = formt den Kernel-Output in die natürliche
                                   einsum-Shape zurück (für verify).
    Output-dtype = `acc_dtype` (kein Akku-Loop; nur der Store castet). Acc-Regeln
    werden auch hier früh erzwungen (Stufe-2-Prüfung).
    """
    err = check_dtype_combo(config.dtype, config.acc_dtype)
    if err:
        raise NotImplementedError(err)
    if config.acc_dtype not in _TORCH_DTYPE:  # Sicherheitsnetz (Regeln decken das ab)
        raise NotImplementedError(f"acc-dtype {config.acc_dtype!r} nicht nach torch auflösbar.")
    torch.manual_seed(0)
    out_dt = _TORCH_DTYPE[config.acc_dtype]

    if isinstance(ir, ElementwiseIR):
        # Op bestimmt die Arity (copy=1, add/mul=2) — muss zum Ausdruck passen.
        expected_arity = 1 if config.op == "copy" else 2
        if ir.arity != expected_arity:
            raise ValueError(
                f"Elementwise-Op {config.op!r} erwartet {expected_arity} Operanden, "
                f"Ausdruck '{config.expr}' hat {ir.arity}."
            )
        nat = [_build_operand(config.dtype, ir.shape) for _ in range(ir.arity)]
        rows, cols = ir.rows, ir.cols
        # 2D-Sicht (rows, cols) für den gekachelten Kernel (View — nat ist kontig.).
        kern_ins = [t.reshape(rows, cols) for t in nat]
        C = torch.empty(rows, cols, dtype=out_dt, device="cuda")
        shape = ir.shape
        return (tuple(kern_ins) + (C,), nat, lambda c: c.reshape(shape))

    if isinstance(ir, ReductionIR):
        A_nat = _build_operand(config.dtype, ir.in_shape)
        # Permute auf [kept…, reduced…], kontiguieren, auf (kept, reduced) falten.
        # Die evtl. Kopie ist Setup (außerhalb der Zeitmessung + der analytischen
        # Roofline-Metriken → verfälscht nichts).
        order = ir.kept_dims + ir.reduced_dims
        perm = [ir.input_axes.index(d) for d in order]
        A_2d = A_nat.permute(*perm).contiguous().reshape(ir.kept_size, ir.reduced_size)
        C = torch.empty(ir.kept_size, dtype=out_dt, device="cuda")
        out_shape = ir.out_shape
        return ((A_2d, C), [A_nat], lambda c: c.reshape(out_shape))

    raise NotImplementedError(f"memory-bound: unbekannte IR {type(ir).__name__}")


def _memory_bound_metrics(config: RunConfig, ir, run_ms: float) -> dict:
    """Family-abhängige Kennzahlen (GB/s primär) für Elementwise/Reduktion."""
    if isinstance(ir, ElementwiseIR):
        return compute_metrics_elementwise(ir.num_elements, ir.arity, config.op,
                                           run_ms, config.dtype, config.acc_dtype)
    if isinstance(ir, ReductionIR):
        return compute_metrics_reduction(ir.kept_size, ir.reduced_size, run_ms,
                                         config.dtype, config.acc_dtype)
    raise NotImplementedError(f"memory-bound-Metrik: unbekannte IR {type(ir).__name__}")


def _provenance(config: RunConfig) -> dict:
    """Statische Provenienz-Basis. `sizes` wird nach dem Parsen, `gpu_state`
    (Takt/Temp/Power via nvidia-smi) nach der Messung ergänzt (TZ 4)."""
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": config.dtype,
        "acc_dtype": config.acc_dtype,
        "sizes": {},  # nach dem Parsen mit M/N/K gefüllt
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run(config: RunConfig, progress=None) -> RunResult:
    """Führe einen vollständigen Lauf aus und liefere ein `RunResult`.

    ``progress`` ist ein optionaler Callback ``(done, iters)``, der während der
    warmen Messung nach jeder getakteten Iteration aufgerufen wird (Live-Anzeige
    „k/N" in der GUI). Ohne Callback (CLI/Tests) unverändert.
    """
    provenance = _provenance(config)
    accuracy: dict = {}
    timing: dict = {}
    metrics: dict = {}
    kernel_path = None
    kernel_source = None

    def _result(status: str, error: str | None = None) -> RunResult:
        r = RunResult(
            status=status, config=config.to_dict(), kernel_path=kernel_path,
            kernel_source=kernel_source,
            accuracy=accuracy, timing=timing, metrics=metrics,
            provenance=provenance, error=error,
        )
        # Persistenz darf den Core↔GUI-Vertrag ("run() wirft nie") NICHT brechen:
        # ein Store-Fehler (Platte voll/Rechte) wird notiert, das RunResult aber
        # trotzdem geliefert. (Modul-Attribut-Aufruf → in Tests patchbar.)
        try:
            store.append_result(r)
        except Exception as store_e:  # noqa: BLE001
            note = f"store: {type(store_e).__name__}: {store_e}"
            r.error = f"{r.error} | {note}" if r.error else note
        return r

    # ===================================================================
    # Memory-bound-Familien (Elementwise/Reduktion, TZ 7): eigener, additiver
    # Zweig — KEIN B1-Reshape/Kanonisierung, family-abhängige Operanden/Verify/
    # Metriken, variable Launch-Arity. Kehrt in allen Pfaden zurück; der
    # Kontraktions-Flow darunter bleibt dadurch **unberührt**.
    # ===================================================================
    if config.family in ("elementwise", "reduction"):
        # 1) IR (family-typisiert) + family-geformte Größen.
        try:
            ir = parse(config)
            provenance["sizes"] = _memory_bound_sizes(ir)
        except Exception as e:
            return _result(STATUS_COMPILE_ERROR, error=f"{type(e).__name__}: {e}")

        # 2) Operanden (natürlich → 2D für den Kernel), OHNE B1/Kanonisierung.
        try:
            operands, ref_operands, out_reshaper = _build_memory_bound_inputs(config, ir)
        except Exception as e:
            return _result(STATUS_COMPILE_ERROR, error=f"input build: {type(e).__name__}: {e}")

        # 3) Quelltext → ladbarer Kernel (persistiert + gecacht).
        try:
            comp = load_kernel(config)
            kernel_path = store.store_relpath(comp.kernel_path)
            try:
                kernel_source = Path(comp.kernel_path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                kernel_source = None
        except Exception as e:
            return _result(STATUS_COMPILE_ERROR, error=f"{type(e).__name__}: {e}")

        # 4) Kalt-Lauf = compile_ms (füllt den Output für verify). Arity variabel.
        try:
            timing["compile_ms"] = round(time_first_launch(comp.launch, *operands), 3)
        except ct.TileError as e:
            return _result(STATUS_COMPILE_ERROR, error=f"cuTile-JIT: {type(e).__name__}: {str(e)[:400]}")
        except Exception as e:
            return _result(STATUS_RUN_ERROR, error=f"kalt-launch: {type(e).__name__}: {str(e)[:400]}")

        # 5) verify-before-trust: Output in die natürliche Shape zurückformen, dann
        #    gegen die family-/op-abhängige fp32-Referenz prüfen (variadisch).
        try:
            out_nat = out_reshaper(operands[-1])   # letzter Operand = Output
            accuracy = verify(out_nat, ref_operands, config)
        except NotImplementedError as e:
            return _result(STATUS_COMPILE_ERROR, error=f"verify: {type(e).__name__}: {e}")
        except Exception as e:
            return _result(STATUS_RUN_ERROR, error=f"verify: {type(e).__name__}: {str(e)[:400]}")
        if not accuracy["passed"]:
            return _result(
                STATUS_VERIFY_FAILED,
                error=(f"max_abs_err={accuracy['max_abs_err']:.4g} überschreitet Toleranz "
                       f"(atol={accuracy['atol']}, rtol={accuracy['rtol']})"),
            )

        # 6) Warme Messung + family-abhängige Metriken (GB/s primär).
        try:
            b = benchmark(comp.launch, *operands,
                          warmup=config.bench_warmup, iters=config.bench_iters,
                          progress=progress)
            timing["run_ms"] = round(b["run_ms"], 5)
            timing["min_ms"] = round(b["min_ms"], 5)
            timing["p90_ms"] = round(b["p90_ms"], 5)
            timing["sigma_ms"] = round(b["sigma_ms"], 5)
            timing["bench_iters"] = b["iters"]
            metrics = _memory_bound_metrics(config, ir, b["run_ms"])
            metrics["tflops"] = round(metrics["tflops"], 3)
        except Exception as e:
            return _result(STATUS_RUN_ERROR, error=f"bench: {type(e).__name__}: {str(e)[:400]}")

        provenance["gpu_state"] = gpu_state()
        # Keine GEMM-Baselines für memory-bound (torch.matmul/gemm_flops passen nicht).
        return _result(STATUS_OK)

    # 1) IR → kanonische Größen + B1-View-Spezifikation
    try:
        ir = parse(config)
        canonical = to_canonical(ir)
        M, N, K, B = canonical.M, canonical.N, canonical.K, canonical.B
        provenance["sizes"] = {"M": M, "N": N, "K": K, "B": B}
    except Exception as e:
        return _result(STATUS_COMPILE_ERROR, error=f"{type(e).__name__}: {e}")

    # 2) Deterministische Eingaben (natürliche Operanden) + kanonische Kernel-
    #    Tensoren. Validiert dtype/Größen FRÜH, bevor ein (evtl. irreführendes)
    #    Kernel-Artefakt geschrieben wird.
    try:
        A_nat, B_nat, A_c, B_c, C_c = _build_inputs(config, canonical)
    except Exception as e:
        return _result(STATUS_COMPILE_ERROR, error=f"input build: {type(e).__name__}: {e}")

    # 3) Quelltext → ladbarer Kernel (persistiert + gecacht). load_kernel emittiert
    #    lazy selbst (nur bei Cache-Miss) → kein doppeltes emit auf dem Cache-Pfad.
    try:
        comp = load_kernel(config)
        kernel_path = store.store_relpath(comp.kernel_path)
        # Quelltext für die GUI-Code-Anzeige mitführen (das persistierte Artefakt =
        # exakt was compiliert wurde). Lesefehler darf einen sonst gesunden Lauf
        # NICHT kippen → still auf None. Neben OSError (Datei weg/Rechte) auch
        # UnicodeDecodeError (korrupte/nicht-UTF-8-Datei; KEIN OSError) abfangen,
        # sonst würde ein compilierter Kernel fälschlich als compile_error markiert.
        try:
            kernel_source = Path(comp.kernel_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            kernel_source = None
    except Exception as e:
        return _result(STATUS_COMPILE_ERROR, error=f"{type(e).__name__}: {e}")

    # 4) Kalt-Lauf = compile_ms (host-seitiger cuTile-JIT); füllt C für verify.
    try:
        timing["compile_ms"] = round(time_first_launch(comp.launch, A_c, B_c, C_c), 3)
    except ct.TileError as e:
        return _result(STATUS_COMPILE_ERROR, error=f"cuTile-JIT: {type(e).__name__}: {str(e)[:400]}")
    except Exception as e:
        return _result(STATUS_RUN_ERROR, error=f"kalt-launch: {type(e).__name__}: {str(e)[:400]}")

    # 5) verify-before-trust: kanonischen Output in die natürliche einsum-Shape
    #    zurückführen, dann gegen die fp32-`torch.einsum`-Referenz prüfen.
    try:
        C_nat = from_canonical_output(canonical, C_c)   # (1,M,N) → natürliche Output-Shape
        accuracy = verify(C_nat, [A_nat, B_nat], config)
    except NotImplementedError as e:
        return _result(STATUS_COMPILE_ERROR, error=f"verify: {type(e).__name__}: {e}")
    except Exception as e:
        return _result(STATUS_RUN_ERROR, error=f"verify: {type(e).__name__}: {str(e)[:400]}")
    if not accuracy["passed"]:
        return _result(
            STATUS_VERIFY_FAILED,
            error=(f"max_abs_err={accuracy['max_abs_err']:.4g} überschreitet Toleranz "
                   f"(atol={accuracy['atol']}, rtol={accuracy['rtol']})"),
        )

    # 6) Warme Messung (=run_ms) + Metriken (TFLOP/s)
    try:
        b = benchmark(comp.launch, A_c, B_c, C_c,
                      warmup=config.bench_warmup, iters=config.bench_iters,
                      progress=progress)
        timing["run_ms"] = round(b["run_ms"], 5)      # Median (unveränderter Key)
        timing["min_ms"] = round(b["min_ms"], 5)      # schnellste Iteration
        timing["p90_ms"] = round(b["p90_ms"], 5)      # 90.-Perzentil (Ausreißer-Kopf)
        timing["sigma_ms"] = round(b["sigma_ms"], 5)  # Streuung über die Iterationen
        timing["bench_iters"] = b["iters"]
        # compute_metrics-dict komplett übernehmen (nicht neu bauen) → die
        # TZ-4-Keys (GB/s, arithm. Intensität, %-Peak) fließen automatisch mit;
        # dtype/acc_dtype werden für Bytes/Peak gebraucht.
        metrics = compute_metrics(M, N, K, b["run_ms"], config.dtype, config.acc_dtype, B=B)
        metrics["tflops"] = round(metrics["tflops"], 3)
    except Exception as e:
        return _result(STATUS_RUN_ERROR, error=f"bench: {type(e).__name__}: {str(e)[:400]}")

    # GPU-Zustand direkt NACH der Messung (Takt/Temp/Power spiegeln die gemessene
    # Last) — additiv in provenance, pro Lauf. Graceful (leer, falls nvidia-smi fehlt).
    provenance["gpu_state"] = gpu_state()

    # 7) Optionale Baselines (cuBLAS-Obergrenze / naive-cuTile-Untergrenze) —
    #    additiv in metrics["baselines"]. Optional & sekundär → ein Fehler hier
    #    kippt den bereits verifizierten+gemessenen ok-Lauf NICHT (graceful).
    if config.baselines:
        try:
            metrics["baselines"] = measure_baselines(config.baselines, A_c, B_c, C_c, config)
        except Exception as e:  # noqa: BLE001
            metrics["baselines"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    return _result(STATUS_OK)
