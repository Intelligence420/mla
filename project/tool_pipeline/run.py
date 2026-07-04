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
from .intermediate_representation.parse import parse
from .intermediate_representation.reshape import to_canonical
from .measure.bench import benchmark, time_first_launch
from .measure.metrics import compute_metrics
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


def _build_operands(dtype: str, M: int, N: int, K: int):
    """Baue A=(M,K), B=(K,N) im Compute-`dtype` (deterministisch; Seed außen).

    Wächst pro TZ-3-dtype: fp16/bf16 sind native `torch.randn`-dtypes; tf32
    (torch.float32 + Kernel-Cast) und fp8 (fp16→`.to(fp8)`) kommen als eigene
    Zweige in den folgenden Teil-Schritten dazu.
    """
    if dtype in ("fp16", "bf16"):
        t = _TORCH_DTYPE[dtype]
        return (torch.randn(M, K, dtype=t, device="cuda"),
                torch.randn(K, N, dtype=t, device="cuda"))
    if dtype == "tf32":
        # tf32-Operanden sind normale fp32-Tensoren; die tf32-Reduktion macht
        # der Kernel-Cast (ct.astype .. ct.tfloat32), NICHT der Input-dtype.
        return (torch.randn(M, K, dtype=torch.float32, device="cuda"),
                torch.randn(K, N, dtype=torch.float32, device="cuda"))
    if dtype in ("fp8e4m3", "fp8e5m2"):
        # torch.randn kann fp8 NICHT direkt erzeugen -> fp16 bauen und host-seitig
        # casten (genau wie in analysis/dtype_analyse.py bewiesen). Der Kernel
        # rechnet die fp8-Tiles direkt (kein in-Kernel-Cast).
        fp8 = torch.float8_e4m3fn if dtype == "fp8e4m3" else torch.float8_e5m2
        return (torch.randn(M, K, dtype=torch.float16, device="cuda").to(fp8),
                torch.randn(K, N, dtype=torch.float16, device="cuda").to(fp8))
    raise NotImplementedError(
        f"input-dtype {dtype!r} noch nicht implementiert."
    )


def _build_inputs(config: RunConfig, M: int, N: int, K: int):
    """Deterministische Eingaben A=(M,K), B=(K,N) + Output C=(M,N).

    Output-dtype = `acc_dtype` (ehrliches Ergebnis, bewahrt Akku-Präzision).
    Die Acc-Regeln werden HIER hart erzwungen (Stufe-2-Frühprüfung, erste
    Verteidigungslinie gegen still falsche Format-Kombis) — `measure.verify`
    prüft dieselben Kombis später ein zweites Mal über seine Toleranztabelle.
    """
    err = check_dtype_combo(config.dtype, config.acc_dtype)
    if err:
        raise NotImplementedError(err)
    if config.acc_dtype not in _TORCH_DTYPE:  # Sicherheitsnetz (Regeln decken das ab)
        raise NotImplementedError(f"acc-dtype {config.acc_dtype!r} nicht nach torch auflösbar.")
    torch.manual_seed(0)
    A, B = _build_operands(config.dtype, M, N, K)
    out_dt = _TORCH_DTYPE[config.acc_dtype]
    C = torch.empty(M, N, dtype=out_dt, device="cuda")
    return A, B, C


def _provenance(config: RunConfig) -> dict:
    """Leichte Provenienz (TZ 1). GPU-Takt/Temp/Power via nvidia-smi = TZ 4."""
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": config.dtype,
        "acc_dtype": config.acc_dtype,
        "sizes": {},  # nach dem Parsen mit M/N/K gefüllt
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run(config: RunConfig) -> RunResult:
    """Führe einen vollständigen Lauf aus und liefere ein `RunResult`."""
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

    # 1) IR → kanonische Größen
    try:
        ir = parse(config)
        canonical = to_canonical(ir)
        M, N, K = canonical.M, canonical.N, canonical.K
        provenance["sizes"] = {"M": M, "N": N, "K": K}
    except Exception as e:
        return _result(STATUS_COMPILE_ERROR, error=f"{type(e).__name__}: {e}")

    # 2) Deterministische Eingaben — validiert dtype/Größen FRÜH, bevor ein
    #    (evtl. irreführendes) Kernel-Artefakt geschrieben wird.
    try:
        A, B, C = _build_inputs(config, M, N, K)
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
        timing["compile_ms"] = round(time_first_launch(comp.launch, A, B, C), 3)
    except ct.TileError as e:
        return _result(STATUS_COMPILE_ERROR, error=f"cuTile-JIT: {type(e).__name__}: {str(e)[:400]}")
    except Exception as e:
        return _result(STATUS_RUN_ERROR, error=f"kalt-launch: {type(e).__name__}: {str(e)[:400]}")

    # 5) verify-before-trust: gegen fp32-Referenz, bevor Zahlen getraut werden.
    try:
        accuracy = verify(C, A, B, config)
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
        b = benchmark(comp.launch, A, B, C)
        timing["run_ms"] = round(b["run_ms"], 5)
        timing["bench_iters"] = b["iters"]
        # compute_metrics-dict weiterreichen (nicht neu bauen) → künftige Schlüssel
        # (TZ 4: GB/s, %-Peak) überleben ohne weitere Edit-Stelle hier.
        metrics = compute_metrics(M, N, K, b["run_ms"])
        metrics["tflops"] = round(metrics["tflops"], 3)
    except Exception as e:
        return _result(STATUS_RUN_ERROR, error=f"bench: {type(e).__name__}: {str(e)[:400]}")

    return _result(STATUS_OK)
