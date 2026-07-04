"""tool_pipeline.cli — headless-Runner (kein GUI) für einen einzelnen Lauf.

Stößt die eine Naht `run(config)` an, druckt eine lesbare Zusammenfassung und
macht den **echten** Store-Append (`results.jsonl`) + Kernel-Write — die
headless-Demonstration der TZ-1-Definition of Done. Später (TZ 8) wachsen hier
Batch-Sweeps für die Report-Plots.

Aufruf (aus `project/`, venv aktiv)::

    python -m tool_pipeline.cli                    # Default: ik,kj->ij, 512^3, fp16->fp32
    python -m tool_pipeline.cli --M 256 --N 384 --K 128
    python -m tool_pipeline.cli --show-kernel      # generierten Quelltext mitdrucken

TZ 1: fest `ik,kj->ij`, fp16→fp32, Tile 128/128/64 (aus den `RunConfig`-Defaults);
verstellbar sind nur die Größen. dtype/Tile/Swizzle/Ausdruck folgen in TZ 3/4/6.
Exit-Code 0 bei `ok`, sonst 1 (skript-/CI-tauglich).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .run import run
from .schema import STATUS_OK, RunConfig
from .store import store


def build_config(args: argparse.Namespace) -> RunConfig:
    """CLI-Argumente → `RunConfig` (TZ 1: nur Größen verstellbar)."""
    # ik,kj->ij: i=M (Zeilen), k=K (Kontraktion), j=N (Spalten).
    return RunConfig(dim_sizes={"i": args.M, "k": args.K, "j": args.N})


def _fmt(x, spec: str, default: str = "—") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else default


def print_summary(res) -> None:
    """Lesbare Zusammenfassung eines `RunResult` auf stdout."""
    c = res.config
    t = c.get("tile", {})
    s = res.provenance.get("sizes", {})
    acc, tim, met = res.accuracy, res.timing, res.metrics

    print("=== einsum/GEMM Performance Explorer — Lauf ===")
    print(f"Ausdruck : {c.get('expr')}   (M={s.get('M','?')}, N={s.get('N','?')}, K={s.get('K','?')})")
    print(f"Format   : {c.get('dtype')} -> {c.get('acc_dtype')}   "
          f"Tile: TM={t.get('TM')} TN={t.get('TN')} TK={t.get('TK')}   swizzle={c.get('swizzle')}")
    print(f"GPU      : {res.provenance.get('gpu')}   @ {res.provenance.get('timestamp')}")
    print(f"Status   : {res.status.upper()}")

    if res.kernel_path:
        print(f"Kernel   : {res.kernel_path}")
    if acc:
        tag = "PASS" if acc.get("passed") else "FAIL"
        print(f"Verify   : {tag}   max_abs_err={_fmt(acc.get('max_abs_err'), '.3e')}   "
              f"(atol={acc.get('atol')}, rtol={acc.get('rtol')})")
    if tim:
        print(f"Timing   : compile={_fmt(tim.get('compile_ms'), '.1f')} ms   "
              f"run(median)={_fmt(tim.get('run_ms'), '.4f')} ms")
    if met.get("tflops") is not None:
        print(f"Durchsatz: {_fmt(met.get('tflops'), '.2f')} TFLOP/s")
    if res.error:
        print(f"Fehler   : {res.error}")
    print(f"Store    : {store.store_relpath(store.RESULTS_JSONL)}  (+1 Zeile)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tool_pipeline.cli",
        description="Ein GEMM-Lauf (ik,kj->ij, fp16->fp32) end-to-end: "
                    "generieren → verifizieren → messen → speichern.",
    )
    p.add_argument("--M", type=int, default=512, help="Zeilen M (Index i), Default 512")
    p.add_argument("--N", type=int, default=512, help="Spalten N (Index j), Default 512")
    p.add_argument("--K", type=int, default=512, help="Kontraktion K (Index k), Default 512")
    p.add_argument("--show-kernel", action="store_true",
                   help="den generierten Kernel-Quelltext mit ausgeben")
    args = p.parse_args(argv)

    res = run(build_config(args))
    print_summary(res)

    if args.show_kernel and res.kernel_path:
        path = store.PROJECT_DIR / res.kernel_path
        if path.exists():
            print("\n--- generierter Kernel-Quelltext " + "-" * 40)
            print(path.read_text(encoding="utf-8"))

    return 0 if res.status == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
