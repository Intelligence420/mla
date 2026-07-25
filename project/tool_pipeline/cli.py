"""tool_pipeline.cli — headless-Runner (kein GUI): Einzellauf ODER Batch-Sweep.

**Einzellauf** (TZ 1): eine ``RunConfig`` → die eine Naht ``run(config)`` →
lesbare Zusammenfassung + echter Store-Append (``results.jsonl``) + Kernel-Write.

**Batch-Sweep** (TZ 8): ein *kuratierter* Satz Konfigurationen über **alle drei
Familien** — inkl. **GROUP_M**-Varianten, **mehrerer Tiles** und **einer n-ären
Ketten-Kontraktion** —, aus dem der Sphinx-Report seine Figuren/Tabellen zieht.
Reproduzierbar, unter **EINEM GPU-Lock** serialisiert, mit **verify-before-trust**
(nur ``status=="ok"``-Läufe zählen für den Report). Die Config-Erzeugung lehnt sich
an ``app.components.controls.configs_from_selection`` an — damit dieselben
Kreuzprodukt-Slugs entstehen wie in der GUI (ein Cache, eine Datenquelle).

Aufruf (aus ``project/``, venv aktiv)::

    python -m tool_pipeline.cli                     # ein GEMM-Lauf (ik,kj->ij, fp16->fp32)
    python -m tool_pipeline.cli --M 256 --N 384 --K 128
    python -m tool_pipeline.cli --family elementwise --op add --expr ij,ij->ij --size 4096
    python -m tool_pipeline.cli --sweep             # kompletter Report-Sweep (GPU, ~2 min)
    python -m tool_pipeline.cli --show-configs      # nur die Sweep-Configs listen (kein GPU)

Exit-Code 0 bei Erfolg (Einzellauf ``ok`` bzw. Sweep alle ``ok``), sonst 1
(skript-/CI-tauglich); 2, wenn der GPU-Lock nicht frei wurde.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime

from .schema import STATUS_OK, RunConfig
from .store import store

# ``app.components.controls`` liefert die Validierungs-/Config-Bau-Helfer, die GUI und CLI
# teilen (ein Kreuzprodukt, ein Slug, ein Cache). Es zieht aber ``dash`` +
# ``dash_bootstrap_components`` auf Modulebene — ein Import auf Modulebene würde die
# headless-CLI an den GUI-Stack koppeln (Batch-Node/CI ohne Dash: ModuleNotFoundError).
# Deshalb LAZY, in den Funktionen, die die Helfer wirklich brauchen: der Modul-Import von
# ``tool_pipeline.cli`` bleibt dash- UND torch-frei.


def _controls():
    """Lazy-Zugriff auf die geteilten Controls-Helfer (zieht dash nach — s. oben)."""
    from .app.components import controls
    return controls

# ---------------------------------------------------------------------------
# Report-Sweep — Größen (klein & deterministisch; geteilte 32-GiB-Maschine).
# ---------------------------------------------------------------------------
# Kontraktion 1024³ ⇒ arithmetische Intensität 1024/4 = 256 FLOP/Byte (klar
# compute-bound, rechte Roofline-Seite); memory-bound 4096² ⇒ AI 0,1–0,5 (linke
# Seite); n-är 256⁴ (zwei paarweise fp16-GEMMs — bei größeren Ketten summiert sich
# der fp16-Fehler beider Schritte über die Verify-Toleranz, s. Report/verify-before-trust).
_SWEEP_SIZE_CONTRACTION = 1024
_SWEEP_SIZE_MEMORY = 4096
_SWEEP_SIZE_NARY = 256

# Fusions-Sweep (TZ 9) — DREI Formen, aufsteigend in arithmetischer Intensität. Die
# Fusion spart immer denselben Zwischentensor-Roundtrip (2·4·M·N Bytes); was sich
# ändert, ist wie stark dieser Roundtrip gegenüber der Kontraktion selbst ins Gewicht
# fällt. Deshalb variiert der Sweep die AI und nicht die Arbeitsmenge — die Frage
# „wann lohnt Fusion?" wird damit belegt statt illustriert (gemessen auf GB10, bias):
#   * schmal  M=N=4096, K=64   ⇒ 2,15 GFLOP,  AI  21 → Speedup 2,17x  (memory-bound)
#   * quadratisch 1024³        ⇒ 2,15 GFLOP,  AI 205 → Speedup 1,23x
#   * tief    M=N=1024, K=8192 ⇒ 17,2 GFLOP,  AI 431 → Speedup 1,05x  (compute-dominiert)
# Die erste und die dritte Form haben denselben gesparten Roundtrip (8 MiB bzw. 128 MiB
# bei der schmalen), aber grundverschiedene Kontraktions-Laufzeiten ⇒ der Trend ist
# monoton fallend. Der A04-Befund (0,984x, assignments/04_assignment/src/task_02.py)
# liegt jenseits der dritten Form: dort erschlägt die Kontraktion (12,83 ms) den Epilog
# (0,067 ms) so weit, dass der Launch-Overhead der Fusion sie leicht negativ macht.
_SWEEP_FUSION_NARROW = {"M": 4096, "N": 4096, "K": 64}      # memory-bound
_SWEEP_FUSION_SQUARE = 1024                                 # quadratisch (Mittelfeld)
_SWEEP_FUSION_DEEP = {"M": 1024, "N": 1024, "K": 8192}      # compute-dominiert

_GPU_LOCK_REL = ".cache/gpu.lock"   # relativ zu store.PROJECT_DIR (wie in der GUI)
_LOCK_TIMEOUT = 60                  # s — danach „GPU belegt" statt endlos zu warten


# ---------------------------------------------------------------------------
# Einzellauf: CLI-Argumente → RunConfig
# ---------------------------------------------------------------------------
def _resolve_op(family: str, op):
    """Family-abhängige Op (Reduktion=sum, Elementwise=gewählt, Kontraktion=None)."""
    if family == "reduction":
        return "sum"
    if family == "elementwise":
        return op
    return None


def build_config(args: argparse.Namespace) -> RunConfig:
    """CLI-Argumente → eine ``RunConfig`` (Einzellauf).

    ``--size`` ist die einheitliche Größe je Index; ``--M/--N/--K`` überschreiben
    (falls angegeben) die Indizes ``i``/``j``/``k`` des Ausdrucks — das erhält den
    klassischen GEMM-Aufruf (``ik,kj->ij``: i=M, k=K, j=N) rückwärtskompatibel und
    generalisiert zugleich auf beliebige Ausdrücke/Familien.

    ``--epilog`` (TZ 9) setzt die Fusion; ohne Angabe bleibt ``epilog=None`` ⇒ Config,
    Kernel-Quelltext und Slug sind byte-identisch zu TZ 1-8. Ein Epilog auf einer
    memory-bound-Familie oder einer n-ären Kette wird hier laut abgelehnt (statt still
    verworfen zu werden), spiegelbildlich zur GUI-Validierung.
    """
    controls = _controls()
    idx = controls.expr_indices(args.expr)
    sizes = {d: args.size for d in idx}
    for d, v in (("i", args.M), ("j", args.N), ("k", args.K)):
        if v is not None and d in sizes:
            sizes[d] = v
    epilog = getattr(args, "epilog", None)
    err = controls.validate_epilog(epilog, args.family, args.expr)
    if err:
        raise SystemExit(f"Ungültiger Epilog: {err}")
    return RunConfig(family=args.family, op=_resolve_op(args.family, args.op),
                     epilog=epilog,
                     expr=controls.resolve_expr(args.expr, args.family), dim_sizes=sizes)


# ---------------------------------------------------------------------------
# Report-Sweep: der kuratierte Config-Satz (deterministisch, headless)
# ---------------------------------------------------------------------------
def sweep_configs(size_c: int = _SWEEP_SIZE_CONTRACTION,
                  size_m: int = _SWEEP_SIZE_MEMORY,
                  size_n: int = _SWEEP_SIZE_NARY,
                  size_f: int = _SWEEP_FUSION_SQUARE,
                  narrow: dict | None = None,
                  deep: dict | None = None) -> list[RunConfig]:
    """Kuratierter Report-Sweep: ein Satz ``RunConfig``s über alle drei Familien,
    der genau die Report-Geschichten erzeugt. **Deterministisch** (keine
    Zufälligkeit, stabile Slugs) und **headless** (kein GPU/torch) — die Ausführung
    macht ``run_sweep``. Aufgebaut aus fokussierten Teil-Sweeps, jeder beantwortet
    eine Report-Frage; zusammen ergeben sie die Roofline mit **beiden** Seiten.

    ``narrow``/``size_f``/``deep`` steuern den Fusions-Sweep (TZ 9): drei Formen
    aufsteigend in arithmetischer Intensität (schmal ⇒ memory-bound, quadratisch
    ``size_f``³ ⇒ Mittelfeld, tief ⇒ compute-dominiert). Beide Epiloge (bias/relu)
    laufen auf allen drei Formen ⇒ der Fusions-Gewinn wird als **Trend** über die AI
    sichtbar, nicht als Einzelbefund.
    """
    controls = _controls()
    ck = controls.combo_key
    cfgs: list[RunConfig] = []
    dims_c = {"i": size_c, "k": size_c, "j": size_c}
    narrow = dict(narrow or _SWEEP_FUSION_NARROW)
    deep = dict(deep or _SWEEP_FUSION_DEEP)

    # (1) Kontraktion — Format-Vergleich @ size_c³, Tile 128/128/64, ohne Swizzle,
    #     inkl. cuBLAS-Baseline (Obergrenze). → Durchsatz je Format · Genauigkeit↔
    #     Durchsatz · compute-bound-Seite der Roofline (vier Punkte weit rechts).
    fmt = [ck("fp16", "fp32"), ck("bf16", "fp32"), ck("tf32", "fp32"), ck("fp8e4m3", "fp16")]
    cfgs += controls.configs_from_selection(
        "ik,kj->ij", dims_c, fmt,
        tiles=[{"TM": 128, "TN": 128, "TK": 64}], swizzle_configs=[(False, 8)],
        baselines=["cublas"], family="contraction")

    # (2) Kontraktion — Tile-Vergleich (fp16→fp32): kleines vs. großes Tile
    #     (das 128er-Tile steckt bereits in (1)).
    fp16 = [ck("fp16", "fp32")]
    cfgs += controls.configs_from_selection(
        "ik,kj->ij", dims_c, fp16,
        tiles=[{"TM": 64, "TN": 64, "TK": 32}, {"TM": 256, "TN": 128, "TK": 64}],
        swizzle_configs=[(False, 8)], family="contraction")

    # (3) Kontraktion — L2-Swizzle/GROUP_M-Vergleich (fp16→fp32, Tile 128/128/64):
    #     G8 (Default) · G16 · G32 (ohne-Swizzle steckt bereits in (1)).
    cfgs += controls.configs_from_selection(
        "ik,kj->ij", dims_c, fp16,
        tiles=[{"TM": 128, "TN": 128, "TK": 64}],
        swizzle_configs=[(True, 8), (True, 16), (True, 32)], family="contraction")

    # (4) Elementwise (add) — memory-bound @ size_m², drei native Formate.
    mem = [ck("fp16", "fp32"), ck("bf16", "fp32"), ck("fp32", "fp32")]
    cfgs += controls.configs_from_selection(
        "ij,ij->ij", {"i": size_m, "j": size_m}, mem, family="elementwise", op="add")

    # (5) Reduktion (Zeilensumme) — memory-bound @ size_m², drei native Formate.
    cfgs += controls.configs_from_selection(
        "ij->i", {"i": size_m, "j": size_m}, mem, family="reduction", op="sum")

    # (6) n-äre Ketten-Kontraktion ij,jk,kl->il → EIN aggregierter Roofline-Punkt
    #     (paarweise Zerlegung durch den bewiesenen 2-Op-GEMM-Pfad).
    cfgs.append(RunConfig(expr="ij,jk,kl->il",
                          dim_sizes={"i": size_n, "j": size_n, "k": size_n, "l": size_n},
                          dtype="fp16", acc_dtype="fp32"))

    # (7) Fusions-Sweep (TZ 9) — beide Epiloge × drei Formen (AI aufsteigend). Der
    #     sequentielle Vergleichspunkt entsteht NICHT als eigene Config: er wird
    #     innerhalb jedes fused-run() mitgemessen (metrics["fusion"], measure/fusion.py).
    #     Zusätzlich je Form die unfusionierte Kontraktion als Bezugspunkt — die
    #     quadratische steckt schon in (1), schmal und tief kommen hier dazu.
    dims_narrow = {"i": narrow["M"], "k": narrow["K"], "j": narrow["N"]}
    dims_square = {"i": size_f, "k": size_f, "j": size_f}
    dims_deep = {"i": deep["M"], "k": deep["K"], "j": deep["N"]}
    for dims in (dims_narrow, dims_deep):            # unfusionierte Bezugspunkte
        cfgs += controls.configs_from_selection(
            "ik,kj->ij", dims, fp16, family="contraction")
    for epilog in ("bias", "relu"):
        for dims in (dims_narrow, dims_square, dims_deep):
            cfgs += controls.configs_from_selection(
                "ik,kj->ij", dims, fp16, family="contraction", epilog=epilog)
    return cfgs


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------
def _fmt(x, spec: str, default: str = "—") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else default


def print_summary(res) -> None:
    """Lesbare, **family-geformte** Zusammenfassung eines ``RunResult`` auf stdout:
    memory-bound zeigt GB/s primär (die aussagekräftige Metrik), compute-bound
    (Kontraktion, inkl. n-är) TFLOP/s primär."""
    c = res.config
    fam = c.get("family", "contraction")
    t = c.get("tile", {})
    s = res.provenance.get("sizes", {})
    acc, tim, met = res.accuracy, res.timing, res.metrics
    memory_bound = fam in ("elementwise", "reduction")

    print("=== cuTile Performance Lab — Lauf ===")
    op_txt = f"  ·  op={c.get('op')}" if c.get("op") else ""
    # Epilog-Fusion (TZ 9) nur zeigen, wenn gesetzt → unfusionierte Ausgabe unverändert.
    ep_txt = f"  ·  epilog={c.get('epilog')}" if c.get("epilog") else ""
    print(f"Familie  : {fam}{op_txt}{ep_txt}")
    print(f"Ausdruck : {c.get('expr')}   (Größen: {s})")
    sw_txt = f" G{c.get('group_m')}" if c.get("swizzle") else ""
    print(f"Format   : {c.get('dtype')} -> {c.get('acc_dtype')}   "
          f"Tile: TM={t.get('TM')} TN={t.get('TN')} TK={t.get('TK')}   "
          f"swizzle={c.get('swizzle')}{sw_txt}")
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
    if met:
        ai = met.get("arithmetic_intensity")
        if memory_bound:
            # memory-bound: GB/s ist die Primärmetrik (TFLOP/s ist hier winzig/irrelevant).
            print(f"Durchsatz: {_fmt(met.get('gbps'), '.2f')} GB/s   "
                  f"(AI={_fmt(ai, '.2f')} FLOP/B, "
                  f"{_fmt(met.get('percent_peak_bw'), '.1f')} % Peak-BW)")
        else:
            print(f"Durchsatz: {_fmt(met.get('tflops'), '.2f')} TFLOP/s   "
                  f"(AI={_fmt(ai, '.1f')} FLOP/B, "
                  f"{_fmt(met.get('percent_peak_flops'), '.1f')} % Peak-FLOP, "
                  f"{_fmt(met.get('gbps'), '.1f')} GB/s)")
    # Fusions-Zeile (TZ 9): fused vs. sequentiell + gesparter Zwischentensor-Roundtrip.
    # Nur bei gesetztem Epilog; nicht verfügbar ⇒ Grund statt stiller Lücke.
    fus = (met or {}).get("fusion")
    if isinstance(fus, dict):
        if fus.get("available"):
            sp = fus.get("speedup")
            verdict = ("Fusion gewinnt" if isinstance(sp, (int, float)) and sp > 1.02 else
                       "neutral" if isinstance(sp, (int, float)) and sp >= 0.98 else
                       "sequentiell schneller")
            print(f"Fusion   : {_fmt(sp, '.3f')}x   "
                  f"(fused={_fmt(fus.get('fused_ms'), '.4f')} ms vs. "
                  f"sequentiell={_fmt(fus.get('sequential_ms'), '.4f')} ms — {verdict})")
            print(f"           gespart: {fus.get('saved_bytes', 0) / 2**20:.1f} MiB "
                  f"Zwischentensor-Roundtrip   AI "
                  f"{_fmt(fus.get('sequential_ai'), '.1f')} → "
                  f"{_fmt(fus.get('fused_ai'), '.1f')} FLOP/B")
        else:
            print(f"Fusion   : kein Vergleich ({fus.get('note', 'unbekannt')})")
    if res.error:
        print(f"Fehler   : {res.error}")
    print(f"Store    : {store.store_relpath(store.RESULTS_JSONL)}  (+1 Zeile)")


# ---------------------------------------------------------------------------
# Sweep-Ausführung (GPU, unter Lock)
# ---------------------------------------------------------------------------
def run_sweep(configs: list[RunConfig], name: str | None = None) -> int:
    """Alle Configs unter **EINEM** GPU-Lock + **EINER** Batch-Identität fahren
    (wie ein „Vergleichen"-Klick der GUI). Je Config eine family-geformte
    Zusammenfassung. Rückgabe 0 nur, wenn **alle** Läufe ``ok`` sind (report-tauglich),
    1 bei mindestens einem Fehlschlag, 2 wenn der GPU-Lock nicht frei wurde."""
    from filelock import FileLock, Timeout   # lazy (nur der Sweep braucht den Lock)
    from .run import run                       # lazy → cli-Import bleibt torch-frei

    lock = store.PROJECT_DIR / _GPU_LOCK_REL
    lock.parent.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    created_at = datetime.now().isoformat(timespec="seconds")
    run_name = name or f"CLI-Report-Sweep · {created_at}"
    total = len(configs)
    print(f"=== Report-Sweep: {total} Konfigurationen  (run_id={batch_id[:8]}…, "
          f"run_name={run_name!r}) ===\n")

    results: list = []
    try:
        with FileLock(str(lock)).acquire(timeout=_LOCK_TIMEOUT):
            for i, cfg in enumerate(configs, 1):
                sw = f" · sw G{cfg.group_m}" if cfg.swizzle else ""
                op = f" · {cfg.op}" if cfg.op else ""
                ep = f" · ep {cfg.epilog}" if cfg.epilog else ""
                print(f"----- [{i}/{total}] {cfg.family} · {cfg.expr} · "
                      f"{cfg.dtype}→{cfg.acc_dtype}{op}{ep} · "
                      f"TM{cfg.tile['TM']}/{cfg.tile['TN']}/{cfg.tile['TK']}{sw} -----")
                res = run(cfg, run_id=batch_id, run_name=run_name, created_at=created_at)
                print_summary(res)
                print()
                results.append(res)
    except Timeout:
        print(f"GPU belegt: ein anderer Lauf hält den Lock seit über {_LOCK_TIMEOUT}s. "
              f"Abbruch (bereits gefahrene Läufe sind gespeichert).")
        return 2

    n_ok = sum(1 for r in results if r.status == STATUS_OK)
    print(f"=== Sweep fertig: {n_ok}/{total} ok  →  "
          f"{store.store_relpath(store.RESULTS_JSONL)} ===")
    if n_ok < total:
        for r in results:
            if r.status != STATUS_OK:
                print(f"    FEHLGESCHLAGEN: {r.config.get('family')} · "
                      f"{r.config.get('expr')} · {r.config.get('dtype')} → "
                      f"{r.status}: {r.error}")
    return 0 if n_ok == total else 1


# ---------------------------------------------------------------------------
# Einstieg
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tool_pipeline.cli",
        description="Einzellauf ODER Report-Batch-Sweep end-to-end: "
                    "generieren → verifizieren → messen → speichern.",
    )
    # Einzellauf-Achsen (additiv, alle mit sinnvollem Default).
    p.add_argument("--family", default="contraction",
                   choices=["contraction", "elementwise", "reduction"],
                   help="Operations-Familie (Default: contraction)")
    p.add_argument("--op", default=None,
                   help="Elementwise-Op (add/mul/copy); Reduktion nutzt immer sum, "
                        "Kontraktion keine Op")
    p.add_argument("--expr", default="ik,kj->ij",
                   help="einsum-Ausdruck (Default: ik,kj->ij). Für andere Familien "
                        "passenden Ausdruck angeben, z. B. ij,ij->ij oder ij->i")
    p.add_argument("--size", type=int, default=512,
                   help="einheitliche Größe je Index (Default 512)")
    p.add_argument("--M", type=int, default=None, help="überschreibt Index i (Zeilen M)")
    p.add_argument("--N", type=int, default=None, help="überschreibt Index j (Spalten N)")
    p.add_argument("--K", type=int, default=None, help="überschreibt Index k (Kontraktion K)")
    p.add_argument("--epilog", default=None, choices=["bias", "relu"],
                   help="Epilog-Fusion auf dem Akku-Tile vor dem Store (nur Kontraktion, "
                        "2-Operanden): bias = acc+D, relu = max(acc,0). Ohne Angabe keine "
                        "Fusion (Kernel byte-identisch zu TZ 1-8)")
    # Sweep-Modus.
    p.add_argument("--sweep", action="store_true",
                   help="kompletten Report-Sweep fahren (alle Familien, GROUP_M/Tiles/n-är)")
    p.add_argument("--show-configs", action="store_true",
                   help="nur die Sweep-Konfigurationen auflisten (kein GPU-Lauf)")
    p.add_argument("--show-kernel", action="store_true",
                   help="den generierten Kernel-Quelltext mit ausgeben (nur Einzellauf)")
    args = p.parse_args(argv)

    # --- Sweep / nur Configs listen ---
    if args.sweep or args.show_configs:
        configs = sweep_configs()
        if args.show_configs:
            print(f"=== Report-Sweep: {len(configs)} Konfigurationen (kein GPU) ===")
            for i, cfg in enumerate(configs, 1):
                print(f"[{i:2}] {store.config_slug(cfg):<58}  family={cfg.family:<11} "
                      f"dims={cfg.dim_sizes} baselines={cfg.baselines}")
            return 0
        return run_sweep(configs)

    # --- Einzellauf ---
    from .run import run   # lazy → cli-Import (z. B. für Tests von sweep_configs) torch-frei
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
