"""Headless-Tests des CLI-Sweeps (TZ 8-3): die Config-Erzeugung ist deterministisch
und **GPU-/torch-frei** prüfbar (``sweep_configs`` ruft kein ``run()``). Belegt, dass
der Report-Sweep alle drei Familien, mehrere Tiles, GROUP_M-Varianten und eine n-äre
Kette abdeckt — und dieselben (bedingten) Slugs wie die GUI erzeugt.

Standalone (``python tests/test_cli.py`` aus ``project/``) **oder** via pytest.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline import cli  # noqa: E402  (importiert controls; run bleibt lazy)
from tool_pipeline import report_figures as RF  # noqa: E402  (torch-frei; Agg-Backend)
from tool_pipeline.schema import RunResult  # noqa: E402
from tool_pipeline.store import store as S  # noqa: E402

_SP = os.environ.get("SP", "/tmp")


def _args(**kw) -> argparse.Namespace:
    """Einzellauf-Argumente mit den CLI-Defaults; Overrides via kw."""
    base = dict(family="contraction", op=None, expr="ik,kj->ij",
                size=512, M=None, N=None, K=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_sweep_configs_covers_all_families():
    """Der Sweep deckt alle drei Familien ab (Roofline mit beiden Seiten)."""
    cfgs = cli.sweep_configs()
    fams = {c.family for c in cfgs}
    assert fams == {"contraction", "elementwise", "reduction"}, fams
    assert len(cfgs) >= 12, f"unerwartet wenige Configs: {len(cfgs)}"


def test_sweep_configs_has_nary_chain():
    """Genau eine n-äre Ketten-Kontraktion (zwei Kommas → drei Operanden)."""
    nary = [c for c in cli.sweep_configs() if c.expr.count(",") >= 2]
    assert len(nary) == 1, [c.expr for c in nary]
    assert nary[0].expr == "ij,jk,kl->il" and nary[0].family == "contraction"


def test_sweep_configs_has_multiple_tiles_and_group_m():
    """Mehrere Tiles UND GROUP_M-Varianten (≠ 8) sind vertreten; der bedingte
    ``__sw_g<N>``-Slug erscheint nur bei swizzle & group_m != 8."""
    cfgs = cli.sweep_configs()
    tiles = {(c.tile["TM"], c.tile["TN"], c.tile["TK"]) for c in cfgs}
    assert len(tiles) >= 3, f"zu wenige verschiedene Tiles: {tiles}"
    # GROUP_M != 8 mit Swizzle → Slug trägt __sw_g<N>.
    g_variants = [c for c in cfgs if c.swizzle and c.group_m != 8]
    assert g_variants, "keine GROUP_M≠8-Swizzle-Variante im Sweep"
    slugs = [S.config_slug(c) for c in g_variants]
    assert all("__sw_g" in s for s in slugs), slugs
    # Der Default-Swizzle (G8) bleibt byte-identisch (bares __sw, kein _g8).
    g8 = [c for c in cfgs if c.swizzle and c.group_m == 8]
    assert g8 and all(S.config_slug(c).endswith("__sw") for c in g8)


def test_sweep_configs_memory_bound_no_swizzle_no_baselines():
    """memory-bound-Familien haben kein Swizzle und keine GEMM-Baselines."""
    mem = [c for c in cli.sweep_configs() if c.family in ("elementwise", "reduction")]
    assert mem, "keine memory-bound-Configs im Sweep"
    for c in mem:
        assert c.swizzle is False, f"{c.expr}: memory-bound darf kein Swizzle haben"
        assert c.baselines == [], f"{c.expr}: memory-bound darf keine Baselines haben"
    # Elementwise trägt op=add, Reduktion op=sum.
    assert any(c.family == "elementwise" and c.op == "add" for c in mem)
    assert any(c.family == "reduction" and c.op == "sum" for c in mem)


def test_sweep_configs_no_duplicate_work():
    """Keine zwei Configs machen dieselbe Arbeit doppelt.

    Der Slug identifiziert das **Kernel-Artefakt**, nicht den Lauf: M/N/K sind
    Launch-Argumente, keine Quelltext-Literale — derselbe Kernel bedient also bewusst
    mehrere Formen (genau das nutzt der Fusions-Sweep aus TZ 9: EIN ``__ep_bias``-Kernel
    für drei Shapes, einmal compilen, dreimal messen). Die Invariante ist deshalb
    ``(Slug, Größen, Baselines)`` eindeutig — nicht der Slug allein.

    (Vor TZ 9 waren die Slugs allein eindeutig, weil der Sweep jede Kernel-Variante nur
    auf genau einer Form fuhr. Der Fusions-Sweep bricht diese zufällige Eigenschaft
    absichtlich — die geprüfte Zusage „keine doppelte GPU-Arbeit" bleibt dieselbe.)"""
    keys = [(S.config_slug(c), tuple(sorted(c.dim_sizes.items())), tuple(c.baselines))
            for c in cli.sweep_configs()]
    dupes = [k for k in keys if keys.count(k) > 1]
    assert len(keys) == len(set(keys)), f"doppelte Läufe: {dupes}"


def test_sweep_configs_baselines_only_on_first_format_combo():
    """cuBLAS-Baseline hängt nur an der ersten (Tile,Swizzle)-Kombi je Format
    (nicht an jedem Tile erneut) — Vertrag von configs_from_selection."""
    contr = [c for c in cli.sweep_configs() if c.family == "contraction" and c.baselines]
    # Genau ein Config je Format der Format-Vergleichs-Gruppe trägt cuBLAS (4 Formate).
    assert all(c.baselines == ["cublas"] for c in contr)
    assert len(contr) == 4, f"erwartet 4 Baseline-Configs (je Format eins), bekam {len(contr)}"


def test_sweep_configs_has_fusion_trend():
    """Der Fusions-Sweep (TZ 9) fährt **beide** Epiloge über **drei** Formen mit
    aufsteigender arithmetischer Intensität — damit ist der Fusions-Gewinn ein Trend
    (memory-bound → compute-dominiert), kein Einzelbefund. Je Epilog+Form ein Config,
    plus unfusionierte Bezugspunkte für die schmale und die tiefe Form."""
    cfgs = cli.sweep_configs()
    fused = [c for c in cfgs if c.epilog]
    assert {c.epilog for c in fused} == {"bias", "relu"}, {c.epilog for c in fused}
    # Drei Formen je Epilog (schmal / quadratisch / tief).
    shapes = {(c.dim_sizes["i"], c.dim_sizes["k"], c.dim_sizes["j"]) for c in fused}
    assert shapes == {(4096, 64, 4096), (1024, 1024, 1024), (1024, 8192, 1024)}, shapes
    assert len(fused) == 6, [c.dim_sizes for c in fused]
    # Alle drei Formen haben auch einen unfusionierten Bezugspunkt im Sweep.
    plain = {(c.dim_sizes["i"], c.dim_sizes["k"], c.dim_sizes["j"])
             for c in cfgs if c.family == "contraction" and not c.epilog
             and c.expr == "ik,kj->ij" and c.dtype == "fp16"}
    assert shapes <= plain, f"Bezugspunkt fehlt für {shapes - plain}"
    # Fusion ist nur bei der Kontraktion (memory-bound bleibt epilog-frei).
    assert all(c.epilog is None for c in cfgs if c.family in ("elementwise", "reduction"))
    # Der Epilog steht im Slug (kein stiller Cache-Treffer aufs unfusionierte Artefakt).
    assert all("__ep_" in S.config_slug(c) for c in fused)


def test_build_config_epilog_flag_and_validation():
    """``--epilog`` landet additiv in der RunConfig; ohne Flag bleibt sie unverändert.
    Ein Epilog auf memory-bound oder auf einer n-ären Kette bricht laut ab (SystemExit)
    statt still ignoriert zu werden."""
    assert cli.build_config(_args()).epilog is None
    assert cli.build_config(_args(epilog="bias")).epilog == "bias"
    assert cli.build_config(_args(epilog="relu", size=256)).epilog == "relu"
    for bad in (dict(epilog="bias", family="elementwise", expr="ij,ij->ij", op="add"),
                dict(epilog="bias", expr="ij,jk,kl->il")):
        raised = False
        try:
            cli.build_config(_args(**bad))
        except SystemExit as e:
            raised = "Epilog" in str(e)
        assert raised, f"kein lauter Abbruch für {bad}"


def test_print_summary_shows_fusion_lines(capsys=None):
    """``print_summary`` zeigt die Fusions-Zeilen (Speedup + Einordnung + gesparte
    Bytes) nur bei gesetztem Epilog — unfusionierte Ausgabe bleibt unverändert."""
    import io
    from contextlib import redirect_stdout

    def _out(res) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.print_summary(res)
        return buf.getvalue()

    base = dict(status="ok", kernel_path="results/kernels/x.py",
                accuracy={"max_abs_err": 1e-4, "passed": True, "atol": 0.2, "rtol": 0.02},
                timing={"compile_ms": 300.0, "run_ms": 0.5},
                provenance={"gpu": "GB10", "sizes": {"M": 4096, "N": 4096, "K": 64},
                            "timestamp": "2026-07-25T14:00:00"})
    fused = RunResult(config={"family": "contraction", "epilog": "bias", "expr": "ik,kj->ij",
                             "dtype": "fp16", "acc_dtype": "fp32",
                             "tile": {"TM": 128, "TN": 128, "TK": 64}, "swizzle": False},
                      metrics={"tflops": 4.22, "gbps": 199.8, "arithmetic_intensity": 21.1,
                               "fusion": {"available": True, "epilog": "bias",
                                          "fused_ms": 0.5089, "sequential_ms": 1.1018,
                                          "speedup": 2.165, "saved_bytes": 134217728,
                                          "fused_ai": 21.1, "sequential_ai": 9.1}},
                      **base)
    txt = _out(fused)
    assert "epilog=bias" in txt, txt
    assert "Fusion   : 2.165x" in txt and "Fusion gewinnt" in txt, txt
    assert "128.0 MiB" in txt and "9.1 → 21.1 FLOP/B" in txt, txt
    # Nicht verfügbar ⇒ Grund statt stiller Lücke.
    unavail = RunResult(config=dict(fused.config),
                        metrics={"tflops": 4.22,
                                 "fusion": {"available": False, "note": "verify_failed"}},
                        **base)
    assert "kein Vergleich (verify_failed)" in _out(unavail)
    # Ohne Epilog: keine Fusions-Zeile.
    plain = RunResult(config={"family": "contraction", "expr": "ik,kj->ij",
                             "dtype": "fp16", "acc_dtype": "fp32",
                             "tile": {"TM": 128, "TN": 128, "TK": 64}, "swizzle": False},
                      metrics={"tflops": 4.22}, **base)
    out = _out(plain)
    assert "Fusion" not in out and "epilog" not in out, out


def test_build_config_single_run_default_gemm():
    """Default-Einzellauf = klassischer GEMM ik,kj->ij, fp16→fp32, i=k=j=512."""
    cfg = cli.build_config(_args())
    assert cfg.family == "contraction" and cfg.op is None
    assert cfg.expr == "ik,kj->ij"
    assert cfg.dim_sizes == {"i": 512, "k": 512, "j": 512}


def test_build_config_mnk_overrides():
    """--M/--N/--K überschreiben i/j/k (Rückwärtskompatibilität zum GEMM-Aufruf)."""
    cfg = cli.build_config(_args(M=256, N=384, K=128))
    assert cfg.dim_sizes == {"i": 256, "k": 128, "j": 384}


def test_build_config_memory_bound_family():
    """--family/--op/--expr additiv: Elementwise mul mit einheitlicher --size."""
    cfg = cli.build_config(_args(family="elementwise", op="mul", expr="ij,ij->ij", size=64))
    assert cfg.family == "elementwise" and cfg.op == "mul"
    assert cfg.dim_sizes == {"i": 64, "j": 64}
    # Reduktion setzt op=sum unabhängig vom --op-Flag.
    red = cli.build_config(_args(family="reduction", expr="ij->i", size=64))
    assert red.family == "reduction" and red.op == "sum"


# --- Report-Daten-Auswahl (TZ 8-4): verify-before-trust auf der Figuren-Quelle ----
def test_report_figures_selects_latest_ok_sweep():
    """report_figures.load_report_rows liefert NUR die ok-Läufe der JÜNGSTEN
    CLI-Report-Sweep-Charge — verify_failed-Zeilen, ältere Chargen und Fremdläufe
    (GUI) fallen raus. So tragen die Report-Figuren garantiert nur verifizierte Punkte."""
    from pathlib import Path
    p = Path(_SP) / "report_rows_test.jsonl"
    if p.exists():
        p.unlink()

    def _res(rid, name, ca, status="ok", dtype="fp16"):
        return RunResult(status=status,
                         config={"expr": "ik,kj->ij", "dtype": dtype, "acc_dtype": "fp32",
                                 "family": "contraction"},
                         run_id=rid, run_name=name, created_at=ca,
                         metrics={"tflops": 1.0}, provenance={"timestamp": ca})

    # Fremdlauf (GUI, kein Sweep) + alte Sweep-Charge + neue Sweep-Charge (mit 1 Fehlschlag).
    S.append_result(_res("GUI", "contraction · ik,kj->ij · 10:00", "2026-07-16T10:00:00"), path=p)
    S.append_result(_res("OLD", "CLI-Report-Sweep · 2026-07-16T11:00:00", "2026-07-16T11:00:00"), path=p)
    S.append_result(_res("NEW", "CLI-Report-Sweep · 2026-07-16T12:00:00", "2026-07-16T12:00:00"), path=p)
    S.append_result(_res("NEW", "CLI-Report-Sweep · 2026-07-16T12:00:00", "2026-07-16T12:00:00",
                         status="verify_failed", dtype="bf16"), path=p)

    rows = RF.load_report_rows(p)
    assert len(rows) == 1, f"nur der EINE ok-Lauf der jüngsten Charge, bekam {len(rows)}"
    assert rows[0]["run_id"] == "NEW" and rows[0]["status"] == "ok"
    assert rows[0]["config"]["dtype"] == "fp16"   # der bf16-verify_failed ist draußen
    p.unlink()


def test_report_figures_empty_without_data():
    """Keine Datei / keine ok-Läufe ⇒ leere Auswahl (kein Crash)."""
    from pathlib import Path
    assert RF.load_report_rows(Path(_SP) / "does_not_exist_xyz.jsonl") == []


# --- Fusions-Figur (TZ 9) + Abgrenzung der bestehenden Figuren --------------------
def _fusion_row(epilog="bias", M=4096, N=4096, K=64, ai=21.1, speedup=2.22,
                fused=0.496, seq=1.101, available=True):
    """Eine ok-Store-Zeile eines fused Laufs (wie run() sie schreibt)."""
    fusion = ({"available": True, "epilog": epilog, "fused_ms": fused,
               "sequential_ms": seq, "speedup": speedup, "saved_bytes": 2 * 4 * M * N,
               "fused_ai": ai, "sequential_ai": ai / 2}
              if available else {"available": False, "note": "verify_failed"})
    return {"status": "ok",
            "config": {"family": "contraction", "expr": "ik,kj->ij", "epilog": epilog,
                       "dtype": "fp16", "acc_dtype": "fp32", "swizzle": False,
                       "tile": {"TM": 128, "TN": 128, "TK": 64}},
            "metrics": {"tflops": 4.2, "gbps": 200.0, "arithmetic_intensity": ai,
                        "fusion": fusion},
            "provenance": {"sizes": {"M": M, "N": N, "K": K, "B": 1}}}


def test_fusion_rows_sorted_by_ai_and_skips_unavailable():
    """``_fusion_rows`` liefert nur Läufe mit **verfügbarem** Vergleich, aufsteigend
    nach AI — die Reihenfolge trägt die Aussage der Figur (memory-bound → compute-
    dominiert), sie darf nicht von der Store-Reihenfolge abhängen."""
    rows = [_fusion_row(ai=431.2, speedup=1.06),
            _fusion_row(ai=21.1, speedup=2.22),
            _fusion_row(ai=204.8, speedup=1.25),
            _fusion_row(ai=99.9, available=False)]        # kippt raus
    got = RF._fusion_rows(rows)
    assert [r["metrics"]["arithmetic_intensity"] for r in got] == [21.1, 204.8, 431.2]


def test_fused_runs_excluded_from_existing_figures():
    """Die fused Läufe (TZ 9) dürfen NICHT in Format-/Tile-/Roofline-Vergleich rutschen:
    sie tragen denselben Ausdruck ik,kj->ij und dasselbe Tile, sind aber eine andere
    Rechnung. Ebenso die unfusionierten Bezugspunkte auf schmaler/tiefer Form (nicht
    quadratisch) — sonst erschienen sie als zusätzliche „fp16"-Einträge."""
    fused = _fusion_row()
    assert RF._is_contraction_gemm(fused) is False, "fused Lauf gehört nicht in den Vergleich"
    # Unfusioniert, aber nicht quadratisch ⇒ kein Format-/Tile-Vergleichspunkt.
    narrow = _fusion_row(epilog=None)
    narrow["config"]["epilog"] = None
    narrow["metrics"].pop("fusion")
    assert RF._is_contraction_gemm(narrow) is True     # ist ein GEMM …
    assert RF._is_square(narrow) is False              # … aber nicht die Vergleichsform
    square = _fusion_row(epilog=None, M=1024, N=1024, K=1024)
    square["config"]["epilog"] = None
    assert RF._is_square(square) is True
    # Und die Format-Gruppe zieht wirklich nur die quadratischen unfusionierten Läufe.
    grp = RF._format_group([fused, narrow, square])
    assert grp == [square], grp


def test_fig_fusion_writes_png_and_skips_without_data(tmp_path=None):
    """``fig_fusion`` schreibt die PNG aus Fusions-Zeilen und gibt ohne solche Zeilen
    ``None`` zurück (die Figur fehlt dann still — sie wird nie leer erzeugt)."""
    from pathlib import Path
    out = Path(_SP) / "fig_fusion_test"
    rows = [_fusion_row("bias", ai=21.1, speedup=2.22),
            _fusion_row("bias", M=1024, N=1024, K=8192, ai=431.2, speedup=1.06,
                        fused=0.364, seq=0.385),
            _fusion_row("relu", M=4096, N=4096, K=64, ai=31.5, speedup=2.72,
                        fused=0.348, seq=0.946)]
    p = RF.fig_fusion(rows, out)
    assert p is not None and p.name == "fusion.png" and p.stat().st_size > 5000
    p.unlink()
    # Keine Fusions-Daten ⇒ keine Figur (kein leeres PNG).
    assert RF.fig_fusion([_fusion_row(available=False)], out) is None
    assert RF.fig_fusion([], out) is None


def _main() -> int:
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
