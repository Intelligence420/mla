"""tool_pipeline.report_figures — Report-Figuren aus ``results.jsonl`` (headless).

Liest die **jüngste** ``CLI-Report-Sweep``-Charge aus dem Store (verify-before-trust:
**nur** ``status=="ok"``) und rendert vier PNGs nach ``sphinx/source/_static/gsc/``:

  1. ``durchsatz_formate.png``    — Kontraktion Durchsatz je Zahlenformat (cuTile vs cuBLAS)
  2. ``genauigkeit_durchsatz.png`` — Genauigkeit ↔ Durchsatz (Trade-off je Format)
  3. ``roofline.png``             — die Headline: memory- vs compute-bound (beide Seiten)
  4. ``tile_swizzle.png``         — Tile- & GROUP_M-Vergleich (fp16, TZ-7.5-Multi-Config)

**Torch-/GPU-frei** (nur ``matplotlib``/``json`` + die reinen ``hardware``-Kennwerte):
die PNGs werden **vorab** erzeugt und eingecheckt, damit ``cd sphinx && make html``
ohne GPU/torch durchläuft (CI-tauglich). Die Palette ist die validierte,
CVD-sichere Light-Mode-Palette des dataviz-Maßstabs.

Aufruf (aus ``project/``, venv aktiv)::

    python -m tool_pipeline.report_figures            # nach dem CLI-Sweep
    python -m tool_pipeline.report_figures --out <dir>  # anderes Ausgabeverzeichnis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless (kein Display, kein Backend-Fenster)
import matplotlib.pyplot as plt   # noqa: E402

from .hardware import MEM_BANDWIDTH_GBPS, PEAK_TFLOPS   # noqa: E402  (torch-frei)
from .store import store   # noqa: E402  (nur Pfad/Reader; pandas lazy)

# --- dataviz-Palette (Light-Mode, validiert, CVD-sicher; s. dataviz/references) ---
_BLUE, _AQUA, _YELLOW, _GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
_VIOLET, _RED, _MAGENTA, _ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"
_INK, _SECOND, _MUTED = "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS, _SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Standard-Ausgabeort: sphinx/source/_static/gsc/ (relativ zum Repo-Root über project/).
_DEFAULT_OUT = store.PROJECT_DIR.parent / "sphinx" / "source" / "_static" / "gsc"

# Anzeige-Label je Format.
_FMT_LABEL = {"fp16": "fp16", "bf16": "bf16", "tf32": "tf32",
              "fp8e4m3": "fp8 e4m3", "fp8e5m2": "fp8 e5m2", "fp32": "fp32"}


# ---------------------------------------------------------------------------
# Daten laden (nur ok-Läufe der jüngsten Report-Sweep-Charge)
# ---------------------------------------------------------------------------
def load_report_rows(path: Path = store.RESULTS_JSONL) -> list[dict]:
    """Alle Zeilen der **jüngsten** ``CLI-Report-Sweep``-Charge mit ``status=="ok"``
    (verify-before-trust). Fällt auf *alle* ok-Zeilen zurück, wenn keine solche
    Charge existiert (z. B. Läufe aus der GUI)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    sweeps = [r for r in rows if (r.get("run_name") or "").startswith("CLI-Report-Sweep")]
    if sweeps:
        latest = max(sweeps, key=lambda r: r.get("created_at") or "")["run_id"]
        rows = [r for r in rows if r.get("run_id") == latest]
    return [r for r in rows if r.get("status") == "ok"]


def _cfg(r: dict) -> dict:
    return r.get("config") or {}


def _is_contraction_gemm(r: dict) -> bool:
    """2-Operand-Kontraktion ik,kj->ij (kein n-är) **ohne Epilog**.

    Die Epilog-Ausschlussklausel ist wichtig: seit TZ 9 fährt der Sweep denselben
    Ausdruck auch fusioniert (``__ep_bias``/``__ep_relu``). Diese Läufe gehören in die
    Fusions-Figur, NICHT in den Format-/Tile-/Roofline-Vergleich — sonst tauchten sie
    dort als zusätzliche „fp16"-Balken auf und verfälschten die Aussage."""
    c = _cfg(r)
    return (c.get("family") == "contraction" and c.get("expr") == "ik,kj->ij"
            and not c.get("epilog"))


def _sizes(r: dict) -> dict:
    return (r.get("provenance") or {}).get("sizes") or {}


def _is_square(r: dict) -> bool:
    """M==N==K — die Form des Format-/Tile-/Swizzle-Vergleichs (1024³).

    Seit TZ 9 enthält der Sweep zusätzlich zwei **unfusionierte** Bezugspunkte auf
    anderen Formen (schmal 4096·4096·64, tief 1024·1024·8192). Sie sind legitime
    ok-Läufe derselben Config-Familie und würden ohne diesen Filter als weitere
    „fp16 / Tile 128/128/64"-Einträge in die bestehenden Figuren rutschen."""
    s = _sizes(r)
    m, n, k = s.get("M"), s.get("N"), s.get("K")
    return None not in (m, n, k) and m == n == k


def _is_nary(r: dict) -> bool:
    return _cfg(r).get("family") == "contraction" and _cfg(r).get("expr", "").count(",") >= 2


def _format_group(rows: list[dict]) -> list[dict]:
    """Die vier Format-Vergleichs-Configs (Tile 128/128/64, ohne Swizzle), sortiert
    in der kanonischen Format-Reihenfolge fp16, bf16, tf32, fp8e4m3."""
    order = {"fp16": 0, "bf16": 1, "tf32": 2, "fp8e4m3": 3}
    grp = [r for r in rows if _is_contraction_gemm(r) and _is_square(r)
           and not _cfg(r).get("swizzle")
           and _cfg(r).get("tile") == {"TM": 128, "TN": 128, "TK": 64}]
    return sorted(grp, key=lambda r: order.get(_cfg(r).get("dtype"), 99))


# ---------------------------------------------------------------------------
# Gemeinsames Styling (Light-Surface, magere Marken, zurückhaltende Achsen)
# ---------------------------------------------------------------------------
def _new_fig(w: float = 8.0, h: float = 4.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=140)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.title.set_color(_INK)
    return fig, ax


def _title(ax, title: str, subtitle: str | None = None):
    """Titel (Ink) + optionaler Untertitel (Sekundär) OHNE Überlappung: der
    Untertitel steht als eigene Achsen-Annotation knapp über der Plotfläche."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=26, loc="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xytext=(0, 10), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9.5, color=_SECOND)


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figur 1 — Durchsatz je Format (cuTile vs cuBLAS, gruppierte Balken)
# ---------------------------------------------------------------------------
def fig_durchsatz_formate(rows: list[dict], out: Path) -> Path | None:
    grp = _format_group(rows)
    if not grp:
        return None
    labels = [_FMT_LABEL.get(_cfg(r).get("dtype"), _cfg(r).get("dtype")) for r in grp]
    cutile = [r["metrics"].get("tflops") for r in grp]
    cublas = [(r["metrics"].get("baselines") or {}).get("cublas", {}).get("tflops") for r in grp]

    fig, ax = _new_fig()
    x = range(len(grp))
    bw = 0.38
    b1 = ax.bar([i - bw / 2 for i in x], cutile, bw, label="cuTile (dieses Tool)",
                color=_BLUE, zorder=3)
    # cuBLAS nur, wo verfügbar (fp8 hat keinen matmul-Pfad → Lücke).
    xb = [i + bw / 2 for i, v in zip(x, cublas) if v is not None]
    yb = [v for v in cublas if v is not None]
    b2 = ax.bar(xb, yb, bw, label="cuBLAS (Obergrenze)", color=_AQUA, zorder=3)

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        if h:
            ax.annotate(f"{h:.0f}", (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.5, color=_SECOND)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Durchsatz  [TFLOP/s]", color=_SECOND, fontsize=10)
    ax.set_ylim(0, max(v for v in cutile + yb if v) * 1.20)   # Kopf-Freiraum für Labels/Legende
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=_SECOND)
    _title(ax, "Kontraktion: Durchsatz je Zahlenformat",
           "GEMM ik,kj->ij · 1024³ · Tile 128/128/64 · NVIDIA GB10")
    return _save(fig, out, "durchsatz_formate.png")


# ---------------------------------------------------------------------------
# Figur 2 — Genauigkeit ↔ Durchsatz (Trade-off je Format)
# ---------------------------------------------------------------------------
def fig_genauigkeit_durchsatz(rows: list[dict], out: Path) -> Path | None:
    grp = _format_group(rows)
    if not grp:
        return None
    colors = {"fp16": _BLUE, "bf16": _AQUA, "tf32": _VIOLET, "fp8e4m3": _RED}
    fig, ax = _new_fig()
    for r in grp:
        d = _cfg(r).get("dtype")
        x = r["metrics"].get("tflops")
        y = r["accuracy"].get("max_abs_err")
        if x is None or y is None:
            continue
        y = max(y, 1e-6)   # log-Achse: 0 (exakt) auf den Boden heben
        ax.scatter([x], [y], s=90, color=colors.get(d, _MUTED), zorder=3,
                   edgecolor=_SURFACE, linewidth=1.2)
        ax.annotate(_FMT_LABEL.get(d, d), (x, y), xytext=(7, 4),
                    textcoords="offset points", fontsize=9.5, color=_INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("Durchsatz  [TFLOP/s]  →  schneller", color=_SECOND, fontsize=10)
    ax.set_ylabel("max. abs. Fehler vs. fp32  ↑  ungenauer", color=_SECOND, fontsize=10)
    ax.margins(x=0.14, y=0.30)   # Rand-Freiraum, damit Direkt-Labels (z. B. fp8) nicht anstoßen
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _title(ax, "Genauigkeit ↔ Durchsatz",
           "Kontraktion je Format — unten-rechts = schnell UND genau")
    return _save(fig, out, "genauigkeit_durchsatz.png")


# ---------------------------------------------------------------------------
# Figur 3 — Roofline (memory- vs compute-bound, beide Seiten) — die Headline
# ---------------------------------------------------------------------------
def fig_roofline(rows: list[dict], out: Path) -> Path | None:
    # Punkte je Familie sammeln: (AI, TFLOP/s).
    fams = {
        "contraction": {"color": _BLUE, "marker": "o", "label": "Kontraktion (GEMM)", "pts": []},
        "elementwise": {"color": _AQUA, "marker": "s", "label": "Elementwise", "pts": []},
        "reduction":   {"color": _VIOLET, "marker": "^", "label": "Reduktion", "pts": []},
        "nary":        {"color": _RED, "marker": "*", "label": "n-äre Kette", "pts": []},
    }
    for r in rows:
        m = r["metrics"]
        ai, tf = m.get("arithmetic_intensity"), m.get("tflops")
        if ai is None or tf is None or tf <= 0:
            continue
        key = "nary" if _is_nary(r) else _cfg(r).get("family")
        # Kontraktion nur die vier Format-Punkte (Tile/Swizzle-Varianten liegen
        # aufeinander → kein Roofline-Mehrwert, nur Klumpen).
        if key == "contraction" and not (_is_contraction_gemm(r) and _is_square(r)
                                          and not _cfg(r).get("swizzle")
                                          and _cfg(r).get("tile") == {"TM": 128, "TN": 128, "TK": 64}):
            continue
        if key in fams:
            fams[key]["pts"].append((ai, tf))
    if not any(f["pts"] for f in fams.values()):
        return None

    fig, ax = _new_fig(8.4, 5.2)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Roofline-Decken: Bandbreiten-Schräge (TFLOP/s = 0.273·AI) + Compute-Decken.
    ai_min, ai_max = 0.05, 2000.0
    import numpy as np
    xs = np.logspace(-1.3, 3.3, 200)
    slope = (MEM_BANDWIDTH_GBPS / 1000.0) * xs
    ax.plot(xs, slope, color=_MUTED, linewidth=1.8, zorder=2)
    for dtype, peak in (("fp16", PEAK_TFLOPS["fp16"]), ("tf32", PEAK_TFLOPS["tf32"])):
        ax.axhline(peak, color=_AXIS, linewidth=1.2, linestyle="--", zorder=1)
        ax.annotate(f"Peak {dtype}  {peak:.0f} TFLOP/s", xy=(ai_max, peak),
                    xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=_MUTED)
    ax.annotate(f"Bandbreite {MEM_BANDWIDTH_GBPS:.0f} GB/s", xy=(6, (MEM_BANDWIDTH_GBPS / 1000.0) * 6),
                xytext=(6, -14), textcoords="offset points", rotation=32,
                fontsize=8, color=_MUTED)

    # Messpunkte je Familie.
    for f in fams.values():
        if not f["pts"]:
            continue
        xs_p = [p[0] for p in f["pts"]]
        ys_p = [p[1] for p in f["pts"]]
        sz = 240 if f["marker"] == "*" else 90
        ax.scatter(xs_p, ys_p, s=sz, color=f["color"], marker=f["marker"],
                   label=f["label"], zorder=4, edgecolor=_SURFACE, linewidth=1.0)

    # Regionen-Beschriftung (memory-bound links / compute-bound rechts).
    ax.axvspan(ai_min, 2.0, color=_AQUA, alpha=0.05, zorder=0)
    ax.annotate("memory-bound\n(Elementwise, Reduktion)", xy=(0.14, 60),
                fontsize=8.5, color=_SECOND, ha="center")
    ax.annotate("compute-nah\n(Kontraktion)", xy=(220, 0.9),
                fontsize=8.5, color=_SECOND, ha="center")

    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.02, 400)
    ax.set_xlabel("Arithmetische Intensität  [FLOP/Byte]", color=_SECOND, fontsize=10)
    ax.set_ylabel("Durchsatz  [TFLOP/s]", color=_SECOND, fontsize=10)
    ax.grid(True, which="both", color=_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=_SECOND)
    _title(ax, "Roofline (GB10): memory- vs compute-bound",
           "beide Familien-Seiten — die Bandbreiten-Schräge dominiert (Ridge ≈ 780 FLOP/B)")
    return _save(fig, out, "roofline.png")


# ---------------------------------------------------------------------------
# Figur 4 — Tile- & GROUP_M-Vergleich (fp16, TZ-7.5-Multi-Config)
# ---------------------------------------------------------------------------
def fig_tile_swizzle(rows: list[dict], out: Path) -> Path | None:
    gemm = [r for r in rows if _is_contraction_gemm(r) and _is_square(r)
            and _cfg(r).get("dtype") == "fp16"]
    if not gemm:
        return None
    tiles, swz = [], []
    for r in gemm:
        c, tf = _cfg(r), r["metrics"].get("tflops")
        t = c.get("tile", {})
        if c.get("swizzle"):
            swz.append((f"G{c.get('group_m')}", tf))
        else:
            tiles.append((f"{t.get('TM')}/{t.get('TN')}/{t.get('TK')}", tf))
    tiles.sort(key=lambda kv: kv[1] or 0)          # Tiles nach Durchsatz (worst→best)
    swz.sort(key=lambda kv: int(kv[0][1:]))        # Swizzle numerisch: G8 < G16 < G32

    labels = [f"Tile {k}" for k, _ in tiles] + [f"Swizzle {k}" for k, _ in swz]
    vals = [v for _, v in tiles] + [v for _, v in swz]
    colors = [_BLUE] * len(tiles) + [_AQUA] * len(swz)

    fig, ax = _new_fig(8.4, 4.8)
    bars = ax.bar(range(len(vals)), vals, 0.62, color=colors, zorder=3)
    for rect, v in zip(bars, vals):
        if v:
            ax.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.5, color=_SECOND)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Durchsatz  [TFLOP/s]", color=_SECOND, fontsize=10)
    ax.set_ylim(0, max(v for v in vals if v) * 1.22)   # Kopf-Freiraum für Legende/Labels
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Zwei-Farben-Legende (Tile-Größe vs. L2-Swizzle-Gruppengröße).
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_BLUE, label="Tile-Größe (ohne Swizzle)"),
                       Patch(color=_AQUA, label="L2-Swizzle GROUP_M (Tile 128/128/64)")],
              frameon=False, fontsize=9, loc="upper left", labelcolor=_SECOND)
    _title(ax, "Tuning-Raum (fp16, 1024³): Tile-Größe & L2-Swizzle",
           "gleicher verifizierter Kernel — nur Kachelung/Block-Umordnung variiert")
    return _save(fig, out, "tile_swizzle.png")


# ---------------------------------------------------------------------------
# Figur 5 — Fusion: fused vs. sequentiell über die arithmetische Intensität (TZ 9)
# ---------------------------------------------------------------------------
def _fusion_rows(rows: list[dict]) -> list[dict]:
    """Die fused Läufe mit verfügbarem Vergleich, sortiert nach AI (aufsteigend) —
    also von memory-bound nach compute-dominiert. Ein Eintrag je (Epilog, Form)."""
    out = []
    for r in rows:
        f = (r.get("metrics") or {}).get("fusion")
        if not isinstance(f, dict) or not f.get("available"):
            continue
        if not f.get("fused_ms") or not f.get("sequential_ms"):
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["metrics"].get("arithmetic_intensity") or 0)


def _shape_label(r: dict) -> str:
    s = _sizes(r)
    return f"{s.get('M')}·{s.get('N')}·{s.get('K')}"


def fig_fusion(rows: list[dict], out: Path) -> Path | None:
    """Fusions-Figur: **Speedup über arithmetischer Intensität**, eine Linie je Epilog.

    Die Frage der Figur ist „wann lohnt Fusion?", also eine Trend-Frage über eine
    kontinuierliche Größe ⇒ Linie mit Markern, nicht Balken. Die Fusion spart immer
    denselben Zwischentensor-Roundtrip (2·4·M·N Bytes); was sich ändert, ist sein
    Gewicht gegenüber der Kontraktion selbst. Links (niedrige AI, memory-bound) lohnt
    Fusion deutlich, rechts (compute-dominiert) läuft sie gegen 1,0 — die Referenzlinie.

    Farbe = Identität der zwei Epiloge (kategorial, feste Reihenfolge Blau→Aqua aus der
    validierten CVD-sicheren Palette); die Serien sind zusätzlich direkt beschriftet,
    Zahlen und Labels tragen Text-Farben (nie die Serien-Farbe)."""
    fus = _fusion_rows(rows)
    if not fus:
        return None

    # Je Epilog eine nach AI sortierte Serie (_fusion_rows liefert bereits sortiert).
    series: dict[str, list[tuple[float, float, str]]] = {}
    for r in fus:
        f = r["metrics"]["fusion"]
        ai = r["metrics"].get("arithmetic_intensity")
        if ai is None or f.get("speedup") is None:
            continue
        series.setdefault(f.get("epilog") or "?", []).append(
            (ai, f["speedup"], _shape_label(r)))
    if not series:
        return None

    colors = {"bias": _BLUE, "relu": _AQUA}
    fig, ax = _new_fig(8.6, 5.0)
    ax.set_xscale("log")

    all_ai = [p[0] for pts in series.values() for p in pts]
    all_sp = [p[1] for pts in series.values() for p in pts]
    x_lo, x_hi = min(all_ai) / 2.2, max(all_ai) * 2.6

    # Referenzlinie: kein Gewinn. Darunter wäre die Fusion langsamer (A04-Bereich).
    ax.axhline(1.0, color=_AXIS, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("kein Gewinn  (fused = sequentiell)", xy=(x_lo, 1.0),
                xytext=(4, 5), textcoords="offset points", fontsize=8.5, color=_MUTED)

    # Für die Label-Platzierung: liegt ein Punkt nahe (Faktor 1.4 in x) an einem Punkt
    # der anderen Serie, wird der NIEDRIGERE nach unten beschriftet — sonst überdecken
    # sich die Werte dort, wo die Serien zusammenlaufen (rechter Rand).
    def _label_below(xv: float, yv: float) -> bool:
        for pts in series.values():
            for ai2, sp2, _shape in pts:
                if ai2 == xv and sp2 == yv:
                    continue
                if max(xv, ai2) / min(xv, ai2) < 1.4 and sp2 > yv:
                    return True
        return False

    for name, pts in series.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        col = colors.get(name, _VIOLET)
        ax.plot(xs, ys, color=col, linewidth=2.0, zorder=3, label=f"Epilog {name}")
        ax.scatter(xs, ys, s=95, color=col, zorder=4,
                   edgecolor=_SURFACE, linewidth=2.0)     # 2px Surface-Ring
        # Direktes Serien-Label am LINKEN Ende (dort liegen die Serien weit auseinander;
        # rechts laufen sie zusammen). Text-Farbe, nicht Serien-Farbe.
        ax.annotate(f"{name} ", (xs[0], ys[0]), xytext=(-10, 0),
                    textcoords="offset points", fontsize=9.5, color=_SECOND,
                    ha="right", va="center")
        # Speedup je Punkt (n=3 je Serie — jeder Punkt IST die Aussage).
        for xv, yv in zip(xs, ys):
            dy = -16 if _label_below(xv, yv) else 11
            ax.annotate(f"{yv:.2f}×", (xv, yv), xytext=(0, dy),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=_SECOND)

    # Die zugehörigen Formen (M·N·K) stehen bewusst NICHT in der Figur: auf der
    # Log-Achse rücken 1024³ (AI ~230) und 1024·1024·8192 (AI ~443) so nah zusammen,
    # dass die Labels sich überdecken. Die Zuordnung Form ↔ AI trägt die Tabelle im
    # Report (report.rst) — die Figur beantwortet allein die Trend-Frage.
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.88, max(all_sp) * 1.14)
    ax.set_xlabel("Arithmetische Intensität des fusionierten Kernels  [FLOP/Byte]",
                  color=_SECOND, fontsize=10)
    ax.set_ylabel("Speedup  fused / sequentiell", color=_SECOND, fontsize=10)
    ax.grid(True, which="major", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=_SECOND)
    _title(ax, "Wann lohnt Fusion? Speedup über arithmetischer Intensität",
           "fused (ein Kernel, Epilog auf dem Akku-Tile) vs. sequentiell "
           "(zwei Kernel, Zwischentensor über DRAM) — fp16→fp32, GB10")
    return _save(fig, out, "fusion.png")


# ---------------------------------------------------------------------------
# Einstieg
# ---------------------------------------------------------------------------
def generate_all(out: Path = _DEFAULT_OUT, path: Path = store.RESULTS_JSONL) -> list[Path]:
    """Alle fünf Figuren erzeugen; Liste der geschriebenen Pfade zurück (überspringt
    Figuren ohne passende Daten still, gibt aber die Anzahl aus)."""
    rows = load_report_rows(path)
    made = []
    for fn in (fig_durchsatz_formate, fig_genauigkeit_durchsatz, fig_roofline,
               fig_tile_swizzle, fig_fusion):
        p = fn(rows, out)
        if p is not None:
            made.append(p)
    return made


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tool_pipeline.report_figures",
        description="Report-Figuren (PNG) aus results.jsonl erzeugen — headless, torch-frei.")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help=f"Ausgabeverzeichnis (Default: {_DEFAULT_OUT})")
    p.add_argument("--results", type=Path, default=store.RESULTS_JSONL,
                   help="Pfad zu results.jsonl")
    args = p.parse_args(argv)

    rows = load_report_rows(args.results)
    if not rows:
        print("Keine ok-Läufe gefunden — zuerst 'python -m tool_pipeline.cli --sweep' fahren.")
        return 1
    made = generate_all(args.out, args.results)
    print(f"{len(made)} Figur(en) aus {len(rows)} ok-Läufen erzeugt:")
    for m in made:
        print(f"  {m}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
