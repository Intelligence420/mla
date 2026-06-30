"""tool_pipeline.store.store — Results-Store + Kernel-Persistenz.

Zwei persistente Artefakte, beide unter `project/results/`:

* `results.jsonl` — **eine JSON-Zeile je Lauf** (ein `RunResult`-dict). Bewusst
  JSON Lines: transparent, git-diff-bar, mit `pandas.read_json(lines=True)`
  ladbar (Report-Datenquelle).
* `kernels/<slug>.py` — der persistierte generierte Kernel-Quelltext. Dient
  zugleich als **Compile-Cache-Artefakt** und als UI-Code-Anzeige (später).

Der `<slug>` ist ein **lesbarer, normalisierter** Name aus genau den Feldern,
die den Quelltext bestimmen — `(expr, dtype, acc_dtype, tile, swizzle)`, z. B.
`ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py`. Bewusst lesbar statt Hash: die
Config steht ohnehin komplett im `results.jsonl`, und ein sprechender Name ist
transparenter + git-diff-freundlich. **Normalisiert** (Whitespace raus, feste
Tile-Reihenfolge), damit logisch gleiche Configs denselben Slug ⇒ Cache-Treffer
ergeben. Bewusst **ohne** `dim_sizes` (M/N/K sind Laufzeit-`ct.Constant`-Args,
der Quelltext ist über die Größen generisch) und **ohne** `family` (die ist
eine Funktion von `expr` — der Router leitet sie deterministisch ab).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from ..schema import RunConfig, RunResult

# ---------------------------------------------------------------------------
# Pfade — relativ zu project/ (store.py liegt in project/tool_pipeline/store/)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]      # .../project
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
KERNELS_DIR = RESULTS_DIR / "kernels"


# ---------------------------------------------------------------------------
# Lesbarer Config-Slug = Cache-Schlüssel + Kernel-Dateiname
# ---------------------------------------------------------------------------
def config_slug(config: Union[RunConfig, dict[str, Any]]) -> str:
    """Lesbarer, normalisierter Name aus den quelltextbestimmenden Feldern.

    Form: ``<expr>__<dtype>-<acc_dtype>__TM<..>_TN<..>_TK<..>[__sw]`` mit im
    `expr` ersetztem `->` (→ `_to_`) und `,` (→ `_`). Deterministisch über die
    Tile-Reihenfolge (TM/TN/TK werden explizit gelesen) und **unabhängig** von
    `dim_sizes`/`baselines` ⇒ logisch gleiche Configs ⇒ gleicher Slug.
    """
    d = config.to_dict() if isinstance(config, RunConfig) else dict(config)
    expr = (d.get("expr") or "").replace(" ", "").replace("->", "_to_").replace(",", "_")
    t = d.get("tile") or {}
    tile = f"TM{t.get('TM')}_TN{t.get('TN')}_TK{t.get('TK')}"
    slug = f"{expr}__{d.get('dtype')}-{d.get('acc_dtype')}__{tile}"
    if d.get("swizzle"):
        slug += "__sw"
    return slug


# ---------------------------------------------------------------------------
# Kernel-Quelltext persistieren
# ---------------------------------------------------------------------------
def kernel_file(slug: str, kernels_dir: Path = KERNELS_DIR) -> Path:
    """Absoluter Pfad der Kernel-Datei für einen Slug (ohne sie zu schreiben)."""
    return kernels_dir / f"{slug}.py"


def save_kernel(src: str, slug: str, kernels_dir: Path = KERNELS_DIR) -> Path:
    """Generierten Quelltext nach `kernels/<slug>.py` schreiben; Pfad zurück.

    Idempotent: überschreibt eine bestehende Datei mit identischem Inhalt
    folgenlos (gleicher Slug ⇒ gleicher Quelltext).
    """
    kernels_dir.mkdir(parents=True, exist_ok=True)
    path = kernel_file(slug, kernels_dir)
    path.write_text(src, encoding="utf-8")
    return path


def store_relpath(path: Union[str, Path]) -> str:
    """Pfad project-relativ als String (portabel im JSONL, maschinenunabhängig)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_DIR))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Results-Store: anhängen + lesen
# ---------------------------------------------------------------------------
def append_result(result: Union[RunResult, dict[str, Any]],
                  path: Path = RESULTS_JSONL) -> Path:
    """Ein `RunResult` als **eine** JSON-Zeile an `results.jsonl` anhängen."""
    d = result.to_dict() if isinstance(result, RunResult) else dict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(d, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def read_results(path: Path = RESULTS_JSONL):
    """Alle Läufe als `pandas.DataFrame` (leer, falls Datei fehlt/leer)."""
    import pandas as pd  # lazy: Schreiben soll auch ohne pandas funktionieren

    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)
