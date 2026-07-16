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
import os
import tempfile
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

    Form: ``<expr>__<dtype>-<acc_dtype>__TM<..>_TN<..>_TK<..>[__<op>][__sw[_g<N>]]``
    mit im `expr` ersetztem `->` (→ `_to_`) und `,` (→ `_`). Deterministisch über die
    Tile-Reihenfolge (TM/TN/TK werden explizit gelesen) und **unabhängig** von
    `dim_sizes`/`baselines` ⇒ logisch gleiche Configs ⇒ gleicher Slug.

    ``op`` (TZ 7) wird nur angehängt, wenn gesetzt: Kontraktion (``op=None``) ⇒
    Slug **byte-identisch** zu TZ 1-6; memory-bound-Familien hängen ihre Op an,
    damit z. B. Elementwise ``add`` und ``mul`` (gleicher Ausdruck!) getrennte
    Kernel-Dateien/Cache-Einträge bekommen (sonst still falsches gecachtes Artefakt).

    ``group_m`` (TZ 7.5) folgt exakt demselben bedingten Muster: nur bei
    ``swizzle=True`` **und** ``group_m != 8`` wird ``__sw`` zu ``__sw_g<N>``. Der
    Default 8 lässt den ``__sw``-Slug (und damit alle bestehenden ``*__sw.py``) byte-
    identisch; ein abweichender GROUP_M erzeugt einen eigenen Slug, damit nicht zwei
    GROUP_M-Werte dieselbe gecachte ``<slug>.py`` treffen (still falsches Artefakt).
    """
    d = config.to_dict() if isinstance(config, RunConfig) else dict(config)
    expr = (d.get("expr") or "").replace(" ", "").replace("->", "_to_").replace(",", "_")
    t = d.get("tile") or {}
    tile = f"TM{t.get('TM')}_TN{t.get('TN')}_TK{t.get('TK')}"
    slug = f"{expr}__{d.get('dtype')}-{d.get('acc_dtype')}__{tile}"
    op = d.get("op")
    if op:
        slug += f"__{op}"
    if d.get("swizzle"):
        # GROUP_M (TZ 7.5) nur BEDINGT: Default 8 ⇒ bares "__sw" (byte-identisch zu
        # TZ 1-6); abweichender Wert ⇒ "__sw_g<N>", damit verschiedene GROUP_M nicht
        # dieselbe gecachte kernels/<slug>.py treffen. Fehlt group_m (Altzeile) ⇒ 8.
        group_m = d.get("group_m")
        group_m = 8 if group_m is None else int(group_m)
        slug += "__sw" if group_m == 8 else f"__sw_g{group_m}"
    return slug


# ---------------------------------------------------------------------------
# Kernel-Quelltext persistieren
# ---------------------------------------------------------------------------
def kernel_file(slug: str, kernels_dir: Path = KERNELS_DIR) -> Path:
    """Absoluter Pfad der Kernel-Datei für einen Slug (ohne sie zu schreiben)."""
    return kernels_dir / f"{slug}.py"


def save_kernel(src: str, slug: str, kernels_dir: Path = KERNELS_DIR) -> Path:
    """Generierten Quelltext nach `kernels/<slug>.py` schreiben; Pfad zurück.

    **Atomar** (TZ 8, Cache-Härtung — Risiko ⑥): erst in eine Temp-Datei im
    SELBEN Verzeichnis schreiben, dann per ``os.replace`` (atomarer Single-Syscall
    auf gleichem Dateisystem) an ihren Platz ziehen — exakt das Muster von
    ``_atomic_rewrite``. So sieht ein paralleler Leser (oder ``inspect.getsourcelines``
    im cuTile-JIT, der die Datei liest) **nie** eine halb geschriebene ``<slug>.py``:
    entweder die alte oder die neue, vollständige Datei. Der geschriebene Byte-Inhalt
    ist identisch zum bisherigen ``write_text`` (utf-8, unveränderter Quelltext) ⇒
    keine Slug-/Cache-Drift, alle bestehenden ``kernels/*.py`` bleiben byte-identisch.

    Idempotent: gleicher Slug ⇒ gleicher Quelltext ⇒ folgenloses Überschreiben.
    """
    kernels_dir.mkdir(parents=True, exist_ok=True)
    path = kernel_file(slug, kernels_dir)
    fd, tmp = tempfile.mkstemp(dir=str(kernels_dir), prefix=".kernel-", suffix=".py.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        os.replace(tmp, path)   # atomar: nie eine halb geschriebene <slug>.py sichtbar
    except BaseException:
        try:
            os.unlink(tmp)      # Temp bei Fehler nicht liegen lassen
        except OSError:
            pass
        raise
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
    """Ein `RunResult` als **eine** JSON-Zeile an `results.jsonl` anhängen.

    `kernel_source` (GUI-Code-Anzeige) wird aus der zu schreibenden Kopie entfernt —
    der Quelltext liegt bereits als `kernels/<slug>.py` vor, gehört also nicht als
    Bloat in jede JSONL-Zeile. Das übergebene RunResult bleibt unverändert.
    """
    d = result.to_dict() if isinstance(result, RunResult) else dict(result)
    d.pop("kernel_source", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(d, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def read_results(path: Path = RESULTS_JSONL):
    """Alle Läufe als `pandas.DataFrame` (leer, falls Datei fehlt/leer). Für den
    Report; die History-Verwaltung nutzt `read_all`/`list_runs` (verlustfreie Objekte)."""
    import pandas as pd  # lazy: Schreiben soll auch ohne pandas funktionieren

    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


# ---------------------------------------------------------------------------
# Results-Store: Testlauf-Verwaltung (TZ 7.5-4) — verlustfreies Lesen + Gruppieren
# + atomare Mutatoren (rename/delete). Bewusst OHNE pandas (Round-Trip wäre lossy:
# Spalten-Union, NaN-Fills, dtype-Coercion) — Zeilen roh via json.loads.
# ---------------------------------------------------------------------------
def _read_rows(path: Path = RESULTS_JSONL) -> list[dict]:
    """Alle JSONL-Zeilen roh als dicts (fidelitätserhaltend). Leer, falls fehlt/leer."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _with_identity(rows: list[dict]) -> list[dict]:
    """Altzeilen ohne `run_id` bekommen beim LESEN einen synthetischen Fallback-Lauf
    (jede Altzeile = eigener Lauf — konservativ, KEINE unzuverlässige Timestamp-
    Gruppierung, KEIN Rewrite der git-Datei). Neue Zeilen (mit run_id) bleiben
    unberührt. Gibt (flache Kopien der) Zeilen zurück."""
    out = []
    for i, d in enumerate(rows):
        d = dict(d)
        if not d.get("run_id"):
            prov = d.get("provenance") or {}
            ts = prov.get("timestamp") or ""
            expr = (d.get("config") or {}).get("expr", "?")
            d["run_id"] = f"legacy-{i}-{ts}"
            d.setdefault("created_at", ts)
            if not d.get("run_name"):
                d["run_name"] = f"(Altlauf) {expr} · {ts}"
        out.append(d)
    return out


def read_all(path: Path = RESULTS_JSONL) -> list[RunResult]:
    """Alle Läufe als `RunResult`-Objekte (via `from_dict`; verlustfrei, NICHT pandas).
    Altzeilen ohne Lauf-Identität bekommen einen synthetischen Fallback-Lauf."""
    return [RunResult.from_dict(d) for d in _with_identity(_read_rows(path))]


def list_runs(path: Path = RESULTS_JSONL) -> list[dict]:
    """Vergangene Läufe nach `run_id` gruppiert (für die History-Auswahl), neueste
    zuerst (`created_at` absteigend). Je Lauf: run_id, run_name, created_at, Familie,
    Expr, n (Zeilen), n_ok (verifiziert)."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for d in _with_identity(_read_rows(path)):
        rid = d.get("run_id")
        g = groups.get(rid)
        if g is None:
            cfg = d.get("config") or {}
            g = groups[rid] = {
                "run_id": rid, "run_name": d.get("run_name") or rid,
                "created_at": d.get("created_at") or "",
                "family": cfg.get("family", "contraction"), "expr": cfg.get("expr", ""),
                "n": 0, "n_ok": 0,
            }
            order.append(rid)
        g["n"] += 1
        if d.get("status") == "ok":
            g["n_ok"] += 1
    # neueste zuerst; bei gleichem/leerem created_at die Einlese-Reihenfolge stabil halten
    return sorted(groups.values(),
                  key=lambda r: (r["created_at"], order.index(r["run_id"])), reverse=True)


def _atomic_rewrite(rows: list[dict], path: Path = RESULTS_JSONL) -> Path:
    """Alle Zeilen atomar zurückschreiben: Temp-Datei im SELBEN Verzeichnis →
    `os.replace` (atomarer Single-Syscall auf same-fs; parallele Leser sehen die
    alte ODER die neue vollständige Datei, nie eine halbe). Byte-Form identisch zu
    `append_result` (ensure_ascii=False, genau ein '\\n' je Zeile, kernel_source raus).
    Leere Liste ⇒ 0-Byte-Datei (NICHT gelöscht) → `read_results` bleibt konsistent leer."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".results-", suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for d in rows:
                d = dict(d)
                d.pop("kernel_source", None)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def rename_run(run_id: str, new_name: str, path: Path = RESULTS_JSONL) -> int:
    """Alle Zeilen eines Laufs (per `run_id`) umbenennen (`run_name` setzen) — atomar.
    Trifft NUR die Gruppe mit diesem `run_id`. :returns: Zahl geänderter Zeilen."""
    rows = _with_identity(_read_rows(path))
    n = 0
    for d in rows:
        if d.get("run_id") == run_id:
            d["run_name"] = new_name
            n += 1
    if n:
        _atomic_rewrite(rows, path)
    return n


def delete_run(run_id: str, path: Path = RESULTS_JSONL) -> int:
    """Alle Zeilen eines Laufs (per `run_id`) löschen — atomar. Löscht **NUR** JSONL-
    Zeilen, **NIE** die geteilte `kernels/<slug>.py`: mehrere Läufe teilen sich einen
    Slug, und `kernels/` ist der (gitignored, teils git-getrackte) Compile-Cache.
    :returns: Zahl gelöschter Zeilen."""
    rows = _with_identity(_read_rows(path))
    keep = [d for d in rows if d.get("run_id") != run_id]
    removed = len(rows) - len(keep)
    if removed:
        _atomic_rewrite(keep, path)
    return removed
