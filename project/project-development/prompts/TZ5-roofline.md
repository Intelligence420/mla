# Auftrag: TZ 5 — Roofline-Chart (cuTile Performance Lab)

Du arbeitest im Repo `/home/mla08/MLA/mla`. Wir bauen die Group-Specific Component
„**cuTile Performance Lab**" (interaktiver einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–4
sind fertig und verifiziert:** die headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → inputs → codegen → compile+Cache → Kalt-Lauf=compile_ms → verify(fp32) → benchmark →
Metriken → **Baselines** → **GPU-Zustand** → Store); die Dash-GUI fährt den Live-Loop als Batch-Vergleich
(Größen · **Tile TM/TN/TK** · **L2-Swizzle (aus/an/beide)** · Zahlenformate · **Baselines**) → je Config
ein `run()` unter einem GPU-Lock → KPIs/Verify/Code + zwei Format-Charts. **Die Mess-Schicht ist
vollständig** (Warmup, L2-Flush/cold-L2, Verteilung Median/min/p90/σ, erreichte GB/s, **arithm. Intensität**,
%-vom-Peak [Compute & BW], Compile/Run getrennt, GPU-Zustand), **Baselines cuBLAS + naive-cuTile** sind
zuschaltbar und im Chart sichtbar, jeder Kernel wird live gegen fp32 verifiziert. Dein Auftrag ist
**ausschließlich Teil-Ziel 5 (TZ 5): das Roofline-Chart.**

TZ 5 ist die **Auszahlung** der ehrlichen Metriken aus TZ 4: es macht die zentrale Erkenntnis des Tools
**sichtbar** — „ist diese Operation **compute-bound** oder **memory-bound**, und wie weit ist sie von der
Hardware-Grenze weg?". **Alle Zutaten liegen bereits vor** (arithm. Intensität + erreichte TFLOP/s je Lauf
in `RunResult.metrics`; Peaks + Bandbreite in `hardware.py`). TZ 5 baut daraus **nur den Chart** — keine neue
Messung, keine neue Metrik.

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly (`dcc.Graph`). Keine Framework-Diskussion.
- **Roofline-Peaks/Bandbreite sind gemessen/geklärt und liegen in `hardware.py`** (PLAN §5, Memory
  `gsc-hardware-dtype-facts`): FP16/BF16 ≈ **213**, FP8 ≈ **214**, TF32 ≈ **53** TFLOP/s; FP32/FP64 = `None`
  (kein Tensor-Core-Peak). Speicherbandbreite **273 GB/s** theoretisch (`MEM_BANDWIDTH_GBPS`), real ~70–85 %
  (`MEM_BANDWIDTH_REAL_FRACTION = (0.70, 0.85)`). **NICHT neu herleiten** — nutzen.
- **Ridge-Point-Mathematik (fix):** die Bandbreiten-Schräge ist `TFLOP/s = (273e9 · AI) / 1e12 = 0.273 · AI`
  (AI in FLOP/Byte). Ridge-Point je dtype = `Peak / 0.273` FLOP/Byte ⇒ BF16 ≈ **780**, FP8 ≈ 784, TF32 ≈ 194.
  GEMM bei 512³ hat AI = **128** FLOP/Byte ⇒ **links vom Ridge** ⇒ **memory-bound** (die zentrale Aussage).
- **Die Roofline-Daten liegen bereits im `RunResult.metrics`:** `arithmetic_intensity` (FLOP/Byte, = x-Achse),
  `tflops` (erreichte TFLOP/s, = y-Achse) — je verifiziertem Lauf. TZ 5 **zeichnet** sie nur.
- **verify-before-trust bleibt Gesetz:** Punkte landen nur aus `status == "ok"`-Läufen im Chart (bereits der
  `_points`-Vertrag) — verifizierte Zahlen zuerst, dann sichtbar.
- **Die Naht bleibt:** `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`;
  Charts sind **reine Funktionen** `figure_*(results, primary_key) -> go.Figure` (Dash-/torch-/cuda-frei,
  headless testbar); Haupt-Prozess bleibt CUDA-frei. **Kein Schema-/Naht-Umbau** — rein additiv.
- **Erweitern statt neu bauen:** TZ 5 baut **direkt** auf `charts.py`/`hardware.py`/`metrics.py` auf und
  übernimmt deren Muster (Palette `_FORMAT_COLOR`, `_style`, `_empty`, `_subtitle`, `_resolve_primary`,
  `save_png`; Anti-Drift; Standalone-Test-Runner). Lies den aktuellen Code und *erweitere* ihn.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§10 „TZ 5"** (Z.139–142: DoD/TODOs/„schaltet frei"),
   **§5** (Peaks/Bandbreite/Ridge-Points), **§3** (Metriksatz), **§6** (Risiken — hier gering, aber die
   verify-before-trust-Regel gilt), **§2** (Dash).
2. `project/project-development/analysis/RESULTS_gb10.md` — gemessene Peaks + Ridge (BF16 ≈ 780) als Beleg.
3. `project/tool_pipeline/hardware.py` — **schon in TZ 4 gefüllt**: `PEAK_TFLOPS` (fp16/bf16=213, fp8=214,
   tf32=53, fp32/fp64=`None`), `MEM_BANDWIDTH_GBPS=273.0`, `MEM_BANDWIDTH_REAL_FRACTION=(0.70,0.85)`,
   `DTYPE_BYTES`, `peak_tflops(dtype)->Optional[float]`, `dtype_bytes(dtype)->int`. **Das sind die
   y-Decken + die Steigung der Bandbreiten-Schräge.** (Ein `ridge_point`-Helfer fehlt noch — optional dazu.)
4. `project/tool_pipeline/measure/metrics.py` — `compute_metrics(M,N,K,run_ms,dtype,acc_dtype)` liefert
   `arithmetic_intensity` (FLOP/Byte, = Roofline-x) und `tflops` (= Roofline-y). **Deterministisch** (AI ist
   eine Funktion von Shape+dtype). Keine Änderung nötig — nur Datenquelle.
5. `project/tool_pipeline/app/components/charts.py` — **hier wächst der Roofline-Chart.** Muster zum Spiegeln:
   `figure_accuracy_throughput` (**log-Achse**, eine Spur je Format, primär hervorgehoben, `_style`), `_points`
   (**führt `arithmetic_intensity` NOCH NICHT** → additiv ergänzen), `_FORMAT_COLOR`/`_PALETTE` (8 Farben, am
   Limit), `_resolve_primary`, `_style`, `_empty`, `_subtitle`, `save_png`. **KEINE** neuen Format-Farben.
6. `project/tool_pipeline/app/callbacks.py` — `render_comparison`: `charts_stacked = html.Div([dcc.Graph(
   figure_throughput...), dcc.Graph(figure_accuracy_throughput...)])` ⟶ **Roofline als dritten `dcc.Graph`
   hier anhängen**; `_GRAPH_CONFIG` (PNG-Export). Fork-Safety/Lock unangetastet.
7. `project/tests/test_app_charts.py` — Standalone-Runner-Muster + Fixtures (`_ok`, `_mixed`) + Assertions auf
   `fig.data`/Achsen/Traces ⟶ Vorlage für die Roofline-Tests.
8. Memory-Index `MEMORY.md` + `gsc-hardware-dtype-facts` (Peaks/Bandbreite), `gsc-gui-tz2` (GUI-Invarianten),
   `gsc-project-plan`. Und der **dataviz-Skill** als Maßstab (log-log-Roofline ist ein bekanntes Muster).

## Die bisherige Implementierung, auf der du aufbaust (konkrete Anker, Ist-Zustand POST-TZ4)
**Core (`tool_pipeline/`):**
- `hardware.py` (torch-frei): `PEAK_TFLOPS: dict[str, Optional[float]]`, `MEM_BANDWIDTH_GBPS: float = 273.0`,
  `MEM_BANDWIDTH_REAL_FRACTION: tuple = (0.70, 0.85)`, `DTYPE_BYTES`, `peak_tflops`, `dtype_bytes`.
  ⟶ **Erweiterungspunkt (optional):** `ridge_point(dtype) = peak_tflops(dtype) / (MEM_BANDWIDTH_GBPS/1000)`
  (FLOP/Byte) — oder die Ridge-Rechnung im Chart halten. Minimal bleiben (keine Chart-Logik in hardware.py).
- `measure/metrics.py`: `compute_metrics(...) -> {"tflops", "gbps", "arithmetic_intensity",
  "percent_peak_flops", "percent_peak_bw"}` (bestätigt). AI + tflops = die Roofline-Koordinaten.

**GUI (`tool_pipeline/app/`):**
- `charts.py`: `_points(results)` liefert je ok-Lauf ein dict mit Keys
  `key,label,swizzle,tflops,rel_err,max_abs_err,gbps,percent_peak_flops,cublas,naive,color`
  ⟶ **`arithmetic_intensity` additiv ergänzen** (aus `metrics`). `figure_throughput`/`figure_accuracy_throughput`
  als Muster; `_FORMAT_COLOR` (feste Palette, Import-Assert am Limit); `_style(fig,title,xaxis_title)`;
  `_empty(msg)`; `_subtitle(fig,text)`; `_resolve_primary(pts,primary_key)`; `save_png(fig,path,...)`.
- `callbacks.py`: `render_comparison(results)` baut `charts_stacked` aus zwei `dcc.Graph`
  ⟶ **dritten `dcc.Graph(figure=charts.figure_roofline(results, primary_key), config=_GRAPH_CONFIG, ...)`**
  ergänzen. `_GRAPH_CONFIG` = Toolbar/PNG-Export. Naht/States unverändert.

## TZ-5-Scope (eng halten!)
1. **`_points`** (`charts.py`): additiv `arithmetic_intensity` mitführen (aus `metrics`), damit die Roofline
   dieselbe verify-before-trust-Punktquelle nutzt wie die anderen Charts. (Kein Zweit-Extraktor.)
2. **`figure_roofline(results, primary_key) -> go.Figure`** (`charts.py`), **log-log**:
   - **x** = arithm. Intensität (FLOP/Byte), **y** = erreichte TFLOP/s.
   - **dtype-Decken**: horizontale Linien bei den Peaks der **vorkommenden** dtypes (213/214/53), beschriftet.
   - **Bandbreiten-Schräge**: Gerade `y = 0.273 · x` (273 GB/s); optional dezentes **reales Band** (70–85 %).
   - **Messpunkte** aus echten Läufen (Farbe = Format via `_FORMAT_COLOR`, primär hervorgehoben, Swizzle-
     Varianten unterscheidbar), Hover mit AI/TFLOP/s/%-Peak.
   - **memory-/compute-bound** ablesbar (Ridge-Punkt: Schnittpunkt Schräge×Decke); dezenter `_subtitle`.
   - Leerfall → `_empty(...)`. `_style` + feste log-Dekaden-Ticks (wie der Scatter).
3. **`render_comparison`** (`callbacks.py`): Roofline als **dritten gestapelten Chart** einhängen.
4. **`hardware.py`** (optional, minimal): `ridge_point(dtype)`-Helfer, falls der Chart ihn sauber braucht.
5. **Tests** (`tests/test_app_charts.py`): headless — Decken vorhanden (bei den vorkommenden dtypes),
   Bandbreiten-Schräge (Steigung/Existenz), Punkte an `(AI, tflops)`, log-Achsen, Leerfall → `_empty`,
   Ridge-Rechnung mit bekannten Zahlen (BF16 ≈ 780); + `save_png` render&look.

## Setup (erster Schritt)
Vermutlich **keine neuen Pakete** (plotly/kaleido da; Roofline-Daten liegen in `metrics`, Peaks in `hardware`).
**Verifiziere headless zuerst**, dass die Bausteine da sind: `compute_metrics(...)` liefert
`arithmetic_intensity` (≠ None für in-scope dtypes); `hardware.peak_tflops(...)` liefert 213/214/53 bzw. None;
`figure_accuracy_throughput` als log-Achsen-Vorlage rendert. Falls etwas fehlt: pinnen, nichts Bleeding-Edge.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Decken-Auswahl:** nur die im aktuellen Ergebnis **vorkommenden** dtype-Peaks zeichnen (empfohlen — weniger
   Clutter) vs. alle in-scope. Und: fp16/bf16 teilen 213, fp8 214 (fast gleich) — als **eine** Linie „Tensor-Core-
   Peak ≈ 213–214" oder getrennt? tf32 (53) separat. Kläre.
2. **Bandbreiten-Darstellung:** nur theoretische 273-GB/s-Gerade (empfohlen als klare Referenz) **plus** dezentes
   reales Band (70–85 %), oder nur die Gerade? Kläre.
3. **Punkt-Identität:** Swizzle-Varianten im Roofline via **Marker-Symbol** (Kreis/Raute) unterscheiden — analog
   zum offen gebliebenen Scatter-Punkt (TZ-4-Review-Befund V3) — oder ignorieren? Primär-Hervorhebung wie im
   Scatter (größerer, umrandeter Marker)? Kläre.
4. **Achsenbereiche (log-log):** datengetrieben mit etwas Rand vs. feste Dekaden; wo/ob ein Ridge-Punkt-Marker.
   Kläre.
5. **Position:** dritter gestapelter Chart in `render_comparison` (empfohlen) vs. eigener Tab. Kläre.

## Scope-Grenzen (was TZ 5 NICHT tut)
- **Keine neue Messung / keine neue Metrik** — AI, erreichte TFLOP/s, Peaks, Bandbreite liegen alle vor; TZ 5
  **zeichnet** nur.
- **Keine allgemeine Kontraktion / kein B1-Reshape / keine Operanden-Liste** — **TZ 6**. Der Ausdruck bleibt
  fest `ik,kj->ij`; die Roofline zeigt vorerst nur **GEMM/compute-bound-Punkte** (die memory-bound-Seite wird
  mit den Elementwise/Reduktions-Familien in **TZ 7** reicher — hier nur die Basis legen).
- **Kein Umbau von Naht/Schema/`run()`/measure** — additiv `arithmetic_intensity` in `_points`, eine neue
  reine `figure_roofline`, eine Graph-Zeile in `render_comparison`.
- **Keine neuen Format-Farben** (Palette am Limit) — Decken/Schräge/Bänder in neutralem Ink, Punkte in den
  bestehenden Format-Farben.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`; Haupt-Prozess bleibt
  **CUDA-frei**. Charts bleiben reine, headless-testbare Funktionen.
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla08/MLA/mla/.venv/bin/python` (Shell-`python` nicht
  im PATH; Shell-State persistiert nicht zwischen Bash-Aufrufen — venv-Pfad direkt nutzen; die Bash-cwd bleibt
  aber erhalten). Start: `python -m tool_pipeline` (GUI), `python -m tool_pipeline.cli` (headless).
- **Harte Regel: NIEMALS `git commit` / `git push`** in diesem Repo (Memory `never-git-commit-or-push`).
- Determinismus / **kleine Größen** (geteilte Maschine; `torch.manual_seed(0)`; App-Default 512³).
- **verify-before-trust:** Roofline-Punkte nur aus `status == "ok"`-Läufen (kein still falscher Punkt).
- **dataviz-Skill** als Maßstab für die log-log-Roofline (Achsen, Ticks, Farb-/Kontrast-Regeln).

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 5 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App in
   lauffähigem, prüfbarem Zustand — z. B. `_points`-AI → `figure_roofline` (headless + render&look) →
   `render_comparison`-Einhängung → Tests). Die Design-Entscheidungen oben **vorab** mit mir klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt bewirkt /
   was das Tool jetzt kann (so, als würdest du es jemandem ohne GPU-/Roofline-Wissen erklären). Dann auf
   **meine Validierung warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-5-Scope bleiben; Scope-Creep (Multi-Input/Elementwise/neue Messung) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf wie in TZ 4 ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 6 (allgemeine 2-Operand-Kontraktion, echter B1-Reshape) — anschauen,
   vorbereiten und einen Session-Prompt + Planungs-MD erstellen** — genau nach *diesem* Muster: gründlich
   einlesen (Workflow), **PLAN §10/TZ 6 maßgeblich**, Anker aus dem *dann* aktuellen Post-TZ5-Ist-Zustand, MD
   unter `project/project-development/prompts/TZ6-*.md`, und **diese Arbeitsweise inkl. der zwei Zusätze (sehr
   einfache Erklärung nach jedem TODO + Planung des übernächsten TZ als letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash und teste sie **headless**: `figure_roofline` als reine Funktion (RunResults →
Figure) — Decken bei den vorkommenden Peaks vorhanden, Bandbreiten-Schräge existiert/Steigung stimmt, Punkte
liegen an `(arithmetic_intensity, tflops)`, Achsen sind log, Leerfall → `_empty`; die Ridge-Rechnung mit
bekannten Zahlen (BF16-Peak 213 → Ridge ≈ 780). Zusätzlich die App real starten und einen Vergleichs-Lauf
durchklicken: die Roofline erscheint als dritter Chart, die GEMM-Punkte liegen (bei 512³, AI=128) klar **links
vom Ridge** = memory-bound; Charts via `save_png` rendern und **ansehen** (dataviz „render & look").

## Definition of Done (TZ 5)
Ein **Roofline-Chart in der GUI**: log-log **arithm. Intensität (FLOP/Byte) vs. erreichte TFLOP/s**, mit
**dtype-Decken** (gemessene Peaks) und **Bandbreiten-Schräge** (273 GB/s, optional reales Band), **Punkte aus
echten Messungen** (Format-Farb-konsistent, primär hervorgehoben), sodass die **compute- vs. memory-bound-
Einordnung sichtbar** wird (GEMM bei üblichen Größen sitzt links vom Ridge = memory-bound); `hardware.py`
liefert Peaks/Bandbreite (schon da, ggf. `ridge_point`); Roofline hängt als dritter Chart in `render_comparison`;
Punkte nur aus verifizierten Läufen; alle Tests grün + App-Smoke. **Zusätzlich:** nach jedem TODO gab es eine
sehr einfache Erklärung, und als letzter Schritt ist **TZ 6 vorbereitet** (Planungs-MD + Session-Prompt erstellt).
