# Auftrag: TZ 4 — Tiling/Swizzle-Achse + volle Mess-Schicht + Baselines (cuTile Performance Lab)

Du arbeitest im Repo `/home/mla08/MLA/mla`. Wir bauen die Group-Specific Component
„**cuTile Performance Lab**" (interaktiver einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–3
sind fertig und verifiziert:** die headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → inputs → codegen → compile+Cache → Kalt-Lauf=compile_ms → verify(fp32) → benchmark=run_ms →
metrics → Store); die Dash-GUI fährt den Live-Loop als **Batch-Vergleich** (Größen + Zahlenformate
wählen → ein Klick → je Format ein `run()` unter **einem** GPU-Lock → KPIs/Verify/Code + zwei
Format-Charts). **fp16/bf16/tf32/fp8 (e4m3/e5m2) sind wählbar, jeder Kernel wird live gegen fp32
verifiziert, Genauigkeit (max/mean/rel) ist gemessen, und der Format-Tradeoff ist in zwei Charts
sichtbar.** Dein Auftrag ist **ausschließlich Teil-Ziel 4 (TZ 4)**.

TZ 4 macht die Pipeline **ehrlich mess-vollständig**: die Kachelung (Tile TM/TN/TK) und Swizzle werden
verstellbar, die Mess-Schicht wird von „nur Median-ms + TFLOP/s" auf den **vollen Metriksatz**
ausgebaut (Warmup, L2-Flush, Verteilung Median/min/p90/σ, erreichte GB/s, arithm. Intensität, %-vom-Peak,
Compile/Run getrennt, GPU-Zustand), und es kommen **Baselines** dazu (cuBLAS-Obergrenze + naive-cuTile-
Untergrenze, je optional zuschaltbar). Das ist die Scheibe, die aus dem Explorer ein **echtes
Performance-Werkzeug** macht: „Was bringt unser Tuning gegenüber cuBLAS, und wo liegt das Limit?"

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix, PLAN §2/§8). Charts = native Plotly (`dcc.Graph`). Keine Framework-Diskussion.
- **Mess-Entscheidungen sind fix** (PLAN §2/§3): eigene **CUDA-Events-Schicht** (`triton.do_bench` allein ist
  zu flach), voller Metriksatz, **cold-L2 als Default** (L2-Flush zwischen Iterationen), **Compile-Zeit
  getrennt** von der Laufzeit. Mess-Pipeline (maßgeblich):
  `verify(fp32) → compile (Zeit separat) → warmup → timed loop (CUDA-Events, L2-Flush) → Metriken →
  Baseline-Vergleich → GPU-Zustand → Results-Store → Charts`.
- **Roofline-Peaks sind gemessen/geklärt** (PLAN §5, Memory `gsc-hardware-dtype-facts`, Belege in
  `analysis/`). **NICHT neu herleiten** — nutzen: FP16/BF16 ≈ **213**, FP8 ≈ **214**, TF32 ≈ **53** TFLOP/s;
  FP32-plain niedrig; FP64 vernachlässigbar. Speicherbandbreite GB10 = **273 GB/s** (theoretisch; real
  ~70–85 %). Ridge-Point BF16 ≈ **780 FLOP/Byte** ⇒ GB10 ist für die meisten Shapes **stark memory-bound**.
- **Baselines-Story ist fix** (PLAN §2): **cuBLAS-Obergrenze** (über `torch.matmul`/cublas-Pfad) +
  **naive-cuTile-Untergrenze**, in der GUI je optional zuschaltbar. Die Erkenntnis: was bringt das Tuning.
- **verify-before-trust bleibt Gesetz:** jeder erzeugte Kernel wird gegen die torch-**fp32**-Referenz geprüft,
  *bevor* seine Zahlen in KPIs/Charts gehen — auch bei neuen Tiles/Swizzle.
- **Die Naht bleibt:** `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`.
  `RunResult.metrics/timing/provenance/accuracy` sind **offene, additive dicts** (`default_factory=dict`) —
  du ergänzt nur Schlüssel, **kein Schema-Umbau**. `RunConfig.tile`/`swizzle`/`baselines` **existieren
  bereits** (heute nur durchgereicht) — kein Schema-Umbau nötig.
- **Determinismus/Persistenz** (PLAN §2): `results/results.jsonl` (ein Objekt je Lauf) + `results/kernels/<slug>.py`;
  `config_slug` enthält bereits `tile` **und** `swizzle` (nicht `dim_sizes`, nicht `baselines`) ⇒ jede
  Tile/Swizzle-Kombi bekommt automatisch eine eigene Kernel-Datei + eigenen Cache-Treffer.
- **Erweitern statt neu bauen (WICHTIG):** TZ 4 baut **direkt** auf der bestehenden TZ-1/2/3-Implementierung
  auf — an genau definierten Nähten (Anker unten) — und übernimmt deren Muster: reine, headless-testbare
  Funktionen; **additive** dict-Erweiterung von `RunResult`; Control-IDs als Konstanten; Standalone-Test-Runner;
  aus `ALLOWED_ACC`/`COMBOS` **abgeleitete** Single-Source-of-Truth-Strukturen (kein Drift). Lies den
  aktuellen Code und *erweitere* ihn; erfinde nichts neu, was schon steht (kein Parallel-Pfad, keine
  Umbauten am `run()`-Ablauf).

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — besonders **§11 „TZ 4"** (maßgeblich: DoD, TODOs, „schaltet frei"),
   **§3** (Mess-Pipeline + voller Metriksatz), **§5** (Roofline-Peaks, Bandbreite, Ridge-Points, Hardware),
   **§6** (Risiken — v. a. ① mma-Orientierung, ⑤ nicht-teilbare Dims [Tile triggert das], ⑥ Compile-Cache),
   **§2/§8** (Dash, Background-Job, GPU-Lock, Versionen pinnen).
2. `project/project-development/analysis/RESULTS_gb10.md` + `analysis/dtype_analyse.py` — **gemessene
   Peaks/Bandbreite/dtype-Idiome** (Quelle der %-Peak-Zahlen; nutzt cold-L2/Event-Timing als Präzedenzfall).
3. `project/README.md` — die eine Naht GUI↔Core.
4. `project/tool_pipeline/schema.py` — `RunConfig.tile/swizzle/baselines` (schon da); `RunResult.metrics/
   timing/provenance/accuracy` (offene dicts). `ALLOWED_ACC`/`check_dtype_combo` (TZ 3, Acc-Regeln).
5. `project/tool_pipeline/measure/bench.py` — `benchmark(launch,A,B,C,warmup=10,iters=30) → {run_ms(=Median),
   iters,warmup}` (**nur Median, KEIN L2-Flush/Verteilung/GB/s** — Kopfkommentar markiert das als TZ 4; der
   Rückgabe-dict ist offen). `time_first_launch → float` (=compile_ms). **Hier wächst die Mess-Schicht.**
6. `project/tool_pipeline/measure/metrics.py` — `compute_metrics(M,N,K,run_ms) → {tflops}`; `gemm_flops=2·M·N·K`;
   `tflops(flops,ms)`. **GB/s + arithm. Intensität + %-Peak kommen hier dazu** (Peaks aus `hardware.py`).
7. `project/tool_pipeline/measure/provenance.py` (**Stub** → nvidia-smi/nvml Takt/Temp/Power) und
   `measure/baselines.py` (**Stub** → cuBLAS/torch + naive-cuTile) und `measure/hardware.py` (**existiert
   NICHT** → für die Peaks/Bandbreite neu anlegen). Die heute genutzte Provenienz liegt inline in
   `run.py::_provenance`.
8. `project/tool_pipeline/run.py` — Stufen 4/5/6. **Wichtig:** Stufe 6 reicht das `compute_metrics`-dict
   **ganz** weiter (GB/s/%-Peak fließen automatisch durch, sobald `compute_metrics` sie liefert), aber
   `timing` mappt nur `run_ms`+`iters` → **neue Verteilungs-Keys (min/p90/σ) musst du explizit in `timing`
   übernehmen**. Baselines brauchen **aktiven Code** (neue/erweiterte Stufe). Naht wirft NIE nach außen.
9. `project/tool_pipeline/store/store.py` — `config_slug` bettet `tile`+`swizzle` ein (nicht `baselines`);
   `append_result` persistiert alle dicts außer `kernel_source` ⇒ **neue TZ-4-Keys landen automatisch im
   JSONL, kein Store-Umbau. Baselines dürfen NICHT in den Slug** (Kernel-Quelltext ändert sich durch sie nicht).
10. `project/tool_pipeline/codegen/templates/contraction.py` — `build_gemm_module(tile,dtype,acc_dtype)`:
    **Tile TM/TN/TK sind bereits Parameter** (als Literale in den Quelltext gebacken); dtype-Pfade
    (`_INPUT_CAST` tf32-astype, fp8; `_ACC_DTYPE_MAP`) aus TZ 3. **Swizzle ist noch NICHT im Template** — hier
    wächst der Swizzle-Parameter. `emit.py` routet family=contraction hierher.
11. `project/tool_pipeline/app/` (post-TZ3, **KEINE Stubs mehr**): `components/controls.py` (Größen + Format-
    Checkliste `COMBOS`/`ID_DTYPES` + Info-Tooltip; `_fixed_config` zeigt **Tile/Swizzle read-only** →
    hier wandern sie in echte Controls; `configs_from_selection`/`validate_selection`); `components/kpis.py`
    (`render_kpis` = 3 Karten Durchsatz/Median/Compile; `render_context` = Provenienz-Zeile ohne GPU-Zustand;
    `_kpi_card`/`_fmt`); `components/charts.py` (`figure_throughput` Balken + `figure_accuracy_throughput`
    Scatter + `_points` liest nur tflops/rel + `_FORMAT_COLOR`/Palette + `save_png`; **KEINE Roofline**);
    `callbacks.py` (`execute_run(m,n,k,selection,progress)` Batch-Loop + `render_comparison` Tabs je Format
    + `register` mit `progress=`; Fork-Safety/Lock/_alert).
12. `project/tests/` — Standalone-Runner (kein pytest im venv): `test_codegen.py` (`_run_gemm(M,N,K,dtype,acc)`,
    `_assert_matches_fp32` [Toleranz aus `verify._TOLERANCES`], `_assert_orientation`), `test_app_controls.py`,
    `test_app_charts.py`, `test_app_execute.py`, `test_verify.py`.
13. Memory-Index `MEMORY.md` + `gsc-hardware-dtype-facts` (Peaks/Bandbreite/API-Gotchas), `gsc-codegen-risks`
    (silent-wrong-answer-Watchlist), `gsc-gui-tz2` (GUI-Invarianten), `gsc-project-plan`.

## Die bisherige Implementierung, auf der du direkt aufbaust (konkrete Anker)
**Erweitere diese vorhandenen Nähte; übernimm die Muster. Für Datei-Interna gilt: lies die Datei — hier
stehen die verifizierten externen Verträge, an die du andockst. Ist-Zustand ist POST-TZ3.**

**Core (`tool_pipeline/`) — verifizierte Signaturen/Muster:**
- Naht `run(config) -> RunResult` (`run.py`), 6 Stufen, gibt IMMER ein RunResult (Status {ok,verify_failed,
  compile_error,run_error}), wirft nie. **Stufe 6:** `b = benchmark(...); timing["run_ms"]=round(b["run_ms"],5);
  timing["bench_iters"]=b["iters"]; metrics = compute_metrics(M,N,K,b["run_ms"])` — das metrics-dict wird
  **ganz** zugewiesen (künftige Keys überleben ohne Edit). `_provenance(config)` ist inline (kein
  `provenance.py`-Aufruf heute).
- `measure/bench.py`: `benchmark(...) → {"run_ms"(Median), "iters", "warmup"}`; **Erweiterungspunkt:** L2-Flush
  zwischen Iterationen + additive Keys `min_ms/p90_ms/sigma_ms` (aus den `times_ms` via `statistics`).
- `measure/metrics.py`: `compute_metrics(M,N,K,run_ms) → {"tflops"}`; `gemm_flops=2·M·N·K`. **Erweiterungspunkt:**
  `gemm_bytes(M,N,K,dtype)` + Keys `gbps`, `arithmetic_intensity`, `percent_peak_flops`, `percent_peak_bw`
  (Peaks aus neuem `hardware.py`).
- `schema.py`: `RunConfig(family,expr,dim_sizes,dtype,acc_dtype,tile{TM,TN,TK},swizzle,baselines[])`;
  `RunResult.{accuracy,timing,metrics,provenance}` offene dicts. `ALLOWED_ACC`+`check_dtype_combo` (torch-frei,
  von run/verify/controls geteilt — Muster für weitere Regel-Tabellen).
- `codegen/templates/contraction.py`: `build_gemm_module(tile,dtype,acc_dtype)` — Tile schon parametrisiert,
  byte-stabiler emit, Orientierung `a=(TM,TK) b=(TK,TN) ct.mma(a,b,acc)` (NICHT umbauen). **Swizzle-Parameter
  kommt hier dazu.** `store.config_slug` = `<expr>__<dtype>-<acc_dtype>__TM.._TN.._TK..[__sw]`.
- `measure/{provenance,baselines}.py` = **Stubs** (Docstring-Zeile); `measure/hardware.py` = **fehlt**.

**GUI (`tool_pipeline/app/`) — Muster, die du spiegelst:**
- `controls.py`: **IDs als Konstanten** (`ID_M/ID_N/ID_K, ID_DTYPES, ID_DTYPE_INFO, ID_RUN, ID_CANCEL,
  ID_PROGRESS, ID_STATUS`); Format-Auswahl = **Checkliste über `COMBOS`** (aus `ALLOWED_ACC` abgeleitet);
  reine Helfer `validate_selection`, `configs_from_selection(m,n,k,sel)` (setzt **heute nur** dim_sizes/dtype/
  acc_dtype → **hier befüllst du tile/swizzle/baselines**). `_fixed_config()` zeigt **Tile + Swizzle read-only**
  → ⟶ **genau das Muster, mit dem TZ 3 dtype/acc aus `_fixed_config` in echte Controls geholt hat: neue
  ID-Konstanten (`ID_TILE_TM/TN/TK`, `ID_SWIZZLE`, `ID_BASELINES`), Bau-Helfer (`_tile_select`/`_baseline_select`
  analog `_dtype_select`), reine Validierer.** Info-Tooltip-Muster (`dbc.Tooltip` auf `ID_*_INFO`) vorhanden.
- `kpis.py`: `render_kpis` = 3 Karten via wiederverwendbarem `_kpi_card(label,value,unit,sub)`; `_fmt`
  (None→„—", bool ausgeschlossen). ⟶ **neue KPI-Karten (GB/s, %-Peak, arithm. Intensität) rein additiv aus
  `result.metrics` lesen; Verteilung (min/p90/σ) als `sub` der Median-Karte oder eigene Karte.** `render_context`
  = Provenienz-Zeile ⟶ **GPU-Zustand (Takt/Temp/Power) hier ergänzen** (neuer provenance-Key).
- `charts.py`: reine `figure_*(results, primary_key) → go.Figure`; `_points(results)` extrahiert je
  `status=="ok"`-Lauf tflops+rel ⟶ **additiv um gbps/percent_peak/baseline erweitern**; `_FORMAT_COLOR` (feste
  Palette, Import-Assert-Headroom). ⟶ **Baseline-Vergleich als zweite Balken-Serie in `figure_throughput`
  (cuBLAS/naive neben cuTile je Format). KEINE Roofline (TZ 5).** `save_png` (kaleido installiert) für
  Report/Inspektion + dataviz-„render & look".
- `callbacks.py`: `execute_run(m,n,k,selection,progress=None)` = reine, headless-testbare Batch-Kernlogik
  (validate → `configs_from_selection` → lazy `from tool_pipeline.run import run` → **ein** `FileLock` über den
  Batch → Loop `run()` + `set_progress("Format i/N…")` → `render_comparison`); `finally set_progress` reset.
  `render_comparison` = Status oben · zwei Charts gestapelt · Status-Badges je Format · **Tabs je Format**
  (KPIs/Verify/Code). `register` = `background=True`, `running=[…]`, `progress=[…]`, `cancel=[…]`.
  ⟶ **Tile/Swizzle/Baseline-States durch `execute_run`/`register` durchreichen; neue KPI-Karten/Chart-Serien
  in `render_comparison` komponieren; Fork-Safety (lazy run-Import), Lock- und `_alert`-Muster beibehalten.**

**Tests — Vorlagen (Standalone-Runner, kein pytest im venv; aus `project/`, venv-Python
`/home/mla08/MLA/mla/.venv/bin/python`):**
- `test_codegen.py`: `_run_gemm(M,N,K,dtype="fp16",acc="fp32")`, `_assert_matches_fp32(...)` (Toleranz aus
  `verify._TOLERANCES`), `_assert_orientation(dtype,acc)` (quadratisch, Doppelgänger > 10) ⟶ **pro Tile-
  Variante spiegeln** (verschiedene TM/TN/TK inkl. **nicht-teilbarer** Größen → Padding-Pfad; Orientierung je Tile).
- `test_app_{controls,charts,execute,render,infra}.py` + `test_verify.py`: `_text`/`_types`-Extraktoren,
  Store-Umleitung in temp-JSONL, Import-Failure-Regression, Anti-Drift-Tests ⟶ Vorlagen für die neuen
  Metrik-/Baseline-/Control-/Chart-Tests.

## TZ-4-Scope (eng halten!)
1. **Codegen-Swizzle** (`codegen/templates/contraction.py`): Swizzle als Parameter in `build_gemm_module`
   (Tile ist schon Parameter). Emit byte-stabil halten; Orientierung nicht anfassen. `emit`/`slug` fließen schon.
2. **Mess-Schicht** (`measure/bench.py` + `run.py` Stufe 6): **L2-Flush** zwischen Iterationen + **Verteilung**
   (`min_ms/p90_ms/sigma_ms`) additiv ins `benchmark`-dict; die neuen Keys **explizit** in `timing` übernehmen
   (nur `run_ms`/`iters` werden heute gemappt). Compile bleibt getrennt (`compile_ms`).
3. **Metriken + `hardware.py`** (`measure/metrics.py` + neue `measure/hardware.py`): `gemm_bytes(M,N,K,dtype)`
   → **erreichte GB/s**, **arithm. Intensität** (FLOP/Byte), **%-vom-Peak** (Compute & BW) mit den GB10-Peaks.
   `hardware.py` **minimal** anlegen (nur Peak-Tabelle je dtype + Bandbreite) — **Grenze zu TZ 5**: nur die
   Zahlen liefern, **kein Roofline-Chart**. `compute_metrics` liefert die Keys → `run.py` reicht sie automatisch weiter.
4. **Baselines** (`measure/baselines.py` + `run.py`): **cuBLAS/torch.matmul** (Obergrenze) + **naive-cuTile**
   (Untergrenze), je optional via `RunConfig.baselines`. Neue/erweiterte `run()`-Stufe, die je gewählter
   Baseline benchmarkt und Werte **additiv** in `metrics` (oder eigenem Key) ablegt. Baselines **nicht** in den Slug.
5. **Provenienz** (`measure/provenance.py` + `run._provenance`): GPU-Zustand (Takt/Temp/Power via nvml/pynvml
   oder `nvidia-smi`-Parse) additiv in `provenance`. Nur Reproduzierbarkeits-Metadaten, keine Kennzahlen.
   Graceful, falls nvml fehlt.
6. **GUI-Controls** (`app/components/controls.py`): **Tile-Control (TM/TN/TK)** + **Swizzle-Toggle** (aus
   `_fixed_config` in echte Controls, wie dtype in TZ 3) + **Baseline-Toggles**; `configs_from_selection`
   befüllt `tile/swizzle/baselines`; reine Validierer (`validate_tile`, …).
7. **GUI-Anzeige** (`app/components/kpis.py` + `charts.py` + `callbacks.py`): neue **KPI-Karten** (GB/s, %-Peak,
   arithm. Intensität, Verteilung), **GPU-Zustand** in `render_context`, **Baseline-Vergleich** als zweite
   Serie in `figure_throughput`. **KEINE Roofline.**
8. **Korrektheitsnetz** (`tests/`): Tile-Parametrisierung (verschiedene TM/TN/TK inkl. nicht-teilbar) +
   Orientierungs-Wächter je Tile; Metrik-Rechnungen headless mit bekannten Werten (gemm_bytes/GB/s/%-Peak/
   Verteilung); Baseline headless; Control-/Chart-Bau-Helfer headless; App-Smoke.

## Setup (erster Schritt)
Vermutlich **keine neuen Pakete** (torch/`torch.matmul`, cuTile, kaleido sind da). **Verifiziere headless
zuerst**, dass die Bausteine existieren, damit du nicht gegen fehlende API baust: `torch.matmul` läuft auf der
GPU (cuBLAS-Baseline); **nvml/pynvml** verfügbar für GPU-Zustand (sonst `nvidia-smi`-Parse; fehlt beides →
graceful Fallback + pinnen, PLAN §8); ein Tile ≠ 128 kompiliert (z. B. `TM=64`). Belege stehen in
`analysis/RESULTS_gb10.md`. Falls etwas fehlt: Versionen pinnen, nichts Bleeding-Edge.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Tile im Vergleich:** Ist Tile Teil des **Batch-Vergleichs** (mehrere Tiles nebeneinander vergleichen,
   analog dtypes) oder **ein fester Tile pro Vergleich** (Slider/Dropdown, ganze dtype-Auswahl nutzt dasselbe
   Tile)? Und die Control-Form: **Slider vs. Dropdown fester Zweierpotenzen** (validierbar: TK teilt K-Tail,
   TM/TN sinnvolle Werte). Kläre a vs. b.
2. **`hardware.py`-Umfang + %-Peak-Quelle:** minimales `hardware.py` in TZ 4 **vorziehen** (nur Peak-Tabelle je
   dtype + 273 GB/s), das TZ 5 dann für die Roofline weiternutzt — **empfohlen** —, ODER Peaks lokal in
   `metrics.py` halten bis TZ 5. Kläre + lege die exakten Peak-Werte/Keys fest.
3. **Baseline-Integration:** Baselines im **selben `run()`** mitmessen (neue Stufe, additiv in `metrics`) vs.
   separater Pfad; und die **Chart-Darstellung** (zweite Balken-Serie je Format vs. Referenzmarker). Und:
   GPU-Zustand **pro Lauf** oder **einmal pro Batch**? Kläre.
4. **Verteilungs-Keys + KPI-Darstellung:** welche `timing`-Keys (`run_ms` bleibt Median; + `min_ms`/`p90_ms`/
   `sigma_ms`) und wie in den KPI-Karten (Median-Karte mit `sub`=„min/p90/σ" vs. eigene Verteilungs-Karte).
   Kläre die genauen Schlüsselnamen (Schema bleibt additiv).

## Scope-Grenzen (was TZ 4 NICHT tut)
- **Keine Roofline** und **kein voller `hardware.py`-Ausbau als Chart** — das ist **TZ 5**. TZ 4 liefert nur die
  **Messwerte** (GB/s, arithm. Intensität, %-Peak); `hardware.py` minimal nur für die %-Peak-Zahlen.
- **Keine allgemeine Kontraktion / kein echter B1-Reshape / kein IR-/Optimizer-Port / keine Operanden-Liste**
  — **TZ 6**. Der Ausdruck bleibt fest `ik,kj->ij`.
- **Keine Elementwise-/Reduktions-Familien, kein Familien-Routing** — **TZ 7**. TZ 4 ist rein GEMM/compute-bound.
- **Naht `run()` und das RunResult/RunConfig-Schema NICHT umbauen** — nur Stufen erweitern (bench/metrics/
  baselines/provenance) und die dicts **additiv** befüllen; Achsen `dtype/acc/tile/swizzle` sind schon verdrahtet.
- Nicht-teilbare Dims: der bestehende **ZERO-Padding-Pfad** deckt das ab — nur **verifizieren** (Test), kein
  Masking-Umbau.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`; Haupt-Prozess bleibt
  **CUDA-frei** (lazy `run`-Import im Callback). Core bleibt headless testbar.
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla08/MLA/mla/.venv/bin/python` (auf der Shell ist
  `python` nicht im PATH; Shell-State persistiert nicht zwischen Bash-Aufrufen — venv-Python-Pfad direkt nutzen).
  Start: `python -m tool_pipeline` (GUI), `python -m tool_pipeline.cli` (headless).
- **Harte Regel: NIEMALS `git commit` / `git push`** in diesem Repo (Memory `never-git-commit-or-push`).
- Determinismus / **kleine Größen** (geteilte Maschine; `torch.manual_seed(0)`; App-Default 512³).
- **Silent-wrong-answer ist der Hauptfeind** (Memory `gsc-codegen-risks`): neue Tiles/Swizzle sind eine neue
  Quelle stiller Falschergebnisse (Orientierung, nicht-teilbare Dims). Deshalb **verify-before-trust je
  Tile + Orientierungs-Wächter im Test**, bevor eine Zahl in einen Chart/eine KPI geht.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel — wie in TZ 3), Verständnis **kurz** bestätigen.
2. TZ 4 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App in
   lauffähigem, prüfbarem Zustand — z. B. bench-Verteilung/L2-Flush → Metriken+`hardware.py` → Baselines →
   Provenienz → Codegen-Swizzle → Controls → Anzeige). Die Design-Entscheidungen oben **vorab** mit mir klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt bewirkt /
   was das Tool jetzt kann (so, als würdest du es jemandem ohne GPU-/Codegen-Wissen erklären). Dann auf
   **meine Validierung warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-4-Scope bleiben; Scope-Creep (Roofline/Multi-Input/Elementwise) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf wie in TZ 3 ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 5 (Roofline) — anschauen, vorbereiten und einen Session-Prompt + Planungs-MD
   erstellen** — genau nach *diesem* Muster: gründlich einlesen (Workflow), **PLAN §12 maßgeblich**, Anker aus
   dem *dann* aktuellen Post-TZ4-Ist-Zustand, MD unter `project/project-development/prompts/TZ5-*.md`, und **diese
   Arbeitsweise inkl. der zwei Zusätze (sehr einfache Erklärung nach jedem TODO + Planung des übernächsten TZ als
   letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash und teste sie **headless**: `gemm_bytes`/GB/s/arithm. Intensität/%-Peak mit
bekannten Zahlen; Verteilungs-Statistik (`min/p90/σ`) auf synthetischen Zeit-Listen; Baseline-Zahlen
(`torch.matmul`) headless; die Tile-/Baseline-Config-Bau- und Validier-Helfer als reine Funktionen; das
Chart-/KPI-Bauen als reine Funktion (RunResults → Figure/Komponenten). Zusätzlich **echte GPU-Läufe je Tile**
(verschiedene TM/TN/TK **inkl. nicht-teilbarer** Größen: jeder muss compilen + gegen fp32 verifizieren —
erweitere `tests/test_codegen.py` um Verify- **und** Orientierungs-Tests je Tile), **Baselines** real gegen
den cuTile-Kernel messen, **und** die App real starten und einen Vergleichs-Lauf durchklicken (KPIs inkl.
GB/s/%-Peak/Verteilung + Baseline-Serie + GPU-Zustand füllen sich; unzulässige Tiles sauber abgefangen).
Charts ggf. via `save_png` rendern und **ansehen** (dataviz „render & look").

## Definition of Done (TZ 4)
**Tile (TM/TN/TK) + Swizzle in der GUI verstellbar**; die **Mess-Schicht ist vollständig** (Warmup + L2-Flush +
Verteilung Median/min/p90/σ + erreichte GB/s + arithm. Intensität + %-vom-Peak [Compute & BW] + Compile/Run
getrennt + GPU-Zustand) und liegt im `RunResult`; **Baselines cuBLAS + naive-cuTile je optional zuschaltbar**
und im Vergleich sichtbar; **jeder** Kernel wird weiterhin **live gegen fp32 verifiziert** (auch neue Tiles);
Fehler-/Unzulässig-Stati sauber angezeigt (kein Crash); alle Tests grün + App-Smoke. Damit ist die
**vollständige, ehrliche Performance-Exploration für GEMM** freigeschaltet. **Zusätzlich:** nach jedem TODO
gab es eine sehr einfache Erklärung, und als letzter Schritt ist **TZ 5 vorbereitet** (Planungs-MD +
Session-Prompt erstellt).
