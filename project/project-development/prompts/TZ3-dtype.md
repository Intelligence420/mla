# Auftrag: TZ 3 — dtype-Achse + Genauigkeits-Story (cuTile Performance Lab)

Du arbeitest im Repo `/home/mla08/MLA/mla`. Wir bauen die Group-Specific Component
„**cuTile Performance Lab**" (interaktiver einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziel 1
(Backbone) und Teil-Ziel 2 (GUI-Live-Skelett) sind fertig und verifiziert** — die headless-Pipeline
läuft über die eine Naht `run(config) → RunResult`, und die Dash-GUI fährt den Live-Loop
(Größen eingeben → Background-Job auf der GPU → KPIs/Verify/Code). Dein Auftrag ist
**ausschließlich Teil-Ziel 3 (TZ 3, „dtype-Achse + Genauigkeits-Story")**.

TZ 3 ist die erste *tiefe* Scheibe, die wieder **Core UND GUI** berührt (nicht nur `app/`):
die Kontraktion soll in allen in-scope Zahlenformaten laufen, jeder Kernel gegen fp32
verifiziert, und der **Format-Tradeoff** (Durchsatz vs. Genauigkeit) in **zwei Charts** sichtbar
werden. Das ist die Headline-Erkenntnis des Tools — sie muss *echt* (gemessen, verifiziert) sein.

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix, PLAN §2/§8). Charts = **native Plotly** (`dcc.Graph`).
  Keine Framework-Diskussion.
- **dtype-Matrix ist empirisch geklärt** (PLAN §5, Memory `gsc-hardware-dtype-facts`, Belege in
  `project/project-development/analysis/`). **NICHT neu herleiten, welche Formate gehen** — nutze die
  bewiesenen Idiome. In-scope (compute→acc):
  - `fp16 → fp32` ✅ (Anker, in TZ 1 fertig) · `bf16 → fp32` ✅ (**acc=fp32 Pflicht**)
  - `tf32 → fp32` ✅ (**kein mma-Flag**: `ct.astype(tile, ct.tfloat32)` VOR `ct.mma`)
  - `fp8 e4m3 → fp16|fp32` ✅ (host-seitig `.to(torch.float8_e4m3fn)`; fp16-acc am schnellsten)
  - `fp8 e5m2 → fp32` ✅ (host-seitig `.to(torch.float8_e5m2)`)
  - Anker/Diagnose optional: `fp32 → fp32` (kein TC), `fp64 → fp64`. **Exkludiert: fp4/int4**
    (keine cuTile-Symbole in diesem Build).
  - **Acc-Regeln erzwingen:** bf16 & tf32 → **fp32**; fp16 & fp8 → **fp16 oder fp32**.
- **verify-before-trust bleibt Gesetz:** jeder generierte dtype-Kernel wird gegen die torch-**fp32**-
  Referenz geprüft, *bevor* Zahlen/Charts angezeigt werden (das ist zugleich das Genauigkeits-Panel).
- **Die Naht bleibt:** `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`.
  `dtype`/`acc_dtype` sind bereits **RunConfig-Felder** — das Schema muss dafür nicht umgebaut werden.
- **Erweitern statt neu bauen (WICHTIG):** TZ 3 baut **direkt auf der bestehenden TZ-1/TZ-2-Implementierung
  auf** — an genau definierten Nähten (konkrete Anker + echte Signaturen unten) — und **übernimmt deren
  Muster**: reine, headless-testbare Funktionen; additive dict-Erweiterung von `RunResult`; Control-IDs als
  Konstanten; Standalone-Test-Runner. Lies den aktuellen Code und *erweitere* ihn; erfinde nichts neu, was
  schon steht (kein Parallel-Pfad, keine Umbauten am `run()`-Ablauf).

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — besonders **§10 TZ 3** (maßgeblich), **§5** (dtype-Matrix,
   Acc-Regeln, Roofline-Peaks: FP16/BF16≈213, FP8≈214, TF32≈53 TFLOP/s), **§3** (Codegen C1+B1),
   **§6** (Codegen-Risiken — v. a. ⑦ dtype-Cast-Pfade und ① mma-Orientierung).
2. `project/project-development/analysis/RESULTS_gb10.md` + `analysis/dtype_analyse.py` — **die
   bewiesenen, lauffähigen cuTile-Idiome je dtype** (tf32-`astype`, fp8-Host-Cast, Acc, Toleranzen).
   **Das ist die Quelle der Wahrheit für die Codegen-dtype-Pfade** — spiegeln, nicht raten.
3. `project/README.md` — die eine Naht GUI↔Core.
4. `project/tool_pipeline/schema.py` — `RunConfig.dtype`/`acc_dtype` (schon da); `RunResult.accuracy`
   (max_abs_err/passed/atol/rtol — hier kommen mean/rel additiv dazu) und `.metrics` (tflops → Balken).
5. `project/tool_pipeline/codegen/templates/contraction.py` — `build_gemm_module(tile, dtype, acc_dtype)`
   (kann heute nur fp16→fp32, lehnt unbekannte acc mit ValueError ab). **Hier wachsen die dtype-Pfade.**
6. `project/tool_pipeline/codegen/emit.py` — Routing (family=contraction → build_gemm_module).
7. `project/tool_pipeline/run.py` — `_build_inputs` (hardcodet fp16, lehnt andere ab → muss erweitert +
   Acc-Regeln erzwingen) und `_TORCH_DTYPE`; die Status-Zweige.
8. `project/tool_pipeline/measure/verify.py` — heutiges verify (fp32-Ref + max_abs_err); **hier kommen
   mean/rel + dtype-abhängige Toleranzen dazu**. `measure/metrics.py` — tflops (Balken-Quelle).
9. `project/tool_pipeline/store/store.py` — `config_slug` enthält bereits `dtype`/`acc_dtype` ⇒ jeder
   dtype bekommt automatisch eine eigene `kernels/<slug>.py` + eigenen Cache-Treffer (**kein Store-Umbau**).
10. Die `app/`-Dateien: `components/controls.py` (dtype-/acc-Controls ergänzen), `components/charts.py`
    (Stub: Balken + Scatter + Roofline — **Roofline ist TZ 5, jetzt nur Balken + Scatter**),
    `callbacks.py` (`execute_run`/`register` — der Vergleich über dtypes), `layout.py` (Charts im Main
    einhängen), `components/kpis.py` (bestehende Render-Funktionen).
11. `project/tests/test_codegen.py` (Korrektheitsnetz — **pro dtype erweitern**), `tests/test_app_*.py`
    (GUI-Test-Muster: Standalone-Runner, reine Logik headless + echter GPU-Lauf).
12. Memory-Index `MEMORY.md` + `gsc-hardware-dtype-facts` (dtype-Matrix + API-Gotchas),
    `gsc-codegen-risks` (silent-wrong-answer-Watchlist), `gsc-gui-tz2` (GUI-Invarianten:
    Fork-Safety/lazy-run-Import, GPU-Lock, execute_run-Naht, running=-Progress), `gsc-project-plan`.

## Die bisherige Implementierung, auf der du direkt aufbaust (konkrete Anker)
**Erweitere diese vorhandenen Nähte; übernimm die Muster. Für Datei-Interna (verify/contraction) gilt:
lies die Datei — hier stehen die verifizierten *externen* Verträge, an die du andockst.**

**Core (`tool_pipeline/`) — verifizierte Signaturen/Muster:**
- Naht `run(config) -> RunResult` (`run.py`), Ablauf: `parse → to_canonical(M,N,K) → _build_inputs →
  load_kernel(+Cache) → time_first_launch (=compile_ms) → verify(fp32) → benchmark (=run_ms) →
  compute_metrics`. Gibt IMMER ein RunResult (Status {ok,verify_failed,compile_error,run_error}).
- `RunConfig`: `family, expr, dim_sizes, dtype, acc_dtype, tile, swizzle, baselines` — **dtype/acc_dtype
  existieren schon** und werden durchgereicht (kein Schema-Umbau nötig).
- `RunResult`: `status`; `accuracy{max_abs_err,passed,atol,rtol}`; `timing{compile_ms,run_ms,bench_iters}`;
  `metrics{tflops}`; `provenance{gpu,dtype,acc_dtype,sizes{M,N,K},timestamp}`; `kernel_path`;
  `kernel_source`; `error`. **accuracy/metrics sind dicts → rein additiv erweiterbar** (genau wie
  `kernel_source` in TZ 2 additiv dazukam → additives Muster übernehmen).
- `run._TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}` und `run._build_inputs(config,M,N,K)`:
  deterministisch (`torch.manual_seed(0)`) `A=(M,K)`, `B=(K,N)`, `C=(M,N)`; **heute NotImplementedError
  für dtype≠fp16 bzw. unbekannten acc** — GENAU HIER die neuen dtypes (bf16/tf32/fp8) + die Acc-Regeln einhängen.
- `codegen.emit.emit(config)` routet `family=="contraction"` → `templates.contraction.build_gemm_module(tile, dtype, acc_dtype)`
  (heute nur fp16→fp32; wirft ValueError bei unbekanntem acc — von `test_codegen` abgedeckt). Der erzeugte
  Kernel nutzt bereits `acc = ct.full((TM,TN), 0, dtype=ct.float32)`, `ct.mma(a,b,acc)` und
  `ct.store(.., tile=ct.astype(acc, C.dtype))` — die **`ct.astype`-Stelle ist der Präzedenzfall** für die
  tf32/fp8-Cast-Pfade. **`build_gemm_module` ist die zentrale Datei, die wächst** (f-String-Struktur dort lesen).
- `measure.verify.verify(C, A, B, config) -> {max_abs_err, passed, atol, rtol}` (gegen torch-fp32-Referenz)
  → **hier mean/rel + dtype-abhängige Toleranztabelle ergänzen**. `measure.metrics.compute_metrics(M,N,K,run_ms)
  -> {tflops, …}` → **tflops ist die Balken-Quelle** (Datei-Interna lesen).
- `store.config_slug` bildet `<expr>__<dtype>-<acc_dtype>__TM..TN..TK..[__sw]` ⇒ **jeder dtype bekommt
  automatisch eine eigene `kernels/<slug>.py` + eigenen Cache — kein Store-Umbau.**

**GUI (`tool_pipeline/app/`) — Muster, die du spiegelst:**
- `components/controls.py`: **IDs als Konstanten** (`ID_M/ID_N/ID_K, ID_RUN, ID_CANCEL, ID_PROGRESS,
  ID_STATUS`); reine Helfer `validate_sizes(m,n,k)->str|None` und `config_from_controls(m,n,k)->RunConfig`
  (setzt heute nur `dim_sizes`); `build_controls()` mit read-only `_fixed_config()`; `_DEFAULT = RunConfig()`.
  ⟶ **Neue dtype/acc-Controls = neue ID-Konstanten + reine Bau-/Prüf-Helfer im gleichen Stil**; dtype/acc
  wandern aus `_fixed_config` (read-only) in echte Controls.
- `callbacks.py`: `execute_run(m,n,k) -> list` ist die **reine, headless-testbare** Kernlogik
  (validate → config → `try{` lazy `from tool_pipeline.run import run`; GPU-`FileLock(_GPU_LOCK)`
  (`_LOCK_TIMEOUT=60`); `run()`; `render_result()` `}` `except Timeout/Exception → _alert`). `render_result(result)`
  komponiert `kpis.render_status/context/kpis/verify` + `code_panel.render_code_panel`. `register(app)` =
  `@app.callback(Output("main","children"), …, background=True, running=[…], cancel=[…])`.
  ⟶ **Der dtype-Vergleich erweitert `execute_run` (Signatur) + `render_result` (Charts dazu); Fork-Safety-,
  Lock- und `_alert`-Fehlerpfad-Muster unbedingt beibehalten** (Audit-Fixes A–F nicht rückgängig machen).
- `components/kpis.py`: reine `render_*` + `_fmt`; `components/code_panel.py`: `render_code_panel(source, kernel_path)`.
  ⟶ **`components/charts.py` (Stub) analog als reine Funktion(en)** `figure(results) -> plotly.Figure`
  (bzw. `dcc.Graph`) bauen und in `layout.build_layout()`/über den Callback im `main`-Bereich einhängen.
- Start: `python -m tool_pipeline` → `app.app.create_app()` (DiskcacheManager, `register(app)`) / `main()`
  (`_host()`/`_port()`, robustes Env-Parsing). Titel = „**cuTile Performance Lab**".

**Tests — Vorlagen (Standalone-Runner, kein pytest im venv):**
- `tests/test_codegen.py`: `_run_gemm(M,N,K)`, `test_gemm_correct_across_sizes` (glatte + ragged Größen),
  `test_gemm_computes_AB_not_transpose` (Orientierungs-Wächter) ⟶ **pro dtype spiegeln** (Verify gegen fp32
  mit dtype-Toleranz + Orientierungs-Test).
- `tests/test_app_{controls,render,execute,infra}.py`: `_text`-Extraktor (Komponentenbaum→Text),
  Store-Umleitung in eine temp-JSONL (kein Pollution), Import-Failure-Regression ⟶ als Vorlage für die
  neuen dtype-/Chart-Tests nutzen.

## TZ-3-Scope (eng halten!)
1. **Codegen** (`codegen/templates/contraction.py`): dtype-Pfade ergänzen — tf32 via `ct.astype(.., ct.tfloat32)`
   vor `ct.mma`; fp8 (e4m3/e5m2) mit Host-Cast + passendem Akku; bf16 nativ (acc fp32); Akku-dtype im
   Kernel (`ct.float32`/`ct.float16`). `run._build_inputs` + `_TORCH_DTYPE` entsprechend erweitern und
   die **Acc-Regeln hart erzwingen** (unzulässige Kombi → klarer Fehler-Status, nie still falsch).
2. **Genauigkeit** (`measure/verify.py`): zusätzlich **mean_abs_err** und **rel_err** (vs. fp32-Referenz)
   messen; **dtype-abhängige Toleranzen** (Tabelle) statt einer festen. `RunResult.accuracy` additiv um
   die neuen Schlüssel erweitern.
3. **GUI-Controls** (`app/components/controls.py`): **dtype-Dropdown** (in-scope-Formate) + Acc-Auswahl,
   mit **erzwungenen Acc-Regeln** (bf16/tf32→fp32 fest; fp16/fp8→fp16|fp32). Reine Helfer (Dash-frei,
   headless testbar) fürs Bauen/Prüfen der dtype-Config.
4. **Charts** (`app/components/charts.py`): **zwei** Plotly-Charts — **Durchsatz je Format (Balken)** und
   **Genauigkeit ↔ Durchsatz (Scatter)**, aktives/primäres Format hervorgehoben, Punkte aus **echten,
   verifizierten** Läufen. In `layout.py`/`callbacks.py` einhängen.
5. **Korrektheitsnetz** (`tests/test_codegen.py`): **pro dtype** ein Verify-gegen-fp32-Test **und** ein
   Orientierungs-Wächter (A@B ≠ Doppelgänger), mit den dtype-passenden Toleranzen.

## Setup (erster Schritt)
Vermutlich **keine neuen Pakete** — `plotly`/`dash` sind installiert, `torch` kann fp8-Storage-dtypes,
cuTile hat `tfloat32`/`float8`-Symbole (fp4/int4 nicht). **Verifiziere headless als Erstes**, dass die
Bausteine da sind (torch `float8_e4m3fn`/`float8_e5m2`/`bfloat16`; cuTile-dtype-Symbole), damit du nicht
gegen fehlende API baust — die Belege stehen in `analysis/RESULTS_gb10.md`. Falls doch etwas fehlt:
Versionen pinnen (PLAN §8), nichts Bleeding-Edge.

## Drei Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Vergleichs-Mess-Modell (womit werden die Charts befüllt?):**
   **(a) Batch-Vergleich (empfohlen):** Multi-Select der dtypes → **ein** „Vergleich"-Klick misst alle
   ausgewählten Formate nacheinander (je ein `run()` **unter einem** GPU-Lock) und füllt beide Charts in
   einem Rutsch → liefert den Format-Tradeoff als eine Aktion; ermöglicht **echten** Per-dtype-Progress
   (`set_progress("dtype 2/5…")`). **(b) Akkumulieren:** ein dtype je Run, Punkte sammeln sich über Läufe
   (`dcc.Store`). Kläre a vs. b (empfohlen: a).
2. **Genauigkeits-Schema:** additive Schlüssel in `RunResult.accuracy` (z. B. `mean_abs_err`, `rel_err`)
   + dtype-Toleranztabelle in `verify.py`. Kläre die genauen Schlüsselnamen und wie `rel_err` definiert
   ist (z. B. relativ zur fp32-Referenznorm) — Schema bleibt additiv (wie `kernel_source` in TZ 2).
3. **Acc-Regeln im UI:** dtype-Dropdown steuert die erlaubte Akku-Auswahl (bf16/tf32 → acc=fp32 fest/
   disabled; fp16/fp8 → fp16|fp32 wählbar), **plus** Core-Validierung als zweite Verteidigungslinie
   (defense-in-depth: unzulässige Kombi → sauberer Fehler-Status statt still falsch). Kläre die
   UI-Darstellung (acc-Feld automatisch gesetzt vs. sichtbar-disabled) — und ob die **KPI-Karten** bei
   einem Mehr-dtype-Vergleich das primäre/aktive Format zeigen oder als kleine Tabelle.

## Scope-Grenzen (was TZ 3 NICHT tut)
- **Kein** Tiling/Swizzle-Control und **keine** volle Mess-Schicht (Median/min/p90/σ, GB/s, arithm.
  Intensität, %-Peak, GPU-Zustand) — das ist **TZ 4**. Tile bleibt fest (128/128/64), `run_ms` = Median wie in TZ 1.
- **Keine** Baselines (cuBLAS/naive) — **TZ 4**.
- **Keine** Roofline — **TZ 5** (auch wenn `charts.py` sie im Docstring erwähnt: jetzt nur Balken + Scatter).
- **Keine** allgemeine Operanden-Liste / Multi-Input / beliebige Kontraktion — **TZ 6**. Der Ausdruck
  bleibt fest `ik,kj->ij`.
- **Keine** Elementwise/Reduktion-Familien — **TZ 7**.
- `run.py`-Ablauf und die Naht **nicht** umbauen — nur die Stufen erweitern (Codegen-dtype-Pfade,
  `_build_inputs`, verify) und `accuracy`/`metrics` additiv befüllen.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert **ausschließlich** `tool_pipeline.run` + `tool_pipeline.schema`; Haupt-Prozess bleibt
  CUDA-frei (lazy `run`-Import im Callback). Core bleibt headless testbar.
- Ausführen aus `project/` (`cd project && python -m tool_pipeline`; headless `python -m tool_pipeline.cli`).
- **Harte Regel: NIEMALS `git commit` / `git push`** in diesem Repo (Memory `never-git-commit-or-push`).
- Determinismus / **kleine Größen** (geteilte Maschine; deterministische Eingaben, `torch.manual_seed`).
- **Silent-wrong-answer ist der Hauptfeind** (Memory `gsc-codegen-risks`): jeder dtype-Pfad ist eine neue
  Quelle stiller Falschergebnisse (Cast-Reihenfolge, Akku-dtype, mma-Orientierung). Deshalb: **verify-
  before-trust je dtype + Orientierungs-Wächter im Test**, bevor irgendeine Zahl in einen Chart geht.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen, Verständnis **kurz** bestätigen.
2. TZ 3 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App in
   lauffähigem, prüfbarem Zustand — z. B. dtype für dtype freischalten, dann Verify-Metriken, dann Controls,
   dann Charts). Die drei Design-Entscheidungen oben **vorab** mit mir klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten, zeigen was du getan hast + **wie verifiziert**, und auf
   **meine Validierung warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-3-Scope bleiben; Scope-Creep (Tile/Baselines/Roofline) widerstehen.

## Verifikation (Hinweis)
Trenne testbare Logik von Dash und teste sie **headless**: dtype/acc-Config-Bau + Acc-Regel-Prüfung als
reine Funktionen; die dtype-Toleranztabelle + verify-Metriken headless; das Chart-Bauen als reine
Funktion (RunResults → Plotly-Figure). Zusätzlich **echte GPU-Läufe pro dtype** (jeder muss compilen +
gegen fp32 verifizieren — erweitere `tests/test_codegen.py` um je einen Verify- und Orientierungs-Test
pro dtype mit passender Toleranz) **und** die App real starten und einen Vergleichs-Lauf durchklicken/-smoken
(beide Charts füllen sich mit verifizierten Punkten; unzulässige Acc-Kombi wird sauber abgefangen).

## Definition of Done (TZ 3)
Alle in-scope Formate (bf16/tf32/fp8 e4m3/e5m2, plus fp16-Anker; Acc-Regeln erzwungen) sind in der GUI
wählbar; **jeder** erzeugte Kernel wird **live gegen fp32 verifiziert** (dtype-passende Toleranzen), *bevor*
Zahlen angezeigt werden; Genauigkeit (max/mean/rel vs. fp32) wird gemessen und liegt im `RunResult`; und
**zwei Charts** zeigen live den Format-Tradeoff — **Durchsatz je Format (Balken)** und
**Genauigkeit ↔ Durchsatz (Scatter)** aus echten Messungen. Fehler-/Unzulässig-Stati werden sauber
angezeigt (kein Crash). Damit ist die **Headline-Erkenntnis** (Format-Tradeoff) real und sichtbar.
