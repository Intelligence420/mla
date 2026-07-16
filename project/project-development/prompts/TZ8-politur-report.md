# Auftrag: TZ 8 — Politur, Robustheit & Report

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen).
Wir bauen die Group-Specific Component „**cuTile Performance Lab**" (interaktiver
einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–7 UND das Zwischen-Teil-Ziel TZ 7.5 sind
fertig und verifiziert:** die headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → **Familien-Router** → reshape/B1 (nur Kontraktion) → emit → compile+Cache →
Kalt-Lauf=compile_ms → verify(fp32) → benchmark → Metriken → Baselines → GPU-Zustand → Store);
die Dash-GUI fährt den Live-Loop als Batch-Vergleich (**Familien-Auswahl** contraction/
elementwise/reduction · Ausdruck via Presets/Freitext + Größen je Index — **inkl. n-är**
(`ij,jk,kl->il`) · Elementwise-Op · **mehrere Tile-Zeilen (+/−) TM/TN/TK · L2-Swizzle-
Konfigurationen als Mehrfachauswahl mit einstellbarem GROUP_M** · Zahlenformate · Baselines) →
**Kreuzprodukt Format × Tile × Swizzle-Konfig**, je Config ein `run()` unter EINEM GPU-Lock →
KPIs/Verify/Code + **drei** Charts (Durchsatz · Genauigkeit↔Durchsatz · Roofline, bei mehreren
Varianten **serien-disambiguiert**). Ein **ausklappbares History-Panel** erlaubt, vergangene
Läufe **anzusehen · zu vergleichen · umzubenennen · zu löschen** (GPU-frei aus dem mutablen Store).
**Alle drei Operations-Familien** laufen end-to-end: Kontraktion (Tensor-Core, compute-bound) —
inkl. **n-ärer Ketten-Kontraktion** (paarweise GEMM-Zerlegung → EIN aggregierter Roofline-Punkt) —
sowie **Elementwise (add/mul/copy)** und **Reduktion (sum über beliebige Achsen)** als
memory-bound Familien — alle **verifiziert gegen die op-/family-abhängige fp32-Referenz** und
mit family-korrekten Metriken (GB/s primär); die Roofline zeigt den Kontrast **memory- vs
compute-bound** sichtbar (memory-bound weit links, AI 0,1–0,5; Kontraktion rechts, AI ~128).

**ABER:** das Deliverable ist noch nicht **poliert und dokumentiert**. Randfälle (nicht-teilbare
Dimensionen) sind zwar per `padding_mode=ZERO`+Store-Clipping abgedeckt, aber nicht systematisch
belegt; der Compile-Cache ist funktional, aber nicht gehärtet (korrupte/parallele Schreibzugriffe);
das Theming ist schlicht (eine `theme.css`), aber nicht durchgängig professionell; die
Fehlerzustände sind vorhanden, aber nicht überall benutzerfreundlich; und der **Sphinx-Report**
(`chapters/group_specific_component/`) ist ein Gerüst ohne die echten Ergebnis-Figuren/Tabellen aus
`results.jsonl`, `projektplan.rst` ist noch nicht aus dem aktuellen `PLAN.md` synchronisiert, und
`cli.py` kann nur **einen** GEMM-Lauf (keine Batch-Sweeps für Report-Plots, keine Familien). Dein
Auftrag ist **ausschließlich Teil-Ziel 8 (TZ 8): Politur, Robustheit & Report** — das erfüllt die
Definition of Done des gesamten Projekts (PLAN §10/TZ 8: „das fertige, dokumentierte Deliverable").

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1** (f-String-Templates → `@ct.kernel`; ein Modul je Familie: `contraction.py`/
  `elementwise.py`/`reduction.py`). **Kein neues Codegen-Paradigma.**
- **Die drei Operations-Familien sind vollständig** (contraction/elementwise/reduction). TZ 8 fügt
  **keine** neuen Familien/Ops hinzu — es poliert, härtet und dokumentiert das Bestehende.
- **Die eine Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie
  Helfer); Haupt-Prozess CUDA-frei; Charts reine, headless-testbare Funktionen. **Kein Naht-Umbau.**
- **Results-Store-Format = JSON Lines** (`project/results/results.jsonl`) + Kernel als
  `results/kernels/<slug>.py` (lesbarer Config-Slug inkl. Op **und — bedingt — GROUP_M**:
  `__sw_g<N>` nur bei `swizzle & group_m!=8`, sonst byte-identisch). **Kein Format-Umbau** — der
  Report liest daraus. **Der Store ist seit TZ 7.5 mutabel** (`rename_run`/`delete_run` via atomarem
  Rewrite `_atomic_rewrite` = temp im selben Verzeichnis → `os.replace`) und trägt eine additive
  Lauf-Identität (`run_id`/`run_name`/`created_at`); Altzeilen ohne diese Felder bleiben ladbar
  (synthetischer Fallback-Lauf beim Lesen). **Der git-getrackte `results.jsonl` wird durch Tests NIE
  verschmutzt** (`$SP`/Monkeypatch — auch für die neuen Mutatoren).
- **verify-before-trust bleibt Gesetz.** Report-Figuren nur aus `status=="ok"`.
- **dataviz-Skill** als Maßstab für alle (neuen) Figuren; memory-bound klar links auf der Roofline.
- **Erweitern statt neu bauen:** Sphinx-Kapitel wachsen in `loesung.rst`/`report.rst`; `cli.py`
  bekommt Batch-Sweeps additiv; `theme.css` wird ergänzt, nicht ersetzt.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§10 „TZ 8"** (Z. 154–157: DoD/TODOs/„schaltet frei"),
   **§2/§9** (Deliverable-Anspruch „Hauptdeliverable, professionell"; Verzeichnisstruktur, u. a. die
   Sphinx-Integration „Plots/Tabellen aus dem Store in den Report", §7 Z. 82), **§6** (Risiko ⑤
   „nicht-teilbare Dimensionen → padden+maskieren", ⑥ „Compile-Cache").
2. `project/project-development/prompts/TZ7-memory-bound.md` **und** `TZ7.5-verbesserungen.md` — die
   **vorherigen Aufträge** (Muster für Aufbau/Arbeitsweise dieses Dokuments; TZ 8 spiegelt genau diese
   Form). TZ 7.5 (Swizzle-GROUP_M · Multi-Config · n-är · Testlauf-Verwaltung) ist **abgeschlossen** —
   sein „Definition of Done" beschreibt exakt die vier Flächen, die TZ 8 nun mitpolieren/-dokumentieren muss.
3. **Sphinx-Report (Ist-Zustand):** `sphinx/source/index.rst` (die `toctree`s; die GSC ist als
   `chapters/group_specific_component/index` eingehängt) → `sphinx/source/chapters/
   group_specific_component/` mit `index.rst`, `pitch.rst`, `presentation.rst`, **`projektplan.rst`**
   (aus einer FRÜHEREN PLAN-Fassung — muss aus dem aktuellen `PLAN.md` re-synchronisiert werden) und
   **`report.rst`** (das **Report-Gerüst**, das mit echten Ergebnissen/Figuren zu füllen ist). Bauen:
   `cd sphinx && make html` (Output `sphinx/build/html/index.html`; Auto-Deploy per
   `.github/workflows/docs.yml` auf push nach main). **Wie Figuren einbinden:** die Assignments-Kapitel
   (z. B. `chapters/04_.../`) zeigen das Muster (`.. figure::`, Bilder unter `_static`/`_images`); ein
   `.. list-table::`-Beispiel steht in `group_specific_component/index.rst`.
4. `project/tool_pipeline/cli.py` — **Single-Run** (`build_config` Z. 29–32 setzt nur `i/k/j`;
   `main` Z. 69–91 ein `run()`; `print_summary` Z. 39–66 ist GEMM-/tflops-geformt). **Hier wachsen
   die Batch-Sweeps** (mehrere Configs → `results.jsonl` → Report-Plots) + Familien-Unterstützung.
   **Post-TZ7.5:** die Sweeps sollten die neuen Achsen abbilden — **GROUP_M** (Swizzle-Gruppengröße),
   **mehrere Tiles je Sweep**, und **mindestens einen n-är-Ketten-Lauf** (`ij,jk,kl->il` → EIN
   Roofline-Punkt) —, damit der Report diese TZ-7.5-Ergebnisse zeigt. Vorbild für die Config-Erzeugung
   ist `controls.configs_from_selection(tiles=…, swizzle_configs=…)`.
5. `project/tool_pipeline/app/assets/theme.css` (Ist-Theming; `:root`-Variablen, `.topbar`, migriert
   aus `layout.py`) + `project/tool_pipeline/app/layout.py` (Grundgerüst + **History-Panel** über der
   `app-body`-Zeile) + `app/components/history.py` (**NEU**, TZ 7.5-4) + die Style-Konstanten in
   `app/components/charts.py`, `controls.py` (**inkl. der dynamischen Tile-Zeilen `_tile_row`/
   `tile_rows` + Swizzle-Konfig-Checkliste**), `kpis.py` (viele inline-styles → Politur-Kandidaten).
6. `project/tool_pipeline/codegen/templates/{contraction,elementwise,reduction}.py` — die
   Randfall-Behandlung nicht-teilbarer Dims: `padding_mode=ct.PaddingMode.ZERO` beim `ct.load` +
   automatisches Clipping durch `ct.store`. **Systematisch belegen** (ragged-Größen-Tests sind für
   Kontraktion schon in `tests/test_codegen.py` — analog für Elementwise/Reduktion).
7. `project/tool_pipeline/codegen/compile.py` (`_MODULE_CACHE` Z. 40; `load_kernel` Z. 79–101:
   idempotenter `save_kernel` nur bei fehlender/abweichender Datei) + `project/tool_pipeline/store/
   store.py` (`save_kernel`, `append_result`) — **Cache-Härtung**: korrupte/halb geschriebene
   `<slug>.py`, parallele Schreibzugriffe (atomar via temp+rename?), veraltete Artefakte.
   **WICHTIG (Post-TZ7.5):** `store.py` besitzt bereits `_atomic_rewrite` (temp im selben Verzeichnis
   → `os.replace`) für `rename_run`/`delete_run` — **dieses Muster für `save_kernel` spiegeln** (nicht
   neu erfinden). Beachte: `config_slug` hängt jetzt bedingt `__sw_g<N>` an (GROUP_M ≠ 8), und n-äre
   Läufe haben einen **synthetischen** Composite-`kernel_path` (die Step-Kernel liegen unter ihren
   eigenen 2-Op-Slugs; `delete_run` fasst `kernels/` grundsätzlich NICHT an).
8. `project/tests/test_measure.py` (+ die übrigen `tests/`) — PLAN §10/TZ 8 nennt `test_measure.py`
   explizit; ergänze Randfall-/Robustheits-/Report-Daten-Tests im bestehenden Dual-Mode-Runner-Muster
   (headless wo möglich; GPU-Tests mit `_has_cuda`-Guard; Store über `$SP`/Monkeypatch isolieren).
9. `project/tool_pipeline/app/callbacks.py` (Fehler-Alerts `_alert`, `execute_run`-Fehlerpfade;
   **Post-TZ7.5:** 11 Callbacks inkl. Tile-+/− und 5 History-Callbacks; die eine nicht-additive Stelle
   ist `Output('main','children', allow_duplicate=True)` beim History-„Laden") + `app/components/
   kpis.py` (`render_status`/`render_verify`) + `app/components/history.py` (`ID_HISTORY_FEEDBACK`) —
   **Fehlerzustände sauber**: sind alle `status`-Fälle (compile_error/verify_failed/run_error/„GPU
   belegt") benutzerfreundlich sichtbar — **plus** die neuen TZ-7.5-Fälle: n-är nicht sauber zerlegbar
   (Hadamard → compile_error/Loud-Fail), doppelte Tile-Zeile, GROUP_M-/Swizzle-Konfig ungültig, die
   weiche „viele Configs"-Warnung (>12), und die History-Rückmeldungen (umbenannt/gelöscht/geladen)?

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ7.5)
**Familien vollständig & verifiziert:** `parse` routet auf `ContractionIR`/`ElementwiseIR`/
`ReductionIR` (family-Parameter); `emit` routet auf die drei Template-Builder (family-abhängiger
Header); `run` hat einen additiven memory-bound-Zweig (kein B1, variable Launch-Arity,
op-/family-abhängige `verify(output, operands: list, config)` + Metriken); `store.config_slug`
hängt die Op an (Kontraktion `op=None` ⇒ Slug byte-identisch); `bench` ist `*operands`-variadisch;
`RunConfig.op` ist additiv im Vertrag. Die GUI hat Familien-Dropdown + Op-Auswahl (nur Elementwise)
+ family-Presets + family-abhängige Validierung/dtype-Gating (memory-bound: fp16/bf16/fp32).

**Metriken family-korrekt:** `compute_metrics` (GEMM), `compute_metrics_elementwise`
(add/mul=1 FLOP/Element, copy=0 → reine Bandbreite), `compute_metrics_reduction` (~kept·reduced
Additionen); GB/s ist die memory-bound-Primärmetrik; die Roofline betont im Hover
`percent_peak_bw` statt `percent_peak_flops` für memory-bound. `copy` (AI=0) erscheint **bewusst
nicht** auf der Roofline (reine Bandbreite), sehr wohl aber im Durchsatz/GB-s.

**TZ-7.5-Erweiterungen (Post-TZ7.5, alle additiv + verifiziert — TZ 8 baut darauf auf):**
- **① Swizzle-GROUP_M einstellbar:** additives `RunConfig.group_m: int = 8`; `contraction.build_gemm_module(…, group_m)`
  backt den Wert als Literal in den Swizzle-Kernel (Default 8 ⇒ byte-identisch, Loud-Fail < 1);
  `config_slug` hängt **bedingt** `__sw_g<N>` an (nur `swizzle & group_m!=8`); GUI-Dropdown + `validate_group_m`;
  in den Charts via Legende/Hover/Symbol disambiguiert. **GPU-bewiesen bijektiv** (GROUP_M=16 == kein-Swizzle
  auf partieller letzter M-Gruppe).
- **② Multi-Config:** `configs_from_selection(tiles=…, swizzle_configs=…)` bildet das **Kreuzprodukt**
  Format × Tile × Swizzle-Konfig (Baselines nur an der ersten Kombi je Format); dynamische Tile-Zeilen (+/−,
  Pattern-Matching `tile_rows`/`tiles_from_state`/`mutate_tile_rows`), Swizzle als GROUP_M-Mehrfachauswahl;
  `charts` generalisiert Serien über `config_slug` + zweiten Kanal (Symbol=Variante), Einzel-Variante
  unverändert; **weiche** Warnung ab >12 Configs. `execute_run` behält den Skalar-Rückfall (Tests/CLI).
- **③ n-äres einsum:** `parse` → `NAryContractionIR` + Pfadplaner (opt_einsum **lazy**, Fallback
  Links-Fold; nicht sauber zerlegbar/Hadamard → **Loud-Fail**); `run` fährt die paarweise Kette durch den
  bewiesenen 2-Op-GEMM-Pfad (Zwischentensoren auf der GPU), `compute_metrics_nary` → **EIN** Roofline-Punkt;
  `kernel_source` = Konkatenation der Step-Kernel. **GPU-verifiziert** gegen `torch.einsum` (`ij,jk,kl->il`).
  `opt_einsum` steht in `requirements.txt`, ist im lokalen venv aber NICHT installiert (Fold reicht).
- **④ Testlauf-Verwaltung (mutabler Store):** `RunResult.run_id/run_name/created_at` (Batch-Identität,
  einmal je „Vergleichen"-Klick); `store.read_all`/`list_runs`/`rename_run`/`delete_run`/`_atomic_rewrite`
  (`os.replace`, **Kernel-Dateien nie angetastet**); `app/components/history.py` (NEU) + History-Panel im
  Layout; **`Output('main','children', allow_duplicate=True)`** beim History-„Laden" (die eine nicht-additive
  Stelle). Altzeilen ohne `run_id` → synthetischer Fallback-Lauf. `$SP`-Isolation deckt auch die Mutatoren ab.

**Tests grün (Post-TZ7.5): 236/236** über **11** Dateien — `test_parse` (34, inkl. n-är-Kette/Hadamard-
Loud-Fail), `test_verify` (13), `test_app_controls` (50, inkl. GROUP_M/Multi-Config/Kreuzprodukt),
`test_measure` (27, inkl. Slug-Bedingtheit + `compute_metrics_nary`), `test_codegen` (40, inkl.
GROUP_M-Bijektivität + n-är end-to-end gegen torch.einsum), `test_app_execute` (10, inkl.
Multi-Config-Kreuzprodukt), `test_app_charts` (36, inkl. Serien-Disambiguierung), `test_store` (8, **NEU** —
`$SP`-isolierte Mutatoren), `test_app_render` (9), `test_app_infra` (2), `test_reshape` (7). Dash bootet mit
11 Callbacks (Familien-/Op-/GROUP_M-/Multi-Config-/History-Verdrahtung). GPU auf der Lab-Maschine verfügbar
⇒ auch die GPU-Tests laufen grün.

**Noch NICHT poliert/dokumentiert (= TZ-8-Arbeit):**
- **Report:** `report.rst` ist ein Gerüst ohne echte Figuren/Tabellen aus `results.jsonl`;
  `projektplan.rst` ist aus einer alten PLAN-Fassung (Re-Sync nötig). Keine automatisch aus dem
  Store erzeugten Report-Plots (Durchsatz je Format, Genauigkeit↔Durchsatz, **die Roofline mit
  beiden Seiten** — der Headline-Chart des Projekts). **Post-TZ7.5 zusätzlich zu erzählen:** die
  GROUP_M-/Tile-Vergleiche (Multi-Config), die n-äre Ketten-Kontraktion (als **ein** Roofline-Punkt)
  und die Testlauf-Verwaltung (History) — inkl. der Serien-Disambiguierung als dataviz-Story.
- **cli.py:** nur ein GEMM-Lauf; keine Batch-Sweeps (mehrere dtypes/Tiles/Familien → `results.jsonl`),
  keine memory-bound-Familien-CLI, `print_summary` ist tflops-/GEMM-geformt (GB/s für memory-bound?).
  **Post-TZ7.5:** die Sweeps sollten auch **GROUP_M**, **mehrere Tiles** und einen **n-är-Ketten-Lauf**
  abdecken (nutze `configs_from_selection(tiles=…, swizzle_configs=…)` als Vorbild).
- **Robustheit:** nicht-teilbare Dims nicht systematisch für Elementwise/Reduktion belegt;
  Compile-Cache nicht gegen korrupte/parallele Schreibzugriffe gehärtet. **Post-TZ7.5:** der atomare
  Rewrite (`_atomic_rewrite` = temp + `os.replace`) existiert bereits im Store (rename/delete) → für
  `save_kernel` **spiegeln**; der OOM-Schutz `_estimate_bytes` deckt n-äre Zwischentensoren bereits ab.
- **Theming/Layout:** `theme.css` schlicht; viele inline-styles verstreut; Roofline-Titel/Untertitel
  überlappen leicht (im PNG sichtbar); kein durchgängiges, professionelles Erscheinungsbild.
  **Post-TZ7.5 zusätzliche Politur-Flächen:** das **History-Panel** (Accordion), die dynamischen
  **Tile-Zeilen (+/−)** und die **Swizzle-Konfig-Checkliste** (neue inline-styles/Komponenten).

## TZ-8-Scope (eng halten!)
1. **Sphinx-Report** (`chapters/group_specific_component/report.rst` + ggf. `loesung.rst`): den
   blogartigen Projektbericht mit echten Ergebnissen füllen — die drei Charts (v. a. die
   **Roofline** memory- vs compute-bound), Tabellen aus `results.jsonl`, die Codegen-/verify-Story,
   die dtype-/Tiling-/Swizzle-Erkenntnisse. Figuren reproduzierbar aus dem Store erzeugen (s. cli).
   **Post-TZ7.5 mit aufnehmen:** die GROUP_M-/Multi-Config-Vergleiche, die n-äre Ketten-Kontraktion
   (als **ein** aggregierter Roofline-Punkt) und die Testlauf-Verwaltung (History).
2. **`projektplan.rst`** aus dem aktuellen `PLAN.md` re-synchronisieren (polierte Fassung; §1–§10).
3. **`cli.py`-Batch-Sweeps:** ein `--sweep`-Modus (mehrere Configs: dtypes × Tiles × Familien) →
   `results.jsonl` + optional die Report-Figuren als PNG (headless, reproduzierbar). Familien-Flag
   (`--family`, `--op`, `--expr`) additiv; `print_summary` family-geformt (GB/s für memory-bound).
   **Post-TZ7.5:** die Achsen um **GROUP_M** und **mehrere Tiles** erweitern und **einen n-är-Ketten-Lauf**
   (`ij,jk,kl->il`) mit sweepen — Config-Erzeugung an `controls.configs_from_selection(tiles=…,
   swizzle_configs=…)` anlehnen, damit dieselben Kreuzprodukt-Slugs entstehen wie in der GUI.
4. **Randfälle padden/maskieren:** systematische ragged-Größen-Tests (Elementwise/Reduktion) gegen
   torch; belegen, dass `padding_mode=ZERO`+Clipping korrekt ist; Doku im Report.
5. **Compile-Cache-Härtung:** atomarer Kernel-Write (temp + `os.replace`), Erkennung korrupter/
   unvollständiger `<slug>.py`, robustes Verhalten bei parallelen Läufen (der GPU-Lock serialisiert
   zwar `run()`, aber der Cache-Write sollte trotzdem atomar sein). Tests dafür. **Post-TZ7.5:** das
   atomare-Rewrite-Muster liegt bereits als `store._atomic_rewrite` vor (rename/delete) → für
   `save_kernel` spiegeln statt neu erfinden.
6. **Fehlerzustände sauber:** alle `status`-Fälle + „GPU belegt" benutzerfreundlich + konsistent in
   der GUI; sinnvolle Meldungen bei ungültigen Familien-/Op-/dtype-Kombis. **Post-TZ7.5 mitprüfen:**
   n-är-Loud-Fail (nicht zerlegbar → compile_error), doppelte Tile-Zeile, ungültiges GROUP_M/Swizzle-
   Konfig, die weiche >12-Config-Warnung und die History-Rückmeldungen (umbenannt/gelöscht/geladen).
7. **Theming/Layout-Politur:** durchgängiges, professionelles Erscheinungsbild (theme.css erweitern;
   inline-styles konsolidieren wo sinnvoll; Chart-Titel/Untertitel-Überlappung fixen; responsives,
   aufgeräumtes Layout). **Post-TZ7.5 mit einbeziehen:** das History-Panel (Accordion), die dynamischen
   Tile-Zeilen (+/−) und die Swizzle-Konfig-Checkliste. **Kein** Vibe-Coding, keine Framework-Diskussion.
8. **Tests:** `tests/test_measure.py` + die übrigen — Randfall-/Cache-Härtungs-/CLI-Sweep-/
   Report-Daten-Tests; alle bestehenden Tests bleiben grün; App-Smoke bleibt grün.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Report-Umfang & -Quelle:** Werden die Report-Figuren **live beim `make html`** erzeugt (Skript
   liest `results.jsonl` → matplotlib/plotly-PNG) oder **vorab** per `cli --sweep` als PNGs
   eingecheckt und nur eingebunden? (Empfohlen: **vorab per cli-Sweep**, PNGs unter
   `sphinx/source/_static/gsc/` eingecheckt → `make html` bleibt torch-/GPU-frei und CI-tauglich.) Kläre.
2. **Report-Struktur:** ein großes `report.rst` vs. Aufteilung (Motivation · Architektur · Codegen ·
   Messung · Ergebnisse/Roofline · Fazit). Kläre den Detailgrad (blogartig, s. PLAN §1).
3. **cli-Sweep-Achsen:** welche Sweeps sind für den Report am aussagekräftigsten? (Empfohlen: je
   Familie ein Sweep — Kontraktion über dtypes+Tiles, Elementwise/Reduktion über dtypes+Größen —
   so entsteht die Roofline mit beiden Seiten.) Kläre Umfang/Größen (geteilte Maschine!).
4. **Cache-Härtung Tiefe:** nur atomarer Write + Korruptions-Erkennung (empfohlen) vs. zusätzlich
   Content-Hash/Versionsstempel im Kernel-Header. Kläre.
5. **Theming-Tiefe:** nur Politur des Bestehenden (empfohlen) vs. eigenes Farbschema/Dark-Mode.
   Kläre, wie viel „professionell" hier bedeutet (Zeitbudget vs. Substanz — PLAN: Substanz > Optik).
6. **projektplan.rst-Sync:** wörtliche Übernahme aus PLAN.md vs. redaktionell gekürzte Fassung.
   (Empfohlen: redaktionell poliert, aber inhaltsgleich.) Kläre.

## Scope-Grenzen (was TZ 8 NICHT tut)
- **Keine neuen Operations-Familien/Ops** (Fusion, Copy/Transpose als eigene Op = Zukunft; **n-är ist
  in TZ 7.5 bereits als Erweiterung der Kontraktions-Familie umgesetzt** — TZ 8 poliert/dokumentiert es nur).
- **Kein Naht-/Schema-/Slug-/Store-Format-Umbau** — additiv, alles Bestehende bleibt grün.
- **Kein neues Mess-/Codegen-Paradigma** — nur Härtung/Randfälle/Doku.
- **Kein Autotuning/keine neuen dtypes** (bewusst gestrichen bzw. abgeschlossen).

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess
  CUDA-frei; Charts reine, headless-testbare Funktionen. **`make html` bleibt GPU-/torch-frei.**
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla07/mla/.venv/bin/python` (Shell-`python`
  nicht im PATH; Shell-State persistiert nicht zwischen Bash-Aufrufen). Start: `python -m tool_pipeline`
  (GUI), `python -m tool_pipeline.cli` (headless). Sphinx: `cd sphinx && make html`.
- **Harte Regel: NIEMALS `git commit` / `git push`, außer der Nutzer fordert es direkt** (dann OHNE
  dich selbst als Autor/Co-Autor zu nennen — Repo-Regel in `CLAUDE.md`).
- **Geteilte Maschine:** kleine Größen, `torch.manual_seed(0)`, GPU-Lock (`project/.cache/gpu.lock`)
  respektieren, keine unnötigen GPU-Läufe, OOM vermeiden. **Store-Isolation in Tests über `$SP`**
  (nicht `/tmp`); `results.jsonl`/`kernels/` nicht durch Tests verschmutzen.
- **verify-before-trust:** kein Ergebnis/keine Report-Figur ohne bestandene fp32-Referenz.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 8 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App
   in lauffähigem, prüfbarem Zustand — z. B. cli-Sweep → Report-Figuren-Skript → report.rst füllen →
   projektplan.rst-Sync → Randfall-Tests → Cache-Härtung → Fehlerzustände → Theming → Tests). Die
   Design-Entscheidungen oben **vorab** mit dem Nutzer klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt
   bewirkt / was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären).
   Dann auf **die Validierung des Nutzers warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-8-Scope bleiben; Scope-Creep (neue Familien, Fusion, Autotuning) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 9 — anschauen, vorbereiten und einen Session-Prompt + Planungs-MD
   erstellen.** Der verbleibende erste Zukunftskandidat aus PLAN §10 „Später/optional" ist jetzt
   **Fusion** (Kontraktion + Elementwise-Epilog, A04-Befund 0,98×) — **n-äres einsum ist bereits in
   TZ 7.5 umgesetzt** und fällt als Kandidat weg. Genau nach *diesem* Muster: gründlich einlesen
   (Workflow), **PLAN §10 + der A04-Fusion-Befund maßgeblich**, Anker aus dem *dann* aktuellen
   Post-TZ8-Ist-Zustand, MD unter `project/project-development/prompts/TZ9-*.md`, und **diese
   Arbeitsweise inkl. der zwei Zusätze (sehr einfache Erklärung nach jedem TODO + Planung des
   übernächsten TZ als letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**, wo möglich: die cli-Sweep-Logik
(Config-Erzeugung) ist deterministisch prüfbar; die Report-Figuren-Erzeugung liest `results.jsonl`
(headless, torch-frei) und rendert PNGs — **ansehen** (dataviz „render & look", v. a. die Roofline
mit beiden Seiten); die Cache-Härtung (atomarer Write, Korruptions-Erkennung) headless mit temp-
Dateien; Randfälle (ragged Größen) auf der GPU gegen torch (unter Lock). `make html` muss **ohne
GPU/torch** durchlaufen (CI-tauglich). App real starten (GPU-Lock!) und die polierten Fehlerzustände
+ das Theming durchklicken. Store in Tests über `$SP` isolieren.

## Definition of Done (TZ 8)
Das Deliverable ist **poliert und dokumentiert**: der **Sphinx-Report**
(`chapters/group_specific_component/`) trägt die echten Ergebnisse (die drei Charts inkl. der
Roofline mit **beiden** Seiten, Tabellen aus `results.jsonl`, die Codegen-/verify-/dtype-/Tiling-
Story **sowie die TZ-7.5-Erweiterungen: GROUP_M-/Multi-Config-Vergleiche, die n-äre Ketten-
Kontraktion als ein Roofline-Punkt und die Testlauf-Verwaltung/History**), `projektplan.rst` ist aus
`PLAN.md` synchronisiert; `cli.py` fährt reproduzierbare **Batch-Sweeps** (alle Familien, inkl.
GROUP_M/Multi-Tile/n-är) für die Report-Plots; nicht-teilbare Dimensionen sind systematisch
belegt (padden/maskieren); der Compile-Cache ist gehärtet (atomar, korruptionsfest — Muster aus
`store._atomic_rewrite`); die Fehlerzustände sind benutzerfreundlich; das Theming/Layout ist
durchgängig professionell (kein Vibe-Coding); `make html` läuft GPU-/torch-frei durch; **alle Tests
grün** (die 236 Post-TZ7.5-Tests + neue Randfall-/Cache-/CLI-/Report-Tests) + App-Smoke; die
Operations-Familien (inkl. n-är) bleiben unverändert. **Zusätzlich:** nach jedem TODO gab es eine
sehr einfache Erklärung, und als letzter Schritt ist **TZ 9 (Fusion) vorbereitet** (Planungs-MD +
Session-Prompt).

---

## Session-Prompt (zum Starten von TZ 8 — dies dem Assistenten geben)

> Lies zuerst VOLLSTÄNDIG die Datei
> `project/project-development/prompts/TZ8-politur-report.md` — das ist dein Auftrag (Teil-Ziel 8
> des „cuTile Performance Lab"). Befolge die dort unter „Arbeitsweise (verbindlich)" beschriebene
> Vorgehensweise strikt:
>
> Belese dich zuerst gründlich in die bestehende, verifizierte Implementierung (TZ 1–7 + TZ 7.5) —
> genau die Dateien/Reihenfolge unter „Zuerst lesen" — und bestätige dein Verständnis kurz. (Paralleles Lesen
> per Subagenten/Workflow ist erwünscht.) Zerlege TZ 8 in geordnete TODOs und kläre die
> „Design-Entscheidungen" vorab mit mir. Lege die Aufschlüsselung zur FREIGABE vor, BEVOR du Code
> schreibst. Dann TODO für TODO: nach jedem anhalten und zeigen (a) was du getan hast, (b) wie du es
> verifiziert hast, UND (c) eine SEHR EINFACHE Erklärung in Alltagssprache. Dann auf meine
> Validierung warten. Nicht mehrere TODOs bündeln. Harte Regeln: NIEMALS git commit/push (außer ich
> fordere es direkt; dann ohne dich als Autor/Co-Autor); Prosa/Kommentare/UI auf Deutsch; aus
> `project/` mit dem venv-Python `.venv/bin/python` ausführen; `make html` GPU-/torch-frei;
> verify-before-trust: Report-Figuren nur aus ok-Läufen. Leg los mit dem Einlesen.
