# Auftrag: TZ 7.5 — Verbesserungen & Feedback (Swizzle-GROUP_M · Multi-Config · n-äres einsum · Testlauf-Verwaltung)

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen).
Wir bauen die Group-Specific Component „**cuTile Performance Lab**" (interaktiver einsum/GEMM-Explorer,
GPU/cuTile). **Teil-Ziele 1–7 sind fertig und verifiziert:** die headless-Pipeline läuft über die eine
Naht `run(config) → RunResult` (parse → **Familien-Router** → reshape/B1 (nur Kontraktion) → emit →
compile+Cache → Kalt-Lauf=compile_ms → verify(fp32) → benchmark → Metriken → Baselines → GPU-Zustand →
Store); die Dash-GUI fährt den Live-Loop als Batch-Vergleich (Familien-Auswahl contraction/elementwise/
reduction · Ausdruck via Presets/Freitext + Größen je Index · Elementwise-Op · Tile TM/TN/TK · L2-Swizzle ·
Zahlenformate · Baselines) → je Config ein `run()` unter **einem** GPU-Lock → KPIs/Verify/Code + **drei**
Charts (Durchsatz · Genauigkeit↔Durchsatz · Roofline). Alle drei Operations-Familien laufen end-to-end;
die Roofline zeigt memory- vs compute-bound. **206 Tests grün** über 10 Dateien.

**ABER:** aus der Nutzung/Präsentation sind **vier konkrete Verbesserungswünsche (Feedback)** entstanden,
die *vor* der finalen Politur (TZ 8) das Werkzeug spürbar mächtiger machen. Dieser Auftrag ist ein
**Zwischen-Teilziel TZ 7.5** (schiebt sich zwischen das fertige TZ 7 und das schon vorbereitete TZ 8):

1. **Swizzle-GROUP_M einstellbar.** Der L2-Swizzle ist heute ein reiner Ein/Aus-Schalter mit fest
   verdrahtetem `GROUP_M = 8` (`contraction.py:54`). Der Nutzer soll GROUP_M selbst wählen können — **mit
   sinnvollen Grenzen**.
2. **Mehrere Tile-/Swizzle-Konfigurationen gleichzeitig vergleichen.** Heute variiert der Nutzer nur zwei
   Achsen (Format-Mehrfachauswahl, Swizzle off/on/both) über **ein festes Tile**. Es soll möglich sein,
   **mehrere Tile-Konfigurationen** (Zeile hinzufügen/entfernen) und **mehrere Swizzle-Konfigurationen**
   (verschiedene GROUP_M) **gegeneinander** zu messen.
3. **n-äres einsum** (`abc,bca,cba->…`, `ij,jk,kl->il`). Heute lehnt der Kontraktions-Parser >2 Operanden
   hart ab (`parse.py:280` `NotImplementedError`). n-är ist laut `PLAN.md §10` ein „Später/optional"-
   Zukunftskandidat (opt_einsum → paarweise Kontraktionen) — **wird jetzt vorgezogen**.
4. **Testlauf-Verwaltung.** Alte Läufe sollen sich **wieder ansehen**, **vergleichen**, **umbenennen** und
   **löschen** lassen. Heute ist der Store strikt append-only, ohne Lauf-Identität und ohne History-Ansicht.

TZ 7.5 setzt genau diese vier Wünsche um — **additiv, verify-before-trust, ohne Naht-Umbau** — und
lässt danach TZ 8 (Politur/Report) auf dem erweiterten Stand aufsetzen.

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1** (f-String-Templates → `@ct.kernel`; ein Modul je Familie). **Kein neues Codegen-Paradigma.**
- **Die eine Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer);
  Haupt-Prozess CUDA-frei (DiskcacheManager **forkt** ihn — `run`/`torch`/`cuda` **niemals** im Modulkopf,
  Import bleibt lazy in `execute_run`, `callbacks.py:234`); Charts/kpis/code_panel/controls-Logik sind reine,
  **headless-testbare** Funktionen. **Kein Naht-Umbau** — Schema-Erweiterungen nur additiv.
- **Je Config genau ein `run()` unter EINEM GPU-Lock.** Mehr Vergleichsachsen (Tiles, GROUP_M, n-är) heißt
  **mehr `RunConfig`s in der Liste**, *nicht* eine neue Batch-API. `run(config)→RunResult` bleibt der Vertrag.
- **Results-Store-Format = JSON Lines** (`project/results/results.jsonl`) + Kernel als
  `results/kernels/<slug>.py` (lesbarer Config-Slug; `results/kernels/` ist der **gitignored Compile-Cache**,
  `.gitignore:32` — git-getrackt ist nur `results.jsonl`). **Byte-Identität der bestehenden Slugs ist heilig:**
  neue quelltextbestimmende Felder gehen **nur bedingt** in den Slug (s. u.), damit weder der Compile-Cache
  (falscher Treffer/Kollision) noch die Slug-Referenzen des git-getrackten `results.jsonl` driften.
- **verify-before-trust bleibt Gesetz.** Kein Kernel-Ergebnis/keine Chart-/Report-Figur ohne bestandene
  `torch.einsum`-fp32-Referenz. Chart-Punkte nur aus `status=="ok"`.
- **dataviz-Skill** als Maßstab für jede Chart-Änderung; memory-bound bleibt links, compute-bound rechts.
- **Erweitern statt neu bauen.** Jede der vier Änderungen ist **additiv** an bestehende Muster angedockt
  (Pattern-Matching-Controls, `config_slug`, `from_dict`-Toleranz, variadisches `verify`). Alle 206
  bestehenden Tests bleiben grün.
- **Der git-getrackte `results.jsonl` wird durch Tests NIE verschmutzt** (`$SP`/Monkeypatch) — **auch nicht**
  durch die neuen Mutatoren (rename/delete).

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§2** (Scope: Operationen-Familie; *„n-ary (opt_einsum)"* als
   Greenfield in §4 Z. 52), **§6** (Codegen-Risiken ① mma-Orientierung, ④ B1-View, ⑥ Compile-Cache — alle
   für n-är relevant), **§10** (TZ-Reihenfolge; *„Später/optional: n-ary einsum (opt_einsum → paarweise
   Kontraktionen)"* Z. 160), **§2 Tabelle Zeile „Persistenz/Results-Store"** (Vergleich über Läufe).
2. `project/project-development/prompts/TZ7-memory-bound.md` **und** `TZ8-politur-report.md` — die **Muster**
   für Aufbau/Arbeitsweise dieses Dokuments (TZ 7.5 spiegelt genau diese Form) **und** der *nachgelagerte*
   Auftrag, den du am Ende re-synchronisierst.
3. **Sub-Ziel 1 (Swizzle-GROUP_M):**
   `codegen/templates/contraction.py` (Konstante `_SWIZZLE_GROUP_M = 8` **Z. 51–54**; `build_gemm_module`-
   Signatur **Z. 57–58**; Swizzle-Zweig `group_m = _SWIZZLE_GROUP_M` **Z. 112–132**; `{group_const}`-
   Substitution **Z. 160–163**) · `schema.py` (`RunConfig.swizzle: bool` **Z. 102–106**; `from_dict`-Filter/
   `asdict` **Z. 119–136**) · `codegen/emit.py` (Builder-Aufruf **Z. 69–71**; Header **Z. 41–44**) ·
   `store/store.py` (`config_slug`, `if d.get("swizzle"): slug += "__sw"` **Z. 60–65**) ·
   `app/components/controls.py` (`_SWIZZLE_OPTIONS`/`swizzles_from_value`/`validate_swizzle` **Z. 183–189,
   464–477**; `configs_from_selection` **Z. 485–524**; `_tile_select` **Z. 657–674**) ·
   `app/callbacks.py` (`execute_run` **Z. 159–265**; `_on_run`-States **Z. 340–376**) ·
   `app/components/charts.py` (`_SWZ_SYMBOL` **Z. 387–388**; `_by_format` **Z. 263–281**).
4. **Sub-Ziel 2 (Multi-Config):** dieselben `controls.py`-Stellen wie oben **plus** `tile_from_controls`
   **Z. 480–482**, `_TILE_*_OPTIONS` **Z. 165–168**, `build_controls` **Z. 748–795** · **das
   Pattern-Matching-Vorbild** `index_size_inputs`/`dim_sizes_from_state` (**controls.py:574–591 / 321–329**)
   und der Callback `_rebuild_index_sizes` · `app/layout.py` (statisch, einziger `dcc.Store('_scroll_dummy')`
   **Z. 36–53**) · `app/components/charts.py` (`_PALETTE`/`_FORMAT_COLOR`/`assert len(COMBOS)<=len(_PALETTE)`
   **Z. 35–48**; `_points` **Z. 62–110**; `figure_throughput`-A/B-Umschaltung **Z. 184–187**; Roofline
   **Z. 515–542**).
5. **Sub-Ziel 3 (n-äres einsum):** `intermediate_representation/parse.py` (Familien-Router **Z. 250–259**;
   **2-Op-Gate `if len(inputs) != 2: raise NotImplementedError` Z. 280–284**; M/N/K/Batch-Klassifikation
   **Z. 326–335**; `ContractionIR` **Z. 55–110**) · `intermediate_representation/config.py`
   (`generate_config`, `if len(inputs) != 2: raise ValueError` **Z. 135–136**) ·
   `intermediate_representation/reshape.py` (`to_canonical`/`to_canonical_operands`/`from_canonical_output`,
   fest `ir.inputs[0]/[1]` **Z. 98, 142–155**) · `intermediate_representation/optimizer.py` (nur
   `split/fuse/permute`, **kein** `contract_path` **Z. 40–127**) · `run.py` (`_build_natural_operands`/
   `_build_inputs` **Z. 92–123**; Launch-Arity **Z. 355, 379, 404**; `verify(...)`/`compute_metrics(M,N,K)`/
   `provenance["sizes"]` **Z. 324, 363–365, 390**) · `measure/verify.py` (**schon variadisch**:
   `verify(output, operands, config)`, `torch.einsum(config.expr, *ops_f)` **Z. 75–112**) ·
   `measure/metrics.py` (`gemm_flops`/`gemm_bytes`/`compute_metrics` 2-Operanden **Z. 16–34, 112–126**) ·
   `codegen/templates/contraction.py` (`launch(A,B,C)` **Z. 166, 194–206**) · `controls.py` (`validate_expr`/
   `index_categories`/Presets **Z. 276–289, 564–571**) · `requirements.txt` (**`opt_einsum` Z. 14 — im
   lokalen `.venv` NICHT installiert!**) · `tests/test_parse.py` (`test_rejects_nary` **Z. 112–115**).
6. **Sub-Ziel 4 (Testlauf-Verwaltung):** `store/store.py` (`append_result` **append-only, Z. 100–114**;
   `read_results`→DataFrame, **einzige Lese-API, ohne Konsument, Z. 117–123**; `RESULTS_JSONL`-Konstanten
   **Z. 33–36**) · `schema.py` (`RunResult`-Felder **ohne id/name/session, Z. 142–173**; `from_dict`-Toleranz
   **Z. 170–173**) · `run.py` (`_provenance` `timestamp=datetime.now()` **pro Zeile, Z. 202–211**; `_result`
   ruft `store.append_result` **Z. 228–243**) · `app/callbacks.py` (`execute_run`/`render_comparison`
   **Z. 108–156, 159–265**; **`_on_run` einziger Schreiber auf `Output('main','children')` Z. 340–376**) ·
   `app/layout.py` (kein History-Panel/Result-Store **Z. 36–53**) · `app/components/code_panel.py`
   (`render_code_panel(source, kernel_path)`, `source=None`→„kein Kernel" **Z. 54–67** — Altzeilen haben
   keinen Quelltext) · `tests/test_app_execute.py` (`$SP`-Isolation `_redirect_store` **Z. 24, 67–74**).
7. **Querschnitt (Infra/Wiring/Tests):** `app/app.py` (`create_app`, lazy `callbacks.register`, Fork-Sicherheit
   **Z. 35–67**) · `app/callbacks.py` (`register` = **4 Callbacks** **Z. 281–405**; GPU-Lock
   `project/.cache/gpu.lock`, `filelock`, 60 s **Z. 38–41, 235–238**) · alle `tests/` (**Dual-Mode-Runner**:
   nackte `def test_*` + `_main()` je Datei; **kein** conftest/pytest.ini; `$SP` und `_has_cuda`-Guard nur in
   `test_measure.py:234`) · `CLAUDE.md` + `intermediate_representation/README.md` (harte Regeln/Konventionen).

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ7)

### Gemeinsame Basis
- **Naht & Fork-Sicherheit:** `execute_run` (Dash-frei, `callbacks.py:159–265`) validiert Controls →
  `configs_from_selection` → **lazy** `from tool_pipeline.run import run` (`:234`) → `FileLock` über den
  ganzen Batch → `run(cfg)` je Config → `render_comparison`. `execute_run` **wirft nie** (gibt immer eine
  Alert-/Ergebnis-Liste zurück — `test_execute_survives_run_import_failure`).
- **Schema-Evolution ist additiv-erprobt:** `RunConfig.from_dict`/`RunResult.from_dict` filtern unbekannte
  Felder weg, `asdict` echot neue Felder automatisch → alte `results.jsonl`-Zeilen bleiben ladbar (belegt
  durch bereits gewachsene Felder: frühe Zeilen ohne `op`/`rel_err`/`gpu_state`).
- **`config_slug` (`store.py:42–65`)** ist Cache-Schlüssel **und** Kernel-Dateiname aus
  `(expr, dtype, acc, tile, op, swizzle)` — bewusst **ohne** `dim_sizes`/`baselines`/`family`; `op` nur wenn
  gesetzt (Kontraktion byte-identisch zu TZ 1–6), `swizzle`→`__sw`.
- **Tests:** 206 `test_*`-Funktionen über 10 Dateien — `test_app_charts` 31, `test_app_controls` 42,
  `test_app_execute` 9, `test_app_infra` 2, `test_app_render` 9, `test_codegen` 37, `test_measure` 25,
  `test_parse` 31, `test_reshape` 7, `test_verify` 13. Headless vs. GPU s. „Verifikation".

### Sub-Ziel 1 — Swizzle-GROUP_M (Ist-Zustand)
- `swizzle` ist ein **reiner bool** (`schema.py:106`). `GROUP_M = 8` ist **die einzige** hart verdrahtete
  Stelle: Modul-Konstante `_SWIZZLE_GROUP_M = 8` (`contraction.py:54`), im Swizzle-Zweig als
  `group_m = _SWIZZLE_GROUP_M` (`:113`) gelesen und als Python-Literal in den Kernel gebacken (`:116, 160`).
- Die grouped-M-Rasterung `group_size_m = min(num_pid_m - first_pid_m, GROUP_M)` (`:127`) **klemmt** die
  partielle letzte Gruppe → **mathematisch korrekt & bijektiv für JEDES `GROUP_M ≥ 1`** (belegt durch
  `test_swizzle_equals_noswizzle`). `GROUP_M=1` reproduziert exakt die plain-`bid`-Zuordnung („kein Swizzle").
- **Kollisionsrisiko:** `config_slug` kodiert GROUP_M **nicht** (`store.py:63–64` hängt nur `__sw` an) →
  verschiedene GROUP_M träfen heute **denselben** Slug/dieselbe gecachte `<slug>.py` (**still falsches
  Artefakt**, sobald GROUP_M variabel). Ebenso: `charts._by_format` bündelt je Format in genau zwei Eimer
  `noswz/swz` (`:274`) und `_SWZ_SYMBOL` ist bool (`:387`) → mehrere GROUP_M mit `swizzle=True` **kollidieren**
  (letzter gewinnt). Der headless-Test `test_swizzle_emit_structure` erwartet **wörtlich `GROUP_M = 8`**
  (`test_codegen.py:233`).

### Sub-Ziel 2 — Multi-Config (Ist-Zustand)
- Der Nutzer variiert heute **zwei** Achsen: Format (`ID_DTYPES`, Mehrfach-Checklist) × Swizzle
  (`off/on/both`→`[False]/[True]/[False,True]`). Das **Tile ist EIN Feldsatz** (drei `dbc.Select`
  `ID_TILE_TM/TN/TK`, `_tile_select` `controls.py:657–674`); `tile_from_controls` (`:480–482`) liefert **ein**
  dict; `configs_from_selection` kopiert dieses **eine** Tile in jede Config (`:519–520`). Kein Listen-Pfad,
  keine `+/-`-Zeile.
- **Einziges dynamisches Muster im Tool:** die Größenfelder je Index über Pattern-Matching-IDs
  `{'type': INDEX_SIZE_TYPE, 'index': d}` (`index_size_inputs`/`dim_sizes_from_state`, `controls.py:574–591 /
  321–329`; Callback `_rebuild_index_sizes`). **Das ist die Blaupause** für dynamische Tile-/Swizzle-Zeilen.
- **Chart-Engpass:** `_PALETTE` hat 8 Farben, `assert len(COMBOS) <= len(_PALETTE)` (`charts.py:45`) → **8 == 8**,
  **kein freier Farbkanal** für eine dritte Achse. `_points` liest **weder `tile` noch `group_m`** (`:62–110`)
  → zwei Configs, die sich nur im Tile unterscheiden, sind im Chart **nicht unterscheidbar**; `_by_format`
  überschreibt sie still. Farbe=Format, Symbol=Swizzle, Größe/Rahmen=Primärformat sind **alle** belegt.

### Sub-Ziel 3 — n-äres einsum (Ist-Zustand)
- **Harter Blocker:** `parse.py:280` `if len(inputs) != 2: raise NotImplementedError`; `config.py:135`
  `raise ValueError("nur 2-Operanden…")`. `reshape.to_canonical` liest fest `ir.inputs[0]/[1]`.
  `run._build_inputs`/Launch-Arity sind fest `A,B,C`. `metrics` ist ein Ein-GEMM-Kostenmodell.
- **Schon bereit / nutzbar:** `verify` ist **variadisch** (`verify.py:96` `torch.einsum(expr, *ops_f)` — n-är
  ohne Änderung); `RunConfig.__post_init__` leitet `inputs=['ij','jk','kl']/output='il'` **korrekt** ab
  (`schema.py:119–126`); `config_slug` erzeugt für n-är bereits einen gültigen Namen
  (`ij_jk_kl_to_il__fp16-fp32__TM128_TN128_TK64`); der 2-Op-GEMM-Pfad ist **bewiesen** und je *paarweisem
  Schritt* wiederverwendbar. `optimizer.py` bietet **keine** Pfadplanung.
- **`opt_einsum` steht in `requirements.txt:14`, ist aber im lokalen `.venv` NICHT installiert.** `torch.einsum`
  funktioniert n-är auch ohne (opt_einsum optimiert nur die Reihenfolge). Jeder Nicht-Lazy-Import bräche die
  CUDA-freie GUI/Headless-Tests.
- `tests/test_parse.py:112` `test_rejects_nary` **erwartet die Ablehnung** — muss bewusst umgestellt werden.

### Sub-Ziel 4 — Testlauf-Verwaltung (Ist-Zustand)
- Ein „Lauf" ist heute physisch **eine `results.jsonl`-Zeile** (ein `RunResult` = eine `RunConfig` = ein
  Format). Der „Vergleichen"-Klick fährt aber einen **ganzen Batch** (`execute_run`), dessen `RunResult`-Liste
  **nur im Speicher** lebt und direkt in `render_comparison` geht — **nichts** wird persistent gruppiert oder
  aus `results.jsonl` **zurückgelesen**.
- Der Store ist **strikt append-only** (`open("a")`, `store.py:100–114`); die einzige Lese-API
  `read_results` liefert einen **DataFrame** (keine `RunResult`-Objekte, kein Konsument, `:117–123`).
  **Kein** `run_id`/`run_name`/`session`, **kein** Umbenennen/Löschen. `provenance.timestamp` ist
  **pro Zeile** sekundengenau (`run.py:210`) → als Gruppenschlüssel **unbrauchbar** (Kollisionen + Drift
  innerhalb eines Batches).
- **Die eine nicht-additive Stelle:** `Output('main','children')` hat mit `_on_run` **genau einen** Schreiber
  (`callbacks.py:340`). Ein History-„Laden" auf denselben Output braucht **`allow_duplicate=True`**.
- `kernel_source` steht **nicht** im JSONL → beim Wieder-Ansehen alter Läufe fällt das Code-Panel auf
  „kein Kernel" zurück (es sei denn, die maschinenlokale `kernels/<slug>.py` wird nachgeladen).

## TZ-7.5-Scope (vier Sub-Ziele; jedes eine lauffähige, verifizierte Scheibe)
**Empfohlene Reihenfolge: 1 → 2 → 4 → 3.** Begründung: **1 ist Voraussetzung für 2** (mehrere
Swizzle-Konfigurationen = mehrere GROUP_M). **2 etabliert die Chart-Serien-Generalisierung** (Serien-Key =
`config_slug`, zweiter visueller Kanal), **die 4 wiederverwendet** (mehrere alte Läufe mit gleichem Format
vergleichen). **3 (n-är)** ist der eigenständige, tiefste Core-Eingriff und kommt **zuletzt**, damit er die
GUI-/Chart-Arbeit nicht verschränkt. Jedes Sub-Ziel lässt Pipeline **und** App in prüfbarem Zustand; nach
**jedem TODO** anhalten (s. „Arbeitsweise").

### 7.5-1 · Swizzle-GROUP_M einstellbar (Nutzer-Punkt 1)
1. **`schema.py`:** additives Feld `group_m: int = 8` (nach `swizzle`, `:106`); `swizzle:bool` bleibt der
   Gate, `group_m` zählt nur bei `swizzle=True`. Optional defensive Untergrenze in `__post_init__`.
2. **`contraction.py`:** `build_gemm_module(..., group_m: int = 8)`; im Swizzle-Zweig `:113` die Konstante
   durch den Parameter ersetzen (`_SWIZZLE_GROUP_M` als Default behalten); **Loud-Fail** `ValueError` bei
   `group_m < 1` (analog dtype/acc-Prüfung `:78–90`). Default 8 ⇒ emittierter Quelltext **byte-identisch**.
3. **`emit.py`:** Builder-Aufruf `:70–71` um `group_m=config.group_m` erweitern. **Header (`:41–44`) NICHT
   ändern** (bleibt `swizzle={bool}`; GROUP_M steht im Body-Docstring) — sonst driften die `__sw`-Kernel im
   Compile-Cache und `test_emit_contraction_header_byte_identical` bricht.
4. **`store.py` `config_slug`:** GROUP_M **nur** anhängen, wenn `swizzle` **und** `group_m != 8` — z. B.
   `slug += f"__sw_g{group_m}"` statt `__sw` (bei 8 weiterhin bares `__sw`). Exaktes Mirror des `op`-Suffix.
   **Zentrale Korrektheitsmaßnahme:** ohne Suffix träfe ein geänderter GROUP_M still den `group_m=8`-Kernel aus
   dem Compile-Cache (**falsches Artefakt**); mit bedingtem Suffix bleiben alle Default-8-Slugs byte-identisch.
5. **`controls.py`:** `_SWIZZLE_GROUP_M_OPTIONS = (1,2,4,8,16,32)` + `ID_SWIZZLE_GROUP_M` + Control im
   `_tile_select` (Default 8) + `validate_group_m(v)` (deutscher Fehlertext, Grenzen);
   `configs_from_selection` reicht `group_m` in jede `RunConfig` durch (analog `tile`).
6. **`callbacks.py`:** `execute_run(..., group_m=8)` (Default hält headless-Aufrufe lauffähig),
   `validate_group_m` (nach `validate_swizzle` `:201`), `_on_run` liest `State(ID_SWIZZLE_GROUP_M,'value')`
   (`:352`) und reicht durch (`:375`).
7. **Tests (headless):** `build_gemm_module(group_m=16)` emittiert `GROUP_M = 16`; `ValueError` bei `<1`;
   `config_slug` → `__sw_g16` nur bei `swizzle & !=8`, sonst `__sw`; `validate_group_m`;
   `configs_from_selection` setzt `group_m`. `test_swizzle_emit_structure` (`:222–237`) **parametrisieren**.
   **GPU:** `test_swizzle_equals_noswizzle`-Variante mit `group_m=16` auf einer Größe mit **partieller** letzter
   M-Gruppe (num_pid_m nicht durch 16 teilbar), um Bijektivität des variablen GROUP_M zu belegen.

### 7.5-2 · Mehrere Tile-/Swizzle-Konfigurationen gleichzeitig vergleichen (Nutzer-Punkt 2)
1. **`layout.py`:** ein `dcc.Store` (z. B. `id='tile-rows-store'`) als **Wahrheit** über die aktuelle
   Zeilen-Liste (analog `_scroll_dummy`).
2. **`controls.py` — Tile-Liste:** Tile von „ein Feldsatz" auf „Liste von Tile-Zeilen mit `+/-`" umstellen —
   Pattern-Matching-IDs `{'type':'tile-tm','index':i}` (Blaupause `index_size_inputs`), `tile_rows(values)`-
   Renderer (rein/headless-testbar) + `tiles_from_state(...) -> list[dict]` (Gegenstück zu
   `dim_sizes_from_state`); `validate_tile` je Zeile; **Duplikate abfangen**. Skalar-Helfer als Ein-Zeilen-Fall
   behalten.
3. **`controls.py` — Swizzle-Liste:** eine **Swizzle-Konfiguration = `(swizzle:bool, group_m:int)`** (baut auf
   7.5-1 auf). `off/on/both` durch eine Mehrfachauswahl der GROUP_M-Werte (+ „aus"-Eintrag) ersetzen/ergänzen
   (`swizzle_configs_from_state`).
4. **`controls.py` — Kreuzprodukt:** `configs_from_selection` um zwei Achsen erweitern: `tiles:list[dict]`
   und `swizzle_configs:list[(bool,int)]`; Batch = `selection × tiles × swizzle_configs`. **Baseline-Anhängung
   verallgemeinern** („nur erste `(tile,swizzle)`-Kombi je Format", nicht n-fach messen).
5. **`callbacks.py`:** `_on_run` liest Tile-Zeilen über `State({'type':'tile-tm','index':ALL})` (wie die
   Größenfelder); **neuer Callback** für `+/-` (Input Buttons → mutiert nur den Store → Renderer baut die
   Zeilen). `execute_run`-Signatur auf `(tiles, swizzle_configs)`. **GPU-Lock-Schleife + Progress bleiben** —
   nur mehr Configs. Progress-Label um Tile/GROUP_M ergänzen.
6. **`charts.py` — Serien-Generalisierung (der Kern):** `_points` um `tile`+`group_m`/Swizzle-Variante
   erweitern; **`config_slug` als kanonischen Serien-Key** wiederverwenden (Single-Source-of-Truth, kein
   zweites Key-Schema); `_by_format` (nur `dtype:acc`, 2 Slots) durch eine tile-/swizzle-fähige Gruppierung
   ersetzen. **Zweiter visueller Kanal** für die Zusatzachse (Marker-Symbol=Tile, Muster/Dash=Swizzle-Variante;
   Farbe bleibt Format), Legende/Hover **vollständig disambiguieren** (`fp16→fp32 · TM128/TN128/TK64 · G8`).
   Bei zu vielen Serien: **weiche** UI-Warnung, keine harte Sperre.
7. **Tests:** `configs_from_selection`-Signatur (`tile→tiles`, `swizzle→swizzle_configs`) + betroffene
   `test_app_controls`-/`test_app_execute`-Tests nachziehen; neue headless-Tests: `tiles_from_state`, volles
   Kreuzprodukt (erwartete Config-Anzahl/Reihenfolge), Baseline-nur-einmal-je-Format, Chart-Serien-Kollisions-
   freiheit (mehrere Tiles/GROUP_M erzeugen **verschiedene** Punkte). Palette ggf. erweitern (s. Risiken).

### 7.5-4 · Testlauf-Verwaltung: ansehen · vergleichen · umbenennen · löschen (Nutzer-Punkt 4)
1. **`schema.py`:** `RunResult` additiv um `run_id: Optional[str]=None`, `run_name: Optional[str]=None`,
   `created_at: Optional[str]=None` (Batch-Zeitstempel, **stabiler** Gruppen-Sortierschlüssel; separat vom
   per-Zeile `provenance.timestamp`).
2. **`run.py` + `callbacks.py`:** `run(config, ..., run_id=None, run_name=None)` reicht die Batch-Identität
   durch (`_result` setzt sie); **`execute_run` vergibt EINEN `run_id` (uuid4) + Default-Namen** (Familie+Expr+
   Uhrzeit) **pro „Vergleichen"-Klick**. `created_at` **einmal außen** setzen (nicht je Zeile via `datetime.now`).
   → Definiert „**ein benannter, wieder-ansehbarer Lauf = ein Batch**".
3. **`store.py` — Lesen/Gruppieren:** `read_all(path) -> list[RunResult]` (via `from_dict`, **Objekte**, nicht
   DataFrame; `read_results` bleibt für den Report) + `list_runs(path) -> list[dict]` (nach `run_id` gruppiert:
   id, name, created_at, #Formate, Familie/Expr, n_ok; nach `created_at` sortiert). **Altzeilen** ohne `run_id`
   bekommen beim Lesen einen **synthetischen** Fallback (`expr`+`timestamp`; **jede Altzeile = eigener Lauf** —
   konservativ, keine unzuverlässige Timestamp-Gruppierung, **kein** Rewrite der git-Datei).
4. **`store.py` — Mutatoren:** `rename_run(run_id, new_name, path)` und `delete_run(run_id, path)` + privater
   `_atomic_rewrite(lines, path)`: alle Zeilen lesen → Gruppe umbenennen/herausfiltern → in **Temp-Datei im
   SELBEN Verzeichnis** (`tempfile.NamedTemporaryFile(dir=path.parent)`) schreiben → `os.replace(tmp, path)`.
   **`delete_run` löscht NUR JSONL-Zeilen, NIE die geteilte `kernels/<slug>.py`** (mehrere Läufe teilen einen
   Slug; `kernels/` ist der gitignored Compile-Cache). Als Modul-Attribut aufrufbar (Test-monkeypatchbar).
5. **`app/components/history.py` (NEU):** reine, Dash-/GPU-frei testbare Komponente — Liste vergangener Läufe
   (`list_runs`) als Mehrfachauswahl (Vergleich), Umbenennen-Feld, Löschen-Button **mit Bestätigung**
   (`dcc.ConfirmDialog`/`dbc.Modal`). IDs als Konstanten; importiert nur `schema`/`store`, **kein**
   `run`/`torch`/`cuda`.
6. **`layout.py` + `callbacks.py`:** History-Panel + benötigte `dcc.Store`(s) einhängen. History-Callbacks
   **NORMAL** (nicht `background=True`, da **GPU-frei**): (a) Liste aus `list_runs`, (b) „Laden" → `read_all` +
   Filter auf `run_id`(s) → `render_comparison` **wiederverwenden**, (c) Umbenennen → `rename_run`, (d) Löschen
   → `delete_run`. **Der Load-Callback braucht `allow_duplicate=True` + `prevent_initial_call` auf
   `Output('main','children')`** (die *eine* nicht-additive Stelle).
7. **Mehrfach-Lauf-Vergleich in `charts.py`:** `_points` trägt `run_name`; Label um Lauf-Name erweitern,
   Lauf-Zugehörigkeit über den in **7.5-2** eingeführten zweiten Kanal (Form/Muster) kodieren (Farbe bleibt
   Format). → baut direkt auf der Serien-Generalisierung aus 7.5-2 auf.
8. **Tests (alle GPU-frei):** `read_all`/`list_runs`/`rename_run`/`delete_run` + atomarer Rewrite gegen eine
   Temp-JSONL unter `$SP` (Rückwärtskompatibilität mit Altzeilen; atomare Ersetzung; `delete_run` fasst keine
   Kernel-Datei an; `rename` trifft nur die Gruppe). **Die $SP-Isolation auf die neuen Mutatoren ausweiten.**
   History-Kernlogik headless (`list_runs`→Auswahl, Load→`render_comparison`, rename/delete→Store-Aufruf).

### 7.5-3 · n-äres einsum (Nutzer-Punkt 3) — zuletzt
1. **`parse.py` — n-är-Zweig:** **vor** dem 2-Op-Gate (`:280`) bei `len(inputs) > 2` in eine neue
   `NAryContractionIR` (voller Ausdruck + geplante Folge paarweiser Sub-Ausdrücke) münden **statt zu raisen**;
   der 2-Op-Pfad (`_parse_contraction`) bleibt für `len == 2` **byte-identisch**. Router-Erweiterung `:250–259`.
2. **`parse.py` — Pfadplaner:** `opt_einsum.contract_path` **lazy** importieren (nicht Modulkopf); **Fallback =
   deterministischer Links-nach-rechts-Fold** (Operand0⊗1, Ergebnis⊗2, …), der immer korrekt ist. **Jeder
   paarweise Schritt MUSS ein sauberes 2-Op-Kontraktions-Sub-Problem sein** (beide Operanden da, K
   klassifizierbar, keine Wiederholung). **Nicht sauber zerlegbare Schritte: Loud-Fail** (kein still falsch).
3. **`run.py` — n-är-Wrapper:** dritter Zweig/Wrapper: pro paarweisem Schritt eine Sub-`RunConfig` (2-Op-expr,
   gleiche dtype/tile) durch die **bestehende** Maschinerie (`to_canonical`→`load_kernel`) treiben,
   **Zwischentensor auf der GPU halten** (`torch.empty(acc_dtype)`) und als Operand des nächsten Schritts nutzen;
   ein **Composite-launch-Closure** fährt alle Schritte sequenziell für `time_first_launch`/`benchmark`. Der
   2-Op-Zweig (`:319–408`) bleibt **unberührt**. `compute_metrics_nary(steps, run_ms, dtype, acc)`:
   `total_flops = Σ 2·Bᵢ·Mᵢ·Nᵢ·Kᵢ`, `total_bytes` inkl. Zwischentensor-Traffic → **ein** aggregierter
   Roofline-Punkt; `provenance["sizes"]` family-geformt (`operands`/`path`/`total_flops`/`steps[]`).
4. **`verify`:** finale natürliche Output-Shape gegen `verify(out, [alle n Leaf-Operanden], config)` —
   `verify.py` **braucht keine Änderung**. **Optional (stärkeres Netz):** je Zwischenschritt zusätzlich gegen
   `torch.einsum(sub_expr)` prüfen (fängt Fehl-Zerlegung/Layout-Fehler ab — Risiko ①/④).
5. **`schema.py`/`store.py`:** **minimal-additiv** — `RunResult.kernel_source` = **Konkatenation** der
   Step-Quelltexte (Trennkommentare), `kernel_path` = **ein** synthetischer Composite-Pfad
   `kernels/<full-slug>.py`. **`RunConfig` braucht KEINE neuen Felder** (`expr` trägt alles, `op=None`,
   `family='contraction'`). Die paarweisen Step-Kernel werden über ihre **eigenen** 2-Op-Slugs von
   `load_kernel` automatisch persistiert + gecacht (Wiederverwendung, keine Kollision).
6. **`controls.py`:** `validate_expr`/`index_categories` lehnen n-är **nicht** mehr über das Gate ab (parse
   liefert NAry-IR); ein **n-är-Preset** in `FAMILY_PRESETS['contraction']` (z. B. „Kettenprodukt
   `ij,jk,kl->il`"). Größenfelder sind bereits n-är-fähig. `_estimate_bytes` für die NAry-IR ergänzen
   (**OOM-Schutz inkl. Zwischentensoren** — geteilte 32-GiB-Maschine!).
7. **Setup/Tests:** `opt_einsum` auf der Lab-Maschine per `pip install -r requirements.txt` verfügbar machen
   (Fallback-Fold testet auch ohne). `test_rejects_nary` (`test_parse.py:112`) auf das neue Verhalten umstellen
   (bzw. auf einen weiterhin abgelehnten, **nicht** sauber zerlegbaren Fall verschieben). Neu **headless**:
   parse→NAry-IR + Pfad + Per-Step-2-Op-Klassifikation (opt_einsum-Zweig überspringen, wenn nicht installiert);
   `compute_metrics_nary`; `validate_expr` akzeptiert n-är + Preset; `_estimate_bytes`. Neu **GPU** (Lab):
   `run('ij,jk,kl->il')` end-to-end gegen `torch.einsum` verifiziert.

## Setup (erster Schritt)
- **n-är (7.5-3):** prüfe `opt_einsum` im venv (`ls /home/mla07/mla/.venv/lib/python*/site-packages | grep -i
  opt_einsum`). Fehlt es lokal, ist das **kein** Blocker (Fold-Fallback + `torch.einsum`-Referenz laufen ohne);
  auf der Lab-Maschine `pip install -r requirements.txt`. **Import strikt lazy im Worker** — nie im Modulkopf.
- **Verifiziere zuerst headless/GPU die tragenden Bausteine**, bevor du breit baust: ein
  `build_gemm_module(group_m=16)`-Smoke (emittiert `GROUP_M = 16`); ein `torch.einsum("ij,jk,kl->il", …)`
  liefert die n-är-Referenz; `store.read_all` rekonstruiert eine handgeschriebene Temp-JSONL zu `RunResult`s.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)

**Sub-Ziel 1 (Swizzle-GROUP_M):**
1. **Schema:** `swizzle:bool` **beibehalten** + additives `group_m:int=8` (empfohlen — hält alle bool-Verträge
   in Slug/Charts/Controls/Tests) **vs.** `swizzle` zu `int` umbauen (bräche ~6 Dateien + Tests). Kläre.
2. **Grenzen:** Zweierpotenz-Dropdown `{1,2,4,8,16,32}`, Default 8 (empfohlen — konsistent zu den Tile-Dropdowns;
   `GROUP_M=1` = „kein Swizzle"; fixer Deckel, da GROUP_M zur Codegen-Zeit gebacken wird, `num_pid_m=cdiv(M,TM)`
   aber Laufzeitwert ist) **vs.** freier int / andere Range. Kläre Range + Zweierpotenz-Zwang.
3. **Slug:** GROUP_M nur bei `swizzle & !=8` anhängen (empfohlen — Byte-Identität der Default-8-Slugs +
   Kollisionsfreiheit im Compile-Cache) **vs.** immer anhängen. Kläre.

**Sub-Ziel 2 (Multi-Config):**
4. **Was ist „eine Konfiguration", die man hinzufügt/entfernt?** (a) **Tile-Zeilen** (TM/TN/TK) *und*
   **Swizzle-Zeilen** (GROUP_M) als getrennte Listen, Batch = `selection × tiles × swizzle_configs` (empfohlen —
   passt exakt zum heutigen Modell: dtypes schon mehrfach, Swizzle als Achse) **vs.** vollständig unabhängige,
   je einzeln benannte Gesamt-Configs (mächtiger, aber größerer UI-/Naht-Eingriff). **Kläre — treibt das ganze
   UI-Design.**
5. **Chart-Kodierung der Zusatzachse(n)** (Farbe=Format & Symbol=Swizzle sind belegt, Palette am 8er-Limit):
   (a) Symbol=Tile, Muster/Dash=Swizzle-Variante, Farbe=Format, volle Legende/Hover (empfohlen als Primärweg);
   (b) Small Multiples (ein Chart-Block je Tile); (c) UI-Guard „nur EINE Zusatzachse gleichzeitig". Kläre den
   akzeptablen Komplexitätsgrad.
6. **Serien-Key:** `config_slug` als kanonischer Key (empfohlen) **vs.** eigenes zusammengesetztes Key-Schema.
   Kläre (empfohlen: `config_slug` — keine Drift).
7. **Kombinatorik-Deckel:** `|selection|×|tiles|×|swizzle_configs|` Läufe unter **einem** 60-s-Lock — weiche
   Warnung ab N Configs (empfohlen) **vs.** harte Grenze. Kläre N (geteilte Maschine, OOM!).

**Sub-Ziel 4 (Testlauf-Verwaltung):**
8. **Was ist ein „Testlauf"?** Der ganze „Vergleichen"-Batch = ein benannter Lauf (empfohlen) **vs.** eine
   einzelne Zeile. Kläre.
9. **Identität:** additive Felder `run_id/run_name/created_at` **je JSONL-Zeile** (empfohlen — selbst-
   beschreibend, git-diff-bar, `from_dict`-tolerant) **vs.** Sidecar-Index `runs.jsonl`. Kläre.
10. **Altzeilen** (bestehende ~750 ohne `run_id`): beim Lesen **synthetisch** (jede Zeile = eigener Lauf,
    empfohlen — **kein** Rewrite der git-Datei) **vs.** einmalige Migration. Kläre.
11. **`main.children`-Konflikt:** `allow_duplicate=True` + `prevent_initial_call` für den Load-Callback
    (empfohlen, kleinster Eingriff, `render_comparison` unverändert) **vs.** eigener Anzeige-Bereich **vs.**
    zusammengeführter Callback via `dash.ctx`. Kläre.
12. **`delete_run` und Kernel-Dateien:** Kernel-Dateien **unangetastet** lassen (empfohlen — geteilter
    Slug/Cache, gitignored) **vs.** verwaiste Slugs mitlöschen. Kläre.

**Sub-Ziel 3 (n-äres einsum):**
13. **Lauf-Modell:** **ein** `run()` zerlegt intern in N Schritte, misst die Sequenz, aggregiert zu **einem**
    `RunResult`/**einem** Roofline-Punkt (empfohlen — hält 1-run=1-Punkt-Vertrag) **vs.** je Schritt ein
    eigener Punkt. Kläre.
14. **Scope & die `->…`-Mehrdeutigkeit von `abc,bca,cba->…`:** mit **vollem** Output (`->abc`, keine Reduktion)
    ist das ein **n-äres Hadamard-Produkt** permutierter Operanden (eher elementwise); mit **reduziertem**
    Output eine echte Kontraktion. Empfohlen als **erste Scheibe**: das echte **Kettenprodukt `ij,jk,kl->il`**
    (zerfällt sauber in 2 GEMMs, verifizierbarer Demonstrator); nicht sauber zerlegbare Schritte **Loud-Fail**.
    **Kläre, welche n-är-Fälle der Nutzer wirklich braucht** und ob n-är memory-bound (n-är-elementwise,
    kein Index reduziert) mit rein soll.
15. **Pfadplanung:** `opt_einsum.contract_path` wenn importierbar, sonst Links-nach-rechts-Fold (beides,
    empfohlen). Kläre, ob `opt_einsum` als harte Abhängigkeit auf der Lab-Maschine gilt.
16. **n-är = Erweiterung von `family='contraction'`** (empfohlen — kleinere additive Fläche, `op=None`, kein
    neuer emit/controls-Zweig; Preis: `test_rejects_nary` anpassen) **vs.** neue Familie `nary`. Kläre.

## Scope-Grenzen (was TZ 7.5 NICHT tut)
- **Keine Fusion** (Kontraktion + Elementwise-Epilog) — bleibt Zukunftskandidat (A04-Befund 0,98×).
- **Keine neuen dtypes / kein Autotuning** — die dtype-Achse + Tiling sind fertig.
- **Keine neuen Operations-Familien/Ops** (Copy/Transpose als eigene Op, Diagonalen/Spuren) — n-är ist eine
  **Erweiterung** der Kontraktions-Familie, keine neue Familie.
- **Kein Naht-/Charts-Grund-Umbau, kein Store-Format-Wechsel** — alles **additiv**; `run(config)→RunResult`
  bleibt; die eine bekannte, bewusst nicht-additive Stelle ist `allow_duplicate=True` auf `main.children`.
- **Keine Politur/Report-Arbeit** — Theming, Cache-Härtung, Sphinx-Report, cli-Batch-Sweeps sind **TZ 8**
  (dieses TZ 7.5 *bereitet* TZ 8 nur den erweiterten Stand). Widerstehe Scope-Creep.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess **CUDA-frei**
  (Fork!); `run`/`torch`/`cuda`-Import **niemals** in den Modulkopf ziehen (bleibt lazy, `callbacks.py:234`).
  Charts/kpis/controls-Logik bleiben **reine, headless-testbare** Funktionen. `execute_run` **wirft nie**.
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla07/mla/.venv/bin/python` (Shell-`python` nicht im
  PATH; Shell-State persistiert nicht zwischen Bash-Aufrufen). Start: `python -m tool_pipeline` (GUI),
  `python -m tool_pipeline.cli` (headless). **Tests:** je Datei standalone `.venv/bin/python tests/test_X.py`
  (aus `project/`) **oder** `python -m pytest tests/` (Dual-Mode ist so gebaut; **kein** conftest/pytest.ini).
- **Harte Regel: NIEMALS `git commit` / `git push`, außer der Nutzer fordert es direkt** (dann OHNE dich selbst
  als Autor/Co-Autor — Repo-Regel in `CLAUDE.md`).
- **Geteilte Maschine:** kleine Größen, `torch.manual_seed(0)`, GPU-Lock (`project/.cache/gpu.lock`) respektieren,
  keine unnötigen GPU-Läufe, **OOM vermeiden** (n-är-Zwischentensoren + Kombinatorik in 7.5-2 sind die neuen
  OOM-Quellen — `_estimate_bytes` erweitern). **Store-Isolation in Tests über `$SP`** (nicht `/tmp`) — **auch
  für die neuen Mutatoren** `rename_run`/`delete_run`; der git-getrackte `results.jsonl` darf durch Tests **nicht**
  verschmutzt werden.
- **verify-before-trust:** kein Kernel-Ergebnis/kein Chart-Punkt ohne bestandene `torch.einsum`-Referenz
  (n-är: gegen den vollen Ausdruck; optional je Zwischenschritt).
- **Byte-Identität ist heilig:** `config_slug`-Erweiterungen strikt **bedingt** (nur bei tatsächlich neuem
  quelltextbestimmendem Wert), damit `results/kernels/*.py` und Cache nicht driften; die Anti-Drift-Tests
  (`test_baselines_not_in_slug`, `test_emit_contraction_header_byte_identical`) bleiben grün.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 7.5 in die **vier Sub-Ziele + geordnete TODOs** zerlegen (Reihenfolge **1 → 2 → 4 → 3**; jedes TODO lässt
   Pipeline **und** App in lauffähigem, prüfbarem Zustand). Die **Design-Entscheidungen oben vorab** mit dem
   Nutzer klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt bewirkt /
   was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären). Dann auf **die
   Validierung des Nutzers warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-7.5-Scope bleiben; Scope-Creep (Fusion, neue dtypes, Politur/Report, neue Familien) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen): das
   **bereits vorbereitete TZ 8** (`prompts/TZ8-politur-report.md`) auf den **neuen Post-TZ-7.5-Stand
   re-synchronisieren** — genau nach *diesem* Muster: gründlich gegenlesen (Workflow), und den TZ-8-Auftrag dort
   aktualisieren, wo TZ 7.5 die Oberfläche verändert hat: „Ist-Zustand POST-TZ7" → **POST-TZ7.5**; die neuen
   Flächen aufnehmen (**`group_m` in Schema/Slug/Charts**, **Multi-Config-Kreuzprodukt**, **n-är-Ergebnisse in
   `results.jsonl`** — Report-Figuren/cli-Sweeps müssen sie abbilden, **mutabler Store + `run_id`/History** —
   Cache-Härtung/Fehlerzustände/Theming müssen History-Panel & atomaren Rewrite berücksichtigen). **Diese
   Arbeitsweise inkl. der zwei Zusätze** (sehr einfache Erklärung nach jedem TODO + Vorbereitung/Re-Sync des
   nächsten TZ als letzter Schritt) **weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**, wo möglich:
- **1 (Swizzle):** `build_gemm_module(group_m=16)` emittiert `GROUP_M = 16` (headless); `config_slug`-Byte-
  Identität bei `group_m=8`; **GPU:** variabler GROUP_M == kein-Swizzle-Ergebnis auf einer Größe mit partieller
  letzter Gruppe (Bijektivität).
- **2 (Multi-Config):** `tiles_from_state`/`configs_from_selection`-Kreuzprodukt (deterministisch, headless);
  Chart-Serien sind **kollisionsfrei** (mehrere Tiles/GROUP_M → verschiedene Punkte); Charts via `save_png`
  rendern und **ansehen** (dataviz „render & look" — Legende disambiguiert, nichts verschmilzt); App real starten
  (GPU-Lock!) und zwei Tiles × zwei GROUP_M gegeneinander durchklicken.
- **4 (Verwaltung):** `read_all`/`list_runs`/`rename_run`/`delete_run` + atomarer Rewrite gegen Temp-JSONL unter
  `$SP` (Altzeilen-Kompatibilität; `delete` fasst keine Kernel-Datei an; `rename` trifft nur die Gruppe); App
  real: einen Lauf machen, umbenennen, wieder laden, mit einem zweiten vergleichen, löschen — **alles GPU-frei**
  außer dem ursprünglichen Messen.
- **3 (n-är):** parse→NAry-IR + Pfad/Per-Step-Klassifikation + `compute_metrics_nary` (headless, Fold-Fallback
  ohne `opt_einsum`); **GPU:** `run('ij,jk,kl->il')` == `torch.einsum` (gelockerte fp16-Toleranz), erscheint als
  **ein** Roofline-Punkt; optional je Zwischenschritt gegengeprüft.
- **Alle 206 bestehenden Tests bleiben grün** + neue Tests; App-Smoke bleibt grün. Gemischte neue Testdateien
  tragen den `_has_cuda`-Guard (`test_measure`-Muster), damit headless-Läufe ohne GPU durchlaufen. Store in Tests
  über `$SP` isolieren (auch die Mutatoren).

## Definition of Done (TZ 7.5)
Die vier Feedback-Wünsche laufen end-to-end, **additiv und verifiziert**:
1. **GROUP_M ist in der GUI wählbar** (sinnvolle Grenzen), fließt korrekt in den generierten Kernel, in den
   Slug (bedingt, kollisionsfrei) und in KPIs/Code — Default 8 bleibt byte-identisch.
2. **Mehrere Tile- und Swizzle-Konfigurationen** (Zeile hinzufügen/entfernen) werden in **einem** Batch
   gegeneinander gemessen und in den drei Charts **eindeutig** (disambiguiert) dargestellt — die eine Naht
   (`run()` je Config) bleibt.
3. **n-äres einsum** (mind. `ij,jk,kl->il`) läuft über paarweise Zerlegung durch den bewiesenen GEMM-Pfad,
   **gegen `torch.einsum` verifiziert**, als **ein** aggregierter Roofline-Punkt; nicht sauber zerlegbare Fälle
   scheitern **laut** (kein still falsch).
4. **Alte Läufe** lassen sich **ansehen, vergleichen, umbenennen und löschen** (GPU-frei aus dem Store), der
   Store ist erstmals sicher **mutabel** (atomarer Rewrite), Altzeilen bleiben lesbar, `results.jsonl` wird durch
   Tests nie verschmutzt.

**Alle 206 bestehenden Tests grün** + neue Randfall-/Slug-/Metrik-/Store-/Controls-Tests; App-Smoke grün; die
Kontraktions-/memory-bound-Familien bleiben unverändert. **Zusätzlich:** nach jedem TODO gab es eine sehr
einfache Erklärung, und als letzter Schritt ist **TZ 8 auf den Post-TZ-7.5-Stand re-synchronisiert**.

---

## Session-Prompt (zum Starten von TZ 7.5 — dies dem Assistenten geben)

> Lies zuerst VOLLSTÄNDIG die Datei
> `project/project-development/prompts/TZ7.5-verbesserungen.md` — das ist dein Auftrag (Zwischen-Teilziel
> TZ 7.5 des „cuTile Performance Lab": Swizzle-GROUP_M einstellbar · mehrere Tile-/Swizzle-Konfigurationen
> vergleichen · n-äres einsum · Testlauf-Verwaltung). Befolge die dort unter „Arbeitsweise (verbindlich)"
> beschriebene Vorgehensweise strikt:
>
> Belese dich zuerst gründlich in die bestehende, verifizierte Implementierung (TZ 1–7) — genau die Dateien/
> Reihenfolge unter „Zuerst lesen" — und bestätige dein Verständnis kurz. (Paralleles Lesen per Subagenten/
> Workflow ist erwünscht.) Zerlege TZ 7.5 in die vier Sub-Ziele + geordnete TODOs (empfohlene Reihenfolge
> 1 → 2 → 4 → 3) und kläre die „Design-Entscheidungen" vorab mit mir. Lege die Aufschlüsselung zur FREIGABE vor,
> BEVOR du Code schreibst. Dann TODO für TODO: nach jedem anhalten und zeigen (a) was du getan hast, (b) wie du
> es verifiziert hast, UND (c) eine SEHR EINFACHE Erklärung in Alltagssprache. Dann auf meine Validierung warten.
> Nicht mehrere TODOs bündeln. Harte Regeln: NIEMALS git commit/push (außer ich fordere es direkt; dann ohne dich
> als Autor/Co-Autor); Prosa/Kommentare/UI auf Deutsch; aus `project/` mit dem venv-Python `.venv/bin/python`
> ausführen; die eine Naht + Fork-Sicherheit wahren (kein run/torch/cuda im Modulkopf); Byte-Identität der Slugs
> heilig; verify-before-trust; Store-Isolation in Tests über `$SP` (auch die neuen Mutatoren). Als letzter
> Schritt: TZ 8 (`TZ8-politur-report.md`) auf den Post-TZ-7.5-Stand re-synchronisieren. Leg los mit dem Einlesen.
