# Auftrag: TZ 7 — Operationen-Breite II: memory-bound (Elementwise + Reduktion)

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen).
Wir bauen die Group-Specific Component „**cuTile Performance Lab**" (interaktiver
einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–6 sind fertig und verifiziert:** die
headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → **reshape/B1** → emit → compile+Cache → Kalt-Lauf=compile_ms → verify(fp32) →
benchmark → Metriken → Baselines → GPU-Zustand → Store); die Dash-GUI fährt den Live-Loop
als Batch-Vergleich (**Ausdruck via Presets/Freitext + Größen je Index** · Tile TM/TN/TK ·
L2-Swizzle · Zahlenformate · Baselines) → je Config ein `run()` unter einem GPU-Lock →
KPIs/Verify/Code + **drei** Charts (Durchsatz · Genauigkeit↔Durchsatz · Roofline). **Jede
2-Operanden-Kontraktion** (transponiert, Batched, mehrdimensionale M/N/K, allgemeine
Tensor-Kontraktion) läuft end-to-end über den **echten B1-Reshape** (config/optimizer-getrieben)
auf die kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)`, vom **batched** Codegen emittiert, gegen
`torch.einsum` verifiziert, mit korrekten Batch-Metriken; die Roofline zeigt die Punkte
memory- vs compute-bound.

**ABER:** die gesamte Pipeline ist **hart auf 2-Operanden-Kontraktion mit K-Achse (Tensor-Core)**
verdrahtet, und `family` wird im App-Pfad **nie gesetzt** (immer Default `"contraction"`). Dein
Auftrag ist **ausschließlich Teil-Ziel 7 (TZ 7): die beiden memory-bound-Familien Elementwise
und Reduktion** als eigene Codegen-Templates, über echtes **Familien-Routing**, sodass sie als
**memory-bound Punkte auf der Roofline** erscheinen (der ehrliche Kontrast zur compute-bound
Kontraktion). Das erfüllt die Scope-Entscheidung „vollständiges Operations-Menü" (PLAN §2/§10).

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1** (f-String-Templates → `@ct.kernel`-Quelltext, ein Modul je Operations-Familie).
  Die neuen Familien bekommen **eigene Template-Module** — die Routing-Naht (`emit.py`) steht schon.
- **memory-bound = KEIN Tensor-Core**: Elementwise/Reduktion haben **kein `ct.mma`, keinen
  FP32-Akku-Loop, keinen B1-Reshape**. Elementwise = Load→Op→Store; Reduktion = Load→`ct.sum(axis)`→Store.
  Das ist der ganze Punkt der Familie (niedrige arithmetische Intensität ⇒ Roofline **weit links**).
- **Bewiesene Port-Vorlagen (A02, fix):**
  - **Reduktion** = `assignments/02_assignment/src/task_02.py`: zeilenweise Summe `(M,K)→(M,)` über
    `axis=1`. **Idiom: `ct.sum(tile, axis=1)`** (single-shot), `grid=(M,1,1)`, ein Block je Zeile,
    `TILE_K` = nächste Zweierpotenz ≥ K, ZERO-Padding neutral für die Summe.
  - **Elementwise** = `assignments/02_assignment/src/task_04.py` (Copy/unär, **echtes cdiv-2D-Grid**)
    als Skelett + `task_03.py` (binäre Add) für die Op. Load→`a (op) b`→Store, `grid=(cdiv(M,TM),cdiv(N,TN),1)`.
    **Bewiesen (task_03-Benchmark): die kontiguierten inneren Achsen kacheln** (~3,5× schneller).
- **`ct.sum` ist nativ** (kein manueller Akku-Loop) — solange die reduzierte Achse in eine gepaddete
  Kachel passt. Großes K (Achse > max. Kachel) ⇒ **einzige nicht in A02 bewiesene Stelle** → als
  Fallback das GEMM-K-Loop-Muster (`acc`-Tile + Schleife, `contraction.py`) mit `acc += ct.sum(...)`
  bauen und **klar als solches markieren** (verify-before-trust!).
- **Für memory-bound ist GB/s die Primärmetrik** — `gbps` **und** `arithmetic_intensity` +
  `percent_peak_bw` liegen **bereits** in `metrics.compute_metrics` vor und sind family-agnostisch.
  TZ 7 zählt nur `flops`/`bytes` familienweise anders (kein `2·B·M·N·K`).
- **verify-before-trust bleibt Gesetz**: jeder generierte Kernel wird gegen `torch.einsum`
  (unär **oder** binär) geprüft, **bevor** Zahlen angezeigt werden.
- **Die Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer);
  Haupt-Prozess CUDA-frei; Charts reine, headless-testbare Funktionen. **Kein Naht-Umbau** —
  Schema-Erweiterungen nur additiv (`family` ist schon im Vertrag).
- **Erweitern statt neu bauen:** TZ 7 spiegelt das f-String-Template-Muster von
  `contraction.build_gemm_module` und portiert die A02-Kernel. Lies den bestehenden Code **und**
  die A02-Quellen und übernimm deren Muster/bewiesene Orientierung.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§10 „TZ 7"** (Z. 149–152: DoD/TODOs/„schaltet frei"),
   **§3** (C1-Codegen, „kleiner Klassifikator auf M/N/K/C routet die Familie", Elementwise/Reduktion-
   A02-Templates), **§5** (Bandbreite 273 GB/s = die Roofline-Schräge, an der memory-bound-Punkte
   liegen), **§2** (Scope: memory-bound „macht die Roofline aussagekräftig").
2. `assignments/02_assignment/src/task_02.py` (Reduktion), `task_03.py` (Elementwise binär),
   `task_04.py` (Elementwise unär/Copy, **cdiv-Grid**) — die **Port-Vorlagen**. `task_01.py` ist
   nur Device-Properties (irrelevant).
3. `project/tool_pipeline/codegen/templates/contraction.py` — `build_gemm_module` (Z. 57–207): das
   **f-String-Builder-Muster** (Signatur, Tile-Literale einbacken, `bid_block`/`cast_block`-
   Substitution, `launch(A,B,C)`, `if __name__=="__main__"`-GPU-Selbsttest gegen torch). Die neuen
   Templates spiegeln genau diese Struktur — **ohne** `ct.mma`/K-Loop/FP32-Akku.
4. `project/tool_pipeline/codegen/templates/elementwise.py` + `reduction.py` — **1-Zeilen-Stubs**
   („Vorlage: A02 task_03" bzw. „task_02"). **Hier wachsen die neuen Builder.**
5. `project/tool_pipeline/codegen/emit.py` (Z. 42–49) — der Familien-Router
   (`if family=="contraction" … else NotImplementedError`). **Hier additiv `elif`-Zweige.**
6. `project/tool_pipeline/intermediate_representation/parse.py` — das family-Gate (Z. 109–114) +
   das **„genau 2 Operanden"-Gate (Z. 132–136)** + `ContractionIR` (M/N/K/Batch, Z. 41–97).
   Die K-Klassifikation (Z. 181: „in beiden Operanden, nicht im Output") passt **nicht** für
   1-Operanden/kein-K. Hier: Familien-Router **vor** dem 2-Operanden-Gate + leichte
   `ElementwiseIR`/`ReductionIR` (oder generalisierte Klassifikation).
7. `project/tool_pipeline/run.py` — der Orchestrator. Kontraktions-spezifisch:
   `_build_natural_operands` (**2 Tensoren**, Z. 66–94), `_build_inputs` (kanonisch `(B,M,K)`,
   Z. 97–120), `to_canonical`/B1-View (Z. 164 — für memory-bound **überspringen/Passthrough**),
   Launch-Arity `launch(A_c,B_c,C_c)` (Z. 197/221 — **fest 2 Inputs**), `verify(C_nat,A_nat,B_nat,…)`
   (Z. 207 — **fest 2 Operanden**), `compute_metrics(M,N,K,…)` (Z. 230), `provenance["sizes"]`
   M/N/K (Z. 166). **Alles familienweise zu generalisieren.**
8. `project/tool_pipeline/measure/metrics.py` — `gemm_flops`/`gemm_bytes` (Z. 16–34, GEMM-spezifisch)
   + `compute_metrics` (Z. 51–87). `gbps` (Z. 44–48) + `arithmetic_intensity` (Z. 77) +
   `percent_peak_bw` (Z. 78) sind **schon da, family-agnostisch**. Neu: `elementwise_bytes/flops`,
   `reduction_bytes/flops` + Family-Dispatch.
9. `project/tool_pipeline/measure/verify.py` — `verify(output, A, B, config)` (Z. 70–71) +
   `torch.einsum(config.expr, A.float(), B.float())` (Z. 85). `torch.einsum` kann **unär**;
   die **Signatur** verlangt aber 2 Tensoren. Neu: variadische Operanden (`*operands`). Die
   Toleranztabelle (Z. 37–56, `(dtype,acc)`-gekeyt) ist family-neutral (ggf. straffere Gates).
10. `project/tool_pipeline/app/components/controls.py` — `PRESETS` (Z. 62–68, nur Kontraktion),
    `config_from_controls`/`configs_from_selection` (Z. 237–243/302–328, setzen **kein `family`**),
    `validate_expr`/`index_categories` (Z. 154–185, scheitern bei 1 Operand), `validate_dim_sizes`
    (Z. ~229, OOM-Schätzung GEMM-geformt). `callbacks.py` (Z. ~201) reicht `family` **nicht** durch.
11. `project/tool_pipeline/app/components/charts.py` — `_points` (Z. 62–104) + `figure_roofline`
    (Z. 415–542) sind **family-agnostisch** (nur `tflops`/`arithmetic_intensity`/`gbps`/`dtype`):
    memory-bound-Punkte (niedrige AI) landen **automatisch links**. **Kein Muss-Umbau** — optional
    im Hover `percent_peak_bw` statt `percent_peak_flops` für memory-bound betonen.
12. `project/tool_pipeline/schema.py` (Z. 83 `family`-Feld, Z. 104–111 `__post_init__` — trägt 1
    Operanden schon) + `project/tool_pipeline/codegen/compile.py` (Z. 70–76 `launch(A,B,C)`-
    Consumer-Konvention → Arity generalisieren; `store/store.py` `config_slug` ist family-**un**abhängig).
13. `project/tool_pipeline/intermediate_representation/README.md` (Z. 44–50/67–70: Familien-Routing
    = „TZ 6/7"; memory-bound braucht **keinen** B1-Reshape) + `project/tests/` (die vorhandenen
    standalone-Runner als Test-Muster; `test_reshape.py`/`test_parse.py`/`test_codegen.py`/
    `test_measure.py`/`test_app_controls.py`/`test_app_execute.py`).

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ6)
**Familien-Naht steht als Guard, Logik fehlt:** `emit.py:42-49` und `parse.py:109-114` werfen für
`family != "contraction"` bewusst `NotImplementedError("… ist TZ 7.")`; die Template-Stubs
`elementwise.py`/`reduction.py` sind 1-Zeilen-Docstrings. `RunConfig.family` (`schema.py:83`)
erlaubt laut Kommentar `"elementwise"/"reduction"`, wird aber **nirgends gesetzt** — `controls.py`/
`callbacks.py` bauen jede Config ohne `family` (⇒ Default contraction).

**contraction-spezifische Annahmen (die generalisiert werden müssen):**
- **2 Operanden**: `parse.py:132`, `verify.py:70/85`, `run.py:66/97/207`, `contraction.py:194-201`
  bzw. die `launch(A,B,C)`-Konvention in `compile.py:70`.
- **K-Kontraktion / mma**: die ganze mma-Schleife `contraction.py:179-186`, `parse.py:181`,
  `metrics.gemm_flops/bytes:16-34`, `reshape.to_canonical:110-111`.
- **family-Routing befüllen**: `emit.py:42`, `parse.py:110`, und v. a. das **Setzen** von `family`
  im App-Pfad (`controls.py:243/322-327`, `callbacks.py:~201`), das heute komplett fehlt.

**Schon bereit (nicht anfassen / nur nutzen):** `gbps`/`arithmetic_intensity`/`percent_peak_bw` in
`metrics` (family-agnostisch); `_points`/`figure_roofline` (family-agnostisch → memory-bound-Punkte
erscheinen automatisch links); `RunConfig.family` + `__post_init__` (1 Operand ok); `config_slug`
(family-unabhängig, neuer `expr` ⇒ eigener Slug, keine Kollision); die dtype/acc-Regeln + `COMBOS`.

**Port-Kern (A02, aus der Analyse):**
- **Reduktion** `build_reduction_module`: `@ct.kernel def row_sum(mat, output, TILE_K): pid=ct.bid(0);
  tile=ct.load(mat, index=(pid,0), shape=(1,TILE_K), padding_mode=ZERO); ct.store(output, index=(pid,),
  tile=ct.sum(tile, axis=1))`; `launch(A, C)`, `TILE_K`=next-pow2(K), `grid=(M,1,1)`.
- **Elementwise** `build_elementwise_module`: `i=bid(0); j=bid(1); a=load(A,(i,j),(TM,TN),ZERO);
  b=load(B,(i,j),(TM,TN),ZERO); ct.store(C,(i,j), tile=ct.astype(a (op) b, C.dtype))`;
  `launch(A,B,C)` (binär) bzw. `launch(A,C)` (unär), `grid=(cdiv(M,TM),cdiv(N,TN),1)`. Op als
  f-String-Fragment substituieren (analog `cast_block`/`bid_block`). **Kein `ct.mma`, kein FP32-Akku.**

## TZ-7-Scope (eng halten!)
1. **`parse.py`**: Familien-Router (bei `family`-RunConfig auf elementwise/reduction verzweigen,
   **vor** dem 2-Operanden-Gate); 1-Operanden-Ausdrücke zulassen (`ij->ij`, `ij->i`); leichte
   `ElementwiseIR`/`ReductionIR` (Achsen-Größen; Reduktion: `kept_dims`/`reduced_dims`).
2. **`codegen/templates/elementwise.py` + `reduction.py`**: die zwei Builder nach dem
   `build_gemm_module`-Muster (f-String, Tile-Literale, `__main__`-Selbsttest), Port aus A02.
3. **`codegen/emit.py`**: `elif family=="elementwise"/"reduction"` → neue Builder; `_header` so, dass
   fehlende Tile-Keys (kein TN/TK) sauber angezeigt werden.
4. **`codegen/compile.py`**: `launch`-Arity generalisieren (`launch(A,C)` für 1-Operanden-Familien).
5. **`run.py`**: family-abhängige Zweige — Operanden in natürlicher Ausdruck-Form (1 vs. 2), **kein**
   B1-GEMM-Reshape für memory-bound, Output-Shape aus dem Ausdruck, Launch-/Verify-Arity,
   family-spezifische Metrik-Signatur; `provenance["sizes"]` family-geformt.
6. **`measure/verify.py`**: variadische Operanden (`torch.einsum(expr, *ops)` mit 1 oder 2 Tensoren).
7. **`measure/metrics.py`**: `elementwise_bytes/flops`, `reduction_bytes/flops` + Family-Dispatch;
   `gbps`/`arithmetic_intensity` (schon da) weiternutzen — GB/s ist die Primärmetrik.
8. **`app/components/controls.py` (+ `callbacks.py`)**: Familien-Auswahl (contraction/elementwise/
   reduction) + family-Presets (`ij->ij`, `ij->i`, …) + `family` durch bis in die `RunConfig`
   durchreichen; `validate_expr`/`index_categories`/OOM-Schätzung family-abhängig (1 Operand zulassen).
9. **Tests**: neue standalone-Runner (`test_codegen` erweitern für die neuen Templates; ein
   `test_measure`-Zweig für elementwise/reduction-flops/bytes; `test_parse`/`test_app_controls` für
   die family-Pfade) + headless, wo möglich; GPU-Selbsttests der Templates.

## Setup (erster Schritt)
Vermutlich **keine neuen Pakete** (torch/cuTile da; `ct.sum` ist im Build vorhanden — task_02 beweist
es). **Verifiziere headless/GPU zuerst**, dass die Bausteine tragen: ein winziger `ct.sum(axis=1)`-
Kernel läuft (spiegelt task_02); `torch.einsum("ij->i", A)` und `torch.einsum("ij,ij->ij", A, B)`
liefern die Referenz. Falls `ct.sum` für große K nicht in eine Kachel passt: den K-Loop-Fallback
(GEMM-Muster) bauen **und markieren**.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Op-Umfang Elementwise:** binär `add`/`mul` (`ij,ij->ij`) + unär `copy` (`ij->ij`)? Wie wird die
   Op gewählt — der einsum-Ausdruck gibt nur die **Struktur** her (nicht „relu"/„mul"): ein `op`-Feld/
   Preset (empfohlen: kleine Menge `add`, `mul`, `copy`) vs. rein additive Presets. Kläre.
2. **Reduktions-Umfang:** nur zeilenweise Summe (`ij->i`, = task_02, empfohlen als Kern) + volle
   Summe (`ij->`)? Andere Achsen (`ij->j`)? Nur `sum` oder auch `max`/`mean`? Kläre (empfohlen: `sum`
   über beliebige reduzierte Achsen, single-shot, mit dokumentiertem Large-K-Fallback).
3. **1-Operanden-parse:** family-Router **vor** dem 2-Operanden-Gate + eigene leichte IRs
   (empfohlen) vs. `ContractionIR` generalisieren. Kläre.
4. **run.py-Generalisierung:** minimale family-Zweige (memory-bound überspringt B1/Kanonisierung;
   Operanden aus natürlicher Ausdruck-Shape; Launch/Verify variadisch) — empfohlen, additiv. Kläre,
   ob ein sauberes `OpPlan`-Objekt (family, operand-shapes, output-shape, launch-arity) den Ablauf
   klarer macht als ein `if family==…`-Baum.
5. **verify variadisch:** `verify(output, operands: list, config)` (empfohlen) vs. `*operands`. Kläre.
6. **Metrik-Zählung:** Elementwise `add` = 1 FLOP/Element, bytes = read(Inputs)+write(Output);
   Reduktion `ij->i` = `M·N` Adds, bytes = read(A)+write(C). AI ⇒ sehr niedrig (das ist die Aussage).
   Kläre, ob `copy` als „0 FLOP" (reine Bandbreite) oder `1` gezählt wird.
7. **GUI:** Familien-Dropdown + per-Familie-Presets + (für Elementwise) Op-Auswahl; `validate_expr`/
   `index_categories` family-abhängig. Kläre den Umfang (wie viele Presets/Ops).
8. **Roofline-Hover (optional):** für memory-bound `percent_peak_bw` statt `percent_peak_flops`
   betonen? Kläre (Charts brauchen sonst **keinen** Umbau).

## Scope-Grenzen (was TZ 7 NICHT tut)
- **Keine Fusion** (Kontraktion + Elementwise-Epilog) — Zukunftskandidat (A04-Befund 0,98×).
- **Kein n-äres einsum**, keine Diagonalen/Spuren (wie in TZ 6 bewusst draußen).
- **Keine neuen dtypes/Autotuning** — die dtype-Achse + Tiling sind fertig; Elementwise/Reduktion
  nutzen dieselben Formate (Akku ist für sie faktisch bedeutungslos — im Builder ignorieren, **nicht**
  über `_ACC_DTYPE_MAP` erzwingen).
- **Kein Umbau der Kontraktions-Pipeline** — additiv (neue Familien-Zweige/Templates). Die GEMM-
  Familie bleibt byte-nah/unverändert; alle bestehenden Tests bleiben grün.
- **Kein Charts-/Schema-/Slug-Umbau** — Roofline/`_points`/`config_slug` profitieren automatisch.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess
  CUDA-frei; Charts reine, headless-testbare Funktionen.
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla07/mla/.venv/bin/python` (Shell-`python`
  nicht im PATH; Shell-State persistiert nicht zwischen Bash-Aufrufen — venv-Pfad direkt nutzen; die
  Tests fügen `project/` selbst in `sys.path` ein, laufen also aus jedem cwd). Start:
  `python -m tool_pipeline` (GUI), `python -m tool_pipeline.cli` (headless).
- **Harte Regel: NIEMALS `git commit` / `git push`** in diesem Repo.
- **Geteilte Maschine:** kleine Größen, `torch.manual_seed(0)`, GPU-Lock respektieren, keine
  unnötigen GPU-Läufe, OOM vermeiden (OOM crasht die Maschine). **Store-Isolation in Tests:** über
  die Umgebungsvariable `SP` (Test-`_redirect_store` schreibt nach `$SP/…jsonl`) — **nicht** nach
  `/tmp` (dort kollidieren parallele Jobs/Fremdprozesse → `PermissionError`); der git-getrackte
  `project/results/results.jsonl` darf durch Tests **nicht** verschmutzt werden.
- **verify-before-trust:** kein Kernel-Ergebnis ohne bestandene `torch.einsum`-Referenz in die
  Anzeige (Elementwise/Reduktion: unär **oder** binär). Roofline-Punkte nur aus `status=="ok"`.
- **dataviz-Skill** als Maßstab; memory-bound-Punkte müssen klar **links** (niedrige AI) sitzen.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 7 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App
   in lauffähigem, prüfbarem Zustand — z. B. parse-family-Router → reduction-Template (GPU-Selbsttest
   gegen `torch.sum`) → elementwise-Template → emit/compile-Arity → run-family-Zweige → metrics-family
   → controls-Familienauswahl → Tests). Die Design-Entscheidungen oben **vorab** mit mir klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt
   bewirkt / was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären).
   Dann auf **meine Validierung warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-7-Scope bleiben; Scope-Creep (Fusion, n-är, neue dtypes, Autotuning) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 8 (Politur, Robustheit & Report) — anschauen, vorbereiten und einen
   Session-Prompt + Planungs-MD erstellen** — genau nach *diesem* Muster: gründlich einlesen (Workflow),
   **PLAN §10/TZ 8 maßgeblich** (professionelles Theming/Layout; Randfälle nicht-teilbare Dims →
   padden/maskieren; Compile-Cache gehärtet; Fehlerzustände sauber; **Sphinx-Report** aus
   `results.jsonl` + `projektplan.rst` aus PLAN; `cli.py`-Batch-Sweeps für Report-Plots; `tests/
   test_measure.py`), Anker aus dem *dann* aktuellen Post-TZ7-Ist-Zustand, MD unter
   `project/project-development/prompts/TZ8-*.md`, und **diese Arbeitsweise inkl. der zwei Zusätze
   (sehr einfache Erklärung nach jedem TODO + Planung des übernächsten TZ als letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**, wo möglich: `parse()` klassifiziert
`ij->ij`/`ij->i` korrekt (family, kept/reduced); die Metrik-Zählung (elementwise/reduction bytes/flops)
ist deterministisch prüfbar; die Controls-Naht baut aus Familie+Ausdruck+Größen die richtige
`RunConfig` (mit `family`!). Die **Templates** GPU-verifizieren: der generierte Elementwise-Kernel ==
`A (op) B` bzw. `A` (torch), der Reduktions-Kernel == `torch.sum(A, dim=…)` (gelockerte fp16-Toleranz).
Zusätzlich die App real starten (GPU-Lock beachten!) und je eine Elementwise-/Reduktions-Op live
durchklicken: verifiziert, erscheint in den Charts, und der Punkt sitzt **memory-bound (weit links)**
auf der Roofline — der Kontrast zur compute-bound Kontraktion. Charts via `save_png` rendern und
**ansehen** (dataviz „render & look"). Store in Tests über `$SP` isolieren.

## Definition of Done (TZ 7)
**Elementwise und Reduktion laufen als eigene Familien** end-to-end: über echtes Familien-Routing
(parse/emit/run) auf ihre memory-bound-Templates (kein `ct.mma`, `ct.sum`-Idiom für Reduktion, cdiv-
Grid für Elementwise), gegen die `torch.einsum`-Referenz (unär/binär) **live verifiziert**, mit
family-korrekten Metriken (GB/s als Primärmetrik); die GUI hat eine **Familien-Auswahl + Presets**;
die drei Charts inkl. Roofline zeigen die neuen Ops als **memory-bound Punkte (weit links vom Ridge)**
— der Kontrast zur compute-bound Kontraktion ist sichtbar (die zentrale Roofline-Aussage wird reicher);
alle Tests grün (inkl. neuer Template-/Metrik-/parse-/controls-Tests) + App-Smoke; die Kontraktions-
Familie bleibt unverändert. **Zusätzlich:** nach jedem TODO gab es eine sehr einfache Erklärung, und
als letzter Schritt ist **TZ 8 vorbereitet** (Planungs-MD + Session-Prompt).
