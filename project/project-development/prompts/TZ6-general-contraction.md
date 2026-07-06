# Auftrag: TZ 6 — Allgemeine 2-Operand-Kontraktion (echter B1-Reshape)

Du arbeitest im Repo `/home/mla08/MLA/mla` (bzw. dem aktuellen Checkout, z. B. `/home/mla07/mla` —
egal, Pfade relativ nehmen). Wir bauen die Group-Specific Component „**cuTile Performance Lab**"
(interaktiver einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–5 sind fertig und verifiziert:** die
headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → **reshape/B1** → emit → compile+Cache → Kalt-Lauf=compile_ms → verify(fp32) → benchmark →
Metriken → Baselines → GPU-Zustand → Store); die Dash-GUI fährt den Live-Loop als Batch-Vergleich
(Größen · Tile TM/TN/TK · L2-Swizzle · Zahlenformate · Baselines) → je Config ein `run()` unter einem
GPU-Lock → KPIs/Verify/Code + **drei** Charts (Durchsatz · Genauigkeit↔Durchsatz · **Roofline**). Die
Mess-Schicht ist vollständig, Baselines (cuBLAS + naive) sind zuschaltbar, jeder Kernel wird live gegen
fp32 verifiziert, und die Roofline (TZ 5) zeigt memory- vs compute-bound.

**ABER:** der Ausdruck ist bisher **fest `ik,kj->ij`** (Plain-GEMM). `parse.py` akzeptiert nur genau 2
Operanden mit explizitem Output ohne Diagonalen; `reshape.to_canonical()` ist ein **reiner Passthrough**
(Batch=1) und lehnt alles ab, was eine echte Umformung bräuchte. Dein Auftrag ist **ausschließlich
Teil-Ziel 6 (TZ 6): die allgemeine 2-Operand-Kontraktion über den echten B1-Reshape.**

TZ 6 macht **B1 tragend**: *jede* 2-Operanden-Kontraktion (z. B. `ki,kj->ij` mit transponiertem A,
`bik,bkj->bij` Batched GEMM, `acspx,bspy->abcyx` allgemeine Tensor-Kontraktion) wird host-seitig auf die
kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)` gebracht ⇒ der Codegen emittiert weiter **eine** bewiesene
Struktur. Die Roofline (TZ 5) bekommt dadurch reichere compute-bound-Punkte; die memory-bound-Seite
kommt mit TZ 7.

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1 + B1** (PLAN §3): C1 = f-String-Templates → `@ct.kernel`-Quelltext; **B1 = host-seitiger
  Reshape** jeder Kontraktion auf kanonisches Batched-GEMM. TZ 6 füllt B1 aus — **kein** neuer Codegen-Ansatz.
- **Kanonische Zielform (fix):** `(B,M,K)×(B,K,N)→(B,M,N)`. M/N/K sind die **fusionierten** Größen
  (Produkt je Kategorie), B das Produkt der Batch-Indizes. Der Codegen emittiert diese eine Struktur.
- **M/N/K/Batch-Klassifikation ist bereits allgemein** (`parse.py`, POST-TZ1): Batch = Index in *beiden*
  Operanden **und** Output; K = in beiden, **nicht** im Output; M = in Operand 0 + Output, nicht in 1;
  N = in Operand 1 + Output, nicht in 0. **NICHT neu herleiten** — nutzen; TZ 6 setzt darauf auf.
- **Bewiesene mma-Orientierung (fix, Risiko ①):** `a=(TM,TK)`, `b=(TK,TN)`, `ct.mma(a,b,acc)→(TM,TN)`,
  **KEIN** Operanden-Swap, **KEIN** Permute im Kernel (drei unabhängige GB10-Verifikationen). Der
  **Reshape passiert host-seitig** (View/Permute), der Kernel bleibt bei dieser Orientierung.
- **verify-before-trust bleibt Gesetz** (Risiko ①/④): jeder Ausdruck wird gegen die fp32-`torch.einsum`-
  Referenz geprüft, **bevor** Zahlen angezeigt werden. Die Referenz in `verify.py` ist **schon allgemein**
  (`torch.einsum(config.expr, A.float(), B.float())`) — sie braucht **keine** Änderung, nur die richtig
  geformten Operanden.
- **Die Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer wie
  `hardware`); Haupt-Prozess CUDA-frei; Charts reine, headless-testbare Funktionen. **Kein Naht-Umbau** —
  Schema-Erweiterungen nur **additiv** (neue Felder/Familien, kein Umbau).
- **Erweitern statt neu bauen:** TZ 6 portiert aus `assignments/05|06/src/config.py` + `optimizer.py`
  (die bekannte IR-Transform-Logik: `split_dim` = Tile-Injektion, fuse/permute) und baut auf `parse.py`/
  `reshape.py` auf. Lies den bestehenden Code **und** die A05/06-Quellen und übernimm deren Muster.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§10 „TZ 6"** (Z. 144–147: DoD/TODOs/„schaltet frei"),
   **§3** (B1-Beschreibung: Reshape auf `(B,M,K)×(B,K,N)→(B,M,N)`, Familien-Routing auf M/N/K/C),
   **§4** (Wiederverwendung A05/06 `config.py`/`optimizer.py`, „Lücke: nur 2 Operanden"), **§6** (Risiken
   — hier zentral: ① mma-Orientierung, ④ **B1-Reshape muss korrekter zero-copy View sein**, ⑤ nicht-
   teilbare Dims → padden/maskieren, ⑧ Familien-Routing), **§2** (Dash/GUI-Invarianten).
2. `project/tool_pipeline/intermediate_representation/README.md` — die Frontend→Mitte→Backend-Erklärung,
   das „A4-Blatt"-Bild fürs Reshapen, der Implementierungsstand (parse/reshape ✅ TZ 1, config/optimizer ⏳).
3. `project/tool_pipeline/intermediate_representation/parse.py` — `ContractionIR` (M/N/K/Batch, `M/N/K/B`-
   Properties = fusionierte Größen, `is_canonical_gemm()`), `parse()` mit **strenger Validierung**. Hier
   die Verallgemeinerung: impliziter Output, evtl. mehr Operanden-Formen — **additiv, streng bleiben**.
4. `project/tool_pipeline/intermediate_representation/reshape.py` — `Canonical`-dataclass (M/N/K/B,
   `transform_needed`, `ir`; Kommentar „TZ 6 ergänzt hier **additiv** die Operanden-View-Spezifikationen")
   und `to_canonical()` (heute Passthrough, lehnt Nicht-GEMM ab). **Hier wächst der echte B1-Reshape.**
5. `project/tool_pipeline/intermediate_representation/config.py` **und** `optimizer.py` — **Stubs**
   (nur Docstring). Ziel: Port aus `assignments/05_assignment/src/config.py` + `assignments/06_assignment/src/`
   (bzw. wo A05/06 die IR-Transformation halten). **Lies zuerst die A05/06-Quellen**, dann portiere minimal.
6. `assignments/05_assignment/src/` + `assignments/06_assignment/src/` (`config.py`, `optimizer.py`,
   `kernel.py`) — die **Port-Vorlage** für fuse/split/permute + die Batched-Orientierung (A06 permutiert B
   + tauscht Operanden **nur** wegen dessen `yx`-Output-Layout — Risiko ①; unser Plain-Output braucht das
   NICHT, siehe `RESULTS_gb10.md`).
7. `project/tool_pipeline/run.py` — die Naht: `parse → to_canonical → load_kernel → verify → bench →
   metrics`. **`_build_operands`** baut heute **fest** `A=(M,K)`, `B=(K,N)` — für allgemeine Ausdrücke
   müssen die Operanden in ihrer **natürlichen einsum-Form** (Shape aus `inputs[0]`/`inputs[1]`) gebaut
   werden (sonst stimmt weder `torch.einsum(expr, A, B)` noch der Reshape). Der B1-View erzeugt daraus die
   kanonischen `(B,M,K)`/`(B,K,N)`.
8. `project/tool_pipeline/codegen/templates/contraction.py` — `build_gemm_module` emittiert heute ein
   **2D-Plain-GEMM** (`launch(A,B,C)`, `A=(M,K)`, Grid `(cdiv(M,TM),cdiv(N,TN))`, `bid(0)=M`, `bid(1)=N`).
   **Hier die Batch-Achse ergänzen:** `(B,M,K)×(B,K,N)→(B,M,N)`, 3D-Grid (`bid(2)=Batch`, Offset-Load je
   Batch) — die bewiesene Orientierung/mma **unverändert**. `B=1` muss byte-nah an TZ 1–5 bleiben (Cache!).
9. `project/tool_pipeline/codegen/emit.py` — routet `family=="contraction"` → `build_gemm_module`
   (Routing-Naht steht schon; TZ 6 füllt Kontraktion breiter, TZ 7 die anderen Familien).
10. `project/tool_pipeline/measure/verify.py` — `verify()` nutzt **bereits** die allgemeine
    `torch.einsum(config.expr, …)`-Referenz + eine nach `(dtype,acc)` gekeyte Toleranztabelle. **Keine
    Änderung nötig** — nur die korrekt geformten Operanden füttern.
11. `project/tool_pipeline/measure/metrics.py` — `gemm_flops(M,N,K)`/`gemm_bytes(M,N,K,…)`/`compute_metrics`
    kennen **kein Batch**. **Additiv um B erweitern** (`flops = 2·B·M·N·K`, `bytes ×B`), Default `B=1` →
    TZ 1–5 unverändert. Das hält die Roofline-Punkte (TZ 5) für batched Ausdrücke korrekt.
12. `project/tool_pipeline/app/components/controls.py` — heute feste Config-Anzeige „Ausdruck: ik,kj->ij".
    **Hier die dynamische Operanden-Liste** (Felder je Operand, Auto-Output-Vorschlag, Presets) + die
    Naht-Logik (`config_from_controls`/`configs_from_selection` um den Ausdruck erweitern). Naht-Regel:
    importiert **nur** `schema`.
13. `project/tests/test_reshape.py` — **leerer Stub** (gehört zu TZ 6). Hier die B1-Tests: bekannte
    Ausdrücke → korrekte kanonische Größen + zero-copy-View-Korrektheit (gegen `torch.einsum` numerisch).
14. Memory-Index `MEMORY.md` + relevante Memories (Hardware/dtype-Fakten, GUI-Invarianten, Projektplan,
    `never-git-commit-or-push`). Falls das Memory-Verzeichnis leer ist: die Fakten stehen in `PLAN.md §5`,
    `analysis/RESULTS_gb10.md`, `hardware.py`.

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ5)
**IR (`intermediate_representation/`):**
- `parse.py`: `ContractionIR(expr, inputs, output, m_dims, n_dims, k_dims, batch_dims, dim_sizes)` mit
  `M/N/K/B`-Properties (Produkt je Kategorie) + `is_canonical_gemm()`. `parse()` **lehnt** ab: ≠2 Operanden,
  fehlendes `->` (kein impliziter Output), wiederholte Indizes je Operand (Diagonalen), freie Output-Indizes,
  unbekannte Größen. ⟶ **TZ 6:** impliziten Output ergänzen (einsum-Konvention), Klassifikation bleibt.
- `reshape.py`: `Canonical(M,N,K,B=1,transform_needed=False,ir=None)`; `to_canonical(ir)` = Passthrough,
  wirft `NotImplementedError` für alles Nicht-kanonische. ⟶ **TZ 6:** echten B1-View bauen (Permute+Reshape
  je Operand → `(B,M,K)`/`(B,K,N)`; `transform_needed=True`; Operanden-View-Spezifikation additiv im
  `Canonical`), zero-copy wo möglich (Risiko ④).
- `config.py` / `optimizer.py`: **Stubs** — Port aus A05/06 (`split_dim`=Tile-Injektion, fuse/permute).

**Core (`tool_pipeline/`):**
- `run.py`: Ablauf steht; `_build_operands(dtype,M,N,K)` baut **fest 2D** `(M,K)`/`(K,N)` je dtype-Zweig
  (fp16/bf16/fp32 nativ, tf32=fp32+Kernel-Cast, fp8=fp16→`.to(fp8)`). ⟶ **TZ 6:** Operanden in
  natürlicher einsum-Shape (aus `ir.inputs`) bauen; kanonischen View für den Kernel erzeugen.
- `codegen/templates/contraction.py`: 2D-Plain-GEMM, `launch(A,B,C)`, `A=(M,K)`. ⟶ **TZ 6:** batched
  `(B,M,K)`-Variante (3D-Grid, `bid(2)=Batch`), Orientierung/mma unverändert, `B=1` byte-nah.
- `measure/verify.py`: `torch.einsum(expr,…)`-Referenz — **schon allgemein**. `measure/metrics.py`:
  **kein Batch** → additiv `B` ergänzen.

**GUI (`app/`):** `controls.py` hält den festen Ausdruck read-only; die Naht-Logik erzeugt RunConfigs.
⟶ **TZ 6:** dynamische Operanden-Liste + Auto-Output + Presets; `RunConfig.expr`/`inputs`/`output`/
`dim_sizes` allgemein befüllen (das Schema ist dafür **schon vorgesehen** — `RunConfig` hat `family`,
`expr`, `inputs`, `output`, `dim_sizes` und leitet inputs/output in `__post_init__` aus `expr` ab).

## TZ-6-Scope (eng halten!)
1. **`parse.py`**: impliziter Output (einsum-Konvention: Output = einmal vorkommende Indizes, sortiert),
   weiterhin genau 2 Operanden, strenge Validierung. (Diagonalen/n-är bleiben draußen.)
2. **`config.py` + `optimizer.py`**: minimaler Port aus A05/06 der fuse/split/permute-Logik, soweit für
   den B1-View gebraucht (nicht mehr).
3. **`reshape.py`**: **echter B1** — `ContractionIR` → `Canonical` mit Operanden-View-Spezifikation
   (Permute+Reshape je Operand auf `(B,M,K)`/`(B,K,N)`), zero-copy wo möglich; `transform_needed` gesetzt.
4. **`run.py`**: Operanden in natürlicher Form bauen; B1-View anwenden; kanonische `(B,M,K)`/`(B,K,N)`/
   `(B,M,N)` an den Kernel; `verify` mit dem allgemeinen Ausdruck (unverändert).
5. **`codegen/templates/contraction.py`**: batched GEMM (Batch-Achse), Orientierung unverändert.
6. **`measure/metrics.py`**: additiv `B` (flops/bytes/AI korrekt für Batched).
7. **`app/components/controls.py`**: dynamische Operanden-Liste + Auto-Output + Presets; Naht-Logik.
8. **`tests/test_reshape.py`**: B1-Tests (bekannte Ausdrücke → korrekte kanonische Größen + numerische
   Gleichheit gegen `torch.einsum`); plus headless-Tests der neuen Controls-Logik.

## Setup (erster Schritt)
Vermutlich **keine neuen Pakete** (torch/cuTile da; einsum-Reshape ist reine torch-/View-Mathematik).
**Verifiziere headless zuerst**, dass die Bausteine tragen: `parse()` klassifiziert einen nicht-trivialen
Ausdruck (z. B. `ki,kj->ij`, `bik,bkj->bij`) korrekt in M/N/K/Batch; `torch.einsum(expr, A, B)` liefert die
Referenz; ein von Hand gebauter View reproduziert sie numerisch. Falls A05/06-Quellen fehlen: die
Klassifikation ist schon da — den View-/Stride-Teil dann eigenständig, klein und getestet bauen.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Reshape-Strategie:** strikt **zero-copy View** (Permute+Reshape; Ausdrücke, die keinen contiguous
   View erlauben, sauber ablehnen — empfohlen für ehrliche AI/Roofline) **vs.** `.contiguous()`-Copy
   erlauben (mehr Ausdrücke, aber Extra-Traffic verfälscht die Roofline-AI → dann ehrlich dokumentieren).
2. **Batched-GEMM-Codegen:** die **eine** Template um eine Batch-Achse erweitern (3D-Grid `bid(2)`, Offset —
   empfohlen: eine bewiesene Struktur, `B=1` byte-nah) **vs.** separate batched Template **vs.** Host-
   Schleife über Batch. Kläre.
3. **parse-Verallgemeinerung:** impliziter Output **ja** (empfohlen, einsum-üblich); Diagonalen/Traces
   **nein** (selten, später); n-är **nein** (opt_einsum = „später/optional"). Kläre den Umfang.
4. **UI-Operanden-Liste:** dynamische Felder + editierbarer **Auto-Output-Vorschlag** + **Presets**
   (z. B. GEMM `ik,kj->ij`, Batched GEMM `bik,bkj->bij`, transponiert `ki,kj->ij`, Tensor-Kontraktion
   `acspx,bspy->abcyx`). Welche Presets? Größen-Eingabe je **Index** (nicht mehr nur M/N/K)? Kläre.
5. **Metrik/Batch:** `compute_metrics` additiv um `B` (Default 1 → TZ 1–5 unverändert) — empfohlen, damit
   die Roofline batched Punkte korrekt platziert. Kläre nur, ob der Batch-Traffic voll zählt.
6. **Nicht-teilbare/nicht-contiguous Dims (Risiko ⑤):** fusionierte Dims sind exakte Produkte (teilen
   sauber); Tile-Ränder deckt `PaddingMode.ZERO` schon ab. Kläre, ob darüber hinaus Padding/Masking nötig ist.

## Scope-Grenzen (was TZ 6 NICHT tut)
- **Kein n-äres einsum** (mehr als 2 Operanden, opt_einsum → paarweise Kontraktionen) — „später/optional".
- **Keine Elementwise-/Reduktions-Familien** — das ist **TZ 7** (memory-bound; macht die Roofline reicher).
- **Keine Fusion** (Kontraktion+Elementwise-Epilog) — Zukunftskandidat.
- **Keine Diagonalen/Spuren/Wiederholungen** je Operand (selten; bewusst draußen).
- **Kein Umbau von Naht/Schema/`run()`/measure/charts** — additiv (Batch in metrics/template, allgemeiner
  Ausdruck in parse/reshape/controls). Roofline/hardware unangetastet (profitieren automatisch von batched Punkten).

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess CUDA-frei;
  Charts reine, headless-testbare Funktionen.
- Ausführen aus `project/` mit dem **venv-Python** (`.venv/bin/python`; Shell-`python` nicht im PATH;
  Shell-State persistiert nicht — venv-Pfad direkt nutzen bzw. `PYTHONPATH=…/project`). Start:
  `python -m tool_pipeline` (GUI), `python -m tool_pipeline.cli` (headless).
- **Harte Regel: NIEMALS `git commit` / `git push`** in diesem Repo.
- Determinismus / **kleine Größen** (geteilte Maschine — OOM crasht sie; `torch.manual_seed(0)`;
  App-Default klein). GPU-Lock respektieren; keine unnötigen GPU-Läufe.
- **verify-before-trust:** kein Ausdruck ohne bestandene fp32-Referenz in die Anzeige (Risiko ①/④).
- **Risiko ④ ernst nehmen:** der B1-View muss numerisch exakt sein — jeder Reshape gegen `torch.einsum`
  gegenprüfen, bevor du ihm traust.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 6 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App in
   lauffähigem, prüfbarem Zustand — z. B. parse-impliziter-Output → reshape-B1-View (headless gegen
   einsum) → run-Operanden/View → batched-Template → metrics-Batch → controls-Operandenliste → Tests).
   Die Design-Entscheidungen oben **vorab** mit mir klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt
   bewirkt / was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären). Dann
   auf **meine Validierung warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-6-Scope bleiben; Scope-Creep (n-är, Elementwise/Reduktion, Fusion, Diagonalen) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 7 (Operationen-Breite II: memory-bound = Elementwise + Reduktion) —
   anschauen, vorbereiten und einen Session-Prompt + Planungs-MD erstellen** — genau nach *diesem* Muster:
   gründlich einlesen (Workflow), **PLAN §10/TZ 7 maßgeblich** (Routing auf M/N/K/C; Elementwise/Reduktion
   als eigene Familien; GB/s als Primärmetrik für memory-bound; sie erscheinen als memory-bound Punkte auf
   der Roofline), Anker aus dem *dann* aktuellen Post-TZ6-Ist-Zustand, MD unter
   `project/project-development/prompts/TZ7-*.md`, und **diese Arbeitsweise inkl. der zwei Zusätze (sehr
   einfache Erklärung nach jedem TODO + Planung des übernächsten TZ als letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**: `parse()` klassifiziert bekannte Ausdrücke
korrekt (M/N/K/Batch, fusionierte Größen); der B1-View reproduziert `torch.einsum(expr, A, B)` **numerisch
exakt** für eine Palette von Ausdrücken (`ki,kj->ij`, `bik,bkj->bij`, mehrdim. M/N wie `ijk,kl->ijl`,
`acspx,bspy->abcyx`) — das ist der Kern-Sicherheitsnetz gegen Risiko ④; die Metriken stimmen mit Batch
(flops=2·B·M·N·K); die Controls-Naht baut aus einer Operanden-Liste die korrekte `RunConfig`. Zusätzlich
die App real starten (GPU-Lock beachten!) und einen allgemeinen Ausdruck (z. B. Batched GEMM) live
durchklicken: er verifiziert, erscheint in allen drei Charts, und der batched Punkt sitzt plausibel auf der
Roofline. Charts via `save_png` rendern und **ansehen** (dataviz „render & look").

## Definition of Done (TZ 6)
Eine **beliebige 2-Operand-Kontraktion** (transponierte Operanden, Batch, mehrdimensionale M/N/K; z. B.
`bik,bkj->bij`, `acspx,bspy->abcyx`) läuft **end-to-end**: über den **echten** B1-Reshape (config/optimizer-
getrieben, zero-copy-View) auf die kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)`, vom batched Codegen emittiert,
gegen die allgemeine fp32-`torch.einsum`-Referenz **live verifiziert**, mit korrekten Batch-Metriken; die
GUI hat eine **dynamische Operanden-Liste** (+ Auto-Output + Presets); die drei Charts inkl. Roofline zeigen
den Ausdruck; alle Tests grün (inkl. `test_reshape.py`) + App-Smoke. **Zusätzlich:** nach jedem TODO gab es
eine sehr einfache Erklärung, und als letzter Schritt ist **TZ 7 vorbereitet** (Planungs-MD + Session-Prompt).
