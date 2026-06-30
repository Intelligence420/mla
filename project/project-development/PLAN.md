# Performance Explorer — Projekt- & Fortschrittsplan

> Group-Specific Component (MLA, FSU Jena). Team: Moritz Martin, Oliver Dietzel.
> Lebendes Planungsdokument — Single Source of Truth für Entscheidungen & Erkenntnisse.
> (Polierte Fassung wandert später nach `sphinx/.../group_specific_component/projektplan.rst` + Report.)

## 1. Projekt

**Idee 2 — Interaktiver einsum/GEMM-Performance-Explorer (GPU / cuTile).**
Ein Tool: Nutzer gibt einen einsum-/GEMM-Ausdruck ein → daraus wird **live** ein cuTile-Kernel **generiert**, mit verstellbarem Zahlenformat / Tiling / Swizzle auf der GPU **gemessen** und in interaktiven Graphen (Durchsatz, Genauigkeit, Roofline) **visualisiert**. Substanz laut Pitch: Kernel-Erzeugung + ehrliche Messung (nicht die GUI-Optik).

**Deadlines:** Presentation (20 min, lauffähiger Prototyp + Ergebnisse) **08.07.2026** · Finaler Sphinx-Report (blogartig) **27.07.2026**.

## 2. Getroffene Designentscheidungen

| Thema | Entscheidung | Begründung |
|---|---|---|
| **Mess-Modell** | **Live** (Regler ändern → live auf GPU neu messen) | echtes Explorer-Erlebnis; GUI von Anfang an, Entwicklung *durch* sie |
| **GUI-Framework** | **Plotly Dash** | turnkey Background-Jobs (`background=True`: Progress/Cancel/Run-Disable) lösen den mehrsekündigen cuTile-Compile ohne Bastelei; native Plotly-Charts inkl. Roofline; reif/langlebig. Runner-up: NiceGUI. Bewusst **nicht** Gradio/Streamlit (Demo-Optik vs. „ordentliches Framework") |
| **GUI-Stellenwert** | **Hauptdeliverable**, kein Anhängsel; professionell, nicht das Mockup, nicht vibe-coded | |
| **Operationen** | Kontraktions-Familie (GEMM, Batched GEMM, allg. Kontraktion) **+ memory-bound** (Elementwise, Reduktion, opt. Copy/Transpose) | einsum ist die verbindende Sprache + Presets → Kernel-Familien; memory-bound macht die **Roofline** aussagekräftig (compute- vs. memory-bound) |
| **Fusion** | **Zukunftskandidat** (Kontraktion+Elementwise-Epilog, A04), nicht jetzt | A04-Befund: 0,98× — ehrlich interessant, aber später |
| **Autotuning + Tile-Heatmap** | **Gestrichen** | Scope-Reduktion |
| **Codegen** | **C1 + B1** (s. §3) | Headline „Kernel generieren" wörtlich wahr & sichtbar; Korrektheitsrisiko begrenzt |
| **Generierter Code** | wird **persistiert** (nicht nur angezeigt) | Cache-Schlüssel + reproduzierbares Artefakt + UI-Anzeige + Debugbarkeit |
| **Verify-before-trust** | jeder generierte Kernel wird gegen torch-fp32 geprüft, **bevor** Zahlen angezeigt werden | Schutz vor stillen Falschergebnissen; = zugleich das Genauigkeits-Panel |
| **Messung** | eigene **CUDA-Events-Schicht**, voller Metriksatz | `do_bench` allein zu flach |
| **Baselines** | **cuBLAS-Obergrenze + naive-cuTile-Untergrenze**, in der GUI je **optional zuschaltbar** | aus „wir verlieren gegen cuBLAS" wird konstruktive Story (was bringt unser Tuning) |
| **dtype-Matrix** | empirisch geklärt — s. §5 | Analyse-Testing auf GB10 |
| **Persistenz / Results-Store** | alle Mess-/Analyse-Ergebnisse **strukturiert & reproduzierbar** speichern (nicht temporär) | Wiederverwendung, Report-Datenquelle, Cache, Vergleich über Läufe |
| **Repo-Layout** | top-level `project/`: `tool_pipeline/` (Tool), `project-development/` (Bau-Artefakte: PLAN.md, analysis), `results/` | Group-Specific Component als eigenständiges Deliverable, klar getrennt von `assignments/` |
| **Results-Store-Format** | **JSON Lines** (`project/results/results.jsonl`, ein Objekt je Lauf) + generierte Kernel als `results/kernels/<slug>.py` | simpel, transparent, git-diff-bar, pandas-ladbar; Kernel-Dateiname = **lesbarer, normalisierter Config-Slug** (`expr`/`dtype`/`acc_dtype`/`tile`/`swizzle`, z. B. `ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py`) statt Hash → selbsterklärend + deterministischer Cache-Treffer (Config steht ohnehin im JSONL) |

## 3. Codegen-Architektur (C1 + B1)

- **C1 — Python-Generierung:** f-String-Templates erzeugen echten `@ct.kernel`-Quelltext → `exec` + `ct.launch`. Präzedenz: euer `assignments/10_assignment/src/gen_matmul.py` (gleiche Technik, MLIR). Der emittierte Text **ist** die UI-Code-Anzeige.
- **B1 — Reshape auf kanonisch:** Host-seitig (IR-getrieben via `config`/`optimizer`: fuse/split/permute) wird **jede** Kontraktion auf ein Batched-GEMM `(B,M,K)×(B,K,N)→(B,M,N)` reshapet ⇒ Codegen muss nur **eine bewiesene** Struktur emittieren. Memory-bound-Ops bekommen einfachere A02-Templates; kleiner Klassifikator auf M/N/K/C routet die Familie.
- **Mess-Pipeline:** `verify(fp32) → compile (Zeit separat) → warmup → timed loop (CUDA-Events, L2-Flush) → Metriken → Baseline-Vergleich → GPU-Zustand → Results-Store → Charts`.
- **Metriksatz:** Latenz (Median+min+p90+σ), TFLOP/s, erreichte GB/s, arithm. Intensität, %-vom-Peak (Compute & BW), Fehler (max/mean/rel vs fp32), Compile- vs. Laufzeit, GPU-Zustand (Clock/Temp/Power). cold-L2 default.

## 4. Wiederverwendung aus A01–06 / A10

| Stufe | Quelle | Status |
|---|---|---|
| Parse (einsum→IR, M/N/K/C) | A05/06 `config.py` | wiederverwenden (Lücke: nur 2 Operanden) |
| IR-Transform / Tiling | A05/06 `optimizer.py` (`split_dim` = Tile-Injektion) | wiederverwenden |
| Kontraktions-Kernel-Template | A03/A05/A06 `kernel.py` | Vorlage für Codegen |
| Elementwise/Reduktion-Templates | A02 `task_02/03/04` | Vorlage |
| dtype-Idiome + Toleranzen + fp8-Cast | A03 `task_01/02` | wiederverwenden |
| Codegen-Präzedenz (f-String) | A10 `gen_matmul.py` | Vorbild |
| Messung/Accuracy | `triton.do_bench`, `reference()`, `allclose`, `flops_count` | Bausteine |
| **Greenfield** | echter cuTile-Codegen, Dash-Web-Layer, Results-Store, n-ary (opt_einsum) | neu bauen |
| **Gotcha** | Mockup-Codepanel nutzt **fiktive** API (`@cuda.tile.jit`/`tile.dot`); echt: `@ct.kernel`/`ct.bid`/`ct.load`/`ct.mma`/`ct.launch` | beachten |

## 5. Hardware (GB10 / DGX Spark) & dtype-Matrix — geklärt 30.06.2026

**Maschine:** NVIDIA **GB10** (Grace-Blackwell), Blackwell-GPU **sm_121**, 48 SM, 6144 CUDA-Cores, 192 5th-gen Tensor-Cores, ~2.42 GHz (max 3.0). **128 GB unified LPDDR5x, 273 GB/s** (theoretisch; real ~70–85 %). aarch64 (20 Cores). CUDA 13.0, torch 2.11.0, triton 3.6.0, `cuda.tile` vorhanden.

**Roofline-Peaks (dense, gemessen mmapeak-Forum):** FP16/BF16 ≈ **213** · FP8 ≈ **214** · TF32 ≈ **53** · FP32 (plain, kein TC) niedrig · FP64 ≈ vernachlässigbar. Ridge-Points sehr hoch (BF16 ≈ **780 FLOP/Byte**) ⇒ GB10 ist für die meisten Shapes **stark memory-bound** — die zentrale Roofline-Erkenntnis des Tools.

**dtype-Matrix (alle compilen/laufen/verifizieren auf GB10):**

| dtype (compute→acc) | Status | max_abs_err | API-Hinweis |
|---|---|---|---|
| fp16 → fp32 | ✅ in-scope (Anker) | 1.7e-4 | nativ |
| bf16 → fp32 | ✅ in-scope (acc=fp32 **Pflicht**) | 1.1e-4 | nativ |
| tf32 → fp32 | ✅ in-scope | 3.4e-2 | **kein mma-Flag**: `ct.astype(tile, ct.tfloat32)` vor `ct.mma` |
| fp8 e4m3 → fp16/fp32 | ✅ in-scope (fp16-acc am schnellsten) | 0.16 / 1.5e-5 | host-seitig `.to(torch.float8_e4m3fn)` |
| fp8 e5m2 → fp32 | ✅ in-scope | 1.5e-5 | host-seitig `.to(torch.float8_e5m2)` |
| fp32 → fp32 (plain) | ⚓ Anker/Diagnose (kein TC) | 1.1e-4 | nativ |
| fp64 → fp64 | ⚓ Diagnose (~0.1 TFLOP/s) | 2.1e-13 | nativ |
| fp4 / int4 | ❌ exkludiert | — | keine Symbole in diesem cuTile-Build |

**Acc-Regeln erzwingen:** bf16 & tf32 → fp32; fp16 & fp8 → fp16 oder fp32. Beleg + Skript: `project/project-development/analysis/RESULTS_gb10.md`, `project/project-development/analysis/dtype_analyse.py`.

## 6. Codegen-Risiken (Watch-list — stille Falschergebnisse)

① `ct.mma`-Operanden-Orientierung (A06 permutiert B + tauscht Operanden — naiv kompiliert, liefert falsch) · ② index/shape-Tupel-Korrektheit · ③ `exec` von generiertem Text (immer persistieren/loggen) · ④ B1-Reshape muss korrekter (zero-copy) View sein (Optimizer-Stride-Mathematik) · ⑤ nicht-teilbare Dimensionen → padden+maskieren · ⑥ Compile-Cache (sonst Live zu langsam) · ⑦ dtype-Cast-Pfade (Compute/Akku) · ⑧ Familien-Routing. **Sicherheitsnetz: verify-before-trust auf jedem Kernel.**

## 7. Offene Schritte

- ✅ **Schritt 6 erledigt:** Layout = top-level `project/` (s. §9); Results-Store = JSON Lines + `kernels/<slug>.py` (lesbarer Config-Slug, s. §2). Sphinx-Integration: Plots/Tabellen aus dem Store in den Report.
- **Schritt 7 = die Teil-Ziele (s. §10):** inkrementelle Umsetzungsreihenfolge; jedes Teil-Ziel komplett & korrekt, spätere bauen darauf auf. Aufgabenteilung flexibel (gemeinsam, mit Assistenz) — die Reihenfolge treibt uns, nicht der Kalender.

## 8. Framework-Recherche (Kurz-Fazit)

Web-gestützter Vergleich (Streamlit, Gradio, Dash, Panel, NiceGUI, Reflex, Solara, Shiny, Custom FastAPI+JS) gegen unser Szenario (live, mehrsekündiger Compile auf geteilter GPU, professionell, 2 Python-Devs). Entscheidend: turnkey Long-Job-Handling + Politur + Reife. → **Dash**. Antipattern: Compile inline im async-Handler (friert ein → immer Thread auslagern); Doppel-Klick ohne Lock; Bleeding-Edge-Versionen tracken (pinnen).

## 9. Verzeichnisstruktur (`project/`)

Gegliedert nach der Pipeline (**logisch**), flach mit sprechenden Namen (**einfach**), jede Stufe ein eigenes Subpaket (**skalierbar**: neue Operation = neues Template-Modul, neuer dtype = nur Daten, neuer Chart = neue Komponente).

```
project/
├── README.md · requirements.txt
├── tool_pipeline/                 # die Tool-Pipeline (Python-Paket)
│   ├── run.py                # Orchestrator = Vertrag Core↔GUI: run(config)->result
│   ├── schema.py             # RunConfig / RunResult (zuerst definieren, T0.2)
│   ├── hardware.py           # GB10-Roofline-Peaks (273 GB/s, TFLOP/s je dtype)
│   ├── cli.py                # headless / Batch-Sweeps für den Report
│   ├── ir/                   # parse.py · config.py · optimizer.py · reshape.py (B1)
│   ├── codegen/              # emit.py · compile.py(+Cache) · templates/{contraction,elementwise,reduction}.py
│   ├── measure/              # bench.py(CUDA-Events) · metrics.py · verify.py · baselines.py · provenance.py
│   ├── store/                # store.py (results.jsonl + kernels/)
│   └── app/                  # Dash: app.py · layout.py · callbacks.py · components/{controls,charts,code_panel,kpis}.py · assets/
├── project-development/      # Bau-Artefakte (nicht das ausgelieferte Tool)
│   ├── PLAN.md                # dieser Plan
│   └── analysis/             # RESULTS_gb10.md + dtype_analyse.py
├── results/                  # results.jsonl + kernels/<slug>.py   (Results-Store)
└── tests/                    # test_reshape · test_codegen · test_measure
```

Jede Datei hat aktuell einen Zweck-Docstring + TODO-Hinweis; Inhalte füllen die Tickets (§10, folgt).

## 10. Umsetzung in Teil-Zielen (inkrementelle Reihenfolge)

**Prinzip:** Nicht jede Datei auf einmal halbfertig, sondern **ein Teil-Ziel komplett & korrekt**, dann das nächste darauf aufbauen. Jedes Teil-Ziel ist eine *vertikale Scheibe* durch die ganze Pipeline, die für sich lauffähig und gegen fp32 verifiziert ist. Erst **tief** (GEMM voll ausreizen), dann **breit** (Operationen), dann **Politur**. Aufgabenteilung bewusst flexibel; die Reihenfolge treibt uns, nicht der Kalender.

### TZ 1 — Backbone: eine Operation, ein Format, headless, korrekt
*Fertig, wenn:* `ik,kj->ij` in fp16→fp32 (feste Tile, kein Swizzle) **end-to-end** läuft: generierter `@ct.kernel`-Quelltext → compile → gegen torch-fp32 verifiziert → gemessen (ms, TFLOP/s) → als Zeile in `results.jsonl` + Kernel-Datei persistiert. Angestoßen über `cli.py`.
*TODOs:* `schema.py`, `store/store.py`, `ir/parse.py` (GEMM minimal), `ir/reshape.py` (GEMM-Passthrough), `codegen/templates/contraction.py` (GEMM-Template), `codegen/emit.py`, `codegen/compile.py` (exec+launch, einfacher Cache), `measure/verify.py` (fp32-Ref + max_err), `measure/bench.py` (CUDA-Events, ms), `measure/metrics.py` (TFLOP/s), `run.py`, `cli.py`, `tests/test_codegen.py`.
*Schaltet frei:* die **gesamte Pipeline ist bewiesen** — alles Spätere erweitert nur einzelne Stufen.

### TZ 2 — GUI um genau diese eine Operation (Live-Skelett)
*Fertig, wenn:* Dash-App läuft (`python -m tool_pipeline`): Größen eingeben → „Run" (background-Job mit Progress/Cancel + GPU-Lock) → ruft `run()` → zeigt KPIs, Verify-Status und **generierten Code**. Live auf der GPU.
*TODOs:* `app/app.py`, `app/layout.py`, `app/callbacks.py` (background=True), `app/components/controls.py` (Größen), `app/components/kpis.py`, `app/components/code_panel.py`, `__main__.py`.
*Schaltet frei:* der **GUI↔Core-Vertrag und der Live-Loop** sind bewiesen — ab jetzt entwickeln wir *durch die GUI*.

### TZ 3 — dtype-Achse + Genauigkeits-Story
*Fertig, wenn:* alle in-scope Formate (bf16/tf32/fp8 + Acc-Regeln) wählbar; Genauigkeit (max/mean/rel vs fp32) gemessen und in **zwei Charts** sichtbar: Durchsatz je Format (Balken) + Genauigkeit↔Durchsatz (Scatter).
*TODOs:* `codegen/templates/contraction.py` (dtype-Pfade: `ct.astype` tf32, fp8-Host-Cast, Acc), `measure/verify.py` (mean/rel + dtype-Toleranzen), `app/components/controls.py` (dtype-Dropdown + Acc-Regeln erzwingen), `app/components/charts.py` (Balken + Scatter).
*Schaltet frei:* die **Headline-Erkenntnis** (Format-Tradeoff) ist real.

### TZ 4 — Tiling/Swizzle-Achse + volle Mess-Schicht + Baselines
*Fertig, wenn:* Tile (TM/TN/TK) + Swizzle verstellbar; Mess-Schicht vollständig (Warmup, L2-Flush, Verteilung Median/min/p90/σ, GB/s, arithm. Intensität, %-Peak, Compile/Run getrennt, GPU-Zustand); Baselines cuBLAS + naive-cuTile je optional zuschaltbar.
*TODOs:* `codegen/templates/contraction.py` (Tile/Swizzle-Parameter), `measure/bench.py` (voll), `measure/metrics.py` (GB/s/Intensität/%-Peak), `measure/baselines.py`, `measure/provenance.py`, `app/components/controls.py` (Slider/Toggles/Baseline-Schalter).
*Schaltet frei:* **vollständige, ehrliche Performance-Exploration** für GEMM.

### TZ 5 — Roofline
*Fertig, wenn:* `hardware.py` (GB10-Peaks + 273 GB/s) + Roofline-Chart (arithm. Intensität vs erreichte FLOP/s, dtype-Decken + Bandbreiten-Steigung), Punkte aus echten Messungen.
*TODOs:* `hardware.py`, `app/components/charts.py` (Roofline).
*Schaltet frei:* die **memory-bound-Einordnung** wird sichtbar (wird mit TZ 7 reicher).

### TZ 6 — Operationen-Breite I: allgemeine 2-Operand-Kontraktion (B1 wird tragend)
*Fertig, wenn:* beliebige 2-Operand-Kontraktion (z. B. `acspx,bspy->abcyx`, Batched GEMM) läuft — über den **echten** B1-Reshape (config/optimizer-getrieben) auf die kanonische Form, mit Operanden-Liste im UI; jeder Ausdruck live verifiziert.
*TODOs:* `ir/config.py` + `ir/optimizer.py` (Port aus A05/06), `ir/reshape.py` (echtes B1), `ir/parse.py` (allgemein + impliziter Output), `app/components/controls.py` (dynamische Operanden-Liste MATCH/ALL, Auto-Output, Presets), `tests/test_reshape.py`.
*Schaltet frei:* die **Kontraktions-Familie** vollständig; IR/Optimizer sind tragend.

### TZ 7 — Operationen-Breite II: memory-bound (Elementwise + Reduktion)
*Fertig, wenn:* Elementwise und Reduktion als eigene Familien laufen (Routing auf M/N/K/C); sie erscheinen als memory-bound Punkte auf der Roofline (Kontrast zur compute-bound Kontraktion).
*TODOs:* `codegen/templates/elementwise.py`, `codegen/templates/reduction.py`, `ir/parse.py` (Familien-Routing), `measure/metrics.py` (GB/s als Primärmetrik für memory-bound), Presets.
*Schaltet frei:* das **vollständige Operations-Menü** (Scope-Entscheidung erfüllt); Roofline zeigt beide Seiten.

### TZ 8 — Politur, Robustheit & Report
*Fertig, wenn:* professionelles Theming/Layout; Randfälle (nicht-teilbare Dims → padden/maskieren); Compile-Cache gehärtet; Fehlerzustände sauber; **Sphinx-Report** (Figuren/Tabellen aus `results.jsonl`) + `projektplan.rst` aus diesem PLAN.
*TODOs:* `app/assets/theme.css`, Padding/Masking im Codegen, Cache-Härtung im Store, `cli.py` (Batch-Sweeps für Report-Plots), Sphinx-Kapitel, `tests/test_measure.py`.
*Schaltet frei:* das **fertige, dokumentierte Deliverable**.

### Später / optional (Zukunftskandidaten)
n-ary einsum (opt_einsum → paarweise Kontraktionen), **Fusion** (Kontraktion+Elementwise-Epilog, A04), Copy/Transpose als eigene memory-bound Ops.
