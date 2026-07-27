# cuTile Performance Lab

**Group-Specific Component — MLA, FSU Jena.** Team: Moritz Martin, Oliver Dietzel.

Ein interaktives Tool: einen einsum-Ausdruck eingeben → daraus wird **live ein cuTile-Kernel generiert**, mit verstellbarem Zahlenformat / Tiling / Swizzle auf der GPU **gemessen** und in interaktiven Graphen (Durchsatz, Genauigkeit, Roofline) **visualisiert**. Substanz: Kernel-Erzeugung + ehrliche Messung — die GUI ist die Schauseite davor.

Abgedeckt sind **drei Operations-Familien**: Kontraktion (GEMM/Batched-GEMM/Tensor-Kontraktion auf Tensor-Cores, optional mit **Epilog-Fusion** `bias`/`relu`, n-äre Ketten als paarweise Zerlegung), **Elementwise** (`add`/`mul`/`copy`) und **Reduktion** (`sum`) — die beiden letzten memory-bound, damit die Roofline **beide** Seiten zeigt.

## Was es macht (der Live-Loop)

`Familie + Ausdruck + Formate/Tiles/Swizzle wählen` → `Kernel generieren (cuTile-Quelltext)` → `compilen (gecacht)` → `gegen fp32 verifizieren` → `messen` → `Charts + generierter Code`.

Zwei Regeln tragen das Ganze: **verify-before-trust** — kein Ergebnis wird angezeigt oder gespeichert, das die fp32-Referenz nicht bestanden hat (`status="verify_failed"` statt einer still falschen Zahl) — und **1 Lauf = 1 Punkt**: jeder Lauf ist eine Zeile in `results/results.jsonl` und ein Punkt in den Charts. Ein „Vergleichen"-Klick ist ein **Batch** über das Kreuzprodukt der gewählten Formate × Tiles × Swizzle-Konfigurationen; alle Zeilen des Batches teilen `run_id`/`run_name` und sind über das **History-Panel** wieder ansehbar, umbenennbar und löschbar.

## Architektur & Datenfluss — wo die GUI eingebunden ist

Die GUI hängt nicht verteilt im Code, sondern ist über **genau eine Naht** an den Core gekoppelt: `tool_pipeline/run.py` mit `run(config) -> result`.

```
Browser ──(Dash, WebSocket)── tool_pipeline/app/          ← GUI
  app.py        Dash-Server + DiskcacheManager             Einstieg: python -m tool_pipeline
  layout.py     Topbar · History · Sidebar (Controls) · Main (KPIs/Charts/Code)
  callbacks.py  "Vergleichen" → Kreuzprodukt der Controls → eine RunConfig je Zelle
                → background=True-Job, EIN GPU-Lock über den ganzen Batch
                → ruft je Config  tool_pipeline.run.run(config)
                → rendert die RunResults in Charts/KPIs/Code-Panel + History
  components/   controls · charts · kpis · code_panel · history     (Styling: assets/theme.css)
        │
        │   die EINZIGE Naht:  run(config: RunConfig) -> RunResult
        ▼
tool_pipeline/run.py — ein Ablauf, vier Zweige (2-Op-Kontraktion · n-är · elementwise · reduction)
   parse → [reshape(B1)] → emit(C1) → compile(+Cache) → verify(fp32) → bench → metrics → store
     └ intermediate_representation/    └ codegen/          └ measure/              └ store/
       parse · reshape ·                 emit · compile ·    verify · bench ·        results.jsonl
       config · optimizer (A05/06-Port)  templates/          metrics · baselines ·   + kernels/
                                         (contraction ·      fusion · provenance
                                          elementwise ·
                                          reduction)
```

Nur die Kontraktion braucht den **B1-Reshape** (fuse/permute → kanonisches Batched-GEMM, per Stride-Adjazenz als zero-copy-View belegt); die memory-bound-Familien gehen ohne Kanonisierung direkt in ihr Template, n-äre Ausdrücke werden in eine Kette paarweiser Kontraktionen zerlegt, die durch dieselbe Maschinerie läuft. Alle vier Zweige enden im gleichen `RunResult`.

**Prinzip:** `app/` importiert aus dem Core nur, was **torch-frei** ist: `run.py`, `schema.py` (dtype-Regeln), `hardware.py` (Roofline-Kennwerte), `store/` (zum *Lesen* der History; pandas lazy) und `ir/parse.py` (Live-Validierung des eingegebenen Ausdrucks) — nie `codegen`/`measure` und nie torch/cuda auf Modulebene. `run` wird zudem **lazy** im Callback geholt, damit der Hauptprozess CUDA-frei und fork-sicher bleibt (der DiskcacheManager forkt ihn). Umgekehrt hängt kein Core-Modul auf Modulebene an der GUI: `cli.py` teilt die Config-Bau-Helfer aus `app/components/controls.py`, importiert sie aber lazy — `import tool_pipeline.cli` läuft **ohne Dash und ohne torch** durch. Dadurch ist der Core headless testbar (`tests/`, `cli.py`), die GUI austauschbar, und der Hintergrund-Job umschließt exakt *einen* Aufruf (`run()`) je Config.

`run()` wirft **nie** nach außen: Fehler werden zu `compile_error` / `verify_failed` / `run_error` kategorisiert, damit die GUI sie anzeigt statt abzustürzen.

## Verzeichnisstruktur

- **`tool_pipeline/`** — das Tool (Python-Paket): `intermediate_representation/` (kurz „ir") → `codegen/` → `measure/` → `store/` → `app/`; `run.py` verklammert sie (= Vertrag Core↔GUI), `schema.py` definiert `RunConfig`/`RunResult` (inkl. der dtype-/Akkumulator-Regeln), `hardware.py` die GB10-Roofline-Kennwerte. Headless-Werkzeuge: `cli.py` (Einzellauf + Report-Sweep), `report_figures.py` (Report-PNGs aus dem Store).
- **`project-development/`** — Artefakte, die **nur dem Bau** dienen (kein Auslieferungs-Code): `analysis/` (Hardware-/dtype-Belege: `RESULTS_gb10.md`, `dtype_analyse.py`).
- **`results/`** — Results-Store: `results.jsonl` (ein Lauf je Zeile) + `kernels/<slug>.py` (persistierter generierter Code = Compile-Cache; `<slug>` = lesbarer Config-Name, z. B. `ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py`, mit `__sw`/`__g32`/`__ep_relu`-Suffixen für Swizzle/GROUP_M/Epilog).
- **`tests/`** — Korrektheitstests, 286 Tests über 12 Dateien (Codegen ist eine silent-wrong-answer-Quelle → jede Familie/jeden dtype testen). Der Großteil läuft ohne GPU; die GPU-Tests in `test_app_execute.py` fahren echte Läufe.

## Ausführen (auf dem GPU-Host, venv aktiv)

Zuerst das venv des GPU-Hosts aktivieren — also das, in dem `torch` (mit CUDA), `cuda.tile` und `triton` liegen; auf dem Lab-Rechner ist das das venv im Wurzelverzeichnis des Checkouts (`.venv/`):

```bash
source ../.venv/bin/activate       # venv des GPU-Hosts (Pfad je Maschine)
pip install -r requirements.txt    # einmalig: GUI-/Plot-/Store-Pakete ergänzen

python -m tool_pipeline                # Dash-GUI starten (http://127.0.0.1:8050)
python -m tool_pipeline.cli            # headless: ein GEMM-Lauf (ik,kj->ij, fp16→fp32)
python -m tool_pipeline.cli --sweep    # headless: kompletter Report-Sweep (GPU, ~2 min)
python -m tool_pipeline.report_figures # torch-frei: Report-Figuren aus results.jsonl
python -m pytest tests/                # Korrektheitstests
```

Host/Port der GUI kommen aus `TP_HOST`/`TP_PORT` (Default `127.0.0.1:8050`; `TP_HOST=0.0.0.0` für Zugriff über SSH-Tunnel/LAN). Der Einzellauf der CLI ist parametrisierbar: `--family` / `--op` / `--expr`, Größen über `--size` bzw. `--M/--N/--K`, `--epilog bias|relu`, `--show-kernel` (generierten Quelltext ausgeben). `--show-configs` listet die Sweep-Configs **ohne** GPU.

Zusätzliche Pakete: siehe `requirements.txt` (Dash/Plotly/pandas/matplotlib/filelock). `torch`, `cuda.tile`, `triton`, `cupy` kommen aus dem vorhandenen venv und stehen bewusst **nicht** in `requirements.txt`; `opt_einsum` (Pfad-Planer für n-äre Ketten) ist optional — fehlt es, greift der Links-nach-rechts-Fold.

## Hardware & Zahlenformate

NVIDIA **GB10** (Grace-Blackwell, sm_121, **273 GB/s** → für die meisten Shapes **stark memory-bound**). In-scope dtypes (Tensor-Core, empirisch verifiziert): **fp16, bf16, tf32, fp8 (e4m3, e5m2)**; `fp32` als Genauigkeits-Anker. Die Akkumulator-Regeln (bf16/tf32 **müssen** in fp32 akkumulieren, fp16/fp8 dürfen fp16 oder fp32) liegen als Vertrags-Daten in `schema.ALLOWED_ACC` und werden früh geprüft, statt still falsch zu rechnen. Belege: `project-development/analysis/RESULTS_gb10.md`.

## Bericht & Status

Der ausführliche Projektbericht (Grundlagen · Architektur · Pipeline · Frontend · Beispielanalyse · Bedienung · Anhang) liegt im Sphinx-Teil des Repos: `sphinx/source/chapters/group_specific_component/report/`. **Alle Zahlen dort stammen aus der jüngsten `CLI-Report-Sweep`-Charge in `results/results.jsonl`** (`--sweep` → `report_figures`), nur aus Läufen mit `status == "ok"`.
