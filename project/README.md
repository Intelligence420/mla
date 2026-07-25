# cuTile Performance Lab

**Group-Specific Component — MLA, FSU Jena.** Team: Moritz Martin, Oliver Dietzel.

Ein interaktives Tool: einen einsum-/GEMM-Ausdruck eingeben → daraus wird **live ein cuTile-Kernel generiert**, mit verstellbarem Zahlenformat / Tiling / Swizzle auf der GPU **gemessen** und in interaktiven Graphen (Durchsatz, Genauigkeit, Roofline) **visualisiert**. Substanz: Kernel-Erzeugung + ehrliche Messung — die GUI ist die Schauseite davor.

## Was es macht (der Live-Loop)

`Ausdruck + Format/Tile/Swizzle wählen` → `Kernel generieren (cuTile-Quelltext)` → `compilen (gecacht)` → `gegen fp32 verifizieren` → `messen` → `Charts + generierter Code`.

## Architektur & Datenfluss — wo die GUI eingebunden ist

Die GUI hängt nicht verteilt im Code, sondern ist über **genau eine Naht** an den Core gekoppelt: `tool_pipeline/run.py` mit `run(config) -> result`.

```
Browser ──(Dash, WebSocket)── tool_pipeline/app/          ← GUI
  app.py        Dash-Server + Layout-Mount                  Einstieg: python -m tool_pipeline
  layout.py     Sidebar (Controls) + Main (KPIs/Charts/Code)
  callbacks.py  "Run" → baut RunConfig aus den Controls
                → background=True-Job → ruft  tool_pipeline.run.run(config)
                → rendert RunResult in Charts/KPIs/Code-Panel
  components/   controls · charts · code_panel · kpis
        │
        │   die EINZIGE Naht:  run(config: RunConfig) -> RunResult
        ▼
tool_pipeline/run.py
   parse → reshape(B1) → emit(C1) → compile(+Cache) → verify → measure → store
     └ intermediate_representation/     └ codegen/          └ measure/         └ store/
       (Paket "ir": parse, reshape)     (emit, compile)     (verify, bench)
```

**Prinzip:** `app/` importiert aus dem Core ausschließlich `run.py`, `schema.py` und `store/` (Letzteres nur zum *Lesen* der History; torch-frei, pandas lazy) — nie `ir`/`codegen`/`measure` direkt. `run` wird zudem **lazy** im Callback geholt, damit der Hauptprozess CUDA-frei und fork-sicher bleibt. Umgekehrt hängt kein Core-Modul auf Modulebene an der GUI: `cli.py` teilt die Config-Bau-Helfer aus `app/components/controls.py`, importiert sie aber lazy — `import tool_pipeline.cli` läuft **ohne Dash und ohne torch** durch. Dadurch ist der Core headless testbar (`tests/`, `cli.py`), die GUI austauschbar, und der Hintergrund-Job umschließt exakt *einen* Aufruf (`run()`).

## Verzeichnisstruktur

- **`tool_pipeline/`** — das Tool (Python-Paket): `intermediate_representation/` (kurz „ir") → `codegen/` → `measure/` → `store/` → `app/`; `run.py` verklammert sie (= Vertrag Core↔GUI), `schema.py` definiert `RunConfig`/`RunResult`.
- **`project-development/`** — Artefakte, die **nur dem Bau** dienen (kein Auslieferungs-Code): `PLAN.md` (vollständiger Projekt- & Fortschrittsplan, alle Entscheidungen), `analysis/` (Hardware-/dtype-Belege: `RESULTS_gb10.md`, `dtype_analyse.py`).
- **`results/`** — Results-Store: `results.jsonl` (ein Lauf je Zeile) + `kernels/<slug>.py` (persistierter generierter Code = Compile-Cache; `<slug>` = lesbarer Config-Name, z. B. `ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py`).
- **`tests/`** — Korrektheitstests (Codegen ist eine silent-wrong-answer-Quelle → jede Familie/jeden dtype testen).

## Ausführen (auf dem GPU-Host, venv aktiv)

Zuerst das venv des GPU-Hosts aktivieren — also das, in dem `torch` (mit CUDA), `cuda.tile` und `triton` liegen; auf dem Lab-Rechner ist das das venv im Wurzelverzeichnis des Checkouts (`.venv/`):

```bash
source ../.venv/bin/activate       # venv des GPU-Hosts (Pfad je Maschine)
pip install -r requirements.txt    # einmalig: GUI-/Plot-/Store-Pakete ergänzen

python -m tool_pipeline            # Dash-GUI starten
python -m tool_pipeline.cli --sweep    # headless: Report-Sweep (GPU, ~2 min)
python -m tool_pipeline.report_figures # torch-frei: Report-Figuren aus results.jsonl
```

Zusätzliche Pakete: siehe `requirements.txt` (Dash/Plotly/pandas/matplotlib). `torch`, `cuda.tile`, `triton`, `cupy` kommen aus dem vorhandenen venv und stehen bewusst **nicht** in `requirements.txt`.

## Hardware & Zahlenformate

NVIDIA **GB10** (Blackwell sm_121, **273 GB/s** → für die meisten Shapes **stark memory-bound**). In-scope dtypes (Tensor-Core, empirisch verifiziert): **fp16, bf16, tf32, fp8 (e4m3, e5m2)**; `fp32`/`fp64` als Genauigkeits-Anker. Belege: `project-development/analysis/RESULTS_gb10.md`.

## Plan & Status

Vollständiger Projekt- & Fortschrittsplan, alle Designentscheidungen und Tickets: **`project-development/PLAN.md`**.
