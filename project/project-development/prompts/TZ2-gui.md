# Auftrag: TZ 2 — GUI (Live-Skelett) um genau eine Operation

Repo: `/home/mla08/MLA/mla`. Group-Specific Component „Interaktiver einsum/GEMM-Performance-Explorer" (GPU/cuTile, GUI = **Plotly Dash**). Dein Auftrag ist **ausschließlich Teil-Ziel 2 (TZ 2)**: die grafische Oberfläche um *eine* Operation herum, mit dem **Live-Loop** (Eingabe → Hintergrund-Job auf der GPU → Anzeige).

## Unabhängigkeit von TZ 1 (wichtig!)
Diese Session **darf NICHT voraussetzen, dass TZ 1 (der Core) fertig ist.** Die Kopplung ist genau eine Naht: `tool_pipeline/run.py` mit `run(config: RunConfig) -> RunResult`. Vorgehen:
- Existiert `run()` schon und liefert echte Ergebnisse → verwende es.
- Sonst baue die GUI gegen einen **Mock** `run()`, der ein schema-konformes `RunResult` mit plausiblen Fake-Werten und einem Beispiel-Kernel-Quelltext zurückgibt. Der Mock **simuliert die Dauer** (ein paar Sekunden, mit Fortschritt) und **berührt echt die GPU** mit einer winzigen Operation (kleines `torch.cuda`-Matmul), damit der kritische Pfad „GPU im Hintergrund-Worker" real getestet ist.
- Weil `app/` ausschließlich über `run()` + `schema.py` koppelt, ist der Tausch Mock→echt später **transparent** (keine GUI-Änderung). Genau dafür ist die eine Naht da.

`schema.py` (`RunConfig`/`RunResult`) ist der **gemeinsame Vertrag**: existiert die Datei schon → **lies und nutze sie unverändert**; existiert sie nicht → lege sie **exakt** nach der Skizze unten an (damit sie mit TZ 1 zusammenpasst).

## Zuerst lesen
1. `project/project-development/PLAN.md` — besonders **§10 (TZ-2-Definition)**, §2 (warum Dash), §9 (Struktur).
2. `project/README.md` — Architektur & die **eine Naht** (`app/` importiert NUR `run.py` + `schema.py`, nie `ir`/`codegen`/`measure`).
3. Falls vorhanden: `tool_pipeline/schema.py`, `tool_pipeline/run.py`, `tool_pipeline/cli.py`, `tool_pipeline/store/store.py` — der echte Vertrag/Aufruf. Falls nicht vorhanden → Mock + Skizze unten.
4. Die Stub-Dateien unter `tool_pipeline/app/` (`app.py`, `layout.py`, `callbacks.py`, `components/{controls,kpis,code_panel}.py`) und `tool_pipeline/__main__.py` — die zu füllenden Dateien.
5. Memory (Index `MEMORY.md`): `gsc-project-plan` (Dash-Entscheidung, Messgrößen), `gsc-hardware-dtype-facts`.

## Was du wissen musst
- **Ziel TZ 2 (eng):** Oberfläche für *eine* Operation (GEMM `ik,kj->ij`, fp16, feste Tile). Eingabe: Größen M/N/K. „Run" → **Dash-Hintergrund-Job** (Fortschritt + Abbrechen + Run-Knopf sperren) → ruft `run()` → zeigt: **KPIs** (TFLOP/s, Laufzeit, max_abs_err, Compile-Zeit), **Verify-Status** (Badge ok/verify_failed) und das **generierte Kernel-Panel** (Code, monospaced).
- **Umgebung:** NVIDIA GB10, venv `source /home/mla08/MLA/mla/.venv/bin/activate`. Dash/Plotly ggf. via `project/requirements.txt` installieren (braucht `dash`, `dash-bootstrap-components`, `diskcache`). Dash-Version **pinnen**.
- **Dash-Hintergrund-Job (turnkey):** `background_callback_manager = DiskcacheManager(diskcache.Cache(...))`; `@callback(..., background=True, running=[(Output("run","disabled"),True,False)], cancel=[Input("cancel","n_clicks")], progress=[Output(...)])`; `set_progress(...)` ist das erste Callback-Argument.
- **KRITISCH — CUDA im Hintergrund-Worker:** Der DiskCache-Manager führt den Callback in einem **eigenen Prozess** aus. CUDA verträgt **kein `fork`** → Start-Methode **`spawn`** erzwingen (`multiprocessing.set_start_method("spawn", force=True)` ganz früh). **Teste das zuerst** (Sub-Ziel 1): ein winziges `torch.cuda`-Matmul im Hintergrund-Job muss durchlaufen. Erst wenn der GPU-im-Worker-Pfad steht, die restliche UI bauen. Falls der Separate-Prozess-Pfad zu sperrig ist (CUDA-Init-Overhead), **Fallback:** `run()` in einem **Thread** ausführen (behält den residenten CUDA-Kontext) und Fortschritt/Abbruch selbst verwalten — empirisch entscheiden und die Wahl begründen.
- **Doppelklick-Schutz:** Run-Knopf während des Laufs sperren (`running=[...]`) und/oder Lock/Semaphore, damit nie zwei GPU-Jobs kollidieren.
- **Einstieg:** `python -m tool_pipeline` startet die App (`__main__.py` → `app/app.py:main()`); `app.run(host="127.0.0.1", port=8050)` und **URL ausgeben**. Zugriff über SSH-Port-Forward im Browser.
- **Naht-Disziplin:** `app/` importiert **nur** `tool_pipeline.run` und `tool_pipeline.schema` — nichts aus `ir`/`codegen`/`measure` direkt.
- **Harte Regel:** **niemals** `git commit`/`git push`.

## Notfalls recherchieren
Dash-Doku zu Background Callbacks (`background=True`, `DiskcacheManager`, `running`/`cancel`/`progress`, `set_progress`). Bei Problemen: „CUDA + multiprocessing spawn". Lieber empirisch testen als raten.

## Einordnung ins Gesamtsystem
TZ 2 beweist den **GUI↔Core-Vertrag** und den **Live-Hintergrund-Loop** — die riskanteste GUI-Mechanik (mehrsekündiger GPU-Job im Dash-Worker). Danach entwickeln alle späteren Teil-Ziele *durch die GUI*: TZ 3 ergänzt dtype-Dropdown + zwei Charts, TZ 4 Tile/Swizzle + volle Messgrößen, TZ 5 Roofline, TZ 6 Operanden-Liste. Deshalb Controls & Callback so bauen, dass weitere Eingaben nur **ergänzt** werden (nicht umgebaut).

## Konventionen
- Prosa/Kommentare **Deutsch**; sauberes, professionelles Layout (Sidebar mit Controls + Main mit Ergebnissen) — Feinschliff/Theming ist TZ 8, aber ein klares `dash-bootstrap`-Layout schon jetzt.
- `python -m ...` aus `project/` ausführen; `__main__.py` → App-Start; `app/app.py` enthält `main()`.

## Scope-Grenzen (was TZ 2 NICHT tut)
- **Keine** Charts (`app/components/charts.py` bleibt leer → TZ 3), **kein** dtype-Dropdown/Tile-Slider/Swizzle (TZ 3/4), **keine** allgemeine Kontraktion/Operanden-Liste (TZ 6), **keine** Roofline (TZ 5).
- Core-Dateien (`ir`/`codegen`/`measure`/`store`) **nicht** implementieren (das ist TZ 1). Nur `app/`, `__main__.py`, ggf. `schema.py` (falls fehlend) und ein **klar markierter** Mock-`run()`.
- Controls in TZ 2 = nur die Größen M/N/K (Operation fix angezeigt).

## Start-Skizzen (Vorschlag — anpassen)
`RunConfig`/`RunResult` (identisch zu TZ 1 — falls `schema.py` fehlt, so anlegen):
- `RunConfig`: `family="contraction"`, `expr="ik,kj->ij"`, `inputs=["ik","kj"]`, `output="ij"`, `dim_sizes={"i":…,"k":…,"j":…}`, `dtype="fp16"`, `acc_dtype="fp32"`, `tile={"TM":128,"TN":128,"TK":64}`, `swizzle=False`, `baselines=[]`.
- `RunResult`: `status` ∈ {`ok`,`verify_failed`,`compile_error`,`run_error`}, `config`, `kernel_path`, `kernel_source` (str, fürs Code-Panel), `accuracy={"max_abs_err":…,"passed":bool}`, `timing={"compile_ms":…,"run_ms":…}`, `metrics={"tflops":…}`, `provenance={"gpu":"GB10","dtype":"fp16","sizes":{…},"timestamp":…}`, `error?`.

Mock (nur falls echter Core fehlt), schema-konform:
```
# tool_pipeline/run.py — MOCK, später durch echten run() ersetzt (keine GUI-Änderung nötig)
import time, torch
def run(config):
    x = torch.randn(256, 256, device="cuda"); (x @ x); torch.cuda.synchronize()  # GPU-im-Worker beweisen
    time.sleep(2)  # simuliert Compile+Messung
    return RunResult(status="ok",
                     kernel_source="@ct.kernel\ndef gen(A, B, C, ...):\n    ...",
                     accuracy={"max_abs_err": 1.7e-4, "passed": True},
                     timing={"compile_ms": 1800, "run_ms": 4.7},
                     metrics={"tflops": 71.4}, provenance={"gpu": "GB10", ...})
```

Dash-Grundgerüst (Ausschnitt):
```
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)   # CUDA-safe, ganz früh
import diskcache
from dash import Dash, dcc, html, Input, Output, State, DiskcacheManager
mgr = DiskcacheManager(diskcache.Cache("./.cache"))
app = Dash(__name__, background_callback_manager=mgr)

@app.callback(
    Output("kpis", "children"), Output("code", "children"),
    Input("run", "n_clicks"),
    State("M", "value"), State("N", "value"), State("K", "value"),
    background=True,
    running=[(Output("run", "disabled"), True, False)],
    cancel=[Input("cancel", "n_clicks")],
    progress=[Output("prog", "value"), Output("prog", "label")],
    prevent_initial_call=True,
)
def on_run(set_progress, n, M, N, K):
    set_progress(("25", "generiere & compile…"))
    from tool_pipeline.run import run
    from tool_pipeline.schema import RunConfig
    res = run(RunConfig(dim_sizes={"i": M, "k": K, "j": N}))  # + feste Felder
    set_progress(("100", "fertig"))
    return render_kpis(res), res.kernel_source
```

## Arbeitsweise (verbindlich)
1. Lies die Dateien, bestätige dein Verständnis **kurz**.
2. **Zerlege TZ 2 in Sub-Ziele + geordnete TODOs.** Empfohlenes **erstes** Sub-Ziel: den **GPU-im-Hintergrund-Worker-Pfad** mit einem Minimal-Callback beweisen (`spawn`!), *bevor* die volle UI steht.
3. Lege mir die Aufschlüsselung **zur Freigabe vor, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**. **Nach jedem TODO: anhalten** und zeigen, wie du verifiziert hast. Bei GUI-TODOs heißt Verifikation: App starten, URL nennen, Callback real auslösen (Server-Log/Ausgabe zeigen) — die **visuelle Abnahme im Browser mache ich** (per Port-Forward). Auf **meine Validierung warten**. Nicht bündeln.
5. Strikt im TZ-2-Scope bleiben (Scope-Creep = spätere Teil-Ziele).

## Definition of Done (TZ 2)
`python -m tool_pipeline` startet die Dash-App; ich setze M/N/K und klicke „Run"; ein **Hintergrund-Job** läuft (Fortschritt sichtbar, abbrechbar, Run gesperrt), berührt echt die GPU, und die Seite zeigt **KPIs** (TFLOP/s, ms, max_abs_err, Compile-Zeit), den **Verify-Status** und den **generierten Code**. Läuft **unabhängig von TZ 1** (gegen echten `run()` *oder* den schema-konformen Mock). Beim späteren Einspielen des echten `run()` sind **keine** GUI-Änderungen nötig.
