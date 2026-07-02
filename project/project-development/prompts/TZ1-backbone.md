# Auftrag: TZ 1 — Backbone des einsum/GEMM-Performance-Explorers umsetzen

Du arbeitest im Repo `/home/mla08/MLA/mla`. Wir bauen die Group-Specific Component „Interaktiver einsum/GEMM-Performance-Explorer" (GPU/cuTile). Dein Auftrag ist **ausschließlich Teil-Ziel 1 (TZ 1, „Backbone")** aus dem Projektplan.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — der vollständige Plan. Besonders **§10 (TZ-1-Definition, maßgeblich)**, §2 (Designentscheidungen), §3 (Codegen C1+B1), §5 (Hardware/dtype), §6 (Codegen-Risiken), §9 (Verzeichnisstruktur).
2. `project/README.md` — Architektur & die **eine Naht** GUI↔Core (`tool_pipeline/run.py`).
3. `project/project-development/analysis/RESULTS_gb10.md` **und** `project/project-development/analysis/dtype_analyse.py` — letzteres ist ein **lauffähiges Referenz-Batched-GEMM über `ct.mma`** (inkl. korrekter Operanden-Orientierung). Nimm es als Vorlage.
4. Bewährte Kernel: `assignments/03_assignment/src/task_02.py` (saubere GEMM-Vorlage), `assignments/03_assignment/src/task_01.py` (dtype/Toleranzen), `assignments/05_assignment/src/kernel.py`, `assignments/06_assignment/src/kernel.py` (Hinweis: A06 permutiert B + tauscht mma-Operanden — *nur* wegen dessen Output-Layout `yx`; plain `ik,kj->ij` braucht das **nicht**).
5. `assignments/10_assignment/src/gen_matmul.py` — der f-String-Codegen-Präzedenzfall des Teams.
6. Die Stub-Dateien unter `project/tool_pipeline/` (jede hat einen Zweck-Docstring) — das sind die zu füllenden Dateien.
7. Die Memory-Dateien (Index in `MEMORY.md`): `gsc-project-plan`, `gsc-codegen-risks`, `gsc-hardware-dtype-facts`.

## Was du wissen musst
- **Echte cuTile-API** (NICHT die fiktive Mockup-API `@cuda.tile.jit`/`tile.dot`): `@ct.kernel`, `ct.bid(i)`, `ct.load(arr, index=(...), shape=(...), padding_mode=ct.PaddingMode.ZERO)`, `ct.full((r,c), 0, dtype=ct.float32)`, `ct.mma(x, y, acc)`, `ct.reshape`, `ct.permute`, `ct.astype(t, dtype)`, `ct.store(arr, index=(...), tile=...)`, `ct.launch(torch.cuda.current_stream().cuda_stream, grid, kernel, (args...))`, `ct.Constant[int]`, `ct.cdiv`.
- **Umgebung:** NVIDIA GB10 (Blackwell sm_121). venv: `source /home/mla08/MLA/mla/.venv/bin/activate` (torch 2.11, `cuda.tile`, triton, cupy vorhanden). Die GPU ist verfügbar — du darfst/sollst echt ausführen. Problemgrößen klein halten (geteilte Maschine).
- **TZ-1-Scope (eng halten!):** NUR `ik,kj->ij` (ein GEMM), dtype **fp16 → fp32-Akku**, **feste** Tile (z. B. TM=128, TN=128, TK=64), **kein** Swizzle, **keine** GUI. Headless über `cli.py`. dtypes / Tile-Slider / Swizzle / GUI / allgemeine Kontraktion sind **spätere** Teil-Ziele — **nicht** vorbauen; ABER die Schnittstellen (`RunConfig`/`RunResult`, `run()`) sauber & erweiterbar anlegen.
- **Größtes Risiko:** `ct.mma`-Operanden-Orientierung erzeugt bei Fehler ein **stilles Falschergebnis** (kompiliert, läuft, liefert falsche Zahl — kein Crash). `dtype_analyse.py` hat die korrekte plain-GEMM-Orientierung — spiegele sie. **Verify-before-trust:** jeder generierte Kernel wird gegen eine `torch`-fp32-Referenz geprüft (`torch.allclose` + `max_abs_err`), *bevor* seine Zahlen verwendet/angezeigt werden.
- **Pipeline von TZ 1** (alle Stufen laufen real, minimal): `schema` → `ir/parse` (GEMM) → `ir/reshape` (Passthrough, Batch=1) → `codegen` (emittiert `@ct.kernel`-Quelltext per f-String) → `compile` (`exec` + `ct.launch`, einfacher Hash-Cache) → `measure/verify` (fp32-Ref, max_err) → `measure/bench` + `metrics` (CUDA-Events → ms, TFLOP/s) → `store` (JSONL + Kernel-Datei). `run.py` orchestriert, `cli.py` stößt an.
- **Persistenz:** generierter Quelltext nach `project/results/kernels/<slug>.py` (lesbarer Slug, kein Hash); ein Ergebnis-Objekt je Lauf nach `project/results/results.jsonl`.
- **Harte Regel:** **niemals** `git commit` / `git push` in diesem Repo.

## Notfalls recherchieren
Wenn ein cuTile-Detail nicht zu den Assignments passt: empirisch auf der GPU prüfen (`python -c "import cuda.tile as ct; help(ct.mma)"` o. ä.) oder die cuTile-Doku (docs.nvidia.com/cuda/cutile-python). Lieber empirisch verifizieren als raten.

## Einordnung ins Gesamtsystem
TZ 1 ist das **Rückgrat** — die dünnste vertikale Scheibe durch die *gesamte* Pipeline. Ist es fertig, erweitern alle späteren Teil-Ziele nur einzelne Stufen (TZ 2 = GUI um `run()`; TZ 3 = dtypes; TZ 4 = Tile/Swizzle + volle Messung; TZ 5 = Roofline; TZ 6 = allgemeine Kontraktion via echtem Reshape; TZ 7 = memory-bound Ops; TZ 8 = Politur + Report). Deshalb müssen `RunConfig`/`RunResult` und `run()` **sauber und erweiterbar** sein — TZ 1 etabliert die Schnittstellen, nicht nur „GEMM läuft".

## Konventionen
- Repo-Konvention: Prosa/Kommentare auf **Deutsch** (wie in den Assignments), Code-Stil an die Umgebung anpassen, saubere Docstrings.
- `tool_pipeline` ist ein Paket → aus `project/` heraus ausführen (`cd project && python -m tool_pipeline.cli ...`) oder `PYTHONPATH=project`. `cli.py` braucht eine `main()` + `if __name__ == "__main__"`. `python -m tool_pipeline` (über `__main__.py`) ist später der GUI-Einstieg (TZ 2) — in TZ 1 nicht nötig.
- Eingaben deterministisch: `torch.manual_seed(0)` vor `torch.randn(..., dtype=torch.float16, device="cuda")`.
- TZ-1-Größen **tile-teilbar** wählen (z. B. M=N=K=512 bei Tile 128/128/64) → **kein Padding** nötig (Padding/Masking = TZ 8, Risiko ⑤). Klein halten.
- Sauber trennen: jede Pipeline-Stufe in ihre Datei (siehe §9). `app/` wird in TZ 1 nicht angefasst.

## Scope-Grenzen (was TZ 1 NICHT tut)
- Mess-Schicht in TZ 1 **minimal**: ein paar CUDA-Event-getaktete Iterationen mit Warmup + `torch.cuda.synchronize()` → Median-ms + TFLOP/s. **Keine** L2-Flush / Verteilung / GB/s / arithm. Intensität / %-Peak / Provenienz — das ist TZ 4. `metrics.py` in TZ 1 nur: `TFLOP/s = 2*M*N*K / (ms*1e-3) / 1e12`.
- **Kein** Swizzle, **kein** dtype außer fp16→fp32, **keine** GUI, **keine** allgemeine Kontraktion. `reshape.py` ist in TZ 1 nur ein Passthrough (Batch=1); der echte B1-Reshape (config/optimizer-getrieben) kommt in TZ 6 — jetzt **nicht** vorbauen.
- Aber: `RunConfig`/`RunResult`/`run()` so anlegen, dass spätere Achsen (dtype, Tile, Swizzle, Familie, Baselines) nur **Felder/Zweige ergänzen**, nicht umbauen.

## Start-Skizzen (Vorschlag — anpassen, nicht blind übernehmen)
`RunConfig` (Eingabe): `family="contraction"`, `expr="ik,kj->ij"`, `inputs=["ik","kj"]`, `output="ij"`, `dim_sizes={"i":512,"k":512,"j":512}`, `dtype="fp16"`, `acc_dtype="fp32"`, `tile={"TM":128,"TN":128,"TK":64}`, `swizzle=False`, `baselines=[]`.

`RunResult` (Ausgabe): `status` ∈ {`"ok"`,`"verify_failed"`,`"compile_error"`,`"run_error"`}, `config` (Echo), `kernel_path`, `accuracy={"max_abs_err":…,"passed":bool}`, `timing={"compile_ms":…,"run_ms":…}`, `metrics={"tflops":…}`, `provenance={"gpu":"GB10","dtype":"fp16","sizes":{…},"timestamp":…}`, `error` (optional, bei Fehlern).

Kanonisches Kernel-Körper-Muster (cuTile, fp16-Inputs → fp32-Akku; **gegen `dtype_analyse.py` abgleichen**):
```
acc = ct.full((TM, TN), 0, dtype=ct.float32)
for kk in range(ct.cdiv(K, TK)):
    a = ct.load(A, index=(i, kk), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)  # (M,K)-Kachel
    b = ct.load(B, index=(kk, j), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)  # (K,N)-Kachel
    acc = ct.mma(a, b, acc)            # Orientierung: a=(TM,TK), b=(TK,TN) -> (TM,TN); gegen fp32-Ref prüfen!
ct.store(C, index=(i, j), tile=ct.astype(acc, C.dtype))
# Grid = (ct.cdiv(M, TM), ct.cdiv(N, TN)); i = ct.bid(0); j = ct.bid(1)
```
(Exakte Tupel/Reshapes/Orientierung aus `dtype_analyse.py` übernehmen — nicht raten.)

Kernel-Cache & Dateiname: **lesbarer Slug** aus der normalisierten Config `(expr, dtype, acc_dtype, tile, swizzle)`, z. B. `ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py` → `project/results/kernels/<slug>.py`. Existieren Datei + kompiliertes Objekt schon → wiederverwenden (nicht neu compilen).

Verifikation: `ref = torch.einsum("ik,kj->ij", a.float(), b.float())`; `max_abs_err = (out.float() - ref).abs().max().item()`; fp16-Toleranz ~ `atol=2e-1, rtol=2e-2` (vgl. A03/A05).

Store: je Lauf `json.dumps(result)` als **eine Zeile** an `project/results/results.jsonl` anhängen (mit `pandas.read_json(path, lines=True)` ladbar). Generierten Quelltext zusätzlich als `.py` ablegen (s. Cache).

## Arbeitsweise (verbindlich)
1. Lies die genannten Dateien, bestätige dein Verständnis **kurz**.
2. **Zerlege TZ 1 in sinnvolle Sub-Ziele und konkrete, geordnete TODOs** (jedes TODO lässt den Code in einem lauffähigen, prüfbaren Zustand).
3. Lege mir diese Aufschlüsselung **zur Freigabe vor, BEVOR** du Code schreibst.
4. Setze dann **TODO für TODO** um. **Nach jedem TODO: anhalten**, zeigen was du getan hast + **wie du es verifiziert hast** (real ausführen, Ausgabe zeigen), und auf **meine Validierung warten**, bevor du weitermachst. **Nicht** mehrere TODOs bündeln.
5. Halte dich strikt an den TZ-1-Scope; widerstehe Scope-Creep (alles andere sind spätere Teil-Ziele).

## Definition of Done (TZ 1)
`python -m tool_pipeline.cli ...` (aus `project/`) führt `ik,kj->ij` in fp16 auf der GPU aus und liefert/persistiert: Pfad des generierten Kernels, Verify-Ergebnis (`max_abs_err`, Pass/Fail vs fp32), Laufzeit (ms), TFLOP/s; hängt **eine** Zeile an `results.jsonl` und schreibt die Kernel-`.py`. Korrektheit gegen fp32 verifiziert. `run(config)` ist die einzige Naht und gibt ein vollständiges `RunResult` zurück.
