# Findungen & Verbesserungen — QA-Review cuTile Performance Lab

## 0. Prüfumfang & Methodik

**Gegenstand.** Group-Specific Component „cuTile Performance Lab" (`project/`) +
Sphinx-Kapitel `sphinx/source/chapters/group_specific_component/`.

| | |
|---|---|
| Branch / Commit | `feature/tz9` @ `62cf75f9a53c19a76cdfd7186c37821b257897b3` („TZ 9", 2026-07-25 14:39) |
| Host | GPU **vorhanden** (`torch 2.11.0+cu130`, `torch.cuda.is_available() == True`), venv `/home/mla07/mla/.venv` |
| Code-Umfang | `tool_pipeline/` 7 874 LOC (40 Module) · `tests/` 13 Dateien / 4 508 LOC · `project-development/analysis/` |
| Gelesen (vollständig) | `README.md`, `PLAN.md`, `run.py`, `schema.py`, `parse.py`, `reshape.py`, `verify.py`, `metrics.py`, `bench.py`, `fusion.py`, `hardware.py`, `store.py`, `compile.py`, `emit.py`, alle drei `codegen/templates/*`, `report.rst` |
| Gelesen (gezielt) | `callbacks.py`, `controls.py`, `cli.py`, `report_figures.py`, `charts.py`, `history.py`, `tests/test_measure.py`, `tests/test_codegen.py`, `projektplan.rst`, `index.rst`, `presentation.rst`, `installation_und_benutzung.rst`, `ki_einsatz.rst`, `overview.rst`, `.github/workflows/docs.yml` |

### Ausgeführt

**1) Testsuite** — `cd project && python3 -m pytest tests/ -q`

```
........................................................................ [ 25%]
........................................................................ [ 50%]
........................................................................ [ 75%]
......................................................................   [100%]
286 passed in 5.98s
```

**286 passed, 0 failed, 0 skipped.** Wichtig für die Einordnung: Es wurde **nichts
übersprungen**, weil dieser Host eine GPU hat — die GPU-Tests liefen wirklich
(`pytest tests/test_measure.py --durations`: `test_fusion_metrics_consistent` 1,04 s,
`test_baselines_cublas_naive_real` 0,37 s, echte `run()`-Aufrufe). Die kurze Gesamtlaufzeit
erklärt sich durch den **warmen Compile-Cache** (285 Dateien in `results/kernels/`) plus
prozesslokalen `_MODULE_CACHE`. Auf einem GPU-losen Host wäre das Bild ein anderes — siehe
**F-08**: 8 Tests würden dann still grün durchlaufen, ~55 Tests in `test_codegen.py` würden
dagegen hart scheitern (kein Guard).

**Store-Isolation geprüft:** `results/results.jsonl` md5 **vor** und **nach** dem Lauf
identisch (`e9773b40a2beaea531f74d191106bdf3`, 87 Zeilen), `results/kernels/` unverändert
bei 285 Dateien. `git status` nach dem Testlauf zeigt keine neuen Änderungen. **Kein
Befund** — die Tests patchen `store.append_result` sauber weg.

**2) Sphinx** — `cd sphinx && make html` → `build succeeded.` (war ein No-op: „no targets
are out of date"). Für eine belastbare Warning-Liste zusätzlich ein **erzwungener
Vollbau** `sphinx-build -E -a`:

```
build succeeded, 1 warning.
chapters/10_xdna_whole_npu/loesung.rst:144: WARNING: Failed to create a cross reference.
    A title or caption not found: 'ch10_access' [ref.ref]
```

Die einzige Warning liegt **außerhalb** der Group-Specific Component. Im lokalen
Arbeitsbaum ist das GSC-Kapitel warnungsfrei.

**3) CI-Gegenprobe (entscheidend).** `.github/workflows/docs.yml` baut auf `main` und
`group-specific-component` — nicht auf `feature/tz9`. Ich habe den Baum von
`group-specific-component` per `git archive` ausgepackt und dort gebaut:

```
build succeeded, 5 warnings.
report.rst:134: WARNING: image file not readable: _static/gsc/roofline.png
report.rst:155: WARNING: image file not readable: _static/gsc/durchsatz_formate.png
report.rst:164: WARNING: image file not readable: _static/gsc/genauigkeit_durchsatz.png
report.rst:214: WARNING: image file not readable: _static/gsc/tile_swizzle.png
```

**4) Eigene GPU-Proben** (read-only, außerhalb des Repos in einem Scratch-Verzeichnis):
Akkumulator-dtype der Reduktion, Trennschärfe der verify-Toleranzen, Reproduktion des
`verify_failed`-Werts. Belege stehen bei den jeweiligen Befunden.

**5) Faktenabgleich** Report ↔ `results/results.jsonl` (87 Zeilen, 4 Lauf-Gruppen) per
Skript — jede Zahl der fünf Report-Tabellen einzeln nachgeschlagen.

### Nicht verifiziert / bewusst ausgelassen

* **Dash-GUI im Browser.** Kein Live-Klicktest der Oberfläche (kein Browser/Display in
  dieser Sitzung). Die Callback-**Logik** ist über `execute_run` (Dash-frei) und
  `tests/test_app_*.py` (116 Tests) geprüft, die visuelle Darstellung nicht.
* **Sehr große Shapes / OOM-Verhalten.** Auf der geteilten Maschine bewusst nicht
  provoziert.
* **`opt_einsum`-Pfad.** Das Paket ist auf diesem Host **nicht installiert**, der n-är-Planer
  lief im Links-nach-rechts-Fallback (siehe **F-10**). Der opt_einsum-Zweig
  (`parse.py:141-147`) ist daher **nicht ausgeführt** worden.
* **Reale Bandbreite der GB10.** Die Peak-Zahlen (273 GB/s, 213 TFLOP/s) sind aus
  `PLAN.md` §5 / `RESULTS_gb10.md` übernommen und nicht unabhängig nachgemessen.

---

## 1. Gesamturteil

Das Projekt ist **technisch reif und deutlich überdurchschnittlich dokumentiert**. Die
Architektur ist keine nachträgliche Erzählung, sondern im Code durchgehalten: Die Naht
`run(config) -> RunResult` existiert wirklich, `run()` wirft nie, jeder Ausgang ist ein
Status, und die Charts sind reine Funktionen. Besonders stark ist die **Ehrlichkeit der
Messkette** — verify-before-trust ist kein Slogan, sondern greift nachweislich (ich habe
das Gate mit vertauschten Operanden und Nullausgaben gegengeprüft: es fängt beide, in jeder
getesteten Größe). Der Faktenabgleich Report ↔ `results.jsonl` ist der beste, den ich in
einem Studienprojekt gesehen habe: **alle 35 geprüften Zahlen** der fünf Report-Tabellen
stammen nachweisbar aus **einer** Sweep-Charge (`run_id 0fce270e…`, 24 Configs, 23 ok, 1
verify_failed) und stimmen auf die angegebene Stelle.

Die drei gefährlichsten Schwächen sind: **(1)** Der ausgelieferte Stand existiert so
**nirgends im Git** — sechs Code-/Testdateien und alle fünf Report-Figuren sind untracked,
`project/` fehlt auf `main` vollständig, und die CI baut den Report mit vier kaputten
Bildern. **(2)** Der single-shot-Pfad des Reduktions-Kernels **ignoriert `acc_dtype`** und
summiert im Eingabeformat; dadurch ist eine Genauigkeitszahl im Report falsch attribuiert
und die prominenteste „verify-before-trust in Aktion"-Anekdote erklärt einen
**Codegen-Defekt als Eigenschaft von bf16**. **(3)** Die „headless" CLI importiert
`app/components/controls` und ist ohne Dash nicht ladbar — die Naht ist an dieser Stelle
gebrochen, und `requirements.txt` fehlt `matplotlib`, sodass die dokumentierte
Figuren-Reproduktion in einer frischen Umgebung nicht läuft.

Alles drei ist in zwei Tagen behebbar; **(1)** ist reine Git-Arbeit, **(2)** eine Zeile
Codegen plus zwei Absätze Report.

### Ampel je Achse

| Achse | Ampel | Kurzbegründung |
|---|---|---|
| **Architektur** | 🟡 gelb | Naht real und durchgehalten, aber ein echter Rückwärts-Import (`cli.py` → `app/`) und drei Doku-Abweichungen; `run()` mit 328 Zeilen an der Grenze |
| **Korrektheit** | 🟡 gelb | Kontraktion (alle dtypes, ragged, batched, Orientierung) exzellent belegt; **ein** echter Defekt im Reduktions-Akkumulator (F-02) |
| **Tests** | 🟢 grün | 286 Tests, ragged × dtype × Familie systematisch, Store sauber isoliert, keine Assertion-losen Tests; einziger Mangel: Skip-Politik (F-08) |
| **Doku / Report** | 🟡 gelb | Report ist inhaltlich und sprachlich stark und faktentreu; **eine** falsche Einsicht (F-02), `presentation.rst` ist ein TODO-Stub (F-05), Installationskapitel kennt das Tool nicht (F-09) |
| **Abgabe-Hygiene** | 🔴 rot | Untracked-Blocker, kein Merge, CI baut kaputte Figuren (F-01) |

---

## 2. Befunde

### F-01 — Der ausgelieferte Stand existiert nicht im Git; die CI baut den Report mit kaputten Figuren

| | |
|---|---|
| **Schwere** | **S1 — Blocker** |
| **Fundstelle(n)** | `git status`; `.github/workflows/docs.yml:3-5`; `sphinx/source/chapters/group_specific_component/report.rst:138,159,168,218,387` |

**Beobachtung.** `git status` auf `feature/tz9` zeigt sechs untracked Dateien, die
zwingend zum Tool gehören, plus das komplette Figuren-Verzeichnis:

```
?? project/tool_pipeline/measure/fusion.py          ← von run.py:44 importiert
?? project/tool_pipeline/report_figures.py          ← erzeugt alle Report-Figuren
?? project/tool_pipeline/app/components/history.py
?? project/tests/test_cli.py
?? project/tests/test_store.py
?? sphinx/source/_static/gsc/                       ← alle 5 Report-Figuren
```

Dazu drei getrennte Zustände, die zusammen den Blocker ergeben:

1. **`project/` existiert auf `main` überhaupt nicht.**
   `git ls-tree -d --name-only main -- project` liefert nichts. `feature/tz9` ist
   **53 Commits vor `main`** und **1 Commit vor `group-specific-component`**.
2. **Die CI baut den falschen Stand.** `docs.yml` triggert auf `[main,
   group-specific-component]`. Der TZ-9-Commit (Fusion — Template-Zweig, `fusion.py`,
   das ganze Fusions-Kapitel des Reports) ist in **keinem** der beiden Branches.
3. **Die Figuren sind in keinem Branch.** Verifiziert per `git archive
   group-specific-component` + Vollbau: **4 Missing-Image-Warnings**, der deployte
   Report zeigt vier leere Bildrahmen. Die fünfte (`fusion.png`) fehlt dort gar nicht
   erst, weil das Fusions-Kapitel selbst noch nicht auf dem Branch ist.

Ein frischer Klon von `main` enthält das Tool nicht; ein frischer Klon von
`group-specific-component` enthält es ohne `fusion.py` — und `run.py:44`
(`from .measure.fusion import measure_sequential`) ist ein **Top-Level-Import**, das Tool
startet dort also gar nicht.

**Warum es zählt.** Die Abgabe am 27.07. verweist per `git_link.txt`/`report_link.txt` auf
Repo und GitHub-Pages-Report. Beide zeigen aktuell nicht das, was bewertet werden soll: das
Repo zeigt kein lauffähiges Tool, die Pages-Seite einen Report mit vier kaputten Figuren und
ohne das Fusions-Kapitel — also ohne TZ 9, das inhaltliche Highlight.

**Empfehlung.** Kleinste Änderung, in dieser Reihenfolge:
1. `git add` der sechs Dateien + `sphinx/source/_static/gsc/` (5 PNGs), committen.
2. `feature/tz9` → `group-specific-component` mergen (die CI baut dann den vollständigen
   Report), anschließend → `main`.
3. Nach dem Push den Pages-Build prüfen: **0** Missing-Image-Warnings.
4. Prüfen, ob `.gitignore` die PNGs blockt — tut sie nicht, sie wurden schlicht nie
   hinzugefügt.

**Belegt durch.** `git status`, `git merge-base --is-ancestor`, `git ls-tree main`,
`git archive group-specific-component` + `sphinx-build -E -a` (Ausgabe in §0).

---

### F-02 — Reduktions-Kernel ignoriert `acc_dtype`; der Report erklärt den Defekt als Eigenschaft von bf16

| | |
|---|---|
| **Schwere** | **S1 — Blocker** (falsche Zahl + falsche Einsicht im bewerteten Report) |
| **Fundstelle(n)** | `tool_pipeline/codegen/templates/reduction.py:90-97` (`row_sum_single`) gegen `:100-111` (`row_sum_loop`); `sphinx/source/chapters/group_specific_component/report.rst:278-282, 424-434` |

**Beobachtung.** Das Reduktions-Template hat zwei Pfade. Der **Fallback** akkumuliert
korrekt im Akku-dtype:

```python
acc = acc + ct.sum(ct.astype(tile, ct.float32), axis=1)   # :110 — sauber
```

Der **Produktionspfad** (`launch` wählt ihn, solange `next_pow2(K) <= 16384`, also für
praktisch alle Report-Größen) summiert dagegen **im Eingabeformat** und castet erst danach:

```python
ct.store(out, index=(pid,), tile=ct.astype(ct.sum(tile, axis=1), out.dtype))   # :97
```

`acc_ct` kommt in `row_sum_single` **überhaupt nicht vor** — der `acc_dtype`-Regler ist auf
diesem Pfad wirkungslos. Gemessen (`ij->i`, fp16-Eingabe, `acc_dtype=fp32` angefordert):

| K | gewählter Pfad | `max_abs_err` |
|---|---|---|
| 4 096 | single-shot (Akku **fp16**) | 0,1004 |
| 8 192 | single-shot (Akku **fp16**) | 0,2352 |
| 16 384 | single-shot (Akku **fp16**) | 0,25 |
| 32 768 | K-Loop (Akku **fp32**) | **0,000244** |

Bei **doppelter** Reduktionslänge ist der Loop-Pfad drei Größenordnungen genauer — der
Unterschied ist nicht die Länge, sondern der Akkumulator.

**Die Auswirkung auf den Report ist die eigentliche Schwere.** Zwei Stellen:

* `report.rst:278-282` weist „Reduktion · sum · **fp16 → fp32** · 171 GB/s" aus. Der
  zugehörige Store-Wert `max_abs_err = 0,2203` (`results.jsonl`) ist der **fp16**-Akku,
  nicht der ausgewiesene fp32-Akku.
* `report.rst:427-430` — die Schlussanekdote des Kapitels *verify-before-trust in Aktion*:
  > „Die bf16-Reduktion über 4096 Elemente überschritt die Toleranz (max. abs. Fehler
  > **1,57**), **weil sich der bf16-Rundungsfehler über Millionen Summanden aufaddiert**"

  Das ist die falsche Ursache. Gegenprobe mit demselben Input, 4096²:

  ```
  single-shot (Produktionspfad, Akku = bf16) : max_abs_err = 1.574   ← exakt der Store-Wert
  K-Loop      (Akku = ct.float32)            : max_abs_err = 3.052e-05
  ```

  Mit dem Akkumulator, den `acc_dtype=fp32` zusagt, wäre der Fehler **51 000× kleiner** und
  der Lauf hätte **bestanden**. Der Report präsentiert also einen **Codegen-Defekt als
  physikalische Grenze eines Zahlenformats** — ausgerechnet in dem Abschnitt, der die
  Sorgfalt des Projekts demonstrieren soll.

**Warum es zählt.** (a) Eine im Report veröffentlichte Genauigkeitszahl trägt ein falsches
Etikett. (b) Eine explizit als „Einsicht" formulierte Aussage ist sachlich falsch und im
Review widerlegbar. (c) Der `acc_dtype`-Regler ist für eine der drei Familien eine
Attrappe. Das verify-Gate hat hier übrigens **korrekt** gearbeitet (es hat den bf16-Lauf
verworfen) — das Prinzip funktioniert, nur die Deutung des Fundes ist falsch.

**Empfehlung.** Kleinste Änderung — eine Zeile, exakt analog zum bereits vorhandenen
Loop-Pfad:

```python
# reduction.py:97
ct.store(out, index=(pid,),
         tile=ct.astype(ct.sum(ct.astype(tile, ACC_CT), axis=1), out.dtype))
```

Dann: `python -m tool_pipeline.cli --sweep` neu fahren, `report_figures` neu erzeugen, die
zwei Reduktions-Zeilen der Tabelle aktualisieren — und den Absatz `report.rst:427-434`
umschreiben. **Er wird dadurch stärker, nicht schwächer**: Aus „bf16 ist zu ungenau" wird
„das verify-Gate hat einen echten Kernel-Defekt gefangen, bevor eine Zahl in den Report
kam" — die überzeugendere Geschichte. Sollte der Sweep nach dem Fix keinen `verify_failed`
mehr liefern, muss der Abschnitt ohnehin neu erzählt werden (Rückfrage **R-1**).

*Hinweis:* Nach dem Fix ändert sich der emittierte Quelltext ⇒ die getrackten
`results/kernels/*__sum.py` werden neu geschrieben. Das ist erwartet und in Ordnung, sollte
aber bewusst mitcommittet werden.

**Belegt durch.** Gelesener Code + zwei eigene GPU-Proben (Werte oben, 1,574 exakt
reproduziert) + `results.jsonl`.

---

### F-03 — Die „headless" CLI ist ohne Dash nicht ladbar (Rückwärts-Import Core → app)

| | |
|---|---|
| **Schwere** | **S2 — Major** |
| **Fundstelle(n)** | `tool_pipeline/cli.py:32`; `tool_pipeline/app/components/controls.py:32-33`; `README.md:32`; `PLAN.md` §9 |

**Beobachtung.** `cli.py` — laut `PLAN.md` §9 „headless / Batch-Sweeps für den Report" —
importiert aus der GUI-Schicht:

```python
from .app.components import controls   # torch-frei (nur schema + parse + dash) → headless
```

Der Kommentar nennt die Ursache selbst („… + dash"), zieht aber die falsche Konsequenz.
`controls.py:32-33` importiert `dash_bootstrap_components` und `dash` auf Modulebene.
Gegenprobe:

```
CLI ohne dash importierbar: NEIN -> ModuleNotFoundError: No module named 'dash.development'
```

Damit ist die Schichtung `ir → codegen → measure → store → app` an genau einer Stelle
**rückwärts** verletzt: Ein Modul auf Core-Ebene hängt am GUI-Paket. Zusätzlich widerspricht
das der README-Aussage (`README.md:32`):

> „**Prinzip:** `app/` importiert ausschließlich `run.py` und `schema.py` … Dadurch ist der
> Core headless testbar (`tests/`, `cli.py`) **ohne Dash**"

**Hier ist die Doku falsch bzw. die Realität hat sie überholt** — die Richtung, um die es
geht (`app/` → Core), stimmt zwar bis auf F-04, aber der daraus abgeleitete Nutzen
(„headless ohne Dash") gilt nicht mehr.

**Warum es zählt.** Der CLI-Sweep ist der dokumentierte, reproduzierbare Weg zu allen
Report-Zahlen. Er auf einem Rechner ohne GUI-Stack (Batch-Node, CI, Zweitmaschine) nicht
lauffähig. Und es ist der einzige Punkt, an dem die im Report prominent verkaufte
Architektur-Aussage („die eine Naht") angreifbar ist — ein Prüfer, der `cli.py:32` liest,
findet ihn sofort.

**Empfehlung.** Kleinste Änderung: Die von `cli.py` genutzten Helfer sind reine
Validierungs-/Config-Bau-Funktionen ohne Dash-Bezug. Entweder (a) den Import in `cli.py`
**lazy** in die Funktionen ziehen, die ihn brauchen — dann bleibt der Modul-Import
Dash-frei —, oder (b) sauberer: `dash`/`dbc` in `controls.py` nur dort importieren, wo
Layout gebaut wird. Variante (a) ist die kleinere und für zwei Tage vor Abgabe die richtige.
Danach `README.md:32` präzisieren.

**Belegt durch.** Gelesener Code + Import-Probe (Ausgabe oben).

---

### F-04 — README/PLAN behaupten eine engere Naht, als der Code hat (`app/` → `store`)

| | |
|---|---|
| **Schwere** | **S3 — Minor** |
| **Fundstelle(n)** | `tool_pipeline/app/callbacks.py:40`; `README.md:32`; `PLAN.md` §2 („Die eine Naht") |

**Beobachtung.** README und Auftrag formulieren: „`app/` importiert im Live-Loop **nur**
`run.py` + `schema.py`". Tatsächlich importiert `callbacks.py:40` auf Modulebene zusätzlich

```python
from ..store import store   # torch-frei (pandas lazy) → im Haupt-Prozess GPU-frei ladbar
```

Der **Live-Loop** hält die Zusage (`run` wird bei `callbacks.py:322` bewusst lazy geholt,
der Hauptprozess bleibt CUDA-frei — das ist sauber). Der Store-Import dient der
History-Verwaltung, ist torch-frei und harmlos. Aber die Formulierung „nur run + schema"
stimmt so nicht.

**Warum es zählt.** Kein Laufzeitrisiko. Aber die Naht ist das zentrale Architektur-Argument
des Reports (`report.rst:56-78`); eine Aussage, die beim Nachlesen nicht hält, kostet in der
Bewertung mehr als die Sache wert ist.

**Empfehlung.** Doku angleichen (nicht den Code): In `README.md:32` präzisieren auf
„`app/` importiert aus dem Core ausschließlich `run.py`, `schema.py` und `store/` (Lesen der
History) — `run` zudem lazy, damit der Hauptprozess CUDA-frei bleibt." Der Report muss nicht
angefasst werden; er formuliert bereits vorsichtiger.

**Belegt durch.** `grep` über alle Core-Importe in `app/` (Ergebnis: genau `store`,
`schema`, `run` lazy, `ir.parse` über `controls`).

---

### F-05 — `presentation.rst` ist ein TODO-Stub, obwohl der Vortragstermin 17 Tage zurückliegt

| | |
|---|---|
| **Schwere** | **S2 — Major** |
| **Fundstelle(n)** | `sphinx/source/chapters/group_specific_component/presentation.rst:13-19`; verlinkt aus `.../index.rst:26-28` |

**Beobachtung.** Das Kapitel ist unverändert im Planungszustand:

```rst
.. note:: TODO — wird vorbereitet, sobald das Projekt feststeht.

* **Introduction** — *TODO*
* **Problem formulation** — *TODO*
* **Implemented solution** — *TODO*
* **Results** — *TODO*
```

Der Vortrag war laut `index.rst` am **08.07.2026**, heute ist der **25.07.2026**. Das
Kapitel ist als eigene **Prüfungsleistung** in der Tabelle von `index.rst:14-30` gelistet
und über den Toctree direkt aus der GSC-Startseite erreichbar.

**Warum es zählt.** Vier sichtbare „TODO" plus „sobald das Projekt feststeht" in einer
abgegebenen, öffentlich deployten Dokumentation — direkt neben dem sonst sehr sorgfältigen
Report. Das ist der billigste vermeidbare Punktverlust in der ganzen Abgabe.

**Empfehlung.** Kleinste Änderung: Die vier Bullets durch je 2–3 Sätze ersetzen, was
tatsächlich gezeigt wurde (der Stoff existiert vollständig in `report.rst` und
`slides/`), plus einen Satz zum Demo-Ablauf. 20–30 Zeilen genügen. Alternativ — falls
inhaltlich nichts nachgereicht werden soll — den Abschnitt „Gliederung" streichen und durch
einen Verweis auf die Folien + `report.rst` ersetzen. **Nicht** als TODO stehen lassen.

**Belegt durch.** Gelesene Datei (vollständig, 19 Zeilen).

---

### F-06 — `requirements.txt` fehlt `matplotlib`: die dokumentierte Figuren-Reproduktion läuft in einer frischen Umgebung nicht

| | |
|---|---|
| **Schwere** | **S2 — Major** |
| **Fundstelle(n)** | `project/requirements.txt`; `tool_pipeline/report_figures.py` (Top-Level-Import `matplotlib`, `numpy`); `report.rst:124-126` |

**Beobachtung.** Abgleich aller Top-Level-Drittanbieter-Importe des Pakets gegen
`requirements.txt`:

| Paket | importiert in | in `requirements.txt`? | installiert? |
|---|---|---|---|
| `matplotlib` | `report_figures.py` | **nein** | ja |
| `numpy` | `report_figures.py` | **nein** | ja (transitiv über `pandas`) |
| `opt_einsum` | `ir/parse.py` (lazy) | **nein** | **nein** → s. F-10 |
| dash, dbc, plotly, pandas, diskcache, filelock, kaleido | — | ja | ja |

`matplotlib` ist die echte Lücke (`numpy` kommt über `pandas` mit). `report.rst:124-126`
nennt `python -m tool_pipeline.report_figures` als **den** Weg, die Figuren zu erzeugen —
in einer nach `requirements.txt` gebauten Umgebung scheitert das mit `ModuleNotFoundError`.

**Nebenbefund (S4):** `kaleido==0.2.1` ist mit „serverseitiger PNG-Export der Charts
(`charts.save_png`; Report/Doku)" begründet. `charts.save_png` (`charts.py:700`) existiert,
wird aber von keinem Report-Pfad genutzt — die Report-Figuren kommen aus
`report_figures.py` via matplotlib. Die Abhängigkeit ist damit faktisch ungenutzt; die
Begründung im Kommentar ist irreführend.

**Warum es zählt.** Reproduzierbarkeit ist ein explizites Versprechen des Reports
(„Beide Schritte sind deterministisch und ohne GUI wiederholbar", `report.rst:126`). Genau
dieses Versprechen ist mit der mitgelieferten Paketliste nicht einlösbar.

**Empfehlung.** Zwei Zeilen in `requirements.txt`:
```
matplotlib==<installierte Version>   # Report-Figuren (report_figures.py)
opt_einsum==<version>                # n-är: Kontraktionsreihenfolge (s. F-10)
```
Versionen mit `pip show matplotlib opt_einsum` ablesen und wie die übrigen pinnen. Beim
`kaleido`-Kommentar entweder die Begründung korrigieren oder die Zeile entfernen.

**Belegt durch.** AST-Scan aller `tool_pipeline/**/*.py`-Importe + Import-Probe je Paket.

---

### F-07 — `run()` ist eine 328-Zeilen-Funktion mit drei parallel gepflegten Pipelines

| | |
|---|---|
| **Schwere** | **S3 — Minor** (Wartbarkeit) |
| **Fundstelle(n)** | `tool_pipeline/run.py:298-626`; Teilzweige `:342-408` (memory-bound), `:421-501` (n-är), `:503-626` (2-Op) |

**Beobachtung.** `run()` ist mit **328 Zeilen** die mit Abstand längste Funktion im Projekt
(nächstlängste: `callbacks.register` 249, `build_gemm_module` 215). Sie enthält drei
vollständige, voneinander unabhängige Pipelines, die jeweils dieselbe Sequenz in eigener
Ausprägung wiederholen: IR bauen → Operanden bauen → `load_kernel` → `time_first_launch`
(mit identischem `ct.TileError`/`Exception`-Doppel-`except`) → `verify` → identischer
`if not accuracy["passed"]`-Block → `benchmark` → dieselben sechs `timing[...]`-Zuweisungen
→ family-spezifische Metrik → `_result(STATUS_OK)`.

Der `verify_failed`-Block steht **dreimal** wortgleich (`:384-389`, `:479-484`, `:573-578`),
der Timing-Übernahmeblock **dreimal** (`:396-400`, `:489-493`, `:585-589`), das
`ct.TileError`-Paar **viermal**.

Die Zweige sind bewusst additiv gebaut („der Kontraktions-Flow darunter bleibt UNBERÜHRT")
— das war für die inkrementelle TZ-Entwicklung genau richtig und hat funktioniert. Nach TZ 9
ist der Preis sichtbar: Eine Änderung an der Messkette muss an drei Stellen nachgezogen
werden.

**Warum es zählt.** Kein aktueller Fehler — aber der teuerste Ort für den nächsten. TZ 10
(Copy/Transpose) würde einen **vierten** Zweig desselben Musters hinzufügen.

**Empfehlung.** **Vor dem 27.07. nichts anfassen** — das Risiko/Nutzen-Verhältnis stimmt
zwei Tage vor Abgabe nicht. Danach die kleinste wirksame Maßnahme: die drei identischen
Blöcke in lokale Helfer ziehen (`_apply_timing(timing, b)`, `_verify_or_fail(...)`,
`_cold_launch_or_fail(...)`). Das schrumpft `run()` um ~60 Zeilen, ohne die
Zweig-Struktur — und damit die bewährte Additivität — anzutasten.

**Belegt durch.** AST-Analyse (Funktionslängen) + gelesener Code.

---

### F-08 — Uneinheitliche Skip-Politik: 8 GPU-Tests melden ohne GPU „passed", 55 andere würden hart scheitern

| | |
|---|---|
| **Schwere** | **S3 — Minor** |
| **Fundstelle(n)** | `tests/test_measure.py:265-267, 406-408, 433-435, 457-459, 481-483, 509-511, 535-537, 558-560`; demgegenüber `tests/test_codegen.py` (0 Guards, 11× `device="cuda"`) |

**Beobachtung.** Acht Tests in `test_measure.py` schützen sich so:

```python
if not _has_cuda():
    print("  (übersprungen: keine CUDA-GPU)")
    return
```

Das ist **kein** Skip — pytest zählt sie als **passed**. `pytest.skip` / `mark.skipif`
kommt im gesamten `tests/`-Baum **null** mal vor. Auf einem GPU-losen Host meldet die Suite
also grün für Tests, die nichts geprüft haben, und die `print`-Meldung ist unter `-q`
unsichtbar.

Gleichzeitig hat `test_codegen.py` — die 55 Tests, die den Codegen wirklich gegen `torch`
verifizieren — **keine** Guards und würde ohne GPU mit Fehlern abbrechen. Die Suite zerfällt
also nicht sauber in „headless" und „GPU", wie `report.rst:445` nahelegt („Die headless-Tests
laufen ohne GPU; die GPU-Tests verifizieren die Kernel real gegen `torch`").

**Warum es zählt.** „286 passed" auf einem CPU-Rechner wäre eine falsche Sicherheitsaussage
— genau die Klasse Problem, gegen die das Projekt sonst konsequent vorgeht (loud fail statt
stiller Zahl). Für die Bewertung relevant: Ein Prüfer, der die Suite ohne GPU laufen lässt,
sieht ein anderes Bild als der Report beschreibt.

**Empfehlung.** Kleinste Änderung: die acht `if not _has_cuda(): … return` durch

```python
pytest.importorskip("torch")            # oder:
if not _has_cuda():
    pytest.skip("keine CUDA-GPU", allow_module_level=False)
```

ersetzen — dann meldet pytest ehrlich `8 skipped`. Für `test_codegen.py` genügt vor der
Abgabe ein Satz im Report (`report.rst:445`), der klarstellt, dass die Codegen-Tests eine
GPU **voraussetzen**. Ein `conftest.py` mit einem `gpu`-Marker wäre die saubere Lösung
danach.

**Belegt durch.** AST-/Regex-Scan über `tests/`, `pytest --durations`-Lauf, gelesener Code.

---

### F-09 — Das Installationskapitel der Doku kennt das Tool nicht; `README.md` nennt ein nicht erreichbares venv

| | |
|---|---|
| **Schwere** | **S3 — Minor** |
| **Fundstelle(n)** | `sphinx/source/chapters/installation_und_benutzung.rst` (gesamt); `sphinx/source/chapters/overview.rst`; `project/README.md:44`; `project/requirements.txt:13` |

**Beobachtung.** Zwei zusammenhängende Lücken:

1. **`installation_und_benutzung.rst` erwähnt `project/` mit keinem Wort.** Es beschreibt
   ausschließlich die Assignments 01–10 („GPU-Assignments (01–06)", „NPU-Assignments
   (07–10)"). Kein Treffer für `tool_pipeline`, `project/`, `cuTile Performance` oder
   `python -m`. Auch `overview.rst` nennt die Group-Specific Component nicht. Ein Leser des
   Reports findet in der Doku also **keinen** Weg, das Tool zu installieren oder zu starten.
2. **Der einzige Start-Hinweis steht in `README.md:44` und ist falsch:**
   ```bash
   source /home/mla08/MLA/mla/.venv/bin/activate
   ```
   Der Pfad ist aus diesem Checkout nicht erreichbar (`Permission denied`); das tatsächlich
   genutzte venv liegt unter `/home/mla07/mla/.venv`. Derselbe tote Pfad steht in
   `requirements.txt:13`.

   **Positiv:** Der Pfad steht **ausschließlich in Doku/Kommentaren**, nirgends im Code —
   `store.py:35` leitet alle Pfade korrekt relativ über `Path(__file__).resolve().parents[2]`
   ab. Es gibt keine hartkodierten Pfade im ausführbaren Teil.

**Warum es zählt.** Der Report ist ausdrücklich **selbsttragend** gedacht. Wer ihn liest und
das Tool ausprobieren will, läuft in eine Sackgasse. Der maschinenspezifische venv-Pfad
macht das Projekt zudem an eine fremde Nutzerkennung gebunden.

**Empfehlung.** (a) In `installation_und_benutzung.rst` einen Abschnitt „Group-Specific
Component" mit ~10 Zeilen ergänzen: venv, `pip install -r project/requirements.txt`,
`python -m tool_pipeline` (GUI) und `python -m tool_pipeline.cli --sweep` (headless). (b) In
`README.md:44` und `requirements.txt:13` den absoluten Pfad durch einen relativen bzw. eine
neutrale Formulierung ersetzen („das venv des GPU-Hosts aktivieren, in dem `torch`,
`cuda.tile`, `triton` liegen").

**Belegt durch.** Gelesene Dateien, `grep` über `/home/mla0`-Vorkommen (nur Doku +
`prompts/`, kein Code).

---

### F-10 — `opt_einsum` ist importiert, aber nicht deklariert und nicht installiert: die n-är-Reihenfolge ist umgebungsabhängig

| | |
|---|---|
| **Schwere** | **S3 — Minor** |
| **Fundstelle(n)** | `tool_pipeline/intermediate_representation/parse.py:136-147`; `requirements.txt` |

**Beobachtung.** `_nary_order()` bestimmt die Kontraktionsreihenfolge per
`opt_einsum.contract_path` und fällt bei `ImportError` still auf einen
Links-nach-rechts-Fold zurück:

```python
except Exception:  # noqa: BLE001 — opt_einsum optional; Fold ist immer korrekt
    return None
```

Auf diesem Host ist `opt_einsum` **nicht installiert** — die Report-Zahl für die n-äre Kette
stammt also aus dem **Fallback**. Der in `results.jsonl` gespeicherte Pfad
(`['ij,jk->ik', 'kl,ik->il']`) entspricht exakt dem Fold, nicht notwendigerweise dem, was
opt_einsum gewählt hätte. `report.rst:297-299` nennt diesen Pfad, ohne die Herkunft zu
erwähnen.

**Der Fallback ist korrekt** — `_validate_pairwise_step` prüft jeden Schritt streng, und
`verify` deckt das Endergebnis ab. Es geht nicht um Richtigkeit, sondern um
**Reproduzierbarkeit**: In einer Umgebung *mit* opt_einsum kann eine andere Zerlegung und
damit eine andere TFLOP/s-Zahl und andere Zwischentensor-Bytes herauskommen.

**Warum es zählt.** Die Report-Zahl „1,64 TFLOP/s bei AI 64" ist nur unter der
undokumentierten Bedingung „ohne opt_einsum" reproduzierbar.

**Empfehlung.** Kleinste Änderung: `opt_einsum` in `requirements.txt` aufnehmen (s. F-06)
**und** die tatsächlich verwendete Planungsquelle in die Provenienz schreiben — in
`_nary_sizes()` (`run.py:287-295`) ein Feld `"path_source": "opt_einsum" | "fold"`
ergänzen. Dann steht in jeder JSONL-Zeile, wie geplant wurde. Für den Report genügt ein
Halbsatz bei `report.rst:297`: „(Pfad per Links-nach-rechts-Zerlegung; mit `opt_einsum`
kann er abweichen)".

**Belegt durch.** Gelesener Code, Import-Probe (`opt_einsum: NICHT installiert`),
`results.jsonl`-Provenienz.

---

### F-11 — `results/kernels/` ist halb getrackt, halb ignoriert

| | |
|---|---|
| **Schwere** | **S3 — Minor** |
| **Fundstelle(n)** | `.gitignore:33` (`/project/results/kernels/`); `git ls-files project/results/` (59 `.py` getrackt); `tool_pipeline/store/store.py:275-279` |

**Beobachtung.** `.gitignore` ignoriert `/project/results/kernels/`, aber **59 Kernel-Dateien
sind bereits getrackt** (vor der Regel committet — `.gitignore` wirkt nicht auf getrackte
Dateien). Auf der Platte liegen **285**. Es sind also 226 Cache-Artefakte unsichtbar und 59
im Git, ohne erkennbares Kriterium, welche.

Der Code weiß darum und formuliert es entwaffnend ehrlich (`store.py:278`):
> „`kernels/` ist der (gitignored, teils git-getrackte) Compile-Cache"

Praktische Folge heute: Ein Lauf, der einen der 59 getrackten Kernel neu emittiert
(z. B. nach dem Fix aus **F-02**), erzeugt eine **Git-Änderung** — mitten in der
Abgabephase, und man merkt es nur, wenn man `git status` liest.

**Warum es zählt.** Kein Korrektheitsrisiko (der Cache heilt sich, `compile.py:112`
vergleicht Inhalte). Aber es macht `git status` als Kontrollinstrument unzuverlässig — und
genau darauf stützt sich die Abgabe-Prüfung aus **F-01**.

**Empfehlung.** Eine Entscheidung, beide Wege sind vertretbar:
* **(a) empfohlen, kleiner:** Die 59 Dateien als bewusste Beispiel-Artefakte behalten und
  die `.gitignore`-Zeile entsprechend kommentieren („nur die eingecheckten Referenz-Kernel
  sind versioniert; neu erzeugte bleiben lokal"). Faktenlage dokumentieren statt ändern.
* **(b) sauberer, riskanter kurz vor Abgabe:** `git rm --cached` für alle 59 → der Cache ist
  vollständig lokal. **Nicht** vor dem 27.07. machen — der Report verweist auf
  `results/kernels/<slug>.py` als reproduzierbares Artefakt.

**Belegt durch.** `.gitignore`, `git ls-files`, `git status --ignored`, Dateizählung.

---

### F-12 — `config_slug` enthält `family` nicht; die Begründung im Docstring stimmt nicht

| | |
|---|---|
| **Schwere** | **S4 — Nit** (latent, über die Oberfläche nicht erreichbar) |
| **Fundstelle(n)** | `tool_pipeline/store/store.py:17-19` (Docstring), `:44-86` |

**Beobachtung.** Der Slug wird bewusst ohne `family` gebildet, begründet mit:
> „**ohne** `family` (die ist eine Funktion von `expr` — der Router leitet sie
> deterministisch ab)"

Das trifft nicht zu: `family` ist ein freies `RunConfig`-Feld, und `parse()` routet auf
`config.family`, **nicht** auf `expr` (`parse.py:373-391`). Derselbe Ausdruck kann in
verschiedenen Familien geparst werden — `ij,ij->ij` ist als `elementwise` eine
Hadamard-Operation und als `contraction` ein Batch-GEMM mit M=N=K=1.

Konstruierte Kollision: `family="contraction", op="add", expr="ij,ij->ij"` ergäbe denselben
Slug wie `family="elementwise", op="add", expr="ij,ij->ij"` — und würde damit das gecachte
Elementwise-Artefakt für eine Kontraktion laden.

**Praktisch nicht erreichbar**, und das ist der Grund für S4: `controls._resolve_op()`
(`controls.py:434-441`) erzwingt `op=None` für jede Kontraktion, und GUI wie CLI bauen ihre
Configs ausschließlich darüber. Ein Angriffspfad besteht nur beim direkten Programmieren
gegen `RunConfig`. Zusätzlich ist der Slug in allen **real erreichbaren** Kombinationen
korrekt trennscharf — `op`, `epilog`, `group_m` sind alle bedingt und richtig eingebaut
(nachgeprüft in `test_epilog_in_slug_conditional`).

**Warum es zählt.** Der Docstring nennt eine Invariante, die nicht gilt. Wer sich später
darauf verlässt (z. B. bei TZ 10 Copy/Transpose, das `family="elementwise"` mit neuen Ops
erweitert), baut auf einer falschen Zusage auf.

**Empfehlung.** Kleinste Änderung: **nur den Docstring korrigieren** — „ohne `family`: in
allen über GUI/CLI erreichbaren Configs ist die Familie durch `op` eindeutig mitkodiert
(Kontraktion ⇒ `op=None`, s. `controls._resolve_op`)". Den Slug selbst **nicht** ändern:
`family` anzuhängen würde alle 285 vorhandenen Kernel-Dateinamen invalidieren.

**Belegt durch.** Gelesener Code (`store.py`, `parse.py`, `controls.py`, `emit.py`).

---

### F-13 — 84 Inline-Styles neben einer vorhandenen `theme.css`

| | |
|---|---|
| **Schwere** | **S4 — Nit** |
| **Fundstelle(n)** | `tool_pipeline/app/components/controls.py` (52×), `kpis.py` (10×), `callbacks.py` (9×), `history.py` (7×), `code_panel.py` (5×), `layout.py` (1×); `tool_pipeline/app/assets/theme.css` |

**Beobachtung.** `theme.css` existiert und wird von Dash automatisch geladen; parallel dazu
stehen 84 `style={...}`-Dictionaries im Python-Code, teils als Modulkonstanten
(`_SECTION` in `callbacks.py`), teils inline.

**Warum es zählt.** Reine Wartbarkeit — eine Farb-/Abstandsänderung muss an zwei Orten
gesucht werden. Für die Bewertung irrelevant, das Ergebnis sieht identisch aus.

**Empfehlung.** **Vor der Abgabe nicht anfassen.** Danach, falls das Projekt weitergeht:
die wiederkehrenden Muster (Abschnittsüberschrift, Hinweistext, KPI-Karte) als CSS-Klassen
nach `theme.css` ziehen und im Python nur noch `className=` setzen. Einmalige Feinheiten
dürfen inline bleiben.

**Belegt durch.** `grep -c "style="` je Modul, Verzeichnis-Listing `app/assets/`.

---

## 3. Positiv-Befunde — beim Aufräumen **nicht** anfassen

**P-1 · Der Faktenabgleich Report ↔ Store ist lückenlos.** Ich habe alle Zahlen der fünf
Report-Tabellen einzeln gegen `results.jsonl` geprüft — Format-Tabelle (16 Werte),
Tile/Swizzle (6), memory-bound (15), n-är (2), Fusion (36 in 6 Zeilen). **Kein einziger
Ausreißer.** Beispiele: Report `28,0 / 36,8 / 76 % / 3,2·10⁻⁴ / 109` ↔ Store
`27.962 / 36.802 / 75,98 % / 0.0003204 / 109.23`. Auch die Prosa hält: „Der Sweep umfasste
24 Konfigurationen; 23 bestanden, eine nicht" ↔ Gruppe `0fce270e…` hat exakt `n=24,
ok=23, verify_failed=1`. Das ist selten und sollte so bleiben.

**P-2 · `report_figures.load_report_rows()` (`report_figures.py:52-64`) ist die richtige
Lösung für ein reales Problem.** Es wählt die **jüngste** `CLI-Report-Sweep`-Charge per
`run_id` und filtert auf `status=="ok"`. Dadurch verunreinigen die drei anderen Lauf-Gruppen
in `results.jsonl` (2 GUI-Batches, 7 identitätslose Altzeilen — mit teils *abweichenden*
Fusions-Speedups wie 2,165 statt 2,222) die Figuren **nicht**, und Tabelle und Figur zeigen
garantiert dieselbe Charge. Der `_is_square`-Filter (`:87-97`) löst das Folgeproblem
sauber. **Diese Filterlogik nicht vereinfachen.**

**P-3 · verify-before-trust hält der Gegenprobe stand.** Ich habe das Gate mit den beiden
Fehlerbildern aus PLAN §6 Risiko ① beschossen — vertauschte mma-Operanden und ein Kernel,
der nichts schreibt — über `bf16/fp16/fp32` × `M=N=K ∈ {32, 128, 512}`: **12 von 12
korrekt als Fehler erkannt**, kein falsch-negativ, und kein falsch-positiv beim korrekten
Ergebnis. Die als „verdächtig lasch" aussehenden Toleranzen (`atol=8.0` für fp16→fp16)
sind für grobe Fehler trennscharf, weil `torch.allclose` **elementweise** prüft. Auch F-02
wurde vom Gate korrekt gefangen. **Die Toleranztabelle nicht „vorsichtshalber" verschärfen**
— sie ist empirisch begründet und funktioniert.

**P-4 · Fehlerbehandlung ist durchgehend loud, nie still.** `run()` gibt in **allen**
Pfaden ein `RunResult` mit Status zurück; `execute_run` (`callbacks.py:203-374`) validiert
sechs Eingabeklassen **vor** dem GPU-Lock, fängt `Timeout` (GPU belegt) mit freundlichem
Text und hat einen Catch-all, der die UI nicht crasht. Sämtliche breiten `except` im Projekt
tragen ein `# noqa: BLE001` **mit Begründung**; es gibt **kein** `except: pass`, das einen
Fehler verschluckt (die zwei `pass` in `store.py:121,255` räumen Temp-Dateien auf — korrekt).
Leere Stores werden in den Charts als `_empty("Noch keine verifizierten Läufe.")` gerendert.

**P-5 · Test-Isolation und ragged-Abdeckung.** Die GPU-Tests patchen `store.append_result`
konsequent mit `try/finally` — nachgewiesen: md5 der `results.jsonl` vor/nach dem Lauf
identisch. Ragged-Randfälle sind für **alle drei Familien** getestet
(`test_gemm_epilog_ragged_dtypes`, `test_elementwise_ragged_dtypes`,
`test_reduction_ragged_dtypes`, `test_reduction_loop_fallback_ragged`), mit echten
Krummgrößen wie `(130,100,70)`, `(129,127,65)`, `(1,1,1)`. Die Behauptung in
`report.rst:88-90` ist damit belegt. Kein Test ohne Assertion (die 20 scheinbar
assertion-losen delegieren an `_assert_*`-Helfer) und **kein** weggemocktes `verify` — die
eine Stelle, die `R.verify` ersetzt (`test_codegen.py:940`), tut das absichtlich, um den
`verify_failed`-**Pfad** zu testen, und stellt sauber wieder her.

**P-6 · Die Fusions-Zweitmessung ist fair.** Geprüft: gleiche Shapes, gleiche
`bench`-Schleife mit identischem L2-Flush, **gleiche** `warmup`/`iters` (aus derselben
Config durchgereicht, `run.py:606-611`), und der sequentielle Pfad wird **ebenfalls** gegen
dieselbe fp32-Referenz verifiziert (`fusion.py:103-108`) — bei Fehlschlag verliert der Lauf
nur den Vergleich. Die Byte-Bilanz ist analytisch sauber (D wird in beiden Pfaden gezählt,
`fusion.py:114-125`). Die Speedup-Zahlen im Report sind damit belastbar.

**P-7 · Kanonisierung und Codegen-Orientierung.** Der B1-Reshape ist in
`test_reshape.py:80` numerisch in **fp64** gegen `torch.einsum` geprüft (`atol=1e-9`) — das
ist die richtige Methode für Risiko ④. Im Codegen sichern dedizierte
Orientierungs-Wächter (`test_gemm_computes_AB_not_transpose`, `…_swizzle_orientation`,
`…_bias_index_orientation`, `…_fp8_orientation`) genau das Risiko ab, an dem A06 gescheitert
war. Der Anti-Drift-Test auf Byte-Identität des unfusionierten Quelltexts ist die richtige
Absicherung für die additive TZ-9-Erweiterung.

---

## 4. Unklar / Rückfragen ans Team

**R-1 · Was passiert mit dem Abschnitt „verify-before-trust in Aktion", wenn F-02 behoben
ist?** Mit korrektem fp32-Akku sinkt der bf16-Reduktionsfehler von 1,574 auf 3,05·10⁻⁵ —
der Lauf **besteht** dann, und der Sweep hat keinen `verify_failed` mehr. Damit fällt das
Anschauungsbeispiel weg. Drei Optionen: (a) den Abschnitt auf den *gefundenen Defekt*
umschreiben (inhaltlich die stärkste Variante — „das Gate hat einen echten Kernel-Fehler
gefangen"); (b) einen anderen Fall provozieren, der ehrlich scheitert (z. B. tiefere n-äre
fp16-Kette ab 384⁴, die der Report bereits erwähnt); (c) beides. **Entscheidung liegt beim
Team** — sie hängt davon ab, wie viel Report-Umbau am 26.07. noch tragbar ist.

**R-2 · Ist die Kursanforderung „KI-Einsatz dokumentieren" mit `ki_einsatz.rst` erfüllt?**
Die Datei ist allgemein gehalten (Werkzeuge, Einsatzzwecke, Grundsatz) und nennt die
Group-Specific Component nicht gesondert. Da die GSC ausweislich
`project-development/prompts/` (11 TZ-Aufträge) stark KI-assistiert entstanden ist, könnte
ein Prüfer eine projektspezifische Ergänzung erwarten. Ob das gefordert ist, geht aus dem
Repo nicht hervor — bitte gegen `slides/pruefungsleistungen.pdf` abgleichen.

**R-3 · Soll TZ 10 (Copy/Transpose) vor der Abgabe noch kommen?**
`prompts/TZ10-copy-transpose.md` liegt untracked vor, `PLAN.md` §10 führt Copy/Transpose als
„Später / optional". Der Report erwähnt es nicht — das ist konsistent. Falls es **nicht**
kommt: nichts zu tun. Falls doch: F-07 (viertes `run()`-Zweig-Muster) wird dann relevant,
und der Befund sollte **vorher** adressiert werden.

**R-4 · Ist `_MAX_SINGLE_SHOT = 16384` (`reduction.py:39`) belegt oder geschätzt?** Der
Kommentar sagt „konservativ gewählt". Eine `(1, 16384)`-Kachel in fp32 sind 64 KiB — je nach
Shared-Memory-Budget der GB10 grenzwertig. Auf diesem Host lief K=16384 fehlerfrei durch
(eigene Probe), ein Beleg im Sinne von `RESULTS_gb10.md` fehlt aber. Nach dem F-02-Fix ändert
sich die Akku-Größe — bitte einmal an der Obergrenze gegenprüfen.

**R-5 · Welcher Branch ist der Abgabe-Branch?** `tar/git_tag.txt` / `git_link.txt` waren
nicht Teil dieses Reviews. Für F-01 ist entscheidend, ob die Abgabe auf `main`, auf
`group-specific-component` oder auf ein Tag zeigt — davon hängt ab, wohin gemergt werden
muss.

---

## 5. Umsetzungsplan

Abgabe: **27.07.2026**. Heute: **25.07.2026**. Zwei Arbeitstage.
**AP-1 bis AP-4 sind Pflicht vor der Abgabe**, AP-5 ist optional, AP-6/AP-7 danach.

---

### AP-1 — Reduktions-Akkumulator korrigieren und Messdaten neu erzeugen · **PFLICHT**

| | |
|---|---|
| **Befunde** | F-02 |
| **Priorität** | Zuerst, weil es als einziges Arbeitspaket **Messdaten ändert**. Jede Report-Korrektur (AP-2) und jede Figur muss auf den neuen Zahlen aufsetzen — umgekehrte Reihenfolge bedeutet doppelte Arbeit. |
| **Dateien** | `tool_pipeline/codegen/templates/reduction.py` · `results/results.jsonl` (neue Zeilen) · `results/kernels/*__sum.py` (werden neu geschrieben) · `sphinx/source/_static/gsc/*.png` |
| **Aufwand** | **M** (Codeänderung S, Sweep ~2 min, Neubewertung des Textes M) |
| **Risiko** | **Niedrig-mittel.** Die Änderung betrifft nur `row_sum_single`; Kontraktion und Elementwise sind unberührt (eigene Templates). Zu beachten: Die getrackten `*__sum.py` ändern sich → bewusst mitcommitten. Falls die 64-KiB-Kachel an der Obergrenze klemmt (R-4), greift der bestehende Loop-Fallback. |

**Schritte**
1. `reduction.py:97` — `ct.sum` auf dem in den Akku-dtype gecasteten Tile rechnen, exakt wie
   in `row_sum_loop:110`. `acc_ct` ist im Builder bereits verfügbar; ggf. als
   Modul-Konstante in den erzeugten Quelltext backen.
2. `python3 -m pytest tests/ -q` → weiter 286 passed.
3. `python -m tool_pipeline.cli --sweep` (GPU-Lock beachten).
4. `python -m tool_pipeline.report_figures`.
5. Neue Werte notieren: Reduktion fp16 und fp32 (GB/s, %-BW, `max_abs_err`) und **ob** der
   bf16-Lauf jetzt besteht → Input für AP-2 und R-1.

**Definition of Done**
* `row_sum_single` enthält den Akku-Cast; `grep "ct.float32" ` findet ihn im emittierten
  single-shot-Kernel.
* Probe `ij->i`, bf16, 4096²: `max_abs_err < 1e-3` (vorher 1,574).
* `pytest` grün, `results.jsonl` hat eine neue `CLI-Report-Sweep`-Charge, alle fünf PNGs neu
  erzeugt.

---

### AP-2 — Report auf die neuen Zahlen und die richtige Ursache bringen · **PFLICHT** · *nach AP-1*

| | |
|---|---|
| **Befunde** | F-02 (Report-Teil), F-10 (Halbsatz), F-08 (Halbsatz) |
| **Priorität** | Direkt nach AP-1, solange die neuen Zahlen frisch sind. Das ist der **bewertete** Deliverable — inhaltlich falsche Aussagen wiegen hier am schwersten. |
| **Dateien** | `sphinx/source/chapters/group_specific_component/report.rst` |
| **Aufwand** | **M** |
| **Risiko** | **Niedrig** (reiner Text). Einzige Falle: Tabellenwerte übersehen. |

**Schritte**
1. `report.rst:278-287` — die zwei Reduktions-Zeilen mit den AP-1-Werten aktualisieren
   (GB/s, % Peak-BW, AI), ggf. die bf16-Zeile ergänzen, falls sie jetzt besteht.
2. `report.rst:424-434` — Abschnitt gemäß Entscheidung aus **R-1** neu schreiben. Empfehlung
   (a): erzählen, dass das Gate einen **Kernel-Defekt** gefangen hat, nicht eine
   Format-Grenze — das belegt verify-before-trust stärker als der bisherige Text.
3. `report.rst:289-292` — den Satz zur Reduktion („bandbreiten-effizient in fp32/fp16")
   gegen die neuen Zahlen prüfen.
4. `report.rst:297-299` — Halbsatz zur n-är-Pfadquelle ergänzen (F-10).
5. `report.rst:445` — präzisieren, dass die **Codegen-Tests eine GPU voraussetzen** (F-08).
6. Alle übrigen Zahlen unverändert lassen — sie sind verifiziert korrekt (**P-1**).

**Definition of Done**
* Jede Zahl in `report.rst` ist in der neuen Sweep-Charge auffindbar.
* Der Abschnitt „verify-before-trust in Aktion" nennt eine Ursache, die der Code stützt.
* `cd sphinx && make html` → keine neuen Warnings.

---

### AP-3 — Reproduzierbarkeit herstellen: Requirements, CLI-Import, Installationsdoku · **PFLICHT**

| | |
|---|---|
| **Befunde** | F-06, F-03, F-09, F-10 (requirements-Teil) |
| **Priorität** | Unabhängig von AP-1/AP-2 (parallelisierbar). Muss **vor** AP-4 fertig sein, damit alles in **einem** Commit-Block landet. |
| **Dateien** | `project/requirements.txt` · `tool_pipeline/cli.py` · `project/README.md` · `sphinx/source/chapters/installation_und_benutzung.rst` |
| **Aufwand** | **S** |
| **Risiko** | **Niedrig.** Der Lazy-Import in `cli.py` ist die einzige Codeänderung; `tests/test_cli.py` deckt sie ab. |

**Schritte**
1. `requirements.txt`: `matplotlib` und `opt_einsum` mit gepinnten Versionen ergänzen
   (`pip show`); den `kaleido`-Kommentar korrigieren oder die Zeile entfernen (F-06).
2. `cli.py:32`: `from .app.components import controls` in die nutzenden Funktionen ziehen
   (lazy) — Gegenprobe: `python3 -c "import sys; sys.modules['dash']=None; import
   tool_pipeline.cli"` muss durchlaufen (F-03).
3. `README.md:32` an die Realität angleichen (`store` gehört zur Naht — F-04);
   `README.md:44` + `requirements.txt:13`: absoluten `/home/mla08/…`-Pfad ersetzen (F-09).
4. `installation_und_benutzung.rst`: Abschnitt „Group-Specific Component" mit venv,
   `pip install -r project/requirements.txt`, `python -m tool_pipeline`,
   `python -m tool_pipeline.cli --sweep` (F-09).

**Definition of Done**
* CLI ohne `dash` importierbar (Probe oben).
* `requirements.txt` deckt jeden Top-Level-Import ab — Gegenprobe per AST-Scan oder
  `pip install -r` in einer frischen venv + `python -m tool_pipeline.report_figures`.
* `installation_und_benutzung.rst` nennt einen Startbefehl, der ohne Vorwissen funktioniert.
* `pytest` grün, `make html` ohne neue Warnings.

---

### AP-4 — Abgabe-Blocker schließen: committen, mergen, CI-Build prüfen · **PFLICHT** · *zuletzt, nach AP-1/2/3*

| | |
|---|---|
| **Befunde** | F-01, F-05, F-11 (nur Kommentar) |
| **Priorität** | **Muss das letzte Arbeitspaket sein** — es friert den Stand ein. Gleichzeitig ist es das **wichtigste**: ohne AP-4 ist alles andere unsichtbar. Wenn die Zeit knapp wird, ist die Reihenfolge AP-4 → AP-1/2 → AP-3 (lieber der jetzige Stand vollständig abgegeben als ein besserer, den niemand sieht). |
| **Dateien** | die sechs untracked Dateien · `sphinx/source/_static/gsc/` (5 PNGs) · `presentation.rst` · `.gitignore` (Kommentar) · Branches |
| **Aufwand** | **M** (Git S, `presentation.rst` M) |
| **Risiko** | **Mittel** — 53 Commits Rückstand auf `main`, Merge-Konflikte möglich. Deshalb zuerst nach `group-specific-component` (nur 1 Commit Differenz), dort die CI beobachten, erst dann nach `main`. |

**Schritte**
1. `presentation.rst` ausformulieren (F-05) — der Stoff steht in `report.rst` und `slides/`.
2. `git add` der sechs Dateien + `sphinx/source/_static/gsc/*.png`; vorher mit
   `git status --ignored` prüfen, dass keine PNG von `.gitignore` geblockt wird.
3. `.gitignore:33` kommentieren (F-11 Variante a) — **kein** `git rm --cached`.
4. Committen (nach Rücksprache — Commits sind laut `CLAUDE.md` nur auf Aufforderung erlaubt).
5. `feature/tz9` → `group-specific-component` mergen, pushen, **CI-Log lesen**.
6. Nach grüner CI → `main`, Tag setzen, `tar/*.txt` aktualisieren (**R-5** klären),
   `./create_submission.sh`.

**Definition of Done**
* `git status` ist sauber (nur `Findungs-und-Verbesserungen.md` und `QA1-code-review.md`,
  falls diese nicht mit sollen).
* Der CI-Build auf dem Abgabe-Branch meldet **0** `image file not readable`.
* Die GitHub-Pages-Seite zeigt alle **fünf** Figuren und das Fusions-Kapitel.
* Frischer Klon des Abgabe-Branch: `python -c "import tool_pipeline.run"` läuft (d. h.
  `measure/fusion.py` ist da).
* `presentation.rst` enthält kein „TODO" mehr.

---

### AP-5 — Test-Skip-Politik ehrlich machen · *optional vor Abgabe, sonst danach*

| | |
|---|---|
| **Befunde** | F-08 (Code-Teil; der Report-Teil ist in AP-2 erledigt) |
| **Dateien** | `tests/test_measure.py` (8 Stellen) |
| **Aufwand** | **S** · **Risiko:** sehr niedrig |
| **Abhängigkeit** | keine |

Acht `if not _has_cuda(): … return` durch `pytest.skip(...)` ersetzen. **DoD:** Auf einem
GPU-Host unverändert 286 passed; die Absicht ist auf einem CPU-Host erkennbar
(`8 skipped` statt stiller Grün-Meldung). Nur machen, wenn AP-1..AP-4 sicher stehen.

---

### AP-6 — `run()` entzerren · **nach der Abgabe**

| | |
|---|---|
| **Befunde** | F-07 |
| **Dateien** | `tool_pipeline/run.py` |
| **Aufwand** | **M** · **Risiko:** mittel (die Naht ist der zentrale Vertrag) |
| **Abhängigkeit** | **Nicht vor dem 27.07.** Voraussetzung für TZ 10. |

Die drei wortgleichen Blöcke (`verify_failed`-Rückgabe, Timing-Übernahme, Kalt-Launch mit
`ct.TileError`) in lokale Helfer ziehen. Die **Zweig-Struktur nicht** zusammenlegen — die
Additivität ist bewusst und hat sich bewährt. **DoD:** `run()` < 270 Zeilen, `pytest`
unverändert grün, `results.jsonl`-Zeilen byte-strukturgleich zu vorher.

---

### AP-7 — Kosmetik · **nach der Abgabe**

| | |
|---|---|
| **Befunde** | F-12 (Docstring), F-13 (Inline-Styles), F-11 Variante (b) |
| **Aufwand** | **S** (F-12) / **M** (F-13) · **Risiko:** niedrig |
| **Abhängigkeit** | nach AP-6 |

F-12 ist eine reine Docstring-Korrektur in `store.py:17-19` — **den Slug nicht ändern**
(285 Dateinamen). F-13 nur angehen, wenn das Projekt weiterläuft.

---

### Abhängigkeitsübersicht

```
AP-1 (Kernel-Fix + Sweep)  ──►  AP-2 (Report-Zahlen & Ursache)  ──┐
AP-3 (Requirements/CLI/Doku) ─────────────────────────────────────┼──►  AP-4 (Commit/Merge/CI)
                                                                  │      ▲
AP-5 (Skips, optional) ───────────────────────────────────────────┘      │
                                                        ── Abgabe 27.07. ──
AP-6 (run() entzerren) ──► AP-7 (Kosmetik)                       danach
```

**Wenn die Zeit nicht reicht:** AP-4 hat absoluten Vorrang, dann AP-1+AP-2 (falsche Aussage
im bewerteten Report), dann AP-3. AP-5/6/7 sind verzichtbar.

---

## 6. Folge-Prompt für den Umsetzungs-Agenten

```
# Auftrag: Umsetzung der QA-1-Befunde — cuTile Performance Lab

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ
nehmen), Branch `feature/tz9`. Du bist **Entwickler**, nicht Reviewer: dein
Deliverable sind Codeänderungen, kein weiteres Befunddokument.

## Kontext

Die Group-Specific Component „cuTile Performance Lab" (`project/`) ist ein
interaktiver einsum-/GEMM-Explorer: Aus einem einsum-Ausdruck wird live ein
cuTile-Kernel **generiert**, compiliert, **gegen fp32 verifiziert**, auf der GPU
(NVIDIA GB10, sm_121, 273 GB/s) **gemessen** und in Charts (Durchsatz · Genauigkeit ·
Roofline) dargestellt. Die Teil-Ziele TZ 1–9 sind umgesetzt, der Sphinx-Report ist
geschrieben. Ein QA-Review hat 13 Befunde ergeben; sie stehen mit Belegen in
`project/project-development/Findungs-und-Verbesserungen.md` — **lies dieses Dokument
zuerst vollständig**, insbesondere §2 (Befunde F-01…F-13), §3 (Positiv-Befunde) und
§5 (Umsetzungsplan mit Reihenfolge und Definition of Done je Arbeitspaket).

**Abgabe ist der 27.07.2026, heute ist der 25.07.2026.** AP-1 bis AP-4 müssen vor der
Abgabe fertig sein; AP-5 ist optional, AP-6/AP-7 kommen danach. Halte dich an die
Reihenfolge — sie ist nicht beliebig (AP-1 ändert Messdaten, auf denen AP-2 aufsetzt;
AP-4 friert den Stand ein und muss zuletzt kommen).

## Bereits festgelegt — NICHT neu evaluieren, NICHT vorschlagen

- **GUI-Framework = Plotly Dash** (fix), Charts nativ Plotly. Keine
  Framework-Diskussion.
- **Codegen = C1**: f-String-Templates → `@ct.kernel`, ein Modul je Familie unter
  `codegen/templates/`. Kein anderes Codegen-Paradigma (AST, Jinja, MLIR …).
- **Die eine Naht:** `app/` importiert im Live-Loop nur `run.py` + `schema.py` (plus
  `store` fürs History-Lesen); der Hauptprozess bleibt CUDA-frei; Charts sind reine,
  headless testbare Funktionen. Diese Architektur ist gesetzt.
- **Results-Store = JSON Lines** (`project/results/results.jsonl`) + generierter Kernel
  als `results/kernels/<slug>.py` (Compile-Cache). **Kein Format-Umbau, keine
  Datenbank, keine Slug-Änderung.**
- **Kommentare und Docstrings auf Deutsch**; „Ergebnisse"-Blöcke am Dateiende sind
  gewollte Dokumentation, kein toter Code.
- **verify-before-trust ist Gesetz**: keine Zahl und keine Report-Figur ohne bestandene
  fp32-Referenz.

## Lesereihenfolge

1. `project/project-development/Findungs-und-Verbesserungen.md` — dein Auftrag
   (§2 Befunde, §3 Positiv-Befunde, §5 Umsetzungsplan).
2. `project/README.md` — Architektur, Datenfluss, Start-Kommandos.
3. `project/project-development/PLAN.md` §2 (Designentscheidungen), §6
   (Codegen-Risiken), §10 (TZ 1–9).
4. `project/tool_pipeline/run.py` + `schema.py` — die Naht.
5. `project/tool_pipeline/codegen/templates/reduction.py` — Ort von AP-1.
6. `sphinx/source/chapters/group_specific_component/report.rst` — Ort von AP-2.

## Arbeitspakete (in dieser Reihenfolge)

### AP-1 — Reduktions-Akkumulator korrigieren + Messdaten neu erzeugen  [F-02]
Dateien: `tool_pipeline/codegen/templates/reduction.py`, danach `results/results.jsonl`,
`results/kernels/*__sum.py`, `sphinx/source/_static/gsc/*.png`.

`row_sum_single` (`reduction.py:90-97`) summiert im **Eingabeformat** und ignoriert
`acc_dtype` — `acc_ct` kommt dort nicht vor. `row_sum_loop:110` macht es richtig
(`ct.sum(ct.astype(tile, ct.float32), axis=1)`). Belegt: bf16-Reduktion 4096²
liefert `max_abs_err = 1.574` (single-shot) gegen `3.05e-05` (Loop-Pfad).
Ziehe den Akku-Cast in den single-shot-Pfad nach, exakt analog zum Loop-Pfad.
Danach: `pytest tests/ -q` → 286 passed; `python -m tool_pipeline.cli --sweep`;
`python -m tool_pipeline.report_figures`. Notiere die neuen Reduktionszahlen (GB/s,
%-BW, max_abs_err) und ob der bf16-Lauf jetzt besteht — das brauchst du für AP-2.
Die getrackten `results/kernels/*__sum.py` ändern sich dabei; das ist erwartet.

### AP-2 — Report korrigieren  [F-02, F-10, F-08]  — nach AP-1
Datei: `sphinx/source/chapters/group_specific_component/report.rst`.
- `:278-287` Reduktionszeilen der memory-bound-Tabelle auf die AP-1-Werte setzen.
- `:424-434` — der Abschnitt erklärt den `verify_failed` mit „der bf16-Rundungsfehler
  addiert sich über Millionen Summanden auf". **Das ist die falsche Ursache** (es war
  der bf16-Akkumulator, s. AP-1). Schreibe ihn um: Das verify-Gate hat einen echten
  **Kernel-Defekt** gefangen, bevor eine Zahl in den Report kam — das belegt
  verify-before-trust stärker als die bisherige Erzählung. Falls nach AP-1 kein
  `verify_failed` mehr im Sweep ist, erzähle den gefundenen und behobenen Defekt.
- `:289-292` Fließtext zur Reduktion gegen die neuen Zahlen prüfen.
- `:297-299` Halbsatz ergänzen, dass der n-är-Pfad per Links-nach-rechts-Zerlegung
  geplant wurde und mit `opt_einsum` abweichen kann.
- `:445` präzisieren, dass die Codegen-Tests eine GPU **voraussetzen**.
- **Alle übrigen Zahlen unverändert lassen** — sie sind gegen `results.jsonl`
  verifiziert korrekt.

### AP-3 — Reproduzierbarkeit  [F-06, F-03, F-04, F-09, F-10]  — parallel zu AP-1/2 möglich
- `project/requirements.txt`: `matplotlib` und `opt_einsum` mit gepinnten Versionen
  ergänzen (`pip show` für die Version). `kaleido`-Kommentar korrigieren oder Zeile
  entfernen (`charts.save_png` wird von keinem Report-Pfad genutzt).
- `tool_pipeline/cli.py:32`: `from .app.components import controls` ist ein
  Rückwärts-Import Core→GUI; `controls.py:32-33` zieht `dash` + `dash_bootstrap_components`
  auf Modulebene, die „headless" CLI ist ohne Dash **nicht importierbar**. Ziehe den
  Import lazy in die nutzenden Funktionen. Gegenprobe:
  `python3 -c "import sys; sys.modules['dash']=None; import tool_pipeline.cli"`.
- `project/README.md:32`: Naht-Beschreibung an die Realität angleichen (`app/callbacks.py:40`
  importiert zusätzlich `store` — torch-frei, harmlos, aber die Doku sagt „nur run + schema").
- `project/README.md:44` und `requirements.txt:13`: den absoluten Pfad
  `/home/mla08/MLA/mla/.venv` (aus diesem Checkout nicht erreichbar) durch eine neutrale
  Formulierung ersetzen.
- `sphinx/source/chapters/installation_und_benutzung.rst`: Abschnitt
  „Group-Specific Component" ergänzen (venv, `pip install -r project/requirements.txt`,
  `python -m tool_pipeline`, `python -m tool_pipeline.cli --sweep`). Das Kapitel erwähnt
  das Tool derzeit **gar nicht**.

### AP-4 — Abgabe-Blocker  [F-01, F-05, F-11]  — ZULETZT
- `sphinx/source/chapters/group_specific_component/presentation.rst` ist ein TODO-Stub,
  obwohl der Vortrag am 08.07.2026 war. Ausformulieren (der Stoff steht in `report.rst`
  und `slides/`) oder durch einen Verweis ersetzen — aber **kein „TODO" stehen lassen**.
- Sechs untracked Dateien gehören zum ausgelieferten Stand und sind in **keinem** Branch:
  `project/tool_pipeline/measure/fusion.py` (von `run.py:44` **top-level** importiert!),
  `project/tool_pipeline/report_figures.py`, `project/tool_pipeline/app/components/history.py`,
  `project/tests/test_cli.py`, `project/tests/test_store.py`, sowie
  `sphinx/source/_static/gsc/` (alle 5 Report-Figuren). Ohne sie ist ein Klon nicht
  lauffähig und der CI-Report zeigt 4 leere Bildrahmen (verifiziert).
- `.gitignore:33` (`/project/results/kernels/`) kommentieren: 59 Kernel sind bereits
  getrackt, 285 liegen auf der Platte. **Kein `git rm --cached`** vor der Abgabe.
- Bereite Commit und Merge vor (`feature/tz9` → `group-specific-component` → `main`;
  `.github/workflows/docs.yml` triggert nur auf diese beiden) und **frage nach**, bevor
  du committest — siehe „Verbote".

### AP-5 (optional) — `tests/test_measure.py`: die acht `if not _has_cuda(): … return`
durch `pytest.skip(...)` ersetzen. Sie melden derzeit ohne GPU „passed" statt „skipped".

## Was du NICHT anfassen darfst

- **Die Naht** `run(config) -> RunResult` und ihren Vertrag („wirft nie, kategorisiert
  in Status"). Keine Signaturänderung.
- **Das Store-Format** (JSON Lines) und **`config_slug`** — eine Slug-Änderung
  invalidiert 285 Kernel-Dateinamen. Bei F-12 nur den **Docstring** korrigieren.
- **Das Codegen-Paradigma** (C1, f-String-Templates).
- **Die Toleranztabelle** `measure/verify.py:_TOLERANCES` — sie ist empirisch begründet
  und wurde gegengeprüft (fängt vertauschte mma-Operanden und Nullausgaben in allen
  getesteten Größen). Nicht „vorsichtshalber" verschärfen.
- **`report_figures.load_report_rows()`** und den `_is_square`-Filter — sie sorgen dafür,
  dass Tabelle und Figur dieselbe Sweep-Charge zeigen. Nicht vereinfachen.
- **Die Zweig-Struktur in `run()`** (memory-bound / n-är / 2-Op). Sie ist bewusst
  additiv. AP-6 (Entzerrung) ist **nach** der Abgabe.
- **Alle übrigen Report-Zahlen** außer den Reduktionszeilen — sie sind gegen
  `results.jsonl` verifiziert.
- Alles unter §3 „Positiv-Befunde" des Befunddokuments (P-1 … P-7).

## Arbeitsweise — verbindlich

- **Nach jeder Änderung** ausführen und die Ausgabe prüfen:
  `cd project && python3 -m pytest tests/ -q`   (Erwartung: 286 passed, 0 failed)
  `cd sphinx && make html`                      (Erwartung: keine neuen Warnings;
                                                 die eine bestehende Warning in
                                                 `chapters/10_xdna_whole_npu/loesung.rst:144`
                                                 ist vorbekannt und nicht dein Auftrag)
- Nach GPU-Läufen `git status` prüfen: außer den beabsichtigten Änderungen darf nichts
  in `results/` auftauchen.
- Die GPU ist geteilt. Halte Shapes im Rahmen des bestehenden Sweeps (32 GiB Grenze),
  und fahre `--sweep` nicht mehrfach parallel (der `FileLock` schützt nur innerhalb
  des Tools).
- Arbeite die Arbeitspakete **einzeln** ab und berichte nach jedem kurz: was geändert,
  was die Tests sagen, welche Zahlen sich verschoben haben.

## Verbote

- **Du committest, pushst oder taggst NICHTS von dir aus.** `CLAUDE.md` untersagt das
  ausdrücklich, solange du nicht direkt dazu aufgefordert wirst. Bereite den Commit in
  AP-4 vor (Dateien nennen, Commit-Message vorschlagen) und **frage**.
- Du trägst dich **nie** als Autor oder Co-Autor eines Commits ein.
- Keine Refactorings über den Auftrag hinaus. Wenn dir unterwegs etwas auffällt, das
  nicht in §5 steht: notieren und **melden**, nicht selbst umbauen.

## Definition of Done (Gesamtauftrag)

1. **AP-1:** Der emittierte single-shot-Reduktionskernel enthält den Akku-Cast; Probe
   `ij->i` bf16 4096² liefert `max_abs_err < 1e-3` (vorher 1,574).
2. **AP-1:** `results.jsonl` hat eine neue `CLI-Report-Sweep`-Charge; alle fünf PNGs
   unter `sphinx/source/_static/gsc/` sind daraus neu erzeugt.
3. **AP-2:** Jede Zahl in `report.rst` ist in der neuen Charge auffindbar; der Abschnitt
   „verify-before-trust in Aktion" nennt eine Ursache, die der Code stützt.
4. **AP-3:** `python3 -c "import sys; sys.modules['dash']=None; import tool_pipeline.cli"`
   läuft durch; eine frische venv nach `requirements.txt` kann
   `python -m tool_pipeline.report_figures` ausführen; `installation_und_benutzung.rst`
   nennt einen funktionierenden Startbefehl.
5. **AP-4:** `presentation.rst` enthält kein „TODO"; die sechs Dateien und die fünf PNGs
   sind zum Commit vorbereitet (`git status` gegengeprüft); Merge-Weg und
   Commit-Message liegen dem Team zur Freigabe vor.
6. `python3 -m pytest tests/ -q` → **286 passed** (bzw. 278 passed / 8 skipped, falls
   AP-5 gemacht wurde), 0 failed.
7. `cd sphinx && make html` → `build succeeded`, keine neue Warning.
8. Ein Abschlussbericht (im Chat, keine neue Datei) listet je Befund-ID: erledigt /
   bewusst ausgelassen / offen — mit Begründung.
```

---

## Anhang — Beweis der Unveränderlichkeit des Repos

`git status` **am Ende** dieses Reviews, Repo-Wurzel `/home/mla07/mla`:

```
?? project/project-development/prompts/QA1-code-review.md
?? project/project-development/prompts/TZ10-copy-transpose.md
?? project/tests/test_cli.py
?? project/tests/test_store.py
?? project/tool_pipeline/app/components/history.py
?? project/tool_pipeline/measure/fusion.py
?? project/tool_pipeline/report_figures.py
?? sphinx/source/_static/gsc/
```

**Kommentar.** Die Liste ist **byte-identisch** mit dem Stand vor dem Review — dieselben
acht untracked Einträge, keine modifizierte (`M`) oder gelöschte (`D`) Datei. Dieses
Dokument (`project/project-development/Findungs-und-Verbesserungen.md`) ist die **einzige**
von mir geschriebene Datei; es erscheint in der obigen Ausgabe noch nicht, weil sie vor dem
Schreiben erhoben wurde.

Zusätzlich geprüft, weil ein Testlauf und mehrere GPU-Proben stattgefunden haben:

* `results/results.jsonl` — md5 `e9773b40a2beaea531f74d191106bdf3`, 87 Zeilen: **vor und
  nach** dem `pytest`-Lauf identisch.
* `results/kernels/` — **285** Dateien vor und nach dem Lauf.
* Meine eigenen GPU-Proben liefen ausschließlich in
  `/tmp/…/scratchpad/` und haben weder `results/` noch `tool_pipeline/` berührt; die
  Sphinx-Vollbauten schrieben nach `/tmp/…/scratchpad/sphinx-full*` bzw. das ohnehin
  gitignorierte `sphinx/build/`.
* Es wurde **kein** `git add`, `git commit`, `git push`, `git checkout` oder
  `git worktree` ausgeführt. Für die CI-Gegenprobe kam `git archive` zum Einsatz — ein rein
  lesender Befehl, der den Arbeitsbaum nicht anfasst.
