# Auftrag: QA 1 — Vollständiges Code-Review & Doku-Verifikation der Group-Specific Component

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen),
Branch `feature/tz9`. Du bist **QA-Engineer**, nicht Entwickler: dein Deliverable ist ein
**Befund-Dokument**, kein Patch.

Zu prüfen ist die Group-Specific Component „**cuTile Performance Lab**" (`project/`):
ein interaktiver einsum-/GEMM-Explorer, der aus einem Ausdruck **live einen cuTile-Kernel
generiert**, ihn compiliert, **gegen fp32 verifiziert**, auf der GPU **misst** und in
Charts (Durchsatz · Genauigkeit · Roofline) darstellt. Die Teil-Ziele TZ 1–9 (inkl. TZ 7.5)
gelten laut `PLAN.md` als abgeschlossen; der Sphinx-Report ist geschrieben. **Genau das ist
zu verifizieren** — nicht zu glauben. Die Abgabe des Projektberichts ist der **27.07.2026**,
heute ist der **25.07.2026**: dein Befund muss priorisiert sein, damit in zwei Tagen das
Wichtigste zuerst behoben wird.

---

## Deine Rolle — harte Regeln

1. **READ-ONLY am Projekt.** Du änderst **keine** Datei unter `project/tool_pipeline/`,
   `project/tests/`, `sphinx/` oder sonst im Repo. Einzige Datei, die du schreibst, ist dein
   Bericht (s. „Deliverable"). Keine Refactorings, keine „schnellen Fixes", kein
   `git add`/`commit`/`push` (siehe `CLAUDE.md` — Commits sind ausdrücklich untersagt,
   solange nicht direkt dazu aufgefordert).
2. **Belegpflicht.** Jeder Befund nennt `datei.py:zeile` (oder `datei.rst:zeile`) und, wo
   möglich, die konkrete Beobachtung (Kommandoausgabe, Testfehler, Build-Warning). Keine
   Vermutungen als Befunde verkaufen — Unsicheres kommt in einen eigenen Abschnitt
   „Unklar / nachzufragen".
3. **Keine Stilpolizei ohne Wirkung.** Bewertet wird **Sinnhaftigkeit, Qualität und Struktur**
   mit Auswirkung auf Korrektheit, Wartbarkeit oder Bewertung des Projekts. Reine
   Geschmacksfragen gehören — wenn überhaupt — in die niedrigste Schwere.
4. **Sprache: Deutsch.** Code-Kommentare und Doku im Repo sind Deutsch; dein Bericht ebenso.
5. **Lesen vor Urteilen.** Die Architektur ist bewusst gewählt (s. u. „Bereits festgelegt").
   Ein Befund, der eine getroffene Designentscheidung ignoriert, ist ein Fehlbefund.

## Bereits festgelegt — NICHT als Mangel melden

- **GUI-Framework = Plotly Dash** (fix), Charts nativ Plotly. Keine Framework-Diskussion.
- **Codegen = C1**: f-String-Templates → `@ct.kernel`, ein Modul je Familie unter
  `codegen/templates/`. Kein Vorschlag für ein anderes Codegen-Paradigma (AST, Jinja, MLIR…).
- **Die eine Naht:** `app/` importiert im Live-Loop **nur** `run.py` + `schema.py`; der
  Hauptprozess bleibt CUDA-frei; Charts sind reine, headless testbare Funktionen. Das ist
  Absicht — prüfe, ob sie **eingehalten** wird, nicht ob sie sinnvoll ist.
- **Results-Store = JSON Lines** (`project/results/results.jsonl`) + generierter Kernel als
  `results/kernels/<slug>.py` (Compile-Cache). Kein Format-Umbau, keine Datenbank.
- **Kommentare/Docstrings auf Deutsch**, „Ergebnisse"-Blöcke am Dateiende sind gewollte
  Dokumentation, kein toter Code.
- **verify-before-trust ist Gesetz**: keine Zahl und keine Report-Figur ohne bestandene
  fp32-Referenz. Prüfe die *Einhaltung*.

## Zuerst lesen (in dieser Reihenfolge)

1. `project/README.md` — Architektur, Datenfluss, Verzeichnisrollen, Start-Kommandos.
2. `project/project-development/PLAN.md` — §2 (Designentscheidungen), §6 (Codegen-Risiken:
   stille Falschergebnisse), §9 (Soll-Verzeichnisstruktur), **§10 (TZ 1–9 mit je einer
   Definition of Done)**. Die DoDs sind deine **Soll-Liste**: jede behauptet abgeschlossene
   TZ ist gegen den echten Code zu prüfen.
3. `project/project-development/prompts/TZ*.md` — die erteilten Aufträge, insbesondere
   `TZ8-politur-report.md` (Politur/Robustheit/Report) und `TZ9-fusion.md` (zuletzt
   umgesetzt). `TZ10-copy-transpose.md` ist **noch nicht umgesetzt** — kein Mangel.
4. `project/tool_pipeline/run.py` + `schema.py` — die Naht, der gesamte Ablauf.
5. `sphinx/source/chapters/group_specific_component/report.rst` — der Projektbericht.

---

## Prüfachse A — Code: Sinnhaftigkeit, Qualität, Struktur

Umfang: `project/tool_pipeline/` (~40 Module, ~12,4 kLOC inkl. Tests), `project/tests/`
(13 Dateien, ~4,5 kLOC), `project/project-development/analysis/`.

**A1 — Architektur & Struktur**
- Wird die Naht wirklich eingehalten? Grep nach Imports aus `app/` in
  `intermediate_representation`/`codegen`/`measure`/`store` und umgekehrt; ist der
  Hauptprozess wirklich CUDA-frei (kein Top-Level-`import torch` auf dem GUI-Pfad)?
- Schichtung `ir → codegen → measure → store`: gibt es Rückwärts- oder Querimporte,
  Zyklen, Module am falschen Ort (z. B. `report_figures.py` direkt in `tool_pipeline/`)?
- Passt die Ist-Struktur zu `PLAN.md` §9 und zur Beschreibung im `README.md`? Abweichungen
  sind Befunde — entweder Code oder Doku ist falsch, sag welches.
- Verantwortungsschnitt: übernimmt `run.py` Aufgaben, die in `measure/`/`store/` gehören?
  Ist `callbacks.py` zur God-Datei geworden?

**A2 — Korrektheit & Risiko (höchste Priorität)**
Codegen ist die Hauptquelle **stiller Falschergebnisse** (PLAN §6). Prüfe gezielt:
- **Index-/Achsen-Logik** in `intermediate_representation/parse.py`, `reshape.py`
  (B1-Umformung) und den drei Templates: Reihenfolge der Ausgabeachsen, Batch-Achsen,
  Reduktionsachsen, n-äre Ketten-Zerlegung.
- **Randfälle nicht-teilbarer Dimensionen**: `padding_mode=ZERO` beim `ct.load` +
  Clipping durch `ct.store` — für **alle drei Familien** belegt (Tests!) oder nur für
  Kontraktion?
- **dtype-/Akkumulator-Pfade**: fp16/bf16/tf32/fp8(e4m3,e5m2) × Akkumulator; sind die
  Verifikationstoleranzen pro dtype begründet oder so weit gesetzt, dass sie nichts fangen?
  Eine zu lasche `atol/rtol` ist ein **S1-Befund** (verify-before-trust wird wertlos).
- **`measure/verify.py`**: verifiziert es wirklich gegen fp32, für jede Familie/Op/Epilog?
  Gibt es Pfade, auf denen ein Ergebnis ohne Verify als `status=="ok"` im Store landet?
- **`measure/bench.py`/`metrics.py`**: sind FLOPs/Bytes je Familie korrekt gezählt
  (Kontraktion 2·M·N·K; memory-bound: gelesene + geschriebene Bytes)? Falsche Nenner
  ⇒ falsche Roofline ⇒ falscher Report.
- **`measure/fusion.py`** (TZ 9): ist der fused-vs-sequentiell-Vergleich fair (gleiche
  Shapes, gleiche Warmups, beide verifiziert)?
- **`store/store.py`**: Atomarität (`_atomic_rewrite`), Slug-Kollisionen, Verhalten bei
  korrupter/halb geschriebener `results.jsonl` oder `kernels/<slug>.py`, Altzeilen-Fallback.
- **Cache-Korrektheit**: kann `codegen/compile.py` je einen Kernel liefern, der nicht zur
  angefragten Config passt (Slug deckt nicht alle Parameter ab)? Das wäre ein S1-Befund.

**A3 — Qualität im Kleinen**
Duplikate (dieselbe Logik in mehreren Templates/Callbacks), toter Code, ungenutzte Importe/
Parameter, magische Zahlen ohne Konstante, `except Exception: pass`-Stellen, die Fehler
verschlucken, inkonsistente Namen (deutsch/englisch gemischt), fehlende oder irreführende
Docstrings, Funktionen jenseits vernünftiger Länge/Verschachtelung, inline-Styles statt
`theme.css`. Fasse Gleichartiges zu **einem** Befund mit Fundstellenliste zusammen — keine
40 Einzeltickets für dasselbe Muster.

**A4 — Fehlerzustände & UX-Robustheit**
Sind alle `status`-Fälle (`compile_error`, `verify_failed`, `run_error`, „GPU belegt")
benutzerfreundlich sichtbar? Was passiert bei: ungültigem Ausdruck, nicht zerlegbarer n-ärer
Kette, doppelter Tile-Zeile, unpassender GROUP_M/Swizzle-Kombination, sehr vielen Configs,
fehlender/leerer `results.jsonl`, parallelem „Run"-Klick? **Loud fail statt stiller Zahl.**

**A5 — Tests**
- **Führe sie aus**: `cd project && python3 -m pytest tests/ -q` (venv aktivieren, s.
  `README.md`; ohne GPU laufen nur die headless-Tests — halte im Bericht fest, **welche
  Tests übersprungen wurden und warum**, und markiere alles GPU-Abhängige explizit als
  „auf diesem Host nicht verifiziert").
- Deckung gegen Risiko: gibt es je Familie × dtype × Randfall (ragged) einen Test? Werden
  `parse`/`reshape`-Achsenpermutationen getestet oder nur der Happy Path?
- **Test-Qualität**: Tests, die nur „läuft durch" prüfen statt Werte; Mocks, die die
  eigentliche Logik wegmocken (z. B. verify); Tests ohne Assertion.
- **Isolation**: verschmutzt irgendein Test die getrackte `project/results/results.jsonl`
  oder `results/kernels/`? Prüfe per `git status` **nach** dem Testlauf — jede Änderung dort
  ist ein Befund.
- Es gibt **keine** `conftest.py` und keine pytest-Konfiguration (kein `pytest.ini`/
  `pyproject.toml`). Bewerte, ob das zu Doppel-Setup in den Testdateien oder zu
  Pfad-Fragilität führt.

**A6 — Reproduzierbarkeit & Hygiene**
`project/requirements.txt` vs. tatsächliche Importe (fehlende/überflüssige Pakete);
Hardcodierte Pfade (z. B. das venv `/home/mla08/...` im `README.md`) und ob sie in Code
oder nur Doku stehen; `.gitignore`-Abdeckung (`__pycache__`, `.pytest_cache`, `.cache`);
`cli.py` als reproduzierbarer Report-Sweep (führen die dokumentierten Kommandos zu den
Zahlen im Report?).

**A7 — Git-/Abgabe-Zustand** (bereits beobachtet — verifiziere und bewerte)
`git status` zeigt **untracked, also nicht committete** Dateien, die zum ausgelieferten
Stand gehören:
`project/tool_pipeline/measure/fusion.py`, `project/tool_pipeline/report_figures.py`,
`project/tool_pipeline/app/components/history.py`, `project/tests/test_cli.py`,
`project/tests/test_store.py`, `project/project-development/prompts/TZ10-copy-transpose.md`
und **`sphinx/source/_static/gsc/`** (alle Report-Figuren).
Konsequenz prüfen und benennen: ein Klon des Repos ist **nicht lauffähig** bzw. der
Sphinx-Build auf `main` rendert **Figuren ohne Bilddatei**. Prüfe zusätzlich, ob
`feature/tz9` gegenüber `main`/`group-specific-component` überhaupt gemerged ist und ob
`.github/workflows/docs.yml` (Trigger: `main`, `group-specific-component`) den Stand baut,
der abgegeben wird. Das ist ein **Blocker-Kandidat**.

---

## Prüfachse B — Sphinx-Doku: Vollständigkeit & Bericht-Qualität

Ort: `sphinx/source/chapters/group_specific_component/` (`index.rst`, `pitch.rst`,
`presentation.rst`, `projektplan.rst`, `report.rst`), eingehängt in `sphinx/source/index.rst:31`.

**B1 — Baut es sauber?**
`cd sphinx && make html`. Notiere **jede Warning** (fehlende Referenzen, doppelte Labels,
nicht auflösbare `:ref:`, Bilder ohne Datei, Dateien nicht im `toctree`, RST-Syntaxfehler).
Prüfe, ob jede `.. figure::`/`.. image::`-Quelle existiert **und im Git getrackt ist**
(vgl. A7). Prüfe interne Links und die Toctree-Struktur.

**B2 — Der Bericht muss die Anforderungen erfüllen.** Der Report ist ausdrücklich
**kein Doku-Anhang, sondern ein selbsttragender Projektbericht**. Bewerte `report.rst`
Abschnitt für Abschnitt gegen diese Kriterien:

| Kriterium | Prüffrage |
|---|---|
| **Introduction** | Führt sie ohne Vorwissen ins Thema ein? Ist klar, *was* gebaut wurde und *warum es interessant ist*, bevor Details kommen? |
| **Problem formulation** | Ist das Problem *präzise* gestellt (was ist schwer, welche Frage wird beantwortet), nicht bloß „wir sollten ein Tool bauen"? |
| **Implemented solution** | Wird die Lösung so beschrieben, dass ein fremder Leser Architektur und Kernentscheidungen versteht — inkl. **Begründungen**, nicht nur Aufzählung von Modulen? |
| **Results and insights** | Stehen dort echte Messergebnisse **und daraus abgeleitete Einsichten**? Eine Zahl ohne Interpretation ist kein Ergebnis. Sind die Grenzen der Aussage genannt? |
| **Detailliert, aber prägnant** | Gibt es Redundanz zwischen Abschnitten, Füllsätze, dreimal erklärte Naht? Umgekehrt: Stellen, an denen die Erklärung zu dünn für die Behauptung ist? |
| **Sprache & Lesefluss** | Liest es sich wie ein guter **Blogpost** — durchgehender roter Faden, Übergänge zwischen Abschnitten, aktive Sprache — oder wie Stichpunkte im Fließtext-Kostüm? Notiere konkrete Absätze mit Bruch im Fluss. |
| **Selbsttragend** | Verweist der Text auf Wissen, das nur im Code/`PLAN.md`/in den Assignments steht? Sind Fachbegriffe (Tiling, Swizzle, Roofline, arithmetische Intensität, fp8-Varianten) bei Erstnennung erklärt? |
| **Figuren** | Hat jede Figur Caption, Achsenbeschriftung mit Einheit und einen Satz im Text, der sie *auswertet*? Ist die Aussage der Figur im Text tatsächlich belegt? |

Die Ist-Gliederung von `report.rst` (Introduction · Problem Formulation · Implemented
Solution · Results and Insights mit den Unterkapiteln Roofline, Formate, Tuning, memory-bound,
n-är, Fusion, verify-before-trust, Test-Stand) ist als Rahmen vorhanden — prüfe **Substanz,
Balance und Lesefluss**, nicht die Existenz der Überschriften.

**B3 — Faktenabgleich Doku ↔ Code ↔ Messdaten (wichtig)**
Stichprobenartig, aber ernsthaft: stimmen die Zahlen im Report mit `project/results/results.jsonl`
überein? Sind die Hardware-Angaben (GB10, sm_121, 273 GB/s) konsistent mit
`project-development/analysis/RESULTS_gb10.md` und `tool_pipeline/hardware.py`? Beschreibt der
Report Verhalten, das der Code so nicht (mehr) hat? Ist `projektplan.rst` aus dem **aktuellen**
`PLAN.md` synchronisiert? Deckt sich `README.md` mit der Realität?

**B4 — Drumherum**
`installation_und_benutzung.rst`, `ki_einsatz.rst`, `overview.rst`, `pitch.rst`,
`presentation.rst`: vorhanden, aktuell, widerspruchsfrei zum Report? Folgen die
Installations-/Startanweisungen dem echten Repo-Stand?

---

## Deliverable

**Eine** neue Datei: `project/project-development/Findungs-und-Verbesserungen.md`
(Deutsch, Markdown). Genau diese Struktur:

```markdown
# Findungen & Verbesserungen — QA-Review cuTile Performance Lab

## 0. Prüfumfang & Methodik
Was wurde geprüft (Commit/Branch, Dateien, Zeilen), was ausgeführt (pytest-Kommando,
Sphinx-Build) mit Ergebnis, und was **nicht** verifiziert werden konnte (z. B. GPU-Pfade
ohne GPU) — ehrlich und explizit.

## 1. Gesamturteil
5–10 Sätze: Reifegrad, größte Stärken, die drei gefährlichsten Schwächen.
Plus eine Ampel je Achse: Architektur · Korrektheit · Tests · Doku/Report · Abgabe-Hygiene.

## 2. Befunde
Absteigend nach Schwere, durchnummeriert `F-01`, `F-02`, …
Je Befund als Tabelle oder feste Feldliste:
  - **ID / Titel**
  - **Schwere**: S1 Blocker (falsche Ergebnisse, Abgabe unvollständig) ·
    S2 Major (Korrektheitsrisiko, Report-Anforderung verfehlt) ·
    S3 Minor (Wartbarkeit, Redundanz) · S4 Nit
  - **Fundstelle(n)**: `datei:zeile`
  - **Beobachtung**: was ist da
  - **Warum es zählt**: konkrete Auswirkung (nicht „unsauber")
  - **Empfehlung**: die kleinste Änderung, die es behebt
  - **Belegt durch**: gelesener Code / Testlauf / Build-Output / `git status`

## 3. Positiv-Befunde
Was gut gelöst ist und beim Aufräumen **nicht** angefasst werden darf.

## 4. Unklar / Rückfragen ans Team
Punkte, die ohne Domänen-/Kursinfo nicht entscheidbar sind.

## 5. Umsetzungsplan
Arbeitspakete `AP-1..AP-n`, **in Reihenfolge**, jedes mit: enthaltene Befund-IDs ·
Begründung der Priorität · betroffene Dateien · geschätzter Aufwand (S/M/L) ·
Risiko beim Ändern · **Definition of Done** (woran man sieht, dass es erledigt ist).
Berücksichtige den Abgabetermin **27.07.2026**: markiere klar, was **vor** der Abgabe
zwingend erledigt sein muss und was danach kann. Setze Abhängigkeiten explizit
(„AP-3 erst nach AP-1"). Keine großen Refactorings kurz vor Abgabe empfehlen, wenn ein
kleiner Fix reicht.

## 6. Folge-Prompt für den Umsetzungs-Agenten
Ein **vollständiger, selbsttragender Prompt** in einem ```-Block, den man ohne weiteren
Kontext an einen Agenten geben kann.
```

**Anforderungen an den Folge-Prompt in §6** — er muss dem Stil der `prompts/TZ*.md` folgen und
enthalten: Kontext des Projekts in wenigen Sätzen · „Bereits festgelegt — nicht neu evaluieren"
(die Liste oben) · Lesereihenfolge · die abzuarbeitenden Arbeitspakete mit Befund-IDs und
Dateien · was **nicht** angefasst werden darf (Naht, Store-Format, Codegen-Paradigma,
Positiv-Befunde aus §3) · das Gebot, nach jeder Änderung `pytest` und `make html` laufen zu
lassen · das Verbot, selbst zu committen · eine Definition of Done für den gesamten Auftrag.
Der Prompt darf auf `Findungs-und-Verbesserungen.md` verweisen, muss aber die Aufgaben so
konkret nennen, dass klar ist, was zu tun ist.

## Definition of Done (für dich)

- `pytest` **wurde ausgeführt**, Ergebnis (Anzahl passed/failed/skipped) steht im Bericht.
- `cd sphinx && make html` **wurde ausgeführt**, alle Warnings sind erfasst.
- Jede Achse A1–A7 und B1–B4 ist im Bericht adressiert — auch wenn das Ergebnis
  „keine Befunde" lautet; ausgelassene Achsen werden mit Grund benannt.
- Jeder Befund hat Fundstelle, Auswirkung und Empfehlung.
- §5 ist eine **ausführbare Reihenfolge**, keine Wunschliste.
- §6 ist ein Prompt, den man copy-paste starten kann.
- Am Repo wurde **nichts** verändert außer der einen neuen Datei (`git status` zum Beweis
  am Ende des Berichts, kurz kommentiert).
