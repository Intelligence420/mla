# Auftrag: TZ 10 — Copy/Transpose als eigene memory-bound Ops

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen).
Wir bauen die Group-Specific Component „**cuTile Performance Lab**" (interaktiver
einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–9 (inkl. TZ 7.5) sind fertig, verifiziert
und dokumentiert:** die headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → Familien-Router → reshape/B1 → emit → compile+gehärteter Cache → Kalt-Lauf →
verify(fp32) → benchmark → Metriken → Baselines → **Fusions-Zweitmessung** → GPU-Zustand →
Store); die Dash-GUI fährt den Live-Loop als Batch-Vergleich (Familien-Auswahl · Ausdruck via
Presets/Freitext + Größen je Index inkl. n-är · Elementwise-Op · **Epilog-Fusion** · mehrere
Tile-Zeilen · L2-Swizzle-Konfigs mit GROUP_M · Zahlenformate · Baselines) → Kreuzprodukt
Format × Tile × Swizzle, je Config ein `run()` unter EINEM GPU-Lock → KPIs/Verify/Code + drei
Charts + History-Panel. **TZ 9** hat die **Fusion** (Kontraktion + Elementwise-Epilog) ergänzt:
Epilog `bias`/`relu` auf dem Akkumulator-Tile **vor** `ct.store`, additiv über
`RunConfig.epilog`, `epilog=None` byte-identisch, fused-vs-sequentiell als verifizierte
Zweitmessung, CLI-Sweep über drei Formen, Report-Figur `fusion.png` + Fusions-Kapitel.

**ABER:** Es fehlt der letzte in PLAN §10 „Später/optional" vermerkte Kandidat —
**Copy/Transpose als eigene memory-bound Ops**. Das ist der **äußerste linke Rand der
Roofline**: Operationen mit **null FLOPs**, reine Datenbewegung. Genau deshalb sind sie
interessant: sie messen die **Bandbreiten-Obergrenze der GB10 direkt** (273 GB/s theoretisch)
und liefern damit den Bezugspunkt, gegen den sich alle memory-bound-Ergebnisse des Reports
einordnen lassen — und die Transposition zeigt zusätzlich, was **nicht-koalesziertes**
Speicher-Zugriffsmuster kostet (der Copy liest/schreibt linear, der Transpose bricht die
Koaleszenz auf einer Achse). Dein Auftrag ist **ausschließlich Teil-Ziel 10 (TZ 10)**.

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1** (f-String-Templates → `@ct.kernel`; ein Modul je Familie unter
  `codegen/templates/`). Copy/Transpose gehören konzeptionell in die **Elementwise-Familie**
  (`copy` existiert dort schon als unäre Op!) — kläre, ob Transpose eine weitere **Op** dieser
  Familie wird oder ein eigenes Template braucht (s. Design-Entscheidungen).
- **Die eine Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie
  Helfer); Haupt-Prozess CUDA-frei; Charts reine, headless-testbare Funktionen. **Kein Naht-Umbau.**
- **verify-before-trust bleibt Gesetz.** Kein Ergebnis/keine Report-Figur ohne bestandene
  fp32-Referenz (bei Transpose ist die Referenz `torch.permute(...).contiguous()`, bei Copy
  die Identität — beides exakt, also **strenge** Toleranz möglich; kläre).
- **Results-Store-Format = JSON Lines** + Kernel als `results/kernels/<slug>.py`. **Kein
  Format-Umbau.** Neue Ops hängen — wie `op`/`epilog`/`__sw_g<N>` — **bedingt** am Slug;
  bestehende Slugs bleiben **byte-identisch**.
- **dataviz-Skill** als Maßstab für neue Figuren (Palette per `validate_palette.js` prüfen,
  Figur ansehen). Ehrliche Story: was die Zahlen zeigen, nicht was man hofft.
- **Erweitern statt neu bauen:** Sphinx-Kapitel wächst in `report.rst`; `cli.py` bekommt den
  Sweep additiv; die Elementwise-Familie bekommt eine weitere Op bzw. ein Schwester-Template.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§10** („Später/optional": *Copy/Transpose als
   eigene memory-bound Ops* ist der verbleibende Kandidat; TZ 9 = Fusion ist dort als erledigt
   dokumentiert), **§2** (Operationen-Zeile: „opt. Copy/Transpose" war von Anfang an im Scope
   vorgesehen), **§5** (GB10: 273 GB/s — der Grund, warum reine Datenbewegung die Headline ist).
2. `project/project-development/prompts/TZ9-fusion.md` **und** `TZ8-politur-report.md` — die
   **vorherigen Aufträge** (Muster für Aufbau/Arbeitsweise; TZ 10 spiegelt genau diese Form).
3. `project/tool_pipeline/codegen/templates/elementwise.py` — **maßgeblich**: das `_OPS`-dict
   (`add`/`mul`/`copy`/`relu` mit `arity` + `frag`), der Tile-Loop, `ct.load`/`ct.store` mit
   `padding_mode=ct.PaddingMode.ZERO`, `launch`-Arity je Op. `copy` ist schon da — die Frage
   ist, was **Transpose** braucht (anderes Index-Muster beim Store, nicht nur ein anderes
   Frag!) und ob der reine Copy als eigene *Familie/Op-Semantik* sichtbar werden soll.
4. `project/tool_pipeline/intermediate_representation/parse.py` — `ElementwiseIR` (Zeile ~248),
   `ReductionIR`, das Familien-Routing in `parse()` (~362). Ein Transpose-Ausdruck ist
   `ij->ji` — **derselbe Index-Satz, andere Reihenfolge**: prüfe, wie `parse` das heute
   klassifiziert (Elementwise? Reduktion? Fehler?) — das ist der eigentliche Knackpunkt.
5. `project/tool_pipeline/measure/metrics.py` — `elementwise_flops`/`elementwise_bytes`,
   `compute_metrics_elementwise`, und **wichtig**: `_finish` behandelt `flops==0` bereits
   ausdrücklich (AI = 0 ⇒ GB/s ist die Primärmetrik, der Punkt sitzt roofline-technisch ganz
   links). Genau dieser Pfad trägt TZ 10 — prüfe, ob er für Transpose stimmt.
6. `project/tool_pipeline/measure/verify.py` — `_reference` (Elementwise-Ops inkl. `relu`,
   Kontraktion mit Epilog, `torch.einsum` als Rest); `_TOLERANCES`. Für Copy/Transpose ist die
   Referenz **exakt** — überlege, ob eine strengere Toleranz (oder Bit-Gleichheit) angebracht
   ist, statt die lockeren fp16-Toleranzen zu erben.
7. `project/tool_pipeline/run.py` — der memory-bound-Zweig (`_build_memory_bound_inputs`,
   `expected_arity`) und wie die Metriken family-abhängig gewählt werden.
8. `project/tool_pipeline/app/components/controls.py` — `_OP_OPTIONS`/`_OP_KEYS`,
   `FAMILY_PRESETS`, `_MEMORY_BOUND`, `validate_*` (Muster: `validate_epilog` aus TZ 9 zeigt,
   wie eine neue Achse family-abhängig validiert wird), `index_categories`.
9. **TZ-9-Neubauten** (dein Ausgangspunkt): `project/tool_pipeline/measure/fusion.py`,
   `project/tool_pipeline/cli.py` (`sweep_configs` — Teil-Sweep (7) ist die Fusions-Gruppe;
   `--epilog`; `print_summary` mit Fusions-Zeilen), `project/tool_pipeline/report_figures.py`
   (`fig_fusion`, `_is_contraction_gemm`/`_is_square` — die **Filter**, die verhindern, dass
   neue Läufe alte Figuren verunreinigen; genau daran musst du beim Hinzufügen denken!),
   `sphinx/source/chapters/group_specific_component/report.rst` (Fusions-Kapitel als Vorbild).

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ9)
**Familien vollständig & verifiziert:** `parse` routet auf `ContractionIR`/`ElementwiseIR`/
`ReductionIR`/`NAryContractionIR`; `emit` routet auf drei Template-Builder; `run` hat einen
memory-bound-Zweig, einen n-är-Zweig und den Fusions-Pfad; `store.config_slug` hängt `op`,
**`__ep_<epilog>`** und bedingt `__sw_g<N>` an (Kontraktion `op=None`, `epilog=None`,
`group_m=8` ⇒ Slug byte-identisch zu TZ 1–6). **GPU auf der Lab-Maschine verfügbar**,
`cuda.tile` vorhanden; `opt_einsum` steht in `requirements.txt`, ist im venv aber NICHT
installiert (Fold-Fallback). **pytest ist im venv installiert** (seit TZ 9).

**TZ-9-Erweiterungen (Post-TZ9, alle additiv + verifiziert):**
- **Epilog-Fusion im Codegen:** `contraction.py` `_EPILOGS = {"bias": {"operand": True},
  "relu": {"operand": False}}`; Epilog-Block zwischen `ct.mma`-Loop und `ct.store`; `bias`
  bringt einen vierten Operanden D (Ausgabe-Form) in Kernel-Signatur, `launch` und
  `ct.launch`-Tuple. `epilog=None` ⇒ **byte-identischer** Quelltext (Test vergleicht die
  Einfügestelle byte-exakt).
- **Vertrag:** `RunConfig.epilog: Optional[str] = None`; `emit`/`run` reichen durch;
  `config_slug` hängt `__ep_bias`/`__ep_relu` **bedingt** an (vor `__sw`).
- **verify inkl. Epilog:** `_apply_epilog_reference` (Kontraktion **gefolgt vom** Epilog);
  `metrics.compute_metrics(..., epilog=...)` zählt den D-Read in die Bytes (ehrliche AI).
- **`measure/fusion.py` (NEU):** `measure_sequential()` misst den sequentiellen
  Zwei-Kernel-Pfad (Plain-Kontraktion + Elementwise-Zwilling `add`/`relu`) in derselben
  bench-Schleife, **verifiziert ihn ebenfalls** und liefert
  `metrics["fusion"] = {available, epilog, fused_ms, sequential_ms, speedup, fused_bytes,
  sequential_bytes, saved_bytes, fused_ai, sequential_ai}` — **graceful**: schlägt sie fehl,
  trägt der dict `available=False` + `note`, ohne den fused-Lauf zu kippen.
- **GUI:** Radio „Epilog-Fusion" (`ID_EPILOG`, nur bei `family=contraction` sichtbar, beim
  Familienwechsel zurückgesetzt), `validate_epilog` (Familie + **n-är-Sperre**),
  `epilog_from_controls`, Epilog in `config_from_controls`/`configs_from_selection`; zwei
  KPI-Karten (Speedup mit Einordnung, gesparter DRAM-Umweg + AI-Verschiebung); Tab-Label
  `· ep bias`.
- **CLI:** `--epilog {bias,relu}` (mit derselben Validierung wie die GUI, Loud-Fail),
  Fusions-Zeilen in `print_summary`, Teil-Sweep (7) in `sweep_configs`: beide Epiloge × **drei**
  Formen (`_SWEEP_FUSION_NARROW` 4096·4096·64, `_SWEEP_FUSION_SQUARE` 1024³,
  `_SWEEP_FUSION_DEEP` 1024·1024·8192) + unfusionierte Bezugspunkte ⇒ **24 Configs**.
- **Report:** `fig_fusion` (Speedup über AI, log-x, eine Linie je Epilog, Referenzlinie 1,0)
  → `_static/gsc/fusion.png`; Kapitel „Fusion: wann lohnt ein Epilog auf dem Akkumulator?"
  mit Messtabelle, Trend-Deutung und Einordnung des A04-Befunds; **alle Report-Tabellen auf
  die neue Sweep-Charge aktualisiert**; `PLAN.md` §2/§10 + `projektplan.rst` nachgezogen.
- **Wichtige Nebenwirkung, die du kennen musst:** weil der Sweep denselben Ausdruck
  `ik,kj->ij` jetzt auch fusioniert und auf **nicht-quadratischen** Formen fährt, filtern
  `report_figures._is_contraction_gemm` (kein Epilog) und `_is_square` (M==N==K) die
  bestehenden Figuren. Der Test `test_sweep_configs_no_duplicate_work` prüft
  `(Slug, Größen, Baselines)` als eindeutig — **nicht** mehr den Slug allein (ein Kernel
  bedient bewusst mehrere Formen).

**Ergebnis TZ 9 (GB10, fp16→fp32):** Speedup 2,22×/2,72× (schmal, AI 21/32) → 1,25×/1,33×
(1024³, AI 205/256) → 1,06×/1,03× (tief, AI 431/455). Monoton fallend.

**Tests grün (Post-TZ9): 286/286** über **12** Dateien — `test_codegen` (55), `test_app_controls`
(54), `test_app_charts` (36), `test_parse` (34), `test_measure` (32), `test_cli` (17),
`test_verify` (13), `test_app_execute` (13), `test_store` (12), `test_app_render` (11),
`test_reshape` (7), `test_app_infra` (2). Dash bootet mit 11 Callbacks. `make html` läuft
GPU-/torch-frei durch (`extensions = []`, nur RST + fertige PNGs), **1** vorbestehende
Warnung (`chapters/10_xdna_whole_npu/loesung.rst:144`, Ref-Ziel `ch10_access` fehlt — nicht
Teil dieses Projekts).

**Noch NICHT umgesetzt (= TZ-10-Arbeit):** Copy und Transpose als **sichtbare, eigene**
memory-bound Operationen mit eigener Story. `copy` existiert als Elementwise-Op, ist aber
nirgends als *Bandbreiten-Referenzmessung* geführt; **Transpose gibt es überhaupt nicht** —
weder im Parser (`ij->ji`), noch im Codegen, noch in der GUI, noch im Report.

## TZ-10-Scope (eng halten!)
1. **Transpose im Codegen:** ein Kernel, der `ij->ji` (und möglichst `bij->bji` /
   allgemeinere Permutationen — kläre den Umfang!) als **reine Datenbewegung** ausführt:
   Tile laden, transponiert speichern (`ct.store` mit vertauschtem Index-Paar bzw. eine
   In-Tile-Transposition). **Null FLOPs.** Ragged-Ränder wie in allen Familien
   (`padding_mode=ZERO` + Store-Clipping) — hier besonders wichtig, weil M/N beim Transponieren
   die Rollen tauschen.
2. **Parser/Vertrag:** `ij->ji` muss **eindeutig** klassifiziert werden (heute vermutlich nicht
   — nachsehen!). Additiv: kein Umbau bestehender IR-Klassen, kein Slug-Format-Umbau; die
   bestehenden Slugs bleiben byte-identisch.
3. **verify-before-trust:** Referenz ist `torch.permute(...).contiguous()` bzw. die Identität
   beim Copy — **exakt**. Kläre, ob eine strengere Toleranz (oder `torch.equal`) gilt, statt
   die fp16-Toleranzen zu erben; ein Transpose darf keinen Rundungsfehler haben.
4. **Metriken + Story:** `flops == 0` ⇒ GB/s ist die einzige sinnvolle Metrik (`_finish`
   behandelt das schon). Die Story ist zweiteilig: (a) **Copy = Bandbreiten-Obergrenze** —
   wie nah kommt reine Datenbewegung an 273 GB/s? (b) **Transpose = Preis der fehlenden
   Koaleszenz** — wie viel Bandbreite kostet das nicht-lineare Zugriffsmuster? Der Vergleich
   Copy ↔ Transpose bei **identischer Datenmenge** ist der ehrliche Kern.
5. **GUI:** Transpose als wählbare Operation (Op oder Familie — s. Design-Entscheidungen) mit
   Preset(s); family-abhängige Validierung; die Punkte erscheinen ganz links auf der Roofline.
6. **CLI-Sweep + Report:** Copy- und Transpose-Läufe additiv in `cli.sweep_configs`
   (mind. eine Größe, an der Copy nahe an die Bandbreite kommt); eine Figur (Vorschlag:
   erreichte GB/s je Operation gegen die 273-GB/s-Linie — kläre die Form mit dem
   dataviz-Maßstab); Report-Kapitel. **Denk an die Figuren-Filter** (`_is_square`,
   `_is_contraction_gemm`) und daran, die Roofline-Beschriftung zu prüfen (der linke Rand
   bekommt neue Punkte).
7. **Tests:** Parser-Klassifikation von `ij->ji`; Codegen-Struktur (null FLOPs, kein `ct.mma`);
   **GPU-Verify exakt** (glatt + ragged + dtypes, inkl. nicht-quadratischer Formen wie
   4096·1024, wo Transponieren die Shape ändert); Metrik `flops==0`/AI-Pfad; Slug-Bedingtheit;
   GUI-Validierung; alle 286 bestehenden Tests bleiben grün; App-Smoke grün.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Transpose = neue Op der Elementwise-Familie oder eigene Familie?** (Empfehlung: eigene
   **Op** in der bestehenden memory-bound-Familie wäre naheliegend, aber der Store-Index
   unterscheidet sich strukturell von allen `_OPS`-Frags — ein **eigenes Template**
   `codegen/templates/transpose.py` mit eigener Familie `"transpose"` ist wahrscheinlich
   ehrlicher und hält `elementwise.py` frei von Sonderfällen. Kläre — das ist die
   Hauptentscheidung von TZ 10.)
2. **Umfang der Permutationen:** nur 2D `ij->ji`? Plus batched `bij->bji`? Oder allgemeine
   Permutation beliebiger Rang-n-Tensoren (das wäre eine kleine B1-artige Kanonisierung:
   Permutation → 2D-Transpose auf zusammengefassten Achsen)? (Empfehlung: mit `ij->ji` +
   `bij->bji` starten, allgemeine Permutation nur wenn sie ohne neues Paradigma fällt.)
3. **Copy: eigene Sichtbarkeit oder bestehende Elementwise-Op?** `copy` existiert schon —
   soll TZ 10 ihn nur als **Bandbreiten-Referenz** in Sweep/Report herausstellen (kein neuer
   Code), oder braucht er eine eigene Op-Semantik? (Empfehlung: bestehende Op nutzen, nur
   Sweep/Report/Story ergänzen — weniger Code, gleiche Aussage.)
4. **Toleranz:** exakte Gleichheit (`torch.equal`) für Copy/Transpose, oder die bestehenden
   dtype-Toleranzen? (Empfehlung: exakt — beides ist eine Permutation von Bits, jede Abweichung
   ist ein Bug. Eigener Toleranz-Eintrag oder eine `exact=True`-Weiche in `verify`.)
5. **Sweep-Formen:** welche Größen zeigen (a) Copy nahe der Bandbreiten-Decke und (b) den
   Koaleszenz-Verlust des Transpose am deutlichsten? Quadratisch vs. stark rechteckig
   (z. B. 4096² gegen 16384·1024)? Kläre Größen — **geteilte Maschine, 32 GiB**.
6. **Roofline-Darstellung:** Punkte mit AI = 0 lassen sich auf einer **log**-Achse nicht
   darstellen. Wie werden sie eingezeichnet (kleines ε, eigene Achsen-Region, eigene Figur)?
   Kläre — sonst verschwinden die interessantesten Punkte still.

## Scope-Grenzen (was TZ 10 NICHT tut)
- **Kein** allgemeines Layout-/Permutations-Framework, **kein** Autotuning der
  Transpose-Kachelung (eine gute Kachelwahl belegen genügt), **kein** shared-memory-optimierter
  Transpose-Spezialkernel jenseits des cuTile-Idioms.
- **Kein** Naht-/Schema-/Slug-/Store-Format-Umbau — additiv, alles Bestehende bleibt grün.
- **Keine** neuen dtypes, **keine** Änderungen an Kontraktion/Fusion.
- Nach TZ 10 ist PLAN §10 „Später/optional" **leer** — dann ist der Abschluss des Projekts
  (Abgabe 27.07.2026: `tar/`-Dateien, Tag, `create_submission.sh`) der nächste Schritt, kein
  weiteres Teil-Ziel.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess
  CUDA-frei; Charts reine, headless-testbare Funktionen. **`make html` bleibt GPU-/torch-frei.**
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla07/mla/.venv/bin/python` (Shell-`python`
  nicht im PATH; Shell-State persistiert nicht). Start: `python -m tool_pipeline` (GUI),
  `python -m tool_pipeline.cli` (headless), `python -m tool_pipeline.report_figures` (Figuren),
  `python -m pytest -q` (Tests). Sphinx: `cd sphinx && make html`.
- **Harte Regel: NIEMALS `git commit` / `git push`, außer der Nutzer fordert es direkt** (dann OHNE
  dich selbst als Autor/Co-Autor — Repo-Regel in `CLAUDE.md`).
- **Geteilte Maschine:** kleine Größen, `torch.manual_seed(0)`, GPU-Lock (`project/.cache/gpu.lock`)
  respektieren, keine unnötigen GPU-Läufe, OOM vermeiden. **Store-Isolation in Tests über `$SP`**
  (nicht `/tmp`); der git-getrackte `results.jsonl`/`kernels/` wird durch Tests NIE verschmutzt.
  **Achtung:** läuft parallel die GUI und hält den GPU-Lock, scheitern GPU-Tests mit „GPU belegt"
  — das ist kein Regress (in TZ 9 einmal passiert); dann die GUI stoppen und neu fahren.
- **Byte-Identität ist heilig:** bestehende Slugs und der unfusionierte/unveränderte
  Kontraktions-Quelltext dürfen sich NICHT ändern (Anti-Drift-Tests + die getrackten
  `kernels/*.py` bleiben stabil).
- **verify-before-trust:** kein Ergebnis/keine Report-Figur ohne bestandene fp32-Referenz.

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern parallel), Verständnis **kurz** bestätigen.
2. TZ 10 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und**
   App in lauffähigem, prüfbarem Zustand — z. B. Parser-Klassifikation → Transpose-Template +
   Byte-Identität der übrigen → Vertrag/Slug → run/verify(exakt)/Metriken → GUI → CLI-Sweep →
   Figur/Report → Tests). Die Design-Entscheidungen oben **vorab** mit dem Nutzer klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt
   bewirkt / was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären).
   Dann auf **die Validierung des Nutzers warten**. **Nicht** mehrere TODOs bündeln.
   (Sagt der Nutzer ausdrücklich „mach alle TODOs nacheinander", darfst du durchlaufen — die drei
   Punkte je TODO bleiben trotzdem Pflicht.)
5. Strikt im TZ-10-Scope bleiben; Scope-Creep (Layout-Framework, Autotuning) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen):
   **PLAN §2/§10 und `projektplan.rst` nachziehen** (Copy/Transpose als erledigt; „Später/optional"
   ist danach leer) **und den Abgabe-Schritt vorbereiten** — Checkliste für `tar/contribution.txt`,
   `tar/git_link.txt`, `tar/git_tag.txt`, `tar/report_link.txt`, Tag `submission-0N` und
   `./create_submission.sh` (s. `CLAUDE.md` „Submission flow"). Es gibt **kein TZ 11**; statt eines
   weiteren Teil-Ziel-Prompts also eine **Abgabe-Checkliste** als MD unter
   `project/project-development/prompts/ABGABE-checkliste.md`.

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**, wo möglich: die
Parser-Klassifikation von `ij->ji`, die Codegen-Struktur (kein `ct.mma`, null FLOPs), die
Slug-Bedingtheit, die Metrik bei `flops==0` und die GUI-Validierung sind deterministisch
prüfbar. Die **Korrektheit** (exakt gegen `torch.permute(...).contiguous()`, glatt/ragged/dtypes,
auch nicht-quadratisch) und der **Durchsatz** brauchen die GPU (unter Lock). Die Report-Figur liest
`results.jsonl` (headless, torch-frei) und wird **angesehen** (dataviz „render & look", Palette per
`validate_palette.js`). `make html` muss **ohne GPU/torch** durchlaufen. App real starten
(GPU-Lock!) und die neue Operation durchklicken. Store in Tests über `$SP` isolieren.

## Definition of Done (TZ 10)
**Copy und Transpose** sind als eigene memory-bound Operationen umgesetzt, verifiziert und
dokumentiert: der Transpose-Kernel bewegt Daten ohne FLOPs und stimmt **exakt** gegen
`torch.permute(...).contiguous()` (glatt, ragged, mehrere dtypes, auch nicht-quadratisch); die
Klassifikation von `ij->ji` ist eindeutig und additiv; Slug/Vertrag wachsen bedingt (bestehende
Slugs byte-identisch); `flops==0` liefert GB/s als Primärmetrik; die GUI bietet die Operation an;
`cli.py` fährt einen reproduzierbaren Copy/Transpose-Sweep; es gibt eine Figur, die **Copy gegen
die 273-GB/s-Decke** und **Transpose gegen Copy** stellt; `report.rst` trägt die Story (inkl. des
ehrlichen Preises der fehlenden Koaleszenz); `make html` läuft GPU-/torch-frei durch; **alle Tests
grün** (die 286 Post-TZ9-Tests + neue) + App-Smoke; Kontraktion/Fusion/Elementwise/Reduktion
bleiben unverändert. **Zusätzlich:** nach jedem TODO gab es eine sehr einfache Erklärung, und als
letzter Schritt sind PLAN/`projektplan.rst` nachgezogen und die **Abgabe-Checkliste** erstellt.

---

## Session-Prompt (zum Starten von TZ 10 — dies dem Assistenten geben)

> Lies zuerst VOLLSTÄNDIG die Datei
> `project/project-development/prompts/TZ10-copy-transpose.md` — das ist dein Auftrag
> (Teil-Ziel 10 des „cuTile Performance Lab"). Befolge die dort unter „Arbeitsweise
> (verbindlich)" beschriebene Vorgehensweise strikt:
>
> Belese dich zuerst gründlich in die bestehende, verifizierte Implementierung (TZ 1–9) — genau die
> Dateien/Reihenfolge unter „Zuerst lesen", **inkl. `codegen/templates/elementwise.py` (dort steckt
> `copy` schon) und `intermediate_representation/parse.py` (wie wird `ij->ji` heute klassifiziert?)**
> — und bestätige dein Verständnis kurz. Zerlege TZ 10 in geordnete TODOs und kläre die
> „Design-Entscheidungen" vorab mit mir (vor allem: Transpose als eigene Familie/Template oder als
> Elementwise-Op, und wie AI-0-Punkte auf der Log-Roofline dargestellt werden). Lege die
> Aufschlüsselung zur FREIGABE vor, BEVOR du Code schreibst. Dann TODO für TODO: nach jedem anhalten
> und zeigen (a) was du getan hast, (b) wie du es verifiziert hast, UND (c) eine SEHR EINFACHE
> Erklärung in Alltagssprache. Dann auf meine Validierung warten. Nicht mehrere TODOs bündeln.
> Harte Regeln: NIEMALS git commit/push (außer ich fordere es direkt; dann ohne dich als
> Autor/Co-Autor); Prosa/Kommentare/UI auf Deutsch; aus `project/` mit dem venv-Python
> `.venv/bin/python` ausführen; `make html` GPU-/torch-frei; verify-before-trust: Report-Figuren nur
> aus ok-Läufen; bestehende Slugs und der Kontraktions-Quelltext bleiben byte-identisch. Leg los mit
> dem Einlesen.
