# Auftrag: TZ 9 — Fusion (Kontraktion + Elementwise-Epilog)

Du arbeitest im Repo (aktueller Checkout, z. B. `/home/mla07/mla` — Pfade relativ nehmen).
Wir bauen die Group-Specific Component „**cuTile Performance Lab**" (interaktiver
einsum/GEMM-Explorer, GPU/cuTile). **Teil-Ziele 1–8 (inkl. TZ 7.5) sind fertig, verifiziert
und dokumentiert:** die headless-Pipeline läuft über die eine Naht `run(config) → RunResult`
(parse → Familien-Router → reshape/B1 → emit → compile+**gehärteter** Cache → Kalt-Lauf →
verify(fp32) → benchmark → Metriken → Baselines → GPU-Zustand → Store); die Dash-GUI fährt den
Live-Loop als Batch-Vergleich (Familien-Auswahl · Ausdruck via Presets/Freitext + Größen je
Index inkl. **n-är** · Elementwise-Op · **mehrere Tile-Zeilen** · **L2-Swizzle-Konfigs mit
GROUP_M** · Zahlenformate · Baselines) → Kreuzprodukt Format × Tile × Swizzle, je Config ein
`run()` unter EINEM GPU-Lock → KPIs/Verify/Code + drei Charts (Durchsatz · Genauigkeit↔Durchsatz
· Roofline, serien-disambiguiert) + ausklappbares **History-Panel**. **Alle drei Operations-
Familien laufen end-to-end** (Kontraktion inkl. n-ärer Kette · Elementwise · Reduktion), gegen
fp32 verifiziert, family-korrekt gemessen. **TZ 8** hat das Deliverable poliert und dokumentiert:
gehärteter Compile-Cache (atomarer Write), systematisch belegte Randfälle, CLI-Batch-Sweeps,
der **Sphinx-Report mit echten Figuren** und der re-synchronisierte `projektplan.rst`.

**ABER:** Es fehlt der letzte, in PLAN §2/§10 vorgemerkte Substanz-Schritt — die **Fusion** von
Kontraktion und Elementwise-Epilog. Der A04-Befund (`assignments/04_assignment/src/task_02.py`)
zeigt sie **ehrlich interessant**: fused = 13,05 ms vs. sequentiell = 12,84 ms ⇒ **0,984×** auf
einer *compute-dominierten* Form (die Kontraktion mit 12,83 ms erschlägt den Epilog von 0,067 ms).
Genau das ist die Frage, die TZ 9 sichtbar machen soll: **wann lohnt Fusion?** (kleine/memory-
bound Kontraktion mit relativ teurem Epilog → Fusion spart den DRAM-Umweg des Zwischentensors;
compute-dominierte Kontraktion → Fusion ist neutral bis leicht negativ). Dein Auftrag ist
**ausschließlich Teil-Ziel 9 (TZ 9): Fusion** — die konsequente nächste Stufe der memory-bound-
Story, additiv auf dem bewiesenen 2-Op-GEMM-Pfad.

---

## Bereits festgelegt — NICHT neu evaluieren
- **GUI-Framework = Plotly Dash** (fix). Charts = native Plotly. Keine Framework-Diskussion.
- **Codegen = C1** (f-String-Templates → `@ct.kernel`; ein Modul je Familie unter
  `codegen/templates/`). **Kein neues Codegen-Paradigma** — die Fusion ist ein **Epilog auf dem
  Akkumulator-Tile** im bestehenden Kontraktions-Template (`contraction.py`), kein neues Template.
- **Die eine Naht bleibt:** `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie
  Helfer); Haupt-Prozess CUDA-frei; Charts reine, headless-testbare Funktionen. **Kein Naht-Umbau.**
- **verify-before-trust bleibt Gesetz.** Kein Ergebnis/keine Report-Figur ohne bestandene
  fp32-Referenz (die Referenz ist hier `torch.einsum(...)` **gefolgt vom** Epilog).
- **Results-Store-Format = JSON Lines** + Kernel als `results/kernels/<slug>.py` (lesbarer
  Slug). **Kein Format-Umbau.** Der Slug wächst **bedingt** (wie `op`/`__sw_g<N>`): ein Epilog
  hängt ein Suffix an, `epilog=None` ⇒ Slug **byte-identisch** zu TZ 1–8 (keine Cache-Drift).
- **dataviz-Skill** als Maßstab für neue Figuren; die honest-Story (0,98× je Form) klar erzählen.
- **Erweitern statt neu bauen:** Sphinx-Kapitel wachsen in `report.rst`; `cli.py` bekommt den
  Fusions-Sweep additiv; `contraction.py` bekommt einen **optionalen** Epilog-Zweig.

## Zuerst lesen (in dieser Reihenfolge)
1. `project/project-development/PLAN.md` — **§2** (Zeile „Fusion = **Zukunftskandidat**
   (Kontraktion+Elementwise-Epilog, A04), nicht jetzt; A04-Befund: 0,98× — ehrlich interessant,
   aber später"), **§10** („Später/optional": Fusion als nächster Kandidat; n-är ist in TZ 7.5
   erledigt), **§3** (Codegen C1+B1), **§5** (GB10 memory-bound — der Grund, warum Fusion
   überhaupt eine Story hat).
2. **Der A04-Fusions-Befund (maßgeblich):** `assignments/04_assignment/src/task_02.py` — VOLLSTÄNDIG
   lesen: der `kernel_fused` (Epilog `* D` auf dem Akku-Tile **vor** `ct.store`), die drei
   Varianten (contract · elemwise · fused), der Benchmark und der **Ergebnis-Block am Ende**
   (0,984×). Das ist das Vorbild für den Fusions-Kernel UND die honest-Story.
3. `project/project-development/prompts/TZ7.5-verbesserungen.md` **und** `TZ8-politur-report.md` —
   die **vorherigen Aufträge** (Muster für Aufbau/Arbeitsweise; TZ 9 spiegelt genau diese Form).
4. `project/tool_pipeline/schema.py` — `RunConfig` (wo ein **additives** `epilog`-Feld hingehört,
   analog zu `op`/`group_m`: `Optional[str] = None`, Default-byte-identisch) und `RunResult`.
5. `project/tool_pipeline/codegen/templates/contraction.py` — der GEMM-Kernel: der `ct.mma`-Loop
   und v. a. die **Store-Stelle** (`acc = ct.astype(...); ct.store(C, ...)`), wo der Epilog auf das
   Akku-Tile angewandt wird (z. B. `acc = acc + bias`, `acc = ct.maximum(acc, 0)`, `acc = acc * s`),
   BEVOR geschrieben wird. Beachte die **Byte-Identität** (Anti-Drift-Test
   `test_emit_contraction_header_byte_identical`, `test_swizzle_equals_noswizzle`): `epilog=None`
   MUSS denselben Quelltext liefern wie heute.
6. `project/tool_pipeline/codegen/emit.py` — Familien-Routing (wie der Epilog-Parameter durchgereicht
   wird) + `project/tool_pipeline/store/store.py` `config_slug` (das **bedingte** Epilog-Suffix,
   exakt gespiegelt vom `op`-Suffix / `__sw_g<N>`).
7. `project/tool_pipeline/run.py` — der **2-Op-Kontraktions-Zweig** (wo der Fusions-Pfad einhakt:
   Epilog-Operand(en) mitbauen, verify = Referenz-Kontraktion **gefolgt vom** Epilog, Metriken).
   Für die honest-Story: der **sequentielle** Vergleich (Kontraktion + separater Elementwise-Lauf)
   ist eine **Baseline-artige** Zweitmessung — überlege, ob als Baseline (`measure/baselines.py`)
   oder als zweiter `run()`-Aufruf im Sweep.
8. `project/tool_pipeline/measure/verify.py` + `metrics.py` — Referenz/Toleranzen (der Epilog geht
   in die fp32-Referenz ein) und die FLOP-/Byte-Zählung (Fusion spart **Bytes**: der Zwischentensor
   wird nicht nach DRAM geschrieben+gelesen → das ist die Roofline-/AI-Verschiebung, die die Story trägt).
9. `project/tool_pipeline/app/components/controls.py` (wo ein **Epilog-Auswahl**-Control hingehört,
   analog zur Elementwise-Op-Auswahl; nur bei Kontraktion sichtbar) + `callbacks.py` (Verdrahtung) +
   `charts.py` (fused-vs-sequentiell-Darstellung; ggf. ein vierter Vergleich).
10. **TZ-8-Neubauten** (dein Ausgangspunkt): `project/tool_pipeline/cli.py` (`sweep_configs`/
    `run_sweep`/`--sweep` — der Fusions-Sweep wächst hier), `project/tool_pipeline/report_figures.py`
    (die 4 PNGs — ein Fusions-Chart kommt dazu), `sphinx/source/chapters/group_specific_component/
    report.rst` (wo die Fusions-Story dokumentiert wird).

## Die bisherige Implementierung, auf der du aufbaust (Anker, Ist-Zustand POST-TZ8)
**Familien vollständig & verifiziert:** `parse` routet auf `ContractionIR`/`ElementwiseIR`/
`ReductionIR`/`NAryContractionIR`; `emit` routet auf die drei Template-Builder; `run` hat einen
additiven memory-bound-Zweig und einen n-är-Zweig; `store.config_slug` hängt Op **und** bedingt
`__sw_g<N>` an (Kontraktion `op=None`, `group_m=8` ⇒ Slug byte-identisch); `RunConfig.op`/`group_m`
sind additiv. **GPU auf der Lab-Maschine verfügbar** (`torch.cuda.is_available()`), `cuda.tile`
vorhanden; `opt_einsum` steht in `requirements.txt`, ist im venv aber NICHT installiert (Fold-Fallback).

**TZ-8-Erweiterungen (Post-TZ8, alle additiv + verifiziert — TZ 9 baut darauf auf):**
- **Compile-Cache gehärtet:** `store.save_kernel` schreibt **atomar** (Temp im selben Verzeichnis →
  `os.replace`, Muster von `_atomic_rewrite`); `compile._read_text_or_none` + `load_kernel` erkennen
  korrupte/nicht dekodierbare `<slug>.py` und schreiben sie neu statt zu crashen. Byte-Inhalt
  unverändert (keine Slug-Drift). Tests in `tests/test_store.py` (jetzt 12).
- **Randfälle systematisch belegt:** ragged-GPU-Verify für **Elementwise** und **Reduktion**
  (inkl. Loop-Fallback der Reduktion) in `tests/test_codegen.py` (jetzt 45) — der ZERO-Padding-/
  Store-Clipping-Pfad ist nun für alle drei Familien gegen fp32 bewiesen.
- **CLI-Batch-Sweeps:** `cli.sweep_configs()` (deterministisch, headless, an
  `controls.configs_from_selection` angelehnt) erzeugt einen kuratierten Satz über alle Familien
  inkl. GROUP_M/Multi-Tile/n-är; `cli.run_sweep()` fährt ihn unter EINEM GPU-Lock mit EINER
  Batch-`run_id`; `--sweep` / `--show-configs`; `print_summary` ist **family-geformt** (GB/s primär
  für memory-bound). Flags `--family/--op/--expr/--size` additiv. Tests in `tests/test_cli.py` (9, NEU).
- **Report-Figuren:** `tool_pipeline/report_figures.py` (torch-frei) liest die jüngste
  `CLI-Report-Sweep`-Charge (`status=="ok"`) → 4 PNGs (`durchsatz_formate`, `genauigkeit_durchsatz`,
  `roofline`, `tile_swizzle`) mit der CVD-sicheren dataviz-Palette nach `sphinx/source/_static/gsc/`.
- **Report/Projektplan:** `report.rst` blogartig gefüllt (die 3 Charts inkl. Roofline mit beiden
  Seiten, Tabellen aus `results.jsonl`, Codegen-/verify-/dtype-/Tiling-/Swizzle-/n-är-/History-Story
  **plus** der ehrliche verify-before-trust-Befund: bf16-Reduktion + tiefe n-är-Ketten scheitern);
  `projektplan.rst` aus `PLAN.md` re-synchronisiert (inkl. TZ 7.5, vollem Fortschritts-Log, Ausblick
  „Fusion").
- **Theming/Fehlerzustände poliert:** `theme.css` erweitert (Akzent-System, Marken-Button, History-
  Accordion, `.ctl-section`/`.ctl-label` — konsolidierte Sidebar-Typografie); Fehlerzustände
  benutzerfreundlich (breiteres `compile_error`-Label deckt n-är-Loud-Fail ab, freundliche „Interner
  Fehler"-Meldung, farbcodiertes History-Feedback, `alert-info` scrollt in den Blick,
  `validate_swizzle_configs` verdrahtet).

**Tests grün (Post-TZ8): 254/254** über **12** Dateien — `test_parse` (34), `test_verify` (13),
`test_reshape` (7), `test_store` (12), `test_cli` (9, NEU), `test_measure` (27), `test_codegen`
(45), `test_app_controls` (50), `test_app_charts` (36), `test_app_execute` (10), `test_app_render`
(9), `test_app_infra` (2). Dash bootet mit 11 Callbacks. `make html` läuft GPU-/torch-frei durch.

**Noch NICHT umgesetzt (= TZ-9-Arbeit):** die Fusion selbst. Es gibt keinen Epilog-Zweig im
Kontraktions-Template, kein `epilog`-Feld in `RunConfig`, keine Epilog-Auswahl in der GUI, keinen
fused-vs-sequentiell-Vergleich und keine Fusions-Story im Report.

## TZ-9-Scope (eng halten!)
1. **Epilog-Fusion im Codegen** (`contraction.py`): ein **optionaler** Epilog auf dem
   Akkumulator-Tile **vor** `ct.store` — Start-Set klein halten (Empfehlung: `bias`-Add mit einem
   zusätzlichen Operanden, `relu` = `ct.maximum(acc, 0)`, ggf. `scale`). `epilog=None` ⇒ Quelltext
   **byte-identisch** zu heute (Anti-Drift-Tests bleiben grün).
2. **Vertrag:** additives `RunConfig.epilog: Optional[str] = None` (+ ggf. Epilog-Operand-Form);
   `config_slug` hängt das Epilog-Suffix **bedingt** an (byte-identisch bei `None`); `emit`/`run`
   reichen es durch. Kein Schema-/Naht-/Slug-Format-Umbau.
3. **verify-before-trust:** die fp32-Referenz ist Kontraktion **gefolgt vom** Epilog
   (`torch.einsum(...)` dann `+bias`/`relu`/`*s`), mit den bestehenden dtype-Toleranzen.
4. **Metriken + honest-Story:** die Fusion spart den DRAM-Umweg des Zwischentensors → **weniger
   Bytes** ⇒ höhere arithmetische Intensität; miss **fused vs. sequentiell** (Kontraktion + separater
   Elementwise-Lauf) und zeige, **wann** Fusion gewinnt (kleine/memory-bound Kontraktion) und wann
   nicht (compute-dominiert, A04 0,98×).
5. **GUI:** eine **Epilog-Auswahl** (nur bei Familie=Kontraktion sichtbar, analog zur Elementwise-Op),
   family-abhängige Validierung; der fused-vs-sequentiell-Vergleich in den Charts sichtbar.
6. **CLI-Sweep + Report:** ein Fusions-Sweep in `cli.sweep_configs` (mind. eine Form, wo Fusion
   gewinnt, und eine, wo sie neutral/negativ ist); eine Fusions-Figur in `report_figures.py`; die
   Fusions-Story (inkl. der ehrlichen 0,98×-Einordnung) in `report.rst`.
7. **Tests:** Byte-Identität bei `epilog=None`; Epilog-Codegen-Struktur (headless); ragged-/dtype-
   GPU-Verify der Fusion gegen `torch` (unter Lock); Slug-Bedingtheit; fused-vs-sequentiell-Logik
   headless; alle 254 bestehenden Tests bleiben grün; App-Smoke grün.

## Design-Entscheidungen — vorab klären/vorschlagen (nicht raten)
1. **Epilog-Set:** nur `bias`-Add (+ Operand) — oder zusätzlich `relu`/`scale`? (Empfohlen: mit
   `bias`+`relu` starten — deckt den A04-Fall und den „billigen memory-bound-Epilog"-Fall ab; `scale`
   optional.) Kläre.
2. **Ausdruck der Fusion:** additives `RunConfig.epilog`-Feld (empfohlen — Kontraktion bleibt
   Kontraktions-Familie, `epilog=None` byte-identisch) vs. eine eigene „fused"-Familie. Kläre.
3. **fused-vs-sequentiell-Vergleich:** als **Baseline** (`measure/baselines.py`: „sequentiell" neben
   cuBLAS/naive) — oder als **zweiter Sweep-Lauf** (Kontraktion + Elementwise getrennt) und Vergleich
   im Chart? (Empfohlen: sequentiell als *zweite Messung* im Fusions-Sweep, damit die Roofline beide
   Punkte zeigt.) Kläre.
4. **Bias-Operand-Form:** voller Broadcast-Operand (`C`-Form) vs. Vektor je Ausgabe-Zeile? (Empfohlen:
   die A04-Form — ein Operand der Ausgabe-Form, elementweise; minimal + bewiesen.) Kläre.
5. **Sweep-Formen:** welche zwei/drei Formen zeigen die Story am ehrlichsten (Fusion-Gewinn vs.
   -Neutralität)? Kläre Größen (geteilte Maschine!).

## Scope-Grenzen (was TZ 9 NICHT tut)
- **Keine** Fusion jenseits Kontraktion+Elementwise-Epilog (kein Elementwise→Elementwise, keine
  Multi-Epilog-Ketten, keine Attention/Softmax-Fusion).
- **Kein** Naht-/Schema-/Slug-/Store-Format-Umbau — additiv, alles Bestehende bleibt grün.
- **Kein** neues Mess-/Codegen-Paradigma, **kein** Autotuning, **keine** neuen dtypes/Familien.
- **Copy/Transpose als eigene memory-bound Ops** bleibt der *nächste* Kandidat (TZ 10) — nicht jetzt.

## Konventionen & harte Regeln
- Prosa/Kommentare/UI-Texte auf **Deutsch** (Repo-Konvention). Saubere Docstrings.
- `app/` importiert im Live-Loop **nur** `run` + `schema` (+ torch-freie Helfer); Haupt-Prozess
  CUDA-frei; Charts reine, headless-testbare Funktionen. **`make html` bleibt GPU-/torch-frei.**
- Ausführen aus `project/` mit dem **venv-Python** `/home/mla07/mla/.venv/bin/python` (Shell-`python`
  nicht im PATH; Shell-State persistiert nicht). Start: `python -m tool_pipeline` (GUI),
  `python -m tool_pipeline.cli` (headless), `python -m tool_pipeline.report_figures` (Figuren).
  Sphinx: `cd sphinx && make html`.
- **Harte Regel: NIEMALS `git commit` / `git push`, außer der Nutzer fordert es direkt** (dann OHNE
  dich selbst als Autor/Co-Autor — Repo-Regel in `CLAUDE.md`).
- **Geteilte Maschine:** kleine Größen, `torch.manual_seed(0)`, GPU-Lock (`project/.cache/gpu.lock`)
  respektieren, keine unnötigen GPU-Läufe, OOM vermeiden. **Store-Isolation in Tests über `$SP`**
  (nicht `/tmp`); der git-getrackte `results.jsonl`/`kernels/` wird durch Tests NIE verschmutzt.
- **Byte-Identität ist heilig:** `epilog=None` ⇒ byte-identischer Kontraktions-Quelltext und
  byte-identischer Slug (die Anti-Drift-Tests + die 59 git-getrackten `kernels/*.py` bleiben stabil).
- **verify-before-trust:** kein Ergebnis/keine Report-Figur ohne bestandene fp32-Referenz (inkl. Epilog).

## Arbeitsweise (verbindlich)
1. Genannte Dateien lesen (gern per Workflow/Subagenten parallel), Verständnis **kurz** bestätigen.
2. TZ 9 in **sinnvolle Sub-Ziele + geordnete TODOs** zerlegen (jedes TODO lässt Pipeline **und** App
   in lauffähigem, prüfbarem Zustand — z. B. Epilog-Codegen + Byte-Identität → `epilog` im Vertrag +
   Slug → run/verify/Metriken → fused-vs-sequentiell → GUI-Auswahl → CLI-Sweep → Report-Figur/-Text →
   Tests). Die Design-Entscheidungen oben **vorab** mit dem Nutzer klären.
3. Aufschlüsselung **zur Freigabe vorlegen, BEVOR** du Code schreibst.
4. Dann **TODO für TODO**: nach jedem anhalten und zeigen: (a) **was du getan hast**, (b) **wie du es
   verifiziert hast**, **und (c) eine SEHR EINFACHE Erklärung** — in Alltagssprache, was der Schritt
   bewirkt / was das Tool jetzt kann (als würdest du es jemandem ohne GPU-/einsum-Wissen erklären).
   Dann auf **die Validierung des Nutzers warten**. **Nicht** mehrere TODOs bündeln.
5. Strikt im TZ-9-Scope bleiben; Scope-Creep (Attention-Fusion, Multi-Epilog, Autotuning) widerstehen.
6. **Als LETZTER Schritt** (nach Abnahme aller TODOs; ein Review-Durchlauf ist optional-empfohlen):
   **das nächste Teil-Ziel — TZ 10 — anschauen, vorbereiten und einen Session-Prompt + Planungs-MD
   erstellen.** Der verbleibende erste Zukunftskandidat aus PLAN §10 „Später/optional" ist dann
   **Copy/Transpose als eigene memory-bound Ops** (n-är in TZ 7.5, Fusion in TZ 9 erledigt). Genau
   nach *diesem* Muster: gründlich einlesen (Workflow), **PLAN §10 maßgeblich**, Anker aus dem *dann*
   aktuellen Post-TZ9-Ist-Zustand, MD unter `project/project-development/prompts/TZ10-*.md`, und
   **diese Arbeitsweise inkl. der zwei Zusätze (sehr einfache Erklärung nach jedem TODO + Planung des
   übernächsten TZ als letzter Schritt) weitergeben.**

## Verifikation (Hinweis)
Trenne testbare Logik von Dash/GPU und teste sie **headless**, wo möglich: die Byte-Identität bei
`epilog=None` (Text-Vergleich, kein GPU), die Epilog-Codegen-Struktur (build-Module-Assertions), die
Slug-Bedingtheit und die fused-vs-sequentiell-Config-Erzeugung sind deterministisch prüfbar. Die
Fusions-**Korrektheit** (ragged/dtype gegen `torch.einsum(...)`+Epilog) und der fused-vs-sequentiell-
**Durchsatz** brauchen die GPU (unter Lock). Die Report-Figur liest `results.jsonl` (headless,
torch-frei) und wird **angesehen** (dataviz „render & look"). `make html` muss **ohne GPU/torch**
durchlaufen. App real starten (GPU-Lock!) und die Epilog-Auswahl + den fused-vs-sequentiell-Vergleich
durchklicken. Store in Tests über `$SP` isolieren.

## Definition of Done (TZ 9)
Die **Fusion** (Kontraktion + Elementwise-Epilog) ist umgesetzt, verifiziert und dokumentiert: der
Epilog wird im Kontraktions-Template auf dem Akku-Tile vor `ct.store` angewandt (`epilog=None`
byte-identisch), über ein additives `RunConfig.epilog` gesteuert und bedingt im Slug geführt; die
fp32-Referenz schließt den Epilog ein; der **fused-vs-sequentiell-Vergleich** ist gemessen und macht
die honest-Story sichtbar (wann Fusion gewinnt — memory-bound-Epilog — und wann nicht — A04 0,98×);
die GUI bietet eine Epilog-Auswahl (nur bei Kontraktion); `cli.py` fährt einen reproduzierbaren
Fusions-Sweep; `report_figures.py` erzeugt die Fusions-Figur; `report.rst` trägt die Fusions-Story;
`make html` läuft GPU-/torch-frei durch; **alle Tests grün** (die 254 Post-TZ8-Tests + neue Fusions-/
Byte-Identitäts-/Slug-/verify-Tests) + App-Smoke; die bestehenden Familien bleiben unverändert.
**Zusätzlich:** nach jedem TODO gab es eine sehr einfache Erklärung, und als letzter Schritt ist
**TZ 10 (Copy/Transpose als eigene memory-bound Ops) vorbereitet** (Planungs-MD + Session-Prompt).

---

## Session-Prompt (zum Starten von TZ 9 — dies dem Assistenten geben)

> Lies zuerst VOLLSTÄNDIG die Datei
> `project/project-development/prompts/TZ9-fusion.md` — das ist dein Auftrag (Teil-Ziel 9 des
> „cuTile Performance Lab"). Befolge die dort unter „Arbeitsweise (verbindlich)" beschriebene
> Vorgehensweise strikt:
>
> Belese dich zuerst gründlich in die bestehende, verifizierte Implementierung (TZ 1–8) — genau die
> Dateien/Reihenfolge unter „Zuerst lesen", **inkl. des A04-Fusions-Befunds** in
> `assignments/04_assignment/src/task_02.py` — und bestätige dein Verständnis kurz. (Paralleles Lesen
> per Subagenten/Workflow ist erwünscht.) Zerlege TZ 9 in geordnete TODOs und kläre die
> „Design-Entscheidungen" vorab mit mir. Lege die Aufschlüsselung zur FREIGABE vor, BEVOR du Code
> schreibst. Dann TODO für TODO: nach jedem anhalten und zeigen (a) was du getan hast, (b) wie du es
> verifiziert hast, UND (c) eine SEHR EINFACHE Erklärung in Alltagssprache. Dann auf meine
> Validierung warten. Nicht mehrere TODOs bündeln. Harte Regeln: NIEMALS git commit/push (außer ich
> fordere es direkt; dann ohne dich als Autor/Co-Autor); Prosa/Kommentare/UI auf Deutsch; aus
> `project/` mit dem venv-Python `.venv/bin/python` ausführen; `make html` GPU-/torch-frei;
> verify-before-trust: Report-Figuren nur aus ok-Läufen; `epilog=None` byte-identisch. Leg los mit
> dem Einlesen.
