.. _gsc_report_architektur:

##########################
Teil 2 — Architektur
##########################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Die eine Naht
=============

Das gesamte System hängt an **einer** Schnittstelle:

.. code-block:: text

   run(config: RunConfig) -> RunResult

Zwei Datentypen, eine Funktion. Die Oberfläche baut ein ``RunConfig``, der Kern
liefert ein ``RunResult`` und nichts darüber hinaus wird zwischen beiden
Seiten geteilt. Das ist die wichtigste Entscheidung der Architektur, und sie hat
vier konkrete Konsequenzen:

1. **Der Kern ist ohne Oberfläche vollständig benutzbar.** ``cli.py`` baut
   dieselben ``RunConfig``-Objekte und ruft dieselbe Funktion; der Report-Sweep
   läuft headless. Es gibt keinen Code, der nur „in der GUI" existiert.
2. **Die Oberfläche ist austauschbar** und muss nichts über Codegen, Kacheln oder
   CUDA wissen.
3. **Der Hintergrund-Job umschließt genau einen Aufruf.** Alles, was ein Lauf
   braucht, passiert innerhalb von ``run()`` — es gibt keinen Zustand, den der
   Aufrufer vorher oder nachher herstellen müsste.
4. **Testbarkeit:** Wer ``run()`` testet, testet das Produkt; wer die
   Chart-Funktionen testet, braucht keine GPU.

``run()`` wirft nie
-------------------

Der Vertrag hat eine Zusatzklausel: **``run()`` löst keine Exception
nach außen aus.** Jeder Ausgang wird stattdessen in einen von vier Zuständen
kategorisiert:

.. list-table:: Die vier Zustände eines Laufs
   :header-rows: 1
   :widths: 22 30 48

   * - ``status``
     - Bedeutung
     - Typische Ursache
   * - ``ok``
     - compiliert, verifiziert, gemessen
     - der Normalfall — **nur** dieser Status liefert Kennzahlen
   * - ``compile_error``
     - nicht baubar
     - Ausdruck unparsebar, unzulässige Format-Kombination, cuTile-JIT scheitert
   * - ``verify_failed``
     - läuft, rechnet aber falsch
     - Kernel-Fehler oder Format-Grenze — **liefert keine Durchsatzzahl**
   * - ``run_error``
     - compiliert, crasht beim Ausführen
     - Launch-/Speicher-Fehler zur Laufzeit

Der Grund für diese Klausel ist die Oberfläche: Ein Tool, in dem man beliebige
einsum-Ausdrücke und Kachelgrößen *ausprobieren* soll, produziert zwangsläufig
ungültige Eingaben. Würde ``run()`` werfen, müsste jede Aufrufstelle jeden
Fehlertyp kennen. Stattdessen rendert die GUI vier bekannte Zustände. Ein
Nebeneffekt macht die Sache robust: Selbst ein Fehler beim *Speichern* des
Ergebnisses kippt den Lauf nicht, er wird an ``error`` angehängt, das
``RunResult`` aber trotzdem geliefert.

Der Vertrag im Detail: RunConfig
================================

``RunConfig`` beschreibt einen Lauf **vollständig** — es gibt keinen impliziten
Zustand. Die letzte Spalte ist wichtiger, als sie aussieht: Sie sagt, ob ein Feld
den *erzeugten Quelltext* beeinflusst und damit in den Kernel-Namen (Slug) eingeht.
Felder, die das nicht tun, dürfen sich ändern, ohne den Compile-Cache zu
invalidieren.

.. list-table:: ``RunConfig`` (``schema.py``)
   :header-rows: 1
   :widths: 20 46 34

   * - Feld
     - Bedeutung
     - Im Slug?
   * - ``family``
     - ``contraction`` · ``elementwise`` · ``reduction`` — wählt das Template
     - nein (folgt deterministisch aus ``expr``/``op``)
   * - ``op``
     - ``add``/``mul``/``copy``/``relu`` (Elementwise), ``sum`` (Reduktion),
       ``None`` (Kontraktion)
     - **ja, wenn gesetzt** — sonst träfen ``add`` und ``mul`` beim identischen
       Ausdruck ``ij,ij->ij`` dieselbe Cache-Datei
   * - ``epilog``
     - ``bias``/``relu`` — fusionierte elementweise Op auf dem Akkumulator
     - **ja, wenn gesetzt** (``__ep_bias``)
   * - ``expr``
     - der einsum-Ausdruck, Single Source of Truth
     - ja
   * - ``inputs`` / ``output``
     - aus ``expr`` abgeleitet (reine String-Zerlegung)
     - ja (über ``expr``)
   * - ``dim_sizes``
     - Größe je Index, z. B. ``{"i": 1024, "k": 1024, "j": 1024}``
     - **nein** — M/N/K sind Launch-Argumente, der Quelltext ist über die Größen
       generisch
   * - ``dtype`` / ``acc_dtype``
     - Compute- und Akkumulator-Format
     - ja
   * - ``tile``
     - ``{"TM", "TN", "TK"}`` — als Literale in den Quelltext gebacken
     - ja
   * - ``swizzle`` / ``group_m``
     - L2-Block-Umordnung an/aus, Gruppengröße
     - ja (``group_m`` nur bei Abweichung vom Default 8 — siehe unten)
   * - ``baselines``
     - z. B. ``["cublas"]`` — optionale Vergleichsmessungen
     - **nein** (ändern den Kernel nicht)
   * - ``bench_warmup`` / ``bench_iters``
     - Messaufwand (Default 10 / 30)
     - **nein**

Warum ``group_m`` nur *bedingt* in den Slug eingeht, ist aufgrund einer bewussten jedoch unschöne Entscheidung: Der Default 8 war in der Startphase hart verdrahtet und die veränderung wurde erst spät nachgetragen. Ginge
er nun unbedingt in den Namen ein, hießen alle bestehenden, eingecheckten
Kernel-Dateien plötzlich anders. Deshalb bleibt ``GROUP_M = 8`` das bare ``__sw``, und nur
abweichende Werte erzeugen ``__sw_g16``. Die Regel, auf die es ankommt, ist trotzdem
erfüllt: **verschiedener Quelltext ⇒ verschiedener Slug.**

Der Vertrag im Detail: RunResult
================================

.. list-table:: ``RunResult`` (= eine Zeile in ``results.jsonl``)
   :header-rows: 1
   :widths: 22 78

   * - Feld
     - Inhalt
   * - ``status``
     - einer der vier Zustände
   * - ``config``
     - Echo der Eingabe — jede Zeile ist selbstbeschreibend
   * - ``kernel_path``
     - relativer Pfad des persistierten Kernels (``results/kernels/<slug>.py``)
   * - ``kernel_source``
     - der Quelltext für die Code-Anzeige der GUI. Bewusst **nicht** ins JSONL
       geschrieben (Bloat) — er liegt schon als Datei vor
   * - ``accuracy``
     - ``max_abs_err``, ``mean_abs_err``, ``rel_err`` (L2), ``passed``, ``atol``,
       ``rtol``
   * - ``timing``
     - ``compile_ms`` (kalter Lauf), ``run_ms`` (Median), ``min_ms``, ``p90_ms``,
       ``sigma_ms``, ``bench_iters``
   * - ``metrics``
     - ``tflops``, ``gbps``, ``arithmetic_intensity``, ``percent_peak_flops``,
       ``percent_peak_bw``; optional ``baselines`` und ``fusion``
   * - ``provenance``
     - ``gpu``, ``dtype``, ``acc_dtype``, family-geformte ``sizes``,
       ``timestamp``, ``gpu_state`` (Takt/Temperatur/Leistung/Auslastung)
   * - ``error``
     - Fehlertext, falls ``status != ok``
   * - ``run_id`` · ``run_name`` · ``created_at``
     - Batch-Identität: alle Läufe **eines** „Vergleichen"-Klicks bzw. eines
       CLI-Sweeps teilen sie — dadurch ist ein „Testlauf" eine benennbare,
       wieder-ansehbare Einheit

Die vier Gruppen ``accuracy``/``timing``/``metrics``/``provenance`` sind offene
dicts, und das ist Absicht: Jedes Teil-Ziel des Projekts hat dort **Schlüssel
ergänzt** (GB/s, %-Peak, Verteilung, Fusions-Vergleich), ohne das Schema
umzubauen — alte ``results.jsonl``-Zeilen bleiben lesbar.

Modul-Landkarte
===============

.. code-block:: text

   Browser ──(Dash)── tool_pipeline/app/                     ← Oberfläche
     app.py        Dash-Server, Background-Manager, Layout-Mount
     layout.py     Topbar · Sidebar (Controls) · Main (Ergebnis)
     callbacks.py  "Vergleichen" → RunConfigs → Worker-Prozess → run()
     components/   controls · charts · kpis · code_panel · history
          │
          │   die EINZIGE Naht:  run(config) -> result
          ▼
   tool_pipeline/run.py            ← Orchestrator
     │
     ├── intermediate_representation/   parse.py · reshape.py · config.py · optimizer.py
     ├── codegen/                       emit.py · compile.py · templates/{contraction,elementwise,reduction}.py
     ├── measure/                       verify.py · bench.py · metrics.py · baselines.py · fusion.py · provenance.py
     ├── store/                         store.py  (results.jsonl + kernels/<slug>.py)
     └── hardware.py                    Peaks/Bandbreite (reine Daten)

   tool_pipeline/cli.py            ← headless: Einzellauf + Report-Sweep
   tool_pipeline/report_figures.py ← torch-frei: PNGs aus results.jsonl

Die Import-Regel (und warum sie existiert)
------------------------------------------

Es gibt eine Festlegung, die sich durch alle Module zieht:

.. admonition:: Naht-Regel

   ``app/`` importiert aus dem Kern **ausschließlich** ``run``, ``schema`` und
   ``store`` (Letzteres nur *lesend*, für die History) — nie
   ``intermediate_representation``, ``codegen`` oder ``measure``. Und ``run``
   selbst wird **lazy im Worker-Prozess** geholt, nicht auf Modulebene.

Der zweite Teil klingt nach Pedanterie, verhindert aber einen echten, schwer zu
findenden Fehler. Die Kette:

1. Ein cuTile-Lauf braucht mehrere hundert Millisekunden bis Sekunden (JIT). In
   einer Weboberfläche darf das den Server nicht blockieren ⇒ **Hintergrund-Job**.
2. Dash führt Hintergrund-Jobs über einen ``DiskcacheManager`` in eigenen
   **Worker-Prozessen** aus, die per ``fork`` aus dem Hauptprozess entstehen.
3. Ein **CUDA-Kontext übersteht ``fork`` nicht.** Hätte der Hauptprozess durch
   irgendeinen Import bereits ``torch.cuda`` initialisiert, wären die Worker
   defekt — mit Fehlern, die nach Treiberproblem aussehen, nicht nach
   Importreihenfolge.
4. Also darf der Hauptprozess ``torch``/``cuda.tile`` **nie** importieren. Da ein
   Import von ``run.py`` beides mitzieht, passiert er erst im Callback-Körper —
   also im Worker.

Diese Regel hat einen angenehmen Nebeneffekt, der sich durch den ganzen Bericht
zieht: Weil die Oberfläche torch-frei ist, sind ihre Chart- und
Render-Funktionen **reine Funktionen** ``RunResult → Figur/Komponente`` und ohne
GPU testbar. 

Umgekehrt gilt die Regel auch: Kein Kern-Modul hängt auf Modulebene an der
Oberfläche. ``cli.py`` benutzt die Config-Bau-Helfer aus
``app/components/controls.py`` (damit GUI und CLI *dieselben* Kreuzprodukt-Slugs
erzeugen — ein Cache, eine Datenquelle), importiert sie aber lazy:
``import tool_pipeline.cli`` läuft ohne Dash **und** ohne torch durch.

Der Weg eines Laufs
===================

Ein Lauf durchquert acht Stufen. Interessant ist weniger die Reihenfolge als das,
was an jeder Kante **liegt** — jede Pfeilbeschriftung ist ein echtes Datenobjekt:

.. code-block:: text

   RunConfig
     │  einsum-String + Größen + Format + Tile
     ▼
   [1] parse ─────────────► ContractionIR | ElementwiseIR | ReductionIR | NAryContractionIR
     │                       (Achsen klassifiziert: M/N/K/Batch)
     ▼
   [2] reshape (B1) ──────► Canonical  (M, N, K, B + permute/reshape-Spezifikation)
     │                       nur Kontraktion; memory-bound überspringt diese Stufe
     ▼
   [3] emit (Codegen C1) ─► Quelltext (str)  — ein vollständiges cuTile-Modul
     │
     ▼
   [4] compile + Cache ───► results/kernels/<slug>.py  →  launch(*operanden)
     │                       (echte Datei, weil cuTile den Quelltext liest)
     ▼
   [5] Kalt-Lauf ─────────► compile_ms  (+ gefüllter Output-Tensor)
     │
     ▼
   [6] verify (fp32) ─────► accuracy {max_abs_err, rel_err, passed, atol, rtol}
     │                       passed == False  ⇒  ENDE als verify_failed
     ▼
   [7] benchmark ─────────► timing {run_ms, min_ms, p90_ms, sigma_ms}
     │                       + metrics {tflops, gbps, AI, %-Peak}
     │                       + optional baselines / fusion
     ▼
   [8] store ─────────────► eine JSON-Zeile in results/results.jsonl
     │
     ▼
   RunResult

Drei Eigenschaften dieses Ablaufs sind bewusst so und werden in Teil 3 einzeln
begründet:

* **``verify`` steht vor ``benchmark``**, nicht danach. Ein Kernel, der falsch
  rechnet, wird nie gemessen — es gibt also keine Zahl im System, die nicht durch
  das Gate gegangen ist.
* **Der Kalt-Lauf ist selbst eine Messung.** Er kostet ohnehin den JIT, also wird
  seine Wall-Clock-Zeit als ``compile_ms`` festgehalten und sein Ergebnis
  gleich für ``verify`` benutzt. Der teure Schritt passiert genau einmal.
* **Alles ab Stufe 7 ist optional und „graceful".** Baselines, Fusions-Vergleich
  und GPU-Zustand können fehlschlagen, ohne einen bereits verifizierten und
  gemessenen Lauf zu entwerten — sie tragen dann ``available: false`` plus Grund.
