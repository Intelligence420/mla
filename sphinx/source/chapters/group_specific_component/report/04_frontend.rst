.. _gsc_report_frontend:

##########################
Teil 4 — Das Frontend
##########################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Die Oberfläche ist die Schauseite des Projekts. Sie muss aus einem Webserver heraus einen **mehrsekündigen
GPU-Job** starten, dabei bedienbar bleiben, Fortschritt zeigen, abbrechbar sein
und sich mit anderen Nutzern derselben GPU teilen können. Das ist der eigentliche
Inhalt dieses Teils.

Warum Plotly Dash
=================

Vier Kandidaten standen zur Wahl. Das Auswahlkriterium war nicht „was ist am
schnellsten gebaut", sondern **„was trägt einen langlaufenden Job mit
Fortschritt und Abbruch"**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Kandidat
     - Für
     - Gegen
   * - **Plotly Dash** (gewählt)
     - Echte Hintergrund-Callbacks (``background=True``) mit
       ``set_progress``/``cancel``; native interaktive Charts; Callbacks sind
       normale Funktionen und damit **headless testbar**; explizites
       Input/Output-Modell
     - mehr Boilerplate als Streamlit; Callback-Graph muss man verstehen
   * - Streamlit
     - schnellster Einstieg
     - Das Ausführungsmodell führt bei **jeder** Interaktion das ganze Skript neu
       aus. Für einen Zustand „Job läuft seit 8 s, 12 von 30 Iterationen" ist das
       das falsche Modell; Fortschritt/Abbruch wären Bastelei.
   * - Gradio
     - sehr einfache Widgets
     - auf ML-Demos zugeschnitten (Eingabe → Ausgabe), nicht auf ein Dashboard
       mit vielen gekoppelten Reglern und mehreren Diagrammen
   * - Jupyter-Notebook
     - null Infrastruktur
     - kein Deliverable, das jemand *bedient*; Zustand und Reihenfolge sind beim
       Vorführen fragil

Der Zusatznutzen von Dash im Nachhinein: Weil die Callback-Logik in gewöhnlichen
Funktionen steckt, ist der zentrale Ablauf („Klick → Configs → Läufe → Charts")
in ``tests/test_app_execute.py`` **ohne Browser** prüfbar — mit echtem GPU-Lauf.
Die Chart-Funktionen sind sogar ganz ohne GPU testbar.

Der Hintergrund-Job — und die Fork-Falle
========================================

Ein Lauf besteht aus JIT, Verifikation und 10 + 30 Iterationen.
Das darf den Dash-Server nicht blockieren, also läuft es als Hintergrund-Callback
über einen ``DiskcacheManager``:

.. code-block:: text

   Browser  ──Klick "Vergleichen"──►  Dash-Hauptprozess
                                        │  legt Job in den Diskcache
                                        ▼
                                     Worker-Prozess (fork)
                                        │  import tool_pipeline.run   ← ERST HIER
                                        │  FileLock(.cache/gpu.lock)  ← ein Lock für den ganzen Batch
                                        │  run(cfg₁) … run(cfgₙ)      ← je Format/Tile/Swizzle
                                        │  set_progress(…) je Schritt
                                        ▼
                                     RunResults ──► Charts · KPIs · Verify · Code

Die entscheidende Zeile ist der Kommentar „ERST HIER". Der ``DiskcacheManager``
erzeugt Worker durch ``fork`` des Hauptprozesses, und **ein CUDA-Kontext übersteht
kein ``fork``**. Würde irgendein Import im Hauptprozess ``torch.cuda``
initialisieren, wären alle Worker defekt — mit Symptomen, die nach Treiber- oder
Hardwareproblem aussehen. Deshalb gilt die in :ref:`Teil 2
<gsc_report_architektur>` beschriebene Import-Regel, und deshalb holt der
Callback-Körper ``run`` **lazy**. Diese Regel ist im Code an mehreren Stellen
kommentiert, weil sie beim Lesen wie überflüssige Vorsicht aussieht und beim
Verletzen teuer ist.

Der GPU-Lock
------------

Der Lab-Rechner ist geteilt, und zwei gleichzeitige cuTile-Läufe sind keine gute
Idee — sie verfälschen sich gegenseitig (Takt, Cache, Bandbreite) und können bei
unglücklicher Größe den Speicher sprengen. Deshalb serialisiert ein
``filelock.FileLock`` (``fcntl``-basiert) über ``.cache/gpu.lock`` **alle**
GPU-Läufe prozessübergreifend — die GUI und der CLI-Sweep benutzen denselben Lock.

Zwei Details, die das robust machen:

* **Ein Lock für den ganzen Batch.** Ein „Vergleichen"-Klick über vier Formate ist
  *eine* GPU-Session. Würde jeder Lauf den Lock einzeln nehmen, könnte sich ein
  fremder Prozess dazwischenschieben und die Vergleichbarkeit der vier Zahlen
  zerstören.
* **Bei Prozess-Tod gibt das Betriebssystem den ``flock`` automatisch frei.**
  Klickt man „Abbrechen", beendet Dash den Worker hart — und der Lock ist
  trotzdem weg. Ein verwaister Lock, der die Maschine für alle blockiert, ist damit
  strukturell ausgeschlossen (das wäre bei einer selbstgebauten Lock-Datei die
  wahrscheinlichste Fehlerquelle gewesen).
* Wird der Lock nicht innerhalb von 60 s frei, meldet das Werkzeug „GPU belegt"
  statt endlos zu warten.

Fortschrittsbalken
------------------

Ein unbestimmter Spinner sagt nichts. Die Anzeige ist deshalb **determinat auf zwei
Ebenen**: Der Balken füllt sich pro fertiger Konfiguration („Format 2/4"), und
innerhalb einer Messung meldet die Bench-Schleife über einen Callback ihre
Iterationen („18/30"). Nach dem Lauf bleibt der Balken **voll stehen**, bis der
nächste ihn zurücksetzt — er verschwindet nicht, sodass man am Ende noch sieht,
dass alles durchlief.

Aufbau der Oberfläche
=====================

.. figure:: /_static/gsc/gui_overview.png
   :align: center
   :width: 100%
   :alt: Gesamtansicht der Oberfläche mit Controls-Sidebar und Ergebnis-Bereich

   Die Oberfläche nach einem Vergleich über **12 Konfigurationen**
   (drei Formate × vier Swizzle-Varianten, in *einem* Klick): links die Regler,
   rechts die Erfolgsmeldung „12/12 Formate verifiziert" und darunter die
   Diagramme. Man sieht hier direkt, wofür das Kreuzprodukt gebaut wurde — fp8
   liegt mit 158–162 TFLOP/s deutlich vor fp16 (39–74) und tf32 (9–13), und
   innerhalb jedes Formats trennen sich die Swizzle-Varianten sichtbar.

Die Controls
------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Regler
     - Verhalten
   * - **Familie**
     - Kontraktion · Elementwise · Reduktion. Die Auswahl schaltet die übrigen
       Regler um: memory-bound-Familien zeigen eine **Op**-Auswahl
       (add/mul/copy/relu bzw. sum) und verbergen Swizzle und Baselines — beides
       sind Kontraktions-Konzepte.
   * - **Preset**
     - je Familie eine kuratierte Liste, z. B. ``ik,kj->ij``, ``bik,bkj->bij``
       (batched), ``ki,kj->ij`` (A transponiert), ``ijk,kl->ijl`` (mehrdim. M),
       ``acspx,bspy->abcyx`` (echte Tensor-Kontraktion) und ``ij,jk,kl->il``
       (n-äre Kette).
   * - **Ausdruck**
     - Freitext. Wird strukturell validiert, bevor irgendetwas passiert — ein
       unparsebarer Ausdruck erzeugt eine Meldung, keinen Lauf.
   * - **Größen je Index**
     - **dynamisch**: Die Eingabefelder entstehen aus den Indexbuchstaben des
       Ausdrucks. Tippt man ``bik,bkj->bij``, erscheinen vier Felder (b, i, k, j).
       Zusätzlich prüft eine Abschätzung die Speichergröße gegen eine
       8-GiB-Obergrenze — auf der geteilten Maschine ist ein OOM ein Problem für
       alle.
   * - **Zahlenformate**
     - Mehrfachauswahl von (Compute → Akku)-Kombinationen. Die Liste wird aus
       ``schema.ALLOWED_ACC`` **abgeleitet**: unzulässige Kombinationen existieren
       in der Oberfläche gar nicht. Das ist bewusst so — ein ungültiger Zustand,
       den man nicht auswählen kann, muss auch nicht abgefangen werden.
   * - **Kachelung / Swizzle / GROUP_M**
     - Tile-Größen, dazu **Mehrfach-Vergleich**: mehrere Tiles und mehrere
       Swizzle-Konfigurationen ergeben das **Kreuzprodukt**
       Format × Tile × Swizzle in einem Klick.
   * - **Epilog**
     - keiner · ``bias`` · ``relu`` (nur Kontraktion). „keiner" erzeugt
       garantiert den byte-identischen Kernel wie ohne dieses Feature.
   * - **Baselines**
     - cuBLAS-Obergrenze, naive-cuTile-Untergrenze — optional.
   * - **Messung**
     - ``warmup`` / ``iters``, damit man zwischen „schnell schauen" und „sauber
       messen" wählen kann.

Der Ergebnis-Bereich
--------------------

Er wird nach jedem Lauf komplett neu gerendert und zeigt **immer** in derselben
Reihenfolge: Status → Verify-Chips je Konfiguration → Detail-Auswahl → Kontext →
KPIs → Verify → Diagramme → Code.

.. figure:: /_static/gsc/gui_kpis.png
   :align: center
   :width: 100%
   :alt: Verify-Chips, KPI-Karten, Verify-Block und Code-Panel eines erfolgreichen Laufs

   Der Kopfbereich eines fusionierten Laufs (``acspx,bspy->abcyx`` mit
   ``relu``-Epilog). Ganz oben ein **PASS-Chip je Konfiguration** — die
   Verifikation ist nicht ein Detail im Text, sondern das Erste, was man sieht.

Der Aufbau im Einzelnen:

* **Verify-Chips** für jede Konfiguration des Batches. Ein einziges rotes
  ``FAIL`` fällt hier sofort auf
* **„Detail je Format"** — eine Auswahlleiste, die den Rest des Bereichs
  (KPIs, Verify, Code) auf eine Konfiguration des Batches umschaltet. Die
  Diagramme bleiben dabei vergleichend, die Kennzahlen werden spezifisch.
* **Die Kontextzeile** trägt alles, was den Lauf reproduzierbar macht:
  ``M=262144 · N=4096 · K=4096 · fp16 → fp32 · NVIDIA GB10 · GPU-Zustand:
  2502 MHz · 63 °C · 44,1 W · 96 % Last · Zeitstempel``. Das ist exakt der
  ``provenance``-Block aus dem ``RunResult``.
* **Die KPI-Karten** zeigen nicht nur die Zahl, sondern ihre Einordnung:
  Durchsatz (mit %-vom-Peak), Laufzeit-Median (mit ``min``/``p90``/σ und der
  Iterationszahl — die Verteilung, nicht nur der Median), Compile-Zeit des
  Kalt-Laufs, Bandbreite (mit %-vom-Peak) und arithmetische Intensität. Bei
  gesetztem Epilog kommen zwei Karten dazu: **Fusion vs. sequentiell** (mit
  beiden gemessenen Zeiten) und **gesparter DRAM-Umweg** (mit der AI-Verschiebung,
  im Bild 584 → 1358 FLOP/Byte).
* **Der Verify-Block** nennt ``max_abs_err`` samt der geltenden Toleranz —
  bestanden oder nicht, immer mit der Zahl daneben.
* **Der generierte Kernel-Quelltext** mit Syntaxhervorhebung, dem Pfad des
  Artefakts als Überschrift und einem Kopier-Knopf. Das ist kein Deko-Element:
  Es ist der Beweis, dass die Zahlen darüber von *diesem* Code kommen — derselbe
  Text, der als ``kernels/<slug>.py`` auf der Platte liegt und compiliert wurde.

Die drei Diagramme sind reine Funktionen ``RunResult-Liste → Figur``:

.. figure:: /_static/gsc/gui_durchsatz.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz je Konfiguration in der Oberfläche

   **Durchsatz je Konfiguration.** Die Untertitel-Zeile erklärt die Kodierung:
   *Farbe = Format · Zeile = Format·Tile·Swizzle-Variante · Rahmen =
   Primärformat*. Genau so ist das Farbsystem gedacht — Farbe trägt die
   Format-Identität, die Hervorhebung des primären Laufs läuft über die Form
   (Rahmen), nicht über eine Sonderfarbe.

* **Genauigkeit ↔ Durchsatz** (Streudiagramm, log-Y) — der Trade-off. 
* **Roofline** (log-log) — Bandbreiten-Schräge und Rechen-Decken kommen aus
  ``hardware.py``, die Punkte aus den Läufen.

.. figure:: /_static/gsc/gui_roofline.png
   :align: center
   :width: 100%
   :alt: Interaktives Roofline-Diagramm in der Oberfläche

   Die Roofline in der Oberfläche: dieselben Kennwerte wie in der Report-Figur,
   aber interaktiv und über *alle* Läufe des Batches — inklusive der
   eingezeichneten Ridge-Linie bei ≈ 780 FLOP/Byte, an der man direkt abliest,
   dass fast alle Punkte links davon liegen. Der Untertitel macht die Regel
   explizit: „Punkte nur aus verifizierten Läufen".

Das Farbsystem
--------------

**Eine Farbe je Zahlenformat, stabil über alle Diagramme.** fp16 ist überall blau, bf16 überall
aqua. Die Farben werden aus derselben Kombinationsliste abgeleitet wie die
Auswahl — sie können also nicht zyklisch neu vergeben werden, wenn eine andere
Menge Formate gewählt ist. Das **primäre** Format wird nicht über eine
Sonderfarbe hervorgehoben, sondern über die **Form** (Umrandung, größerer Marker),
damit die Farbkodierung eindeutig bleibt. Die Palette ist CVD-sicher
(farbfehlsichtigkeits-geprüft) — dieselbe, die auch die Report-Figuren benutzen.

Fehlerzustände
--------------

Weil ``run()`` nie wirft, hat die Oberfläche genau vier Fälle zu rendern:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Zustand
     - Darstellung
   * - ``ok``
     - KPIs, Verify-PASS, Diagramme, Code
   * - ``verify_failed``
     - roter Verify-Block mit ``max_abs_err`` und der verletzten Toleranz —
       **keine** Durchsatzzahl, kein Punkt im Diagramm
   * - ``compile_error``
     - Fehlertext (z. B. „Epilog-Fusion ist nur für 2-Operanden-Kontraktionen
       unterstützt") plus Kontext
   * - ``run_error``
     - Fehlertext des Launch/Bench, Kernel-Pfad bleibt sichtbar

Der Punkt dieses Abschnitts: Ein Explorationswerkzeug **muss** Fehler gut
darstellen, weil Ausprobieren dazugehört. Jeder Fehlerfall ist deshalb ein
regulärer Zustand mit einer Anzeige, nicht ein Sonderweg.

History: Läufe verwalten
------------------------

Direkt unter der Topbar liegt ein einklappbares Panel — in der Gesamtansicht oben
als Leiste „Vergangene Läufe — ansehen · vergleichen · umbenennen · löschen"
sichtbar. Es listet die Testläufe aus ``results.jsonl``. Man kann einen Lauf
**ansehen** (die Charts werden aus den gespeicherten Ergebnissen neu gerendert —
ohne GPU!), **vergleichen**, **umbenennen** und **löschen**. Weil das Panel nur
liest bzw. den Store atomar neu schreibt, braucht es weder GPU noch torch — es ist
ein reines Datenwerkzeug über der JSONL-Datei. Damit ist auch der Weg von der
Oberfläche in diesen Bericht geschlossen: Was hier als benannter Lauf steht, ist
dieselbe Charge, aus der ``report_figures.py`` die Figuren zieht.

Testbarkeit
===========

Die Naht-Regel zahlt sich beim Testen aus. Die Suite teilt sich in zwei Klassen:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Klasse
     - Inhalt
   * - **headless, ohne GPU**
     - Parsing, Kanonisierung (gegen ``torch.einsum`` nur auf CPU-Ebene der
       View-Logik), Metrik-Formeln, Store-Mutatoren, Sweep-Config-Erzeugung,
       Chart-Funktionen, Render-Funktionen für alle vier Zustände,
       Controls-Validierung
   * - **GPU-pflichtig**
     - Codegen-Korrektheit (die Kernel werden wirklich compiliert und gegen
       ``torch`` verifiziert), Orientierungs-Wächter, ragged-Randfälle über alle
       Familien, der zentrale ``execute_run``-Ablauf mit echtem Lauf

Ein Beispiel für einen Test, der ohne die Naht-Disziplin nicht möglich wäre: Die
Chart-Funktionen bekommen konstruierte ``RunResult``-dicts und werden auf ihre
Figur geprüft — kein Browser, keine GPU, keine Dash-App.
