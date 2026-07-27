.. _gsc_report_bedienung:

##################################
Teil 6 — Starten und Benutzen
##################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Dieser Teil ist die Bedienungsanleitung: **welche Befehle es gibt, was sie tun und
in welcher Reihenfolge man sie benutzt**.

Es gibt zwei Wege, und sie führen durch **dieselbe** Pipeline: die Oberfläche
(interaktiv, zum Ausprobieren) und die Kommandozeile (headless, für
reproduzierbare Chargen). Beide rufen am Ende genau eine Funktion —
``tool_pipeline.run.run(config)``. Was in der GUI erscheint, ist deshalb dasselbe
Ergebnis, das die CLI in ``results.jsonl`` schreibt.

Schnellstart
============

Vier Zeilen, aus dem Wurzelverzeichnis des Checkouts, auf dem GPU-Host:

.. code-block:: bash

   source .venv/bin/activate          # venv mit torch (CUDA), cuda.tile, triton
   cd project
   pip install -r requirements.txt    # einmalig: Dash/Plotly/pandas/matplotlib
   python -m tool_pipeline            # GUI → http://127.0.0.1:8050

Danach im Browser einen Ausdruck wählen, ein oder mehrere Zahlenformate ankreuzen
und **"▶ Vergleichen"** drücken.

Voraussetzungen
===============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Was
     - Warum / woher
   * - ``torch`` mit CUDA, ``cuda.tile``, ``triton``
     - der eigentliche Stack. Er existiert **auf dem Lab-Rechner** (GB10). Diese Pakete stehen bewusst **nicht**
       in ``project/requirements.txt``, damit ein ``pip install`` das vorhandene
       venv des Hosts nicht überschreibt.
   * - ``project/requirements.txt``
     - ergänzt Dash, Plotly, pandas, matplotlib — die GUI- und Plot-Schicht.
   * - eine NVIDIA-GPU
     - für alles, was compiliert, verifiziert oder misst.
   * - *keine* GPU nötig für
     - Parsen und Validieren von Ausdrücken, die Metrik-Formeln, die
       Chart-Funktionen, die History-Ansicht gespeicherter Läufe,
       ``report_figures``, die headless-Tests und ``make html``.

.. important::

   Alle Aufrufe erfolgen **aus dem Ordner** ``project/``. ``python -m
   tool_pipeline`` von woanders aus findet das Paket nicht, und die relativen
   Pfade des Result-Stores (``results/results.jsonl``,
   ``results/kernels/``) hängen an diesem Arbeitsverzeichnis.

Die Oberfläche starten
======================

.. code-block:: bash

   cd project
   python -m tool_pipeline

Der Dash-Server läuft **blockierend** im Vordergrund und ist danach unter
http://127.0.0.1:8050 erreichbar. Beendet wird er mit ``Ctrl-C``. Zwei
Umgebungsvariablen ändern die Bindung, falls Port oder Host belegt sind:

.. code-block:: bash

   TP_PORT=8060 python -m tool_pipeline     # anderer Port
   TP_HOST=0.0.0.0 python -m tool_pipeline  # im Netz erreichbar (bewusst NICHT Default)

Der Default ist ``127.0.0.1``, also **nur lokal** — auf einer geteilten Maschine
soll die Oberfläche nicht versehentlich im LAN hängen. Ein unbrauchbarer
``TP_PORT`` (leer, keine Zahl, außerhalb 1–65535) führt nicht zum Absturz, sondern
zu einer Warnung und dem Default 8050. Der Dash-Reloader ist **abgeschaltet**:
Er würde den Prozess doppelt starten, und zusammen mit dem Fork-basierten
Hintergrund-Manager wäre das auf der GPU ein Problem.

Einen Lauf durchführen
======================

Die Sidebar wird **von oben nach unten** durchgearbeitet — jeder Regler schaltet
gegebenenfalls die darunter liegenden um:

#. **Familie** wählen — *Kontraktion*, *Elementwise* oder *Reduktion*. Die Wahl
   verändert die Oberfläche: memory-bound-Familien zeigen eine **Op**-Auswahl
   (``add`` · ``mul`` · ``copy`` · ``relu``; die Reduktion rechnet immer ``sum``)
   und verbergen Swizzle und Baselines, weil das Kontraktions-Konzepte sind.
#. **Preset** oder **Ausdruck** — das Preset füllt eine kuratierte Liste je
   Familie (``ik,kj->ij``, ``bik,bkj->bij`` batched, ``ki,kj->ij`` mit
   transponiertem A, ``ijk,kl->ijl``, ``acspx,bspy->abcyx``, die n-äre Kette
   ``ij,jk,kl->il``). Das Feld **Ausdruck** nimmt jeden eigenen einsum-String;
   er wird strukturell geprüft, **bevor** irgendetwas auf die GPU geht.
#. **Größen je Index** — die Eingabefelder entstehen aus den Indexbuchstaben des
   Ausdrucks. Bei ``bik,bkj->bij`` erscheinen vier Felder (b, i, k, j).
#. **Zahlenformate** ankreuzen — Mehrfachauswahl von (Compute → Akku)-Paaren.
   Unzulässige Kombinationen existieren in der Auswahl gar nicht.
#. **Kachelung** (TM/TN/TK) und **L2-Swizzle** (``GROUP_M``) — hier ebenfalls
   mehrfach. Mehrere Formate × mehrere Tiles × mehrere Swizzle-Varianten ergeben
   das **Kreuzprodukt in einem Klick**; das ist der eigentliche Sinn des
   Werkzeugs.
#. Optional **Epilog** (``bias`` oder ``relu``, nur Kontraktion), **Baselines**
   (cuBLAS als Obergrenze) und die **Messparameter** ``warmup``/``iters`` —
   letztere entscheiden zwischen „schnell schauen" und „sauber messen".
#. **▶ Vergleichen** drücken.

Während der Lauf arbeitet, ist der Fortschritt **determinat auf zwei Ebenen**: Der
Balken zählt fertige Konfigurationen („Format 3/4"), die Bench-Schleife meldet ihre
Iterationen („18/30"). **Abbrechen** beendet den Hintergrund-Job hart — der
GPU-Lock wird dabei vom Betriebssystem automatisch freigegeben, es bleibt also
keine blockierte Maschine zurück.

Zwei Wächter greifen **vor** dem GPU-Lauf, nicht danach:

* Ein **unparsebarer Ausdruck** oder eine ungültige Größen-/Tile-/Format-Auswahl
  erzeugt eine Warnung und **keinen** Lauf.
* Eine **Speicher-Abschätzung** stoppt bei über **8 GiB** geschätzter Tensorgröße
  („Zu groß: ~X GiB geschätzt") — auf der geteilten 32-GiB-Maschine ist ein OOM
  ein Problem für alle Nutzer, nicht nur für den eigenen Lauf.

Das Ergebnis lesen
==================

Der Ergebnis-Bereich wird nach jedem Lauf komplett neu gerendert, immer in
derselben Reihenfolge:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Element
     - Was man daran abliest
   * - **Verify-Chips**
     - je Konfiguration ein ``PASS``/``FAIL``. Das ist bewusst das Erste im Bild:
       Ein einziges rotes ``FAIL`` fällt auf, ohne dass man in die Diagramme
       schauen muss. Zu einem ``FAIL`` gehört **keine** Durchsatzzahl und kein
       Punkt im Diagramm.
   * - **Detail je Format**
     - schaltet KPIs, Verify-Block und Code auf eine Konfiguration des Batches um.
       Die Diagramme bleiben vergleichend.
   * - **Kontextzeile**
     - Größen, dtype, GPU, Takt/Temperatur/Leistung/Last und Zeitstempel — der
       ``provenance``-Block, also alles, was den Lauf reproduzierbar macht.
   * - **KPI-Karten**
     - Durchsatz (mit %-vom-Peak), Laufzeit-Median mit ``min``/``p90``/σ und
       Iterationszahl, Compile-Zeit des Kalt-Laufs, Bandbreite und arithmetische
       Intensität; bei gesetztem Epilog zusätzlich *Fusion vs. sequentiell* und
       *gesparter DRAM-Umweg*.
   * - **Verify-Block**
     - ``max_abs_err`` **samt der geltenden Toleranz** — bestanden oder nicht,
       immer mit der Zahl daneben.
   * - **Diagramme**
     - Durchsatz je Konfiguration · Genauigkeit ↔ Durchsatz (log-Y) · Roofline
       (log-log). Eine Farbe je Zahlenformat, stabil über alle drei.
   * - **Kernel-Quelltext**
     - der generierte cuTile-Code mit Pfad als Überschrift und Kopier-Knopf.
       Derselbe Text liegt als ``results/kernels/<slug>.py`` auf der Platte und
       wurde genau so compiliert.

Schlägt etwas fehl, ist das ein regulärer Zustand mit eigener Anzeige, kein
Absturz: ``verify_failed`` (roter Verify-Block mit Fehler und Toleranz),
``compile_error`` und ``run_error`` (jeweils Fehlertext plus Kontext, Kernel-Pfad
bleibt sichtbar).

.. note::

   **Verify-before-trust** ist die Bedienregel, nicht nur ein Implementierungs-
   detail: Eine Durchsatzzahl ohne bestandene fp32-Verifikation ist keine
   Messung, sondern ein Artefakt. Nicht weiterverwenden.

Läufe wiederfinden: die History
-------------------------------

Direkt unter der Topbar liegt das einklappbare Panel **„Vergangene Läufe"**. Es
listet die Chargen aus ``results/results.jsonl``. Man kann einen Lauf **ansehen**
(die Charts werden aus den gespeicherten Ergebnissen neu gezeichnet — **ohne
GPU**), mehrere **vergleichen**, sie **umbenennen** und **löschen**. Das Panel
liest bzw. schreibt nur die JSONL-Datei und braucht deshalb weder GPU noch
``torch``.

Headless: die Kommandozeile
===========================

Dieselbe Pipeline ohne Browser — für Skripte, für einen schnellen Einzeltest und
für die Charge, aus der dieser Bericht seine Zahlen zieht.

.. code-block:: bash

   cd project

   # (1) Einzellauf mit den Defaults (ik,kj->ij bei 512³, fp16 → fp32)
   python -m tool_pipeline.cli

   # (2) Einzellauf, Größen explizit
   python -m tool_pipeline.cli --M 1024 --N 1024 --K 1024

   # (3) andere Familien
   python -m tool_pipeline.cli --family elementwise --op add --expr ij,ij->ij --size 4096
   python -m tool_pipeline.cli --family reduction --expr ij->i --size 4096

   # (4) mit Epilog-Fusion
   python -m tool_pipeline.cli --epilog bias --M 4096 --N 4096 --K 64

   # (5) den generierten Kernel gleich mit ausgeben
   python -m tool_pipeline.cli --M 1024 --N 1024 --K 1024 --show-kernel

   # (6) der komplette Report-Sweep: alle Konfigurationen unter EINEM GPU-Lock (~2–3 min)
   python -m tool_pipeline.cli --sweep

   # (7) nur auflisten, was der Sweep fahren würde — fasst die GPU nicht an
   python -m tool_pipeline.cli --show-configs

Alle Schalter im Überblick:

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Schalter
     - Default
     - Bedeutung
   * - ``--family``
     - ``contraction``
     - ``contraction`` · ``elementwise`` · ``reduction``
   * - ``--op``
     - –
     - Elementwise-Op (``add``/``mul``/``copy``/``relu``). Die Reduktion nutzt
       immer ``sum``, die Kontraktion keine Op.
   * - ``--expr``
     - ``ik,kj->ij``
     - der einsum-Ausdruck. Muss zur Familie passen (z. B. ``ij,ij->ij``
       elementwise, ``ij->i`` als Reduktion).
   * - ``--size``
     - ``512``
     - einheitliche Größe **je Index** — der bequeme Weg für quadratische Fälle.
   * - ``--M`` / ``--N`` / ``--K``
     - –
     - überschreiben einzeln die Indizes ``i`` / ``j`` / ``k``.
   * - ``--epilog``
     - –
     - ``bias`` (``acc+D``) oder ``relu`` (``max(acc,0)``) auf dem Akku-Tile vor
       dem Store; nur 2-Operanden-Kontraktionen. Ohne Angabe entsteht der
       **byte-identische** Kernel wie ohne dieses Feature.
   * - ``--sweep``
     - aus
     - fährt die gesamte Report-Charge statt eines Einzellaufs.
   * - ``--show-configs``
     - aus
     - listet die Sweep-Konfigurationen und beendet sich — **ohne GPU**.
   * - ``--show-kernel``
     - aus
     - hängt den generierten Quelltext an die Ausgabe (nur Einzellauf).

Der **Exit-Code** ist skript- und CI-tauglich:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - Bedeutung
   * - ``0``
     - Erfolg — Einzellauf ``ok`` bzw. im Sweep **alle** Läufe ``ok``
   * - ``1``
     - mindestens ein Fehlschlag (Verify, Compile oder Lauf)
   * - ``2``
     - der GPU-Lock wurde nicht frei — es lief etwas anderes auf der Karte

Jeder erfolgreiche Lauf hängt eine Zeile an ``project/results/results.jsonl`` und
legt seinen Quelltext unter ``project/results/kernels/<slug>.py`` ab. Der Sweep
vergibt für die ganze Charge **eine** ``run_id`` und **einen** ``run_name`` — das
ist die Klammer, an der die Figuren-Erzeugung später die jüngste vollständige
Charge erkennt.

Figuren, Tests und Bericht
==========================

.. code-block:: bash

   cd project
   python -m tool_pipeline.report_figures    # Figuren aus results.jsonl (torch-frei, ohne GPU)
   python -m pytest tests/ -q                # Testsuite (286 Tests, ~6 s)

   cd ../sphinx && make html                 # Bericht → sphinx/build/html/index.html

``report_figures`` liest **nur** ``results.jsonl``, wählt daraus automatisch die
jüngste ``CLI-Report-Sweep``-Charge und darin **nur** die ``ok``-Läufe, und
schreibt die PNGs nach ``sphinx/source/_static/gsc/``. Es braucht weder GPU noch
``torch`` — deshalb baut ``make html`` auch auf einem Rechner ohne GPU
durch.

Die Reihenfolge für neue Messwerte im Bericht ist damit festgelegt:
``cli.sweep_configs`` erweitern → ``--sweep`` fahren → ``report_figures`` →
Tabellen im Bericht angleichen → ``make html``. Von Hand eingetippte Zahlen wären
nicht reproduzierbar.

Von der Testsuite setzt ein Teil eine **CUDA-GPU voraus** (die Codegen- und
Mess-Tests compilieren die generierten Kernel wirklich). Die übrigen laufen
überall. Welche Datei was abdeckt, steht in
:ref:`Teil 7 <gsc_report_anhang>`.

Wenn etwas nicht klappt
=======================

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Ursache und Abhilfe
   * - ``ModuleNotFoundError: cuda.tile`` (oder ``torch``)
     - falsches bzw. kein venv aktiv, oder der Rechner ist nicht der GPU-Host.
       Das ist erwartetes Verhalten — headless-Teile (``--show-configs``,
       ``report_figures``, ``make html``) laufen trotzdem.
   * - ``GPU belegt``
     - ein anderer Lauf hält ``project/.cache/gpu.lock`` seit über **60 s**. Das
       Werkzeug wartet absichtlich nicht endlos: kurz warten und erneut drücken.
       Stirbt ein Prozess, gibt das Betriebssystem den Lock von selbst frei.
   * - Exit-Code ``2`` auf der CLI
     - dasselbe headless — der Lock wurde nicht frei.
   * - ``Zu groß: ~X GiB geschätzt``
     - der 8-GiB-Wächter. Größen verkleinern; die Maschine ist geteilt.
   * - Warnung statt Lauf nach dem Klick
     - Ausdruck, Größen, Tile oder Auswahl sind ungültig. Die Meldung nennt das
       Feld; es wurde **nichts** auf der GPU gestartet.
   * - Port 8050 belegt / Seite lädt nicht
     - ``TP_PORT=8060 python -m tool_pipeline``, und die im Terminal
       ausgegebene URL benutzen.
   * - ``FAIL`` im Verify-Chip
     - der generierte Kernel rechnet außerhalb der Toleranz. Die Zahlen dieses
       Laufs sind **nicht** verwendbar — genau dafür ist das Gate da (es hat
       einen echten Codegen-Bug gefangen, siehe :ref:`Teil 1
       <gsc_report_grundlagen>`).
   * - Befehl findet das Paket nicht
     - nicht aus ``project/`` heraus gestartet.
