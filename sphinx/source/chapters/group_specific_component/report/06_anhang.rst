.. _gsc_report_anhang:

######################################
Teil 6 — Anhang: Reproduzierbarkeit
######################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Alles in diesem Bericht ist mit vier Befehlen nachvollziehbar. Dieser Teil sagt,
welche das sind, was genau gemessen wird und wie die Daten aussehen.

Voraussetzungen
===============

Der volle Stack (torch mit CUDA, ``cuda.tile``, ``triton``) existiert nur auf dem
Lab-Rechner; lokal scheitert bereits der Import. Alles, was **keine** GPU braucht
(Parsing, Metrik-Formeln, Charts, dieser Sphinx-Bericht), läuft überall.

.. code-block:: bash

   source ../.venv/bin/activate       # venv des GPU-Hosts (Pfad je Maschine)
   pip install -r requirements.txt    # einmalig: Dash/Plotly/pandas/matplotlib

``torch``, ``cuda.tile``, ``triton`` und ``cupy`` stehen bewusst **nicht** in
``requirements.txt`` — sie kommen aus dem vorhandenen venv des Hosts und dürfen
nicht versehentlich überschrieben werden.

Die vier Befehle
================

.. code-block:: bash

   # 1) Interaktiv: die Oberfläche starten (Browser auf 127.0.0.1:8050)
   python -m tool_pipeline

   # 2) Headless: ein einzelner Lauf
   python -m tool_pipeline.cli --M 1024 --N 1024 --K 1024
   python -m tool_pipeline.cli --family elementwise --op add --expr ij,ij->ij --size 4096
   python -m tool_pipeline.cli --epilog bias --M 4096 --N 4096 --K 64

   # 3) Der Report-Sweep: alle 33 Konfigurationen unter EINEM GPU-Lock (~2-3 min)
   python -m tool_pipeline.cli --sweep
   python -m tool_pipeline.cli --show-configs    # nur auflisten, ohne GPU

   # 4) Die Figuren neu erzeugen (torch-frei, liest nur results.jsonl)
   python -m tool_pipeline.report_figures

Der Exit-Code ist skript-/CI-tauglich: 0 bei Erfolg (Einzellauf ``ok`` bzw. Sweep
**alle** ``ok``), 1 bei mindestens einem Fehlschlag, 2 wenn der GPU-Lock nicht
frei wurde.

Was der Sweep misst
===================

Der Sweep ist ein **kuratierter**, deterministischer Satz aus neun Teil-Sweeps —
jeder beantwortet genau eine Frage dieses Berichts:

.. list-table:: Die 33 Konfigurationen
   :header-rows: 1
   :widths: 6 22 12 60

   * - #
     - Teil-Sweep
     - Läufe
     - Frage
   * - 1
     - Format-Vergleich
     - 4
     - Wie schnell und wie genau ist jedes Zahlenformat? (1024³, Tile 128/128/64,
       mit cuBLAS-Baseline)
   * - 2
     - Tile-Vergleich
     - 2
     - Wie stark wirkt die Kachelung? (64/64/32 und 256/128/64; 128/128/64 steckt
       in ①)
   * - 3
     - Swizzle bei 1024³
     - 3
     - G8/G16/G32 — und warum sie hier *nicht* wirken
   * - 4
     - Elementwise ``add``
     - 3
     - Wie nah kommt eine memory-bound-Op an die Bandbreite? (4096²)
   * - 5
     - Reduktion ``sum``
     - 3
     - dieselbe Frage mit anderem Lese-/Schreib-Verhältnis
   * - 6
     - n-äre Kette
     - 1
     - Was kostet die paarweise Zerlegung? (``ij,jk,kl->il`` bei 256⁴)
   * - 7
     - Fusions-Trend
     - 8
     - Wann lohnt der Epilog? (2 Epiloge × 3 Formen + 2 unfusionierte
       Bezugspunkte)
   * - 8
     - ``GROUP_M`` bei 4096³
     - 6
     - Was bringt der Swizzle auf einem 32×32-Gitter? (ohne Swizzle + G2/4/8/16/32)
   * - 9
     - ``copy``-Bandbreite
     - 3
     - Wie schnell ist die Maschine wirklich? (0 FLOP, 4096²)

Zwei Eigenschaften machen den Sweep report-tauglich:

* **Keine doppelte Arbeit.** Ein Test stellt sicher, dass keine zwei
  Konfigurationen dieselbe (Slug, Größen, Baselines)-Kombination haben. Derselbe
  *Kernel* darf mehrfach vorkommen — der Fusions-Sweep compiliert ein Artefakt und
  misst es auf drei Formen.
* **Eine Charge, ein Lock, eine Identität.** Alle 33 Läufe teilen ``run_id`` und
  ``run_name`` und laufen unter einem einzigen GPU-Lock. Die Figuren-Erzeugung
  wählt automatisch die **jüngste** solche Charge und darin **nur** die
  ``ok``-Läufe.

Die Artefakte
=============

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Pfad
     - Inhalt
   * - ``project/results/results.jsonl``
     - eine JSON-Zeile je Lauf (siehe Schema unten)
   * - ``project/results/kernels/<slug>.py``
     - der generierte cuTile-Quelltext = Compile-Cache = Beleg
   * - ``sphinx/source/_static/gsc/*.png``
     - die sechs Report-Figuren, vorab erzeugt und eingecheckt (damit
       ``make html`` ohne GPU läuft)
   * - ``project/project-development/analysis/``
     - die Hardware-/dtype-Vorabanalyse (``RESULTS_gb10.md``,
       ``dtype_analyse.py``)

Schema einer ``results.jsonl``-Zeile
------------------------------------

.. code-block:: json

   {
     "status": "ok",
     "config":     { "family": "contraction", "op": null, "epilog": null,
                     "expr": "ik,kj->ij", "inputs": ["ik","kj"], "output": "ij",
                     "dim_sizes": {"i":1024,"k":1024,"j":1024},
                     "dtype": "fp16", "acc_dtype": "fp32",
                     "tile": {"TM":128,"TN":128,"TK":64},
                     "swizzle": false, "group_m": 8,
                     "baselines": ["cublas"],
                     "bench_warmup": 10, "bench_iters": 30 },
     "kernel_path": "results/kernels/ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py",
     "accuracy":   { "max_abs_err": 0.00032, "mean_abs_err": 3.19e-05,
                     "rel_err": 1.29e-06, "passed": true,
                     "atol": 0.2, "rtol": 0.02 },
     "timing":     { "compile_ms": 50.681, "run_ms": 0.07582, "min_ms": 0.0736,
                     "p90_ms": 0.08307, "sigma_ms": 0.0082, "bench_iters": 30 },
     "metrics":    { "tflops": 28.322, "gbps": 110.63,
                     "arithmetic_intensity": 256.0,
                     "percent_peak_bw": 40.5, "percent_peak_flops": 13.3,
                     "baselines": { "cublas": { "available": true,
                                                "run_ms": 0.0553,
                                                "tflops": 38.836 } } },
     "provenance": { "gpu": "NVIDIA GB10", "dtype": "fp16", "acc_dtype": "fp32",
                     "sizes": {"M":1024,"N":1024,"K":1024,"B":1},
                     "timestamp": "2026-07-26T17:57:03",
                     "gpu_state": {"sm_clock_mhz": 2418.0, "temp_c": 38.0,
                                   "power_w": 9.45, "util_pct": 0.0} },
     "error": null,
     "run_id": "61196a065fee…", "run_name": "CLI-Report-Sweep · 2026-07-26T17:57:03",
     "created_at": "2026-07-26T17:57:03"
   }

Das Feld ``kernel_source`` (der Quelltext für die Code-Anzeige) wird bewusst
**nicht** mitgeschrieben — er liegt bereits unter ``kernel_path``. Zeilen aus
älteren Läufen ohne ``run_id``/``epilog``/``group_m`` bleiben lesbar; fehlende
Felder werden beim Laden ergänzt.

Auswerten ohne das Werkzeug
---------------------------

.. code-block:: python

   import pandas as pd
   df = pd.read_json("project/results/results.jsonl", lines=True)
   ok = df[df.status == "ok"]
   ok["tflops"] = ok.metrics.apply(lambda m: m["tflops"])

Tests
=====

``python -m pytest tests/ -q`` — **286 Tests**, Laufzeit rund 6 Sekunden auf dem
Lab-Rechner:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Datei
     - Tests
     - Schwerpunkt
   * - ``test_codegen.py``
     - 55
     - **GPU:** jeder generierte Kernel wird compiliert und gegen ``torch``
       verifiziert — alle Familien, alle dtypes, Orientierungs-Wächter,
       ragged-Randfälle, Anti-Drift (Byte-Identität des unfusionierten Quelltexts)
   * - ``test_app_controls.py``
     - 54
     - Validierung von Ausdruck, Größen, Formaten, Tile, Swizzle, Epilog;
       Kreuzprodukt-Erzeugung
   * - ``test_app_charts.py``
     - 36
     - die Chart-Funktionen (ohne GPU, ohne Browser)
   * - ``test_parse.py``
     - 34
     - M/N/K/Batch-Klassifikation, Family-Router, n-är-Planung, Ablehnungsfälle
   * - ``test_measure.py``
     - 32
     - Metrik-Formeln, Verteilungs-Kennzahlen, Baselines, Fusions-Vergleich
   * - ``test_cli.py``
     - 17
     - Sweep-Erzeugung (Vollständigkeit, keine Doppelarbeit), Figuren-Auswahl
   * - ``test_app_execute.py``
     - 13
     - **GPU:** der zentrale „Klick → Configs → Läufe → Charts"-Ablauf, headless
   * - ``test_verify.py``
     - 13
     - Referenzen je Familie/Op, Toleranztabelle, Epilog-Referenz
   * - ``test_store.py``
     - 12
     - Slug-Bildung, atomares Schreiben, Mutatoren, Cache-Härtung
   * - ``test_app_render.py``
     - 11
     - alle vier Zustände sauber gerendert (fehlende Werte → „—")
   * - ``test_reshape.py``
     - 7
     - der B1-View numerisch gegen ``torch.einsum`` (transponiert, batched,
       mehrdimensional)
   * - ``test_app_infra.py``
     - 2
     - Fork-Sicherheit: der Hauptprozess importiert **kein** torch/cuda

.. important::

   Die Codegen- und Mess-Tests **setzen eine CUDA-GPU voraus** — sie compilieren
   die generierten Kernel wirklich und verifizieren sie real gegen ``torch``. Auf
   einem Host ohne GPU sind sie nicht aussagekräftig. Die übrigen laufen überall.

Den Bericht bauen
=================

.. code-block:: bash

   cd sphinx && make html      # Ausgabe: sphinx/build/html/index.html

Der Build ist **GPU- und torch-frei**: Er liest ausschließlich die eingecheckten
PNGs. Auf ``main`` wird er zusätzlich per GitHub Actions gebaut und auf GitHub
Pages veröffentlicht.

Quellen der Hardware-Kennwerte
==============================

* ``nvidia-smi`` lokal (GB10, ``sm_121``, CUDA 13.0, Treiber 580.159.03)
* DGX-Spark-Hardwaredokumentation (Speichergröße, Bandbreite)
* Ein veröffentlichter ``mmapeak``-Microbenchmark für die Rechen-Peaks — es gibt
  **kein** offizielles GB10-Whitepaper mit Theoriewerten
* cuTile-Python-Dokumentation (``cuda.tile.matmul``, ``mma_scaled``, Datenmodell)
* Eigene Vorabanalyse: ``project/project-development/analysis/RESULTS_gb10.md``
  und ``dtype_analyse.py`` — welche dtypes in diesem cuTile-Build wirklich
  compilen, laufen und rechnen
