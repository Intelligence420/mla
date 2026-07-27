.. _gsc_report_anhang:

######################################
Teil 7 — Anhang: Reproduzierbarkeit
######################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Alles in diesem Bericht ist nachvollziehbar. **Welche Befehle** man dafür braucht
und wie man sie benutzt, steht vollständig in :ref:`Teil 6 — Starten und Benutzen
<gsc_report_bedienung>`. Dieser Teil beantwortet das, was danach kommt: *was*
genau gemessen wird, wie die Daten aussehen und was die Tests abdecken.

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

Die Suite umfasst **286 Tests** und läuft rund 6 Sekunden auf dem Lab-Rechner
(Aufruf: :ref:`Teil 6 <gsc_report_bedienung>`):

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

Der Build (``cd sphinx && make html``, siehe :ref:`Teil 6
<gsc_report_bedienung>`) ist **GPU- und torch-frei**: Er liest ausschließlich die
eingecheckten PNGs. Auf ``main`` wird er zusätzlich per GitHub Actions gebaut und auf GitHub
Pages veröffentlicht.

Quellen
=======

Vorlesungsfolien des Moduls
---------------------------

Grundlage des Projekts sind die Folien des Moduls **Machine Learning
Accelerators** (FSU Jena). Sie liegen als PDF unter
``slides/``; die Prüfungsanforderungen an die Group-Specific Component stehen in
``slides/pruefungsleistungen.pdf``. Für diesen Bericht tragend sind die
GPU-Kapitel:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Foliensatz
     - Was daraus in diesen Bericht eingeht
   * - ``01_tensors_and_einsum.pdf``
     - einsum-Notation, Index-Klassifikation nach M/N/K/Batch — die Sprache, in
       der das Werkzeug seine Eingabe annimmt (Teil 3, Stufe *parse*)
   * - ``02_gpu_architecture.pdf``
     - Speicherhierarchie, Occupancy und das **Roofline-Modell** — die Brille,
       durch die dieser Bericht jede Messung liest (Teil 1)
   * - ``03_matmul_gpu.pdf``
     - getiltes GEMM mit Tensor-Cores, FP16-Eingang mit FP32-Akkumulator — das
       Muster, dem der generierte Kernel folgt (Teil 3, Stufe *codegen*)
   * - ``04_tensor_contract_gpu.pdf``
     - Tensor-Kontraktion als GEMM: Permutieren und Zusammenfassen der Indizes,
       also genau der ``B1``-View der *reshape*-Stufe
   * - ``05_contraction_interface_and_swizzling.pdf``
     - L2-Swizzling und ``GROUP_M`` — Herkunft der Swizzle-Achse, deren Wirkung
       Teil 5 mit 2,03× bei 4096³ belegt
   * - ``06_optimizations_and_multi_input_einsums.pdf``
     - Epilog-Fusion und mehrstufige/n-äre einsums — Grundlage der
       Fusions-Messreihe und der paarweisen Zerlegung

Hardware-Kennwerte
------------------

* ``nvidia-smi`` lokal (GB10, ``sm_121``, CUDA 13.0, Treiber 580.159.03)
* DGX-Spark-Hardwaredokumentation (Speichergröße, Bandbreite)
* Ein veröffentlichter ``mmapeak``-Microbenchmark für die Rechen-Peaks — es gibt
  **kein** offizielles GB10-Whitepaper mit Theoriewerten
* cuTile-Python-Dokumentation (``cuda.tile.matmul``, ``mma_scaled``, Datenmodell)
* Eigene Vorabanalyse: ``project/project-development/analysis/RESULTS_gb10.md``
  und ``dtype_analyse.py`` — welche dtypes in diesem cuTile-Build wirklich
  compilen, laufen und rechnen
