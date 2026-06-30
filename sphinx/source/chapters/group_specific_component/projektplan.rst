.. _gsc_projektplan:

#########################
Projektplan & Fortschritt
#########################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Gewähltes Projekt
=================

Umgesetzt wird **Idee 2 — der interaktive einsum/GEMM-Performance-Explorer**
(GPU/cuTile). Aus einem einsum-/GEMM-Ausdruck wird *live* ein cuTile-Kernel
**generiert**, bei verstellbarem Zahlenformat, Tiling und Swizzling auf der GPU
**gemessen** und in interaktiven Graphen (Durchsatz, Genauigkeit, Roofline)
**visualisiert**.

Diese Idee nutze die GPU-/cuTile-Bausteine der Assignments 01–06 — einsum-Parsing,
Kontraktions-Kernel, Tiling und L2-Swizzling. Diese werden zu einem eigenständigen Werkzeug
gebündelt. Das Thematische Ziel ist dabei: den
Zusammenhang von **Geschwindigkeit, Genauigkeit und Hardware-Peak** einfacher/anschaulicher zu macen.

.. note::

   Die GUI ist die Schauseite — die eigentliche Substanz sind Kernel-Erzeugung,
   ehrliche Messung und die daraus gewonnenen Erkenntnisse.

Designentscheidungen (Überblick)
================================

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Aspekt
     - Entscheidung
   * - Messung
     - **Live** auf der GPU — Regler verstellen, sofort neu messen.
   * - GUI-Framework
     - **Plotly Dash** — robuste Hintergrund-Jobs für den mehrsekündigen
       cuTile-Compile, native interaktive Charts, ausgereift.
   * - Operationen
     - Kontraktions-Familie (GEMM, Batched GEMM, allgemeine Kontraktion)
       **plus** speichergebundene Operationen (Elementwise, Reduktion). einsum
       ist die verbindende Sprache; die speichergebundenen Operationen machen
       die Roofline erst aussagekräftig.
   * - Codegen
     - cuTile-Quelltext per **f-String-Template** erzeugen und ausführen. Jede
       Kontraktion wird vorab auf ein kanonisches Batched-GEMM gebracht, sodass
       nur **eine** bewährte Kernel-Struktur generiert werden muss.
   * - Korrektheit
     - **Verify-before-trust** — jeder generierte Kernel wird gegen eine
       fp32-Referenz geprüft, bevor seine Messwerte angezeigt werden.
   * - Zahlenformate
     - fp16, bf16, tf32, fp8 (e4m3/e5m2) — auf der Ziel-GPU empirisch
       verifiziert; fp32/fp64 als Genauigkeits-Anker.
   * - Messgrößen
     - eigene Messschicht auf Basis von CUDA-Events: Laufzeit, TFLOP/s,
       erreichte Bandbreite, arithmetische Intensität, Anteil am Hardware-Peak
       und Fehler gegenüber fp32.
   * - Vergleich
     - zwei optionale Referenzlinien — cuBLAS/torch als Obergrenze und ein
       naiver cuTile-Kernel als Untergrenze.

Zielhardware
============

Entwicklung und Messung laufen auf einer **NVIDIA GB10** (Grace-Blackwell,
Compute Capability ``sm_121``) mit 128 GB Unified Memory und rund 273 GB/s
Speicherbandbreite. Eine wichtige frühe Erkenntnis: Bei diesem Verhältnis von
Rechen- zu Speicherleistung sind die meisten GEMM-/einsum-Formen **stark
speichergebunden** — genau das soll die Roofline-Darstellung sichtbar machen.

Aufbau & Vorgehen
=================

Das Werkzeug ist als Pipeline organisiert:

   Ausdruck parsen → auf kanonische Form bringen → cuTile-Kernel generieren →
   compilen → gegen fp32 verifizieren → messen → Ergebnisse speichern →
   visualisieren.

Die grafische Oberfläche ist über **genau eine Schnittstelle** an den Kern
gekoppelt (eine ``run(config) → result``-Funktion). Dadurch bleibt der Kern
unabhängig von der Oberfläche testbar, und die Oberfläche misst live durch
dieselbe Pipeline, die auch ohne GUI funktioniert.

Meilensteine
============

.. list-table::
   :header-rows: 1
   :widths: 6 34 60

   * - #
     - Teil-Ziel
     - Ergebnis
   * - 1
     - Backbone (ohne Oberfläche)
     - Ein GEMM (fp16) läuft komplett durch: generieren → verifizieren →
       messen → speichern.
   * - 2
     - Oberfläche um diese eine Operation
     - Live-GUI: Eingabe → Messung auf der GPU → Kennzahlen + generierter Code.
   * - 3
     - Zahlenformat-Achse
     - Alle Formate wählbar; Genauigkeit gegen Durchsatz als Graphen.
   * - 4
     - Tiling/Swizzling + volle Messschicht + Vergleichslinien
     - Alle Stellschrauben und die ehrlichen Messgrößen.
   * - 5
     - Roofline
     - Einordnung relativ zum Hardware-Peak.
   * - 6
     - Allgemeine Kontraktionen
     - Beliebige 2-Operand-Ausdrücke über den kanonischen Reshape.
   * - 7
     - Speichergebundene Operationen
     - Elementwise & Reduktion — rechen- gegen speichergebunden im Vergleich.
   * - 8
     - Politur & Bericht
     - Feinschliff, Robustheit, Dokumentation.

Aufgabenverteilung
==================

Die Bearbeitung erfolgt **gemeinsam und flexibel** entlang der Teil-Ziele
(etwa eine Spur „Kern/Pipeline" und eine Spur „Oberfläche/Visualisierung"),
ohne starre Zuordnung einzelner Dateien.

Fortschritts-Log
================

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Stand
     - Eintrag
   * - Planung
     - Projektidee, Designentscheidungen und Architektur festgelegt;
       Zielhardware (GB10) vermessen und die unterstützten Zahlenformate
       empirisch verifiziert; Code-Gerüst und Umsetzungsreihenfolge
       (Teil-Ziele) stehen.
