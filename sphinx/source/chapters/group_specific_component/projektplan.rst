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

Die Idee nutzt die GPU-/cuTile-Bausteine der Assignments 01–06 — einsum-Parsing,
Kontraktions-Kernel, Tiling und L2-Swizzling — und bündelt sie zu einem
eigenständigen Werkzeug. Das thematische Ziel: den Zusammenhang von
**Geschwindigkeit, Genauigkeit und Hardware-Peak** anschaulich machen.

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
       cuTile-Compile, native interaktive Charts, ausgereift. Die GUI ist
       **Hauptdeliverable**, kein Anhängsel.
   * - Operationen
     - Kontraktions-Familie (GEMM, Batched GEMM, allgemeine und **n-äre**
       Kontraktion) **plus** speichergebundene Operationen (Elementwise,
       Reduktion). einsum ist die verbindende Sprache; die speichergebundenen
       Operationen machen die Roofline erst aussagekräftig.
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
   * - Results-Store
     - **JSON Lines** (``results.jsonl``, ein Objekt je Lauf) plus generierte
       Kernel als ``kernels/<slug>.py`` mit **lesbarem, normalisiertem
       Config-Slug** — transparent, git-diff-bar, zugleich Compile-Cache und
       Report-Datenquelle.
   * - Testlauf-Verwaltung
     - Der Store ist **sicher mutabel** (atomarer Rewrite): vergangene Läufe
       lassen sich ansehen, vergleichen, umbenennen und löschen (GPU-frei).
   * - Zukunfts-Scope
     - Autotuning und Tile-Heatmap bewusst gestrichen; **Fusion**
       (Kontraktion + Elementwise-Epilog) als Zukunftskandidat vorgemerkt.

Zielhardware
============

Entwicklung und Messung laufen auf einer **NVIDIA GB10** (Grace-Blackwell,
Compute Capability ``sm_121``) mit 128 GB Unified Memory und rund 273 GB/s
Speicherbandbreite. Rechen-Peaks (dense): fp16/bf16 ≈ 213 TFLOP/s, fp8 ≈ 214,
tf32 ≈ 53. Eine zentrale frühe Erkenntnis: Der Ridge-Point liegt sehr hoch
(bf16 ≈ 780 FLOP/Byte), weit jenseits typischer GEMM-/einsum-Intensitäten —
die meisten Formen sind daher **stark speichergebunden**. Genau das macht die
Roofline-Darstellung sichtbar.

Aufbau & Vorgehen
=================

Das Werkzeug ist als Pipeline organisiert:

   Ausdruck parsen → auf kanonische Form bringen → cuTile-Kernel generieren →
   compilen → gegen fp32 verifizieren → messen → Ergebnisse speichern →
   visualisieren.

Die grafische Oberfläche ist über **genau eine Schnittstelle** an den Kern
gekoppelt (eine ``run(config) → result``-Funktion). Dadurch bleibt der Kern
unabhängig von der Oberfläche testbar, und die Oberfläche misst live durch
dieselbe Pipeline, die auch ohne GUI (headless über ``cli.py``) funktioniert.

Meilensteine
============

Die Umsetzung erfolgt in **Teil-Zielen** — jedes eine vertikale, gegen fp32
verifizierte Scheibe durch die ganze Pipeline. Erst **tief** (GEMM ausreizen),
dann **breit** (Operationen), dann **Politur**.

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
   * - 7.5
     - Erweiterungen aus dem Feedback
     - Einstellbare Swizzle-Gruppengröße (``GROUP_M``), Mehrfach-Vergleich von
       Tile- und Swizzle-Konfigurationen (Kreuzprodukt), **n-äre** Ketten-
       Kontraktion (als ein Roofline-Punkt) und die Testlauf-Verwaltung.
   * - 8
     - Politur, Robustheit & Bericht
     - Feinschliff des Erscheinungsbilds, systematisch belegte Randfälle,
       gehärteter Compile-Cache, saubere Fehlerzustände, reproduzierbare
       Batch-Sweeps und dieser Sphinx-Bericht.

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
   * - TZ 1–2
     - Backbone end-to-end (ein GEMM: generieren → verifizieren → messen →
       speichern) und die Live-GUI um diese eine Operation; der
       Oberfläche↔Kern-Vertrag ``run(config) → result`` ist bewiesen.
   * - TZ 3–5
     - Zahlenformat-Achse mit Genauigkeits-Story, volle Messschicht
       (Verteilung, GB/s, arithmetische Intensität, %-Peak, GPU-Zustand) plus
       cuBLAS-/naive-Vergleichslinien, und die Roofline.
   * - TZ 6–7
     - Allgemeine 2-Operand-Kontraktion über den echten kanonischen Reshape;
       die speichergebundenen Familien Elementwise und Reduktion — die Roofline
       zeigt nun beide Seiten (compute- vs. memory-bound).
   * - TZ 7.5
     - Einstellbares ``GROUP_M``, Mehrfach-Vergleich von Tiles/Swizzle
       (Kreuzprodukt Format × Tile × Swizzle), n-äre Ketten-Kontraktion und die
       mutable Testlauf-Verwaltung (History) — alles additiv und verifiziert.
   * - TZ 8
     - Politur, Robustheit & Bericht: atomarer/korruptionsfester Compile-Cache,
       systematisch belegte nicht-teilbare Dimensionen (Elementwise/Reduktion),
       reproduzierbare CLI-Batch-Sweeps mit family-geformter Ausgabe, die
       Report-Figuren aus dem Store, benutzerfreundliche Fehlerzustände und ein
       durchgängigeres Erscheinungsbild. Ergebnis: das fertige, dokumentierte
       Deliverable.

Ausblick
========

Als Zukunftskandidat bleibt die **Fusion** von Kontraktion und
Elementwise-Epilog (A04-Befund 0,98×): das Ergebnis-Tile eines GEMM direkt auf
dem Compute-Tile weiterverarbeiten, statt das Zwischenergebnis über den langsamen
Hauptspeicher zu schicken — die konsequente nächste Stufe der memory-bound-Story.
