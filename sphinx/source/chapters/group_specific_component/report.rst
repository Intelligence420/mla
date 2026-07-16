.. _gsc_report:

##############
Project Report
##############

Finale Abgabe am **27.07.2026**. Die Group-Specific Component ist nicht bloß
Dokumentation, sondern ein **eigenständiger Projektbericht** — detailliert, aber
prägnant und gut lesbar (ähnlich einem Blogpost).

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Introduction
============

Das **cuTile Performance Lab** ist ein interaktiver einsum-/GEMM-Explorer für die
NVIDIA **GB10** (Grace-Blackwell, ``sm_121``): Man gibt einen einsum-Ausdruck ein
(z. B. ``ik,kj->ij``), wählt Zahlenformat, Kachelung und Cache-Umordnung — das Tool
**erzeugt daraus automatisch einen cuTile-Kernel**, verifiziert ihn gegen eine
fp32-Referenz, misst ihn auf der GPU und stellt Durchsatz, Genauigkeit und die
Roofline live gegenüber. Es beantwortet damit die Leitfrage jeder GPU-Kontraktion:
*Wie schnell, wie genau, und wie nah am Hardware-Limit?*

Der Kern ist bewusst **kein Mockup**: Die grafische Oberfläche ist nur die
Visualisierungs-Schicht über einer vollständigen, gegen fp32 verifizierten
Pipeline. Dieser Bericht fasst die Architektur, die Codegen-/Mess-Substanz und die
tatsächlichen Ergebnisse zusammen — alle Zahlen und Figuren stammen aus einem
reproduzierbaren Batch-Lauf des Werkzeugs selbst (``python -m tool_pipeline.cli
--sweep``), und es fließt **kein** Ergebnis ein, das die fp32-Verifikation nicht
bestanden hat (*verify-before-trust*).

Problem Formulation
===================

Für jede Kontraktion existiert ein großer Konfigurationsraum: **Zahlenformat**
(fp16, bf16, tf32, fp8 …), **Kachelung** (Tile-Größen ``TM``/``TN``/``TK``),
**L2-Swizzle** (Block→Kachel-Umordnung mit Gruppengröße ``GROUP_M``) und die
**Operations-Familie** (Kontraktion, elementweise, Reduktion). Jede Achse
verschiebt Durchsatz **und** Genauigkeit, oft gegenläufig — und der Zusammenhang zum
Hardware-Peak (Rechen- vs. Bandbreiten-Limit) bleibt ohne Werkzeug unsichtbar.

Zwei Risiken prägen die Umsetzung. Erstens ist **generierter Kernel-Code eine Quelle
stiller Falschergebnisse**: eine vertauschte ``ct.mma``-Orientierung oder ein
falsch behandelter Rand liefert plausible, aber falsche Zahlen. Zweitens sind
**nicht-teilbare Dimensionen** (z. B. 130 Zeilen bei Kachelbreite 128) der Normalfall,
nicht die Ausnahme. Beide adressiert das Tool durch eine fp32-Verifikation auf
**jedem** Kernel und durch systematisch belegte Randfall-Behandlung.

Implemented Solution
====================

Die eine Naht
-------------

Das gesamte System hängt an **einer** Schnittstelle:

.. code-block:: text

   run(config: RunConfig) -> RunResult

``run`` wirft nie, sondern kategorisiert jeden Ausgang in einen von vier Zuständen
(``ok`` · ``verify_failed`` · ``compile_error`` · ``run_error``). Der Ablauf je Lauf:

.. code-block:: text

   parse  →  Familien-Router  →  (Kontraktion: Reshape/Kanonisierung)
          →  Codegen (emit)   →  compile + Cache
          →  Kalt-Lauf (= compile_ms)  →  verify(fp32)
          →  Benchmark (CUDA-Events)   →  Metriken  →  Baselines  →  Store

Die GUI (Plotly Dash) und der headless-CLI bauen nur ``RunConfig`` und lesen
``RunResult`` — der Hauptprozess bleibt CUDA-frei (fork-sicher), sämtliche Charts
sind reine, headless testbare Funktionen. Deshalb baut auch dieser Report
GPU-/torch-frei: ``cd sphinx && make html`` liest nur fertige PNGs.

Codegen (C1) und Randfälle
--------------------------

Der Codegen ist **C1**: f-String-Templates erzeugen pro Familie ein
self-contained cuTile-Modul (``@ct.kernel`` + ``launch``). Nicht-teilbare Dimensionen
werden in allen drei Familien identisch behandelt — ``ct.load(...,
padding_mode=ct.PaddingMode.ZERO)`` füllt den Rand mit Nullen (neutral für
Multiply-Accumulate und Summe), und ``ct.store`` schneidet den Überstand automatisch
ab. Dass dies exakt rechnet, ist per GPU-Test gegen ``torch`` belegt — für die
Kontraktion **und** (neu) systematisch für Elementwise und Reduktion, jeweils über
glatte und ragged Größen inklusive des Loop-Fallback-Pfads der Reduktion.

Drei Operations-Familien (inkl. n-är)
-------------------------------------

* **Kontraktion** (Tensor-Core, compute-nah): GEMM, batched GEMM, transponiert,
  mehrdimensionale Kontraktion — und die **n-äre Kette** ``ij,jk,kl->il``, die per
  paarweiser Zerlegung durch den bewiesenen 2-Operanden-GEMM-Pfad läuft und als
  **ein** aggregierter Roofline-Punkt erscheint.
* **Elementwise** (memory-bound): ``add``/``mul``/``copy``.
* **Reduktion** (memory-bound): Summe über beliebige Achsen.

Die memory-bound-Familien nutzen family-korrekte Metriken (GB/s als Primärmetrik)
und werden gegen die op-abhängige fp32-Referenz verifiziert.

Robustheit: verify-before-trust, Cache-Härtung
-----------------------------------------------

Jeder Kernel wird vor der Messung gegen fp32 geprüft; scheitert er, trägt das
``RunResult`` ``verify_failed`` und **liefert keine Durchsatz-Zahl** — kein
Ergebnis ohne bestandene Referenz. Der persistierte Kernel-Quelltext
(``results/kernels/<slug>.py``) dient zugleich als Compile-Cache; sein Schreiben ist
**atomar** (Temp-Datei + ``os.replace``), sodass nie eine halb geschriebene Datei
sichtbar wird, und beschädigte Artefakte werden erkannt und neu erzeugt statt einen
Lauf abzustürzen.

Reproduzierbare Messung: der CLI-Sweep
--------------------------------------

Für den Report fährt ``python -m tool_pipeline.cli --sweep`` einen kuratierten Satz
Konfigurationen über alle drei Familien — inklusive mehrerer Tiles, der
``GROUP_M``-Varianten und der n-ären Kette — unter **einem** GPU-Lock, und schreibt
die verifizierten Ergebnisse in ``results/results.jsonl``. Ein zweites, torch-freies
Skript (``python -m tool_pipeline.report_figures``) erzeugt daraus die folgenden
Figuren. Beide Schritte sind deterministisch und ohne GUI wiederholbar.

Results and Insights
====================

Alle Zahlen stammen aus einer einzigen verifizierten Sweep-Charge auf der GB10
(``ok``-Läufe). Die Kontraktions-Läufe sind ``ik,kj->ij`` bei :math:`1024^3`, die
memory-bound-Läufe bei :math:`4096^2`.

Die Roofline: GB10 ist memory-bound
-----------------------------------

.. figure:: /_static/gsc/roofline.png
   :align: center
   :width: 100%
   :alt: Roofline-Diagramm der GB10 mit memory-bound- und compute-nahen Punkten

   Roofline (GB10). Die Bandbreiten-Schräge (273 GB/s) dominiert bis zu einem
   Ridge-Point von ≈ 780 FLOP/Byte — jenseits typischer Kontraktions-Intensitäten.
   Memory-bound-Familien (Elementwise, Reduktion) liegen weit links (AI 0,1–0,5),
   die Kontraktion rechts (AI 128–512), die n-äre Kette als ein Punkt dazwischen.

Die zentrale Erkenntnis des Projekts liest man direkt aus der Roofline ab: Der
Ridge-Point der GB10 liegt sehr weit rechts (bf16 ≈ 780 FLOP/Byte), weit jenseits
der arithmetischen Intensität selbst großer GEMMs. **Die Bandbreiten-Schräge ist
also die operative Decke** — sowohl die memory-bound-Familien als auch die
Kontraktion bleiben unter ihr; die flachen Rechen-Peaks (fp16/bf16 ≈ 213 TFLOP/s)
werden nie erreicht. Der Kontrast memory- vs. compute-bound ist sichtbar (linke vs.
rechte Punktwolke), aber beide Seiten teilen dasselbe Bandbreiten-Limit.

Durchsatz und Genauigkeit je Format
-----------------------------------

.. figure:: /_static/gsc/durchsatz_formate.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz je Zahlenformat, cuTile gegen cuBLAS

   Kontraktion je Format: der generierte cuTile-Kernel gegen die cuBLAS-Obergrenze
   (``torch.matmul``). Für fp8 gibt es keinen direkten ``matmul``-Pfad (keine
   cuBLAS-Säule).

.. figure:: /_static/gsc/genauigkeit_durchsatz.png
   :align: center
   :width: 90%
   :alt: Streudiagramm Genauigkeit gegen Durchsatz je Format

   Genauigkeit ↔ Durchsatz. fp16/bf16 sind genau, fp8 ist am schnellsten, aber am
   ungenauesten; tf32 ist hier der schlechteste Kompromiss (langsam **und** ungenau).

.. list-table:: Kontraktion :math:`1024^3`, Tile 128/128/64 (verifiziert)
   :header-rows: 1
   :widths: 26 16 16 14 14 14

   * - Format
     - cuTile [TFLOP/s]
     - cuBLAS [TFLOP/s]
     - Anteil
     - max. abs. Fehler
     - GB/s
   * - fp16 → fp32
     - 27,9
     - 31,3
     - 89 %
     - 3,2·10⁻⁴
     - 109
   * - bf16 → fp32
     - 27,6
     - 37,4
     - 74 %
     - 2,1·10⁻⁴
     - 108
   * - tf32 → fp32
     - 8,4
     - 19,1
     - 44 %
     - 4,6·10⁻²
     - 49
   * - fp8 e4m3 → fp16
     - 43,7
     - —
     - —
     - 3,4·10⁻¹
     - 85

Der einfache f-String-Codegen erreicht bei fp16 rund **89 %** von cuBLAS — ohne
Autotuning bemerkenswert nah. fp8 ist mit Abstand am schnellsten (43,7 TFLOP/s),
zahlt das aber mit dem größten Fehler; fp16/bf16 sind praktisch exakt.

Tuning-Raum: Kachelung und Swizzle
----------------------------------

.. figure:: /_static/gsc/tile_swizzle.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz über Tile-Größen und Swizzle-Gruppengrößen

   fp16-Tuning-Raum (:math:`1024^3`): gleicher verifizierter Kernel, nur Kachelung
   bzw. Block-Umordnung variiert.

.. list-table:: fp16, :math:`1024^3` — Kachelung und L2-Swizzle
   :header-rows: 1
   :widths: 40 20

   * - Konfiguration
     - Durchsatz [TFLOP/s]
   * - Tile 256/128/64
     - 5,6
   * - Tile 64/64/32
     - 24,1
   * - Tile 128/128/64
     - 27,9
   * - + Swizzle G8
     - 28,0
   * - + Swizzle G16
     - 28,4
   * - + Swizzle G32
     - 28,0

Die Kachelwahl ist der stärkste Hebel: ein ungünstiges Tile (256/128/64) bricht auf
**5,6 TFLOP/s** ein, während 128/128/64 das Fünffache erreicht. Der L2-Swizzle ist
eine reine Block-Umordnung (numerisch identisch, per GPU-Test bewiesen) und verändert
den Durchsatz bei dieser Größe kaum — die Gruppengröße ``GROUP_M`` ist einstellbar
(8/16/32) und geht nur bei Abweichung vom Default in den Kernel-Slug ein.

Memory-bound: Bandbreite als Primärmetrik
-----------------------------------------

.. list-table:: Elementwise & Reduktion, :math:`4096^2` (verifiziert)
   :header-rows: 1
   :widths: 26 20 14 16 12

   * - Familie · Op
     - Format
     - GB/s
     - % Peak-BW
     - AI
   * - Elementwise · add
     - fp16 → fp32
     - 214
     - 78 %
     - 0,12
   * - Elementwise · add
     - bf16 → fp32
     - 213
     - 78 %
     - 0,12
   * - Elementwise · add
     - fp32 → fp32
     - 210
     - 77 %
     - 0,08
   * - Reduktion · sum
     - fp16 → fp32
     - 166
     - 61 %
     - 0,50
   * - Reduktion · sum
     - fp32 → fp32
     - 205
     - 75 %
     - 0,25

Die elementweise Addition erreicht rund **78 % der theoretischen Bandbreite**
(273 GB/s) — nahe am praktisch Erreichbaren. Die Reduktion ist bandbreiten-effizient
in fp32/fp16; ihr niedrigerer AI-Wert und Durchsatz spiegeln das Verhältnis von
gelesenen Eingaben zu geschriebenen Ergebnissen.

Die n-äre Kette als ein Punkt
-----------------------------

Die Kette ``ij,jk,kl->il`` (:math:`256^4`) wird in zwei paarweise GEMMs zerlegt
(Pfad ``ij,jk->ik`` dann ``kl,ik->il``), gegen ``torch.einsum`` (fp32) verifiziert
und als **ein** aggregierter Roofline-Punkt gemessen: 1,64 TFLOP/s bei einer
arithmetischen Intensität von 64 FLOP/Byte. Dass ihre Intensität **unter** der eines
einzelnen GEMMs liegt, ist erwartbar — die Zwischentensoren erzeugen zusätzlichen
Speicherverkehr.

verify-before-trust in Aktion
-----------------------------

Der Sweep umfasste 16 Konfigurationen; **15** bestanden, **eine** nicht — und genau
das ist der Wert des Prinzips. Die bf16-Reduktion über 4096 Elemente überschritt die
Toleranz (max. abs. Fehler 1,57), weil sich der bf16-Rundungsfehler über Millionen
Summanden aufaddiert; sie erscheint deshalb **nicht** in den Figuren. Dasselbe gilt
für tiefere n-äre fp16-Ketten: ab :math:`384^4` summiert sich der Fehler beider
GEMM-Schritte über die Toleranz, weshalb der Report die Kette bewusst bei
:math:`256^4` zeigt. Das Tool meldet solche Fälle laut, statt still eine falsche Zahl
zu liefern — dieselbe Verifikation, die auch die Kernel-Erzeugung absichert.

Test- und Reproduzierbarkeits-Stand
-----------------------------------

Die Pipeline ist durch eine breite Testsuite abgesichert (Codegen-Korrektheit inkl.
Orientierungs-Wächter und ragged-Randfällen über alle Familien, family-korrekte
Metriken, Store-Mutatoren und Compile-Cache-Härtung, CLI-Sweep-Erzeugung). Die
headless-Tests laufen ohne GPU; die GPU-Tests verifizieren die Kernel real gegen
``torch``. Alle im Report gezeigten Figuren sind aus ``results.jsonl`` reproduzierbar
und tragen ausschließlich ``ok``-Läufe.
