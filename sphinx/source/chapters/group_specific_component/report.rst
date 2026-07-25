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
**L2-Swizzle** (Block→Kachel-Umordnung mit Gruppengröße ``GROUP_M``), die
**Operations-Familie** (Kontraktion, elementweise, Reduktion) und die Frage, ob eine
nachgelagerte elementweise Operation als eigener Kernel läuft oder in die Kontraktion
**fusioniert** wird. Jede Achse verschiebt Durchsatz **und** Genauigkeit, oft
gegenläufig — und der Zusammenhang zum Hardware-Peak (Rechen- vs. Bandbreiten-Limit)
bleibt ohne Werkzeug unsichtbar.

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
  **ein** aggregierter Roofline-Punkt erscheint. Optional mit **fusioniertem Epilog**
  (``bias``/``relu``) auf dem Akkumulator-Tile (siehe *Fusion* unten).
* **Elementwise** (memory-bound): ``add``/``mul``/``copy``/``relu``.
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
``GROUP_M``-Varianten, der n-ären Kette und der Fusions-Formen — unter **einem**
GPU-Lock, und schreibt
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
     - 28,0
     - 36,8
     - 76 %
     - 3,2·10⁻⁴
     - 109
   * - bf16 → fp32
     - 29,6
     - 39,6
     - 75 %
     - 2,1·10⁻⁴
     - 116
   * - tf32 → fp32
     - 8,0
     - 20,0
     - 40 %
     - 4,6·10⁻²
     - 47
   * - fp8 e4m3 → fp16
     - 46,6
     - —
     - —
     - 3,4·10⁻¹
     - 91

Der einfache f-String-Codegen erreicht bei fp16/bf16 rund **drei Viertel** von cuBLAS —
ohne Autotuning bemerkenswert nah. fp8 ist mit Abstand am schnellsten (46,6 TFLOP/s),
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
     - 5,2
   * - Tile 64/64/32
     - 26,6
   * - Tile 128/128/64
     - 28,0
   * - + Swizzle G8
     - 30,0
   * - + Swizzle G16
     - 29,8
   * - + Swizzle G32
     - 29,9

Die Kachelwahl ist der stärkste Hebel: ein ungünstiges Tile (256/128/64) bricht auf
**5,2 TFLOP/s** ein, während 128/128/64 mehr als das Fünffache erreicht. Der L2-Swizzle ist
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
     - 224
     - 82 %
     - 0,12
   * - Elementwise · add
     - bf16 → fp32
     - 223
     - 82 %
     - 0,12
   * - Elementwise · add
     - fp32 → fp32
     - 225
     - 83 %
     - 0,08
   * - Reduktion · sum
     - fp16 → fp32
     - 171
     - 63 %
     - 0,50
   * - Reduktion · sum
     - fp32 → fp32
     - 216
     - 79 %
     - 0,25

Die elementweise Addition erreicht rund **82 % der theoretischen Bandbreite**
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

Fusion: wann lohnt ein Epilog auf dem Akkumulator?
--------------------------------------------------

Der Zwischentensor der n-ären Kette führt direkt zur letzten Frage des Projekts.
Wer eine Kontraktion und eine anschließende elementweise Operation
(:math:`C = A \cdot B`, dann :math:`+D` oder :math:`\max(\cdot, 0)`) als **zwei**
Kernel fährt, schreibt das Zwischenergebnis nach DRAM und liest es sofort wieder —
auf einer memory-bound Maschine bezahlt man diesen Umweg voll. Die **Fusion** wendet
den Epilog stattdessen auf dem Akkumulator-Tile an, bevor ``ct.store`` es
wegschreibt; der Zwischentensor entsteht nie:

.. code-block:: python

   for kk in range(0, K, TK):        # Kontraktions-Loop, unverändert
       acc = ct.mma(a, b, acc)

   d = ct.load(D, index=(bb, i, j), shape=(1, TM, TN),
               padding_mode=ct.PaddingMode.ZERO)
   acc = acc + ct.astype(d, ct.float32)      # Epilog auf dem Akku-Tile …
   ct.store(C, index=(bb, i, j),             # … VOR dem Store
            tile=ct.reshape(ct.astype(acc, C.dtype), (1, TM, TN)))

Gespart wird damit genau der Roundtrip des Zwischentensors, :math:`2 \cdot 4 \cdot M
\cdot N` Bytes — er wird weder geschrieben noch gelesen. Das hebt die arithmetische
Intensität des Kernels, und **genau daran** hängt, ob sich die Fusion lohnt: die
Ersparnis ist absolut konstant, die Kontraktion selbst wird mit steigendem :math:`K`
immer teurer. Der Sweep variiert deshalb die Intensität und nicht die Arbeitsmenge —
die schmale und die quadratische Form haben mit 2,15 GFLOP dieselbe FLOP-Zahl:

.. list-table:: Fusion vs. sequentiell (fp16 → fp32, verifiziert gegen ``torch.einsum`` + Epilog)
   :header-rows: 1
   :widths: 20 12 10 12 14 14 12

   * - Form (M·N·K)
     - Epilog
     - AI
     - fused [ms]
     - sequentiell [ms]
     - Speedup
     - gespart
   * - 4096·4096·64
     - bias
     - 21
     - 0,496
     - 1,101
     - **2,22×**
     - 128 MiB
   * - 4096·4096·64
     - relu
     - 32
     - 0,348
     - 0,946
     - **2,72×**
     - 128 MiB
   * - 1024·1024·1024
     - bias
     - 205
     - 0,083
     - 0,103
     - 1,25×
     - 8 MiB
   * - 1024·1024·1024
     - relu
     - 256
     - 0,072
     - 0,095
     - 1,33×
     - 8 MiB
   * - 1024·1024·8192
     - bias
     - 431
     - 0,364
     - 0,385
     - 1,06×
     - 8 MiB
   * - 1024·1024·8192
     - relu
     - 455
     - 0,362
     - 0,373
     - 1,03×
     - 8 MiB

.. figure:: /_static/gsc/fusion.png
   :align: center
   :width: 100%
   :alt: Speedup der Fusion über der arithmetischen Intensität, je Epilog eine Kurve

   Fusions-Speedup über der arithmetischen Intensität. Links die schmale Form
   (4096·4096·64), in der Mitte :math:`1024^3`, rechts die tiefe Form
   (1024·1024·8192). Beide Kurven fallen monoton gegen die Referenzlinie 1,0.

Das Ergebnis ist ein klarer Trend statt eines Einzelbefunds: Bei der **schmalen,
memory-bound** Form ist der fusionierte Kernel mehr als **doppelt so schnell**
(2,22× bzw. 2,72×) — dort dominiert der gesparte 128-MiB-Roundtrip die Laufzeit, und
der fused Kernel erreicht mit 196–205 GB/s rund **drei Viertel der Peak-Bandbreite**.
Bei der **tiefen, compute-dominierten** Form schrumpft der Gewinn auf 1,03–1,06×: die
Kontraktion braucht dort 0,36 ms, der gesparte 8-MiB-Roundtrip nur etwa 0,03 ms — er
verschwindet im Rauschen der Rechenzeit.

Damit ordnet sich auch der Befund aus Assignment 04 ein, der dieses Teil-Ziel
motiviert hat: Dort war die Fusion mit **0,984×** minimal *langsamer* als der
sequentielle Pfad (Kontraktion 12,83 ms gegenüber einem Epilog von 0,067 ms). Diese
Form liegt noch weiter rechts als unsere tiefste — jenseits des Punktes, an dem der
gesparte Speicherverkehr überhaupt messbar ist, bleibt nur der zusätzliche Aufwand des
größeren Kernels übrig. Die ehrliche Aussage lautet deshalb nicht „Fusion ist
schneller", sondern: **Fusion zahlt sich in dem Maß aus, in dem die Operation
bandbreiten- und nicht rechenlimitiert ist** — und die GB10 ist mit ihrem
Ridge-Point von ≈ 780 FLOP/Byte eine Maschine, auf der dieser Bereich groß ist.

Zwei Eigenschaften der Umsetzung sind dabei bewusst konservativ. Erstens ist die
Fusion rein **additiv**: ohne gewählten Epilog erzeugt der Codegen byte-identischen
Quelltext und denselben Kernel-Slug wie vorher — durch einen Textvergleich im Test
festgeschrieben, damit die eingecheckten Kernel-Artefakte nicht driften. Zweitens
misst das Tool den sequentiellen Vergleichspfad **selbst** (zweiter Kernel-Paar-Lauf
innerhalb desselben ``run()``, gleiche Messschleife) und verifiziert auch dessen
Ergebnis gegen fp32 — der Speedup ist damit kein Vergleich gegen eine Schätzung,
sondern gegen eine gemessene, verifizierte Alternative. Schlägt diese Zweitmessung
fehl, verliert der Lauf nur den Vergleich, nicht sein eigenes Ergebnis.

verify-before-trust in Aktion
-----------------------------

Der Sweep umfasste 24 Konfigurationen; **23** bestanden, **eine** nicht — und genau
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
Metriken, Store-Mutatoren und Compile-Cache-Härtung, CLI-Sweep-Erzeugung, sowie für
die Fusion: Byte-Identität des unfusionierten Quelltextes, ragged-/dtype-Verify beider
Epiloge gegen ``torch.einsum`` + Epilog, ein Wächter dagegen, dass ein Epilog *still
unangewandt* bliebe, und ein Orientierungs-Wächter für die Bias-Kachel). Die
headless-Tests laufen ohne GPU; die GPU-Tests verifizieren die Kernel real gegen
``torch``. Alle im Report gezeigten Figuren sind aus ``results.jsonl`` reproduzierbar
und tragen ausschließlich ``ok``-Läufe.
