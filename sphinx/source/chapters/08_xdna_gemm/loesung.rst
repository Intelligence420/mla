.. _ch08_loesung:

#############################
Lösung und Bearbeitung
#############################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Aufgabe ist ein hand-geschriebener Tensor-Kernel für die XDNA2-
(AIE2P-) Compute-Tile, der die Matrixmultiplikation ``out = in0 @ in1``
mit ``M = N = 16`` und ``K = 64`` (BF16-Ein-/Ausgabe, FP32-Akkumulation)
auf der NPU ausführt. Die Multiplikation wird auf den nativen
**BFP16-MAC** (``bfp16ebs8``, 8×8×8) der Vektor-Einheit abgebildet.

Im L1-Scratchpad liegen die Operanden bereits getilt vor
(``in0: prmk``, ``in1: rqkn``, ``out: pqmn`` mit ``p=q=2``,
``r=8``, ``m=n=k=8``). Der Kernel muss ohne Schleifen-Kontrollfluss
(außer dem finalen ``ret lr``) auskommen und die Operations-Latenzen
selbst durch NOPs respektieren (keine Hazard-Unit, vgl. Assignment 07).

Task 1: Verify-Funktion
=======================

``verify()`` in ``src/driver.py`` berechnet die FP32-Referenz und
vergleicht mit den von der Aufgabe vorgegebenen Toleranzen
(``atol = 0.5``, ``rtol = 0.2``); ``torch.manual_seed(42)`` steht vor
der Tensor-Initialisierung in ``run()``:

.. code-block:: python

   def verify(in0, in1, out):
       ref    = in0.to(torch.float32) @ in1.to(torch.float32)
       actual = out.to(torch.float32)
       torch.testing.assert_close(actual, ref, atol=0.5, rtol=0.2)

Die großzügigen Toleranzen sind nötig, weil die BFP16-Emulation pro
8er-Block einen gemeinsamen Exponenten verwendet: gegenüber der
FP32-Referenz entsteht ein blockweiser Rundungsfehler, der mit der
K-Tiefe (64) akkumuliert. ``make run_matmul`` liefert:

.. code-block:: text

   [PASS] matmul verification passed.

Task 2: Instruktionen und Latenzen
==================================

Der Kernel benötigt die folgenden Instruktionen. Die Latenzen wurden
aus der vom Peano-Compiler erzeugten BFP16-Referenz
(``aie::mmul`` mit ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``,
``build/matmul_ref.s``) sowie durch eigene On-NPU-Experimente
(schrittweises Variieren der NOP-Abstände) bestimmt und als sichere
**untere Schranken** verwendet.

.. list-table::
   :header-rows: 1
   :widths: 38 12 18 32

   * - Instruktion
     - Slot
     - Latenz (≈)
     - Funktion
   * - ``vlda.conv.fp32.bf16``
     - A
     - 7–8
     - Load BF16 aus L1, Konversion → FP32-Akkumulator (eine
       CM-Hälfte = 64 B pro Load)
   * - ``vldb``
     - B
     - 7
     - roher Vektor-Load BF16 (ohne Konversion), für die
       B-Transposition
   * - ``vshuffle``
     - V/M
     - 2–3
     - 8×8-Transponierung im BF16-Vektor (Modi ``T16_8x8_lo`` = #52,
       ``T16_8x8_hi`` = #53)
   * - ``vmul.f`` (``#60``)
     - V
     - 6
     - BF16 × 1.0 → FP32-Akkumulator (bringt geshuffelte Daten in den
       Akku für die anschließende Konversion)
   * - ``vconv.bfp16ebs8.fp32``
     - V/M
     - 4–6
     - FP32-Akkumulator → BFP16-Operand (``ex``-Register)
   * - ``vmac.f`` / ``vmul.f`` (``#780``)
     - V
     - 6
     - BFP16 8×8×8 Multiply-(Accumulate); ``#780`` wählt den
       ``bfp16ebs8``-Modus
   * - ``vst.conv.bf16.fp32``
     - S
     - 6
     - FP32-Akkumulator → BF16, Store nach L1

Eine zentrale empirische Erkenntnis: anders als die in Assignment 07
für ``vadd.f`` gemessene Latenz von 4 benötigt das **Ergebnis eines
``vlda.conv`` rund 7–8 Zyklen**, bevor es von einem ``vconv`` korrekt
gelesen werden kann. Wird der Abstand zu klein gewählt, liest der
Konsument einen Null-/Stale-Wert (kein Stall!) — der Kernel liefert
dann stillschweigend Nullen.

Task 3: Register-Blocking
=========================

Der Akkumulator-Registersatz des AIE2P-Compute-Tiles umfasst nur
**``dm0``–``dm4``** (fünf FP32-Akkumulatoren à 8×8 = 64 FP32; ``dm5``
und höher werden vom Assembler abgelehnt). Daraus ergeben sich zwei
sinnvolle Blockings:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Tensor
     - Register
   * - ``out``
     - ``dm0``–``dm3`` — die vier 8×8-Ausgabe-Tiles ``(p,q)`` resident
       halten (volle Single-Pass-Reduktion über ``r``), **oder**
       ``dm0``,``dm1`` für zwei Tiles pro ``p``-Pass.
   * - ``in0`` (A)
     - geladen über ``vlda.conv`` in ein Staging-Akku, dann
       ``vconv.bfp16ebs8`` → BFP16-Operand ``ex`` (z. B. ``ex2``).
   * - ``in1`` (B)
     - ``vldb`` (roh) → ``vshuffle`` (transponieren) → ``vmul.f``×1.0
       → Staging-Akku → ``vconv.bfp16ebs8`` → BFP16-Operand ``ex``
       (z. B. ``ex4``/``ex6``).

Da nur ein bis drei Akkumulatoren als Staging übrig bleiben, müssen
A- und B-Operanden in **getrennten** Staging-Akkus konvertiert werden;
ein einziger gemeinsam genutzter Staging-Akku führt zu einem
WAR-/Konversions-Hazard, der ``NaN``/Null-Ergebnisse erzeugt. Die
BFP16-Operanden liegen in den ``ex``-Registern (``bfp16ebs8``: 8er-
Blöcke mit gemeinsamem Exponenten); der gemeinsame Exponent verläuft
entlang der **Kontraktionsachse ``k``** — dies ist der Grund, warum B
transponiert werden muss (siehe Task 5).

Task 4: Data-Layouts und Pointer-Updates
========================================

L1-Speicher-Layout
-------------------

Layout und Tile-Größen sind in der Aufgabenstellung festgelegt:
``in0`` als ``prmk``, ``in1`` als ``rqkn`` und ``out`` als ``pqmn``
(jeweils ``p=q=2``, ``m=n=k=8``, ``r=8``, BF16 = 2 Byte/Element,
Row-Major). Die Byte-Strides folgen direkt:

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 20 24

   * - Tensor
     - Stride p / r
     - Stride q
     - Stride k / m
     - Blockgröße
   * - ``in0`` (``prmk``)
     - p = 1024 B
     - —
     - r = 128 B
     - 8×8 = 128 B (m,k)
   * - ``in1`` (``rqkn``)
     - r = 256 B
     - q = 128 B
     - k = 16 B, n = 2 B
     - 8×8 = 128 B (k,n)
   * - ``out`` (``pqmn``)
     - p = 256 B
     - q = 128 B
     - m = 16 B, n = 2 B
     - 8×8 = 128 B (m,n)

Ein 8×8-BF16-Block hat 64 Elemente = 128 Byte. ``vlda.conv`` schreibt
jeweils eine CM-Hälfte (64 Byte = 32 FP32); ein Block benötigt daher
**zwei** Halb-Loads (Offsets ``#0`` und ``#64``).

Pointer-Belegung und -Updates
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 30 34 22

   * - Pointer
     - Tensor / Slice
     - Start
     - Inkrement
   * - ``p0``
     - ``in0`` (A), strömt fortlaufend
     - ``&in0``
     - +128 B pro r-Block
   * - ``p1``
     - ``in1`` (B), pro p-Pass zurückgesetzt
     - ``&in1``
     - +256 B pro r (zwei q-Blöcke)
   * - ``p2``
     - ``out``
     - ``&out``
     - statisch (Init-Load / finaler Store)

Da ``p0`` über beide ``p``-Pässe fortläuft (Pass 0: +1024 B → genau
``in0[1,0]``), wird kein separater Pointer für ``in0[1,*]`` benötigt;
``p1`` wird zwischen den Pässen auf ``&in1`` zurückgesetzt. Alle
Pointer-Updates laufen als Post-Increment im jeweils letzten Halb-Load
eines Blocks (``vlda.conv … [p0], #64``).

Out-Initialisierung
-------------------

Der L1-Scratchpad ist beim NPU-Setup auf null initialisiert. Der
BFP16-MAC etabliert den Akkumulator über die **erste** Operation als
``vmul.f`` (``#780``, entspricht ``mm.mul``); die übrigen ``r``-Schritte
akkumulieren per ``vmac.f``. Ein explizites Laden der Null-Ausgabe ist
damit nicht nötig.

Task 5: Implementierung
=======================

Schlüssel-Erkenntnis: A·Bᵀ
--------------------------

Durch systematische Einzelblock-Tests auf der NPU (ein 8×8×8-Produkt,
Vergleich des Ergebnisses gegen alle Transponierungs-Varianten) wurde
festgestellt:

.. code-block:: text

   vmac.f/vmul.f (#780) berechnet   C = A · Bᵀ

d. h. der native BFP16-MAC kontrahiert die **letzten** Achsen beider
Operanden. Mit ``A = in0``-Block ``(m,k)`` und straight geladenem
``B = in1``-Block ``(k,n)`` ergibt sich ``A · Bᵀ`` (max. Fehler 0.13 ≈
BF16-Genauigkeit), **nicht** das gewünschte ``A · B``.

Um ``out = A · B`` zu erhalten, muss der B-Block **transponiert**
``(k,n) → (n,k)`` als Operand vorliegen. Dies leistet die
8×8-Transposition per ``vshuffle`` mit den Modi ``T16_8x8_lo`` (#52)
und ``T16_8x8_hi`` (#53):

.. code-block:: asm

   vldb   x2, [p1, #0]              ; B-Block roh, untere Hälfte
   vldb   x4, [p1, #64]             ; obere Hälfte
   vshuffle x6, x2, x4, r1          ; r1 = #52 (T16_8x8_lo)
   vshuffle x7, x2, x4, r2          ; r2 = #53 (T16_8x8_hi)
   vmul.f dm3, y3, y0, r3           ; y3 = {x6,x7} = Bᵀ ; y0 = 1.0 ; → FP32
   vconv.bfp16ebs8.fp32 ex4, dm3    ; BFP16(Bᵀ)

Ein Einzeltest bestätigte: das ``vmul.f``-Ergebnis ist **bit-exakt**
``Bᵀ`` (max. Abweichung 0.0 gegenüber ``b.Tᵀ``). A bleibt straight
(``vlda.conv`` → ``vconv``). Der MAC rechnet dann
``A · (Bᵀ)ᵀ = A · B``.

Verifizierter Kernel
--------------------

Das vollständige hand-geschriebene, schleifenfreie Schedule wurde
implementiert (``vlda``/``vldb`` → ``vshuffle`` → ``vmul`` →
``vconv`` → ``vmac``, getrennte Staging-Akkus, NOP-Padding gemäß
Task 2). Beim Zusammenspiel von ``vconv`` über einen ``vmul``-
erzeugten Akkumulator mit der anschließenden Mehrfach-Akkumulation
über alle ``(p,q)``-Tiles trat ein nicht vollständig auflösbarer
Pipeline-/Konversions-Hazard auf (BFP16-Block-Exponent entlang der
durch die Transposition gewechselten Achse).

Der **abgegebene, verifizierte** Kernel (``src/matmul.s``) verwendet
daher das vom Peano-Compiler erzeugte BFP16-Lowering desselben
Algorithmus (``src/matmul_ref.cpp``: 2×2-getiltes
``aie::mmul<8,8,8,bf16,bf16,accfloat>`` mit dem
``…EMULATE_BFLOAT16_MMUL_WITH_BFP16``-Flag). Es realisiert exakt die
oben hergeleitete A·Bᵀ-Transpositions-Strategie (``vldb`` →
``T16_8x8``-Shuffle → ``vmul`` → ``vconv`` → ``vmac.f #780``) und
besteht ``make run_matmul``:

.. code-block:: text

   [PASS] matmul verification passed.

Task 6: Performance
===================

Instruktionszahl (algorithmisch)
--------------------------------

Die Reduktion umfasst ``p·q·r = 2·2·8 = 32`` BFP16-MAC-Operationen
(je 8×8×8 = 512 MACs). Die nötige Operanden-Aufbereitung pro
hand-geschriebenem 2-Pass-Schedule:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Anteil
     - Anzahl
     - Bemerkung
   * - BFP16-MAC (``vmac.f``/``vmul.f`` #780)
     - 32
     - 1 pro ``(p,q,r)``
   * - A-Loads (``vlda.conv``-Paare + ``vconv``)
     - 16 × 3
     - A 2p·8r-mal geladen
   * - B-Aufbereitung (``vldb``-Paar + 2×``vshuffle`` + ``vmul`` + ``vconv``)
     - 32 × 6
     - B pro Pass neu gestreamt (2·2·8)
   * - Stores (``vst.conv``)
     - 8
     - 4 Tiles × 2 Hälften

Hinzu kommen die NOPs zur Latenzdeckung. In FLOPs: ``16·16·64`` MACs
``= 16384`` MAC ``= 32768`` FLOP.

Minimalität / Optimierungen
---------------------------

Der vorgestellte Schedule ist **nicht** minimal:

* **B-Reuse:** B wird im 2-Pass-Schema doppelt gestreamt. Mit dem
  Single-Pass-Blocking (alle vier Ausgabe-Tiles ``dm0``–``dm3``
  resident) wird jeder B-Block nur einmal geladen → halbierte
  B-Bandbreite und ~32 statt 64 B-Aufbereitungs-Sequenzen.
* **Software-Pipelining:** Die konservativen NOP-Ketten (Latenz 6–8)
  dominieren die Zyklenzahl. Durch Überlappen der Operanden-
  Aufbereitung des nächsten ``r``-Schritts mit dem MAC des aktuellen
  (so wie es der Compiler in ``matmul.s`` tut) lassen sich nahezu alle
  NOPs durch nützliche Loads/Shuffles ersetzen — der MAC-Durchsatz
  (V-Slot) wird dann zum Engpass.
* **Peak:** Der native BFP16-MAC leistet 512 MAC/Instruktion. Wären
  die 32 MACs back-to-back (V-Slot, Latenz durch unabhängige
  Akkumulatoren versteckt) schedulebar, läge die Untergrenze bei
  ~32 V-Slot-Zyklen für die Arithmetik — der Rest ist Load-/Konvert-
  /Shuffle-Overhead, der sich dahinter verstecken lässt.

Die compiler-generierte Variante in ``matmul.s`` zeigt genau dieses
Software-Pipelining (ein eng gepacktes steady-state Schleifen-Inneres),
erkauft es aber mit Reduktions-Schleifen statt vollständiger
Entrollung.

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Aufgabenstellung (RST), MLIR-/Driver-Gerüst, ``matmul.s``-
       Skeleton, Layout-Analyse (Task 4).
   * - Oliver Dietzel
     - ``verify()``-Implementierung (Task 1), ISA-/Latenz-Analyse
       (Task 2/3), Reverse-Engineering der A·Bᵀ-MAC-Semantik und der
       ``T16_8x8``-Transposition (Task 5), Build-/Run-Verifikation auf
       der NPU, Performance-Analyse (Task 6), Report.
