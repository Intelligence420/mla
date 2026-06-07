.. _ch09_loesung:

#############################
Lösung und Bearbeitung
#############################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Berechnet wird ``out += in0 @ in1`` mit ``M = 256``, ``N = 128`` und
``K = 1024`` in BF16 auf einem einzelnen Compute-Tile der XDNA2-
(AIE2P-) NPU. Im Hauptspeicher (L3) liegen die Matrizen row-major vor
(``in0: MK``, ``in1: KN``, ``out: MN``). Da das L1-Scratchpad des
Compute-Tiles nur 64 KiB groß ist, werden die Matrizen während der
Datenbewegung **getilt** und beim Transport ins L1 in ein rechen-
günstiges Layout **umsortiert**. Der eigentliche Tensor-Kernel ist der
aus Assignment 08; der Schwerpunkt liegt hier auf dem im *MLIR-AIE*-
Dialekt geschriebenen Datenfluss und den ``a``/``b``/``c``-Schleifen
um den Kernel (Quelle: ``assignment_09.rst``;
``slides/09_xdna_datamovement.pdf``).

.. _ch09_tiling:

Tiling und Layouts
------------------

Die Aufgabenstellung gibt die Aufteilung der drei Achsen fest vor:

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Achse
     - Zerlegung
     - Index-Rekonstruktion
   * - ``M = 256``
     - ``a·p·m`` mit ``a=16, p=2, m=8``
     - ``M = a*16 + p*8 + m``
   * - ``N = 128``
     - ``b·q·n`` mit ``b=8, q=2, n=8``
     - ``N = b*16 + q*8 + n``
   * - ``K = 1024``
     - ``c·r·k`` mit ``c=16, r=8, k=8``
     - ``K = c*64 + r*8 + k``

Daraus ergeben sich die L3-Views ``in0: apmcrk``, ``in1: crkbqn`` und
``out: apmbqn``. Beim Transport ins L1 wird auf die Tile-Layouts
``in0: prmk`` (``memref<2x8x8x8xbf16>``), ``in1: rqkn``
(``memref<8x2x8x8xbf16>``) und ``out: pqmn``
(``memref<2x2x8x8xbf16>``) umsortiert. Die Achsen ``a``, ``b`` und
``c`` werden sequentiell durch Schleifen abgearbeitet: Das Compute-Tile
iteriert über die Ausgabe-Tiles ``(a,b)`` und akkumuliert über ``c``
(die K-Reduktion). **Vor** der ``c``-Schleife wird das Ausgabe-Tile
genullt; beim Zurückschreiben wird ``out: pqmn`` wieder ins
Matrix-Layout ``MN`` gebracht.

Task 0 — Setup
==============

Der XDNA-Tensor-Kernel aus Assignment 08 (``src/matmul_ref.cpp``,
woraus ``src/matmul.s`` erzeugt wird) wurde übernommen und die
``verify()``-Funktion in ``src/driver.py`` mit den vorgegebenen
Toleranzen versehen: maximaler absoluter Fehler ``atol = 2``,
maximaler relativer Fehler ``rtol = 0.5``.

.. code-block:: python

   def verify(in0, in1, out):
       ref    = in0.to(torch.float32) @ in1.to(torch.float32)
       actual = out.to(torch.float32)
       torch.testing.assert_close(actual, ref, atol=2, rtol=0.5)

Die Toleranzen sind deutlich großzügiger als in Assignment 08
(dort ``atol = 0.5``, ``rtol = 0.2``), weil ``K = 1024`` hier 16-mal
tiefer ist und die Zwischensumme zwischen den ``c``-Blöcken als BF16
vorliegt; der dadurch akkumulierte Rundungsfehler wächst entsprechend
(siehe :ref:`ch09_verifikation`).

Der Kernel akkumuliert (statt zu überschreiben) — siehe
:ref:`ch09_akkumulation`. Damit setzt diese Abgabe direkt eine der
zentralen Forderungen aus dem Assignment-08-Feedback um
(„Es ist ein explizites Laden von ``OUT`` verlangt",
:ref:`ch08_feedback`).

Task 1 — MLIR-AIE-Operationen
=============================

Kurzzusammenfassung der acht in der Aufgabe genannten Operationen
(Quellen: ``xilinx.github.io/mlir-aie/AIEDialect.html`` und
``…/AIEXDialect.html``, IRON Programming Guide, Abschnitt 2; KI-Zusammengesammelt):

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Operation
     - Bedeutung
   * - ``aie.tile(col,row)``
     - Deklariert ein physisches Tile und liefert ein SSA-Handle. Die
       Zeile bestimmt die Art: ``row 0`` = Shim/L3-Anbindung,
       MemTile-Zeile = L2, Compute-Zeilen = Core/L1.
   * - ``aie.core(tile)``
     - VLIW-Programmregion eines Compute-Tiles (L1). Enthält hier die
       ``a``/``b``/``c``-Schleifen mit ``zero``- und ``matmul``-Aufruf.
       Attribute u. a. ``stack_size``, ``link_with``.
   * - ``aie.runtime_sequence(args…)``
     - Instruktionsstrom des Steuer-Prozessors, der den Datenfluss
       L3↔Array konfiguriert und treibt. Die Block-Argumente sind die
       L3-Memrefs. Enthält ``aiex.npu.*``-Operationen, **keine**
       Rechenoperationen.
   * - ``aie.objectfifo @name(prod,{cons},depth) : <memref…>``
     - Benannter, tiefenbegrenzter Ringpuffer (Producer → Consumer).
       ``depth = 2`` ergibt Double-/Ping-Pong-Buffering. Der Elementtyp
       ist das Tile-Layout; optional relayoutet
       ``dimensionsToStream``/``-FromStream`` ein Tile per
       size/stride beim Transfer.
   * - ``aie.objectfifo.link [in] -> [out]([] [])``
     - Verkettet zwei ObjectFIFOs über ein gemeinsames (Mem-)Tile;
       erzeugt einen impliziten DMA-Copy L3→L2→L1 **ohne** Core-Code.
       Die Offset-Listen dienen join/distribute über mehrere FIFOs.
   * - | ``aie.objectfifo.acquire(…)``
       | ``aie.objectfifo.release(…)``
     - Producer/Consumer-Handshake im Core; lowert zu Lock-Operationen.
       Erzwingt die FIFO-Ordnung; die Tiefe begrenzt die Zahl
       gleichzeitig acquired-but-unreleased Puffer.
   * - ``aiex.npu.dma_memcpy_nd(mem[off][size][stride]) {id,metadata}``
     - n-D-„Half-DMA" über die ShimDMA (L3↔Array). Bis zu **4
       Dimensionen**, Angaben in **Elementen**, äußerste Dim links,
       innerste mit Stride 1. ``id`` = Buffer-Descriptor (max. 16),
       ``metadata`` = Ziel-FIFO; Richtung MM2S/S2MM ergibt sich aus dem
       FIFO. Wiederholung über ``stride = 0`` auf der äußersten Dim.
   * - ``aiex.npu.dma_wait {symbol=@fifo}``
     - Blockiert, bis das Completion-Token (TCT) der **ältesten offenen**
       Übertragung dieses Symbols eintrifft; lowert zu ``npu.sync``.
       Nötig vor der Wiederverwendung einer Buffer-Descriptor-ID. Nur
       S2MM erzeugt per Default ein TCT.

Task 2 — Datenfluss und beteiligte Operationen
==============================================

Der Datenfluss verläuft L3 → Shim-Tile → Mem-Tile → Compute-Tile und
für die Ausgabe zurück. Je Stufe ist die beteiligte *MLIR-AIE*-
Operation angegeben:

.. code-block:: text

   L3 (DDR, row-major Matrizen in0/in1/out)
     │  aiex.npu.dma_memcpy_nd   (in der runtime_sequence; getilt/strided)
     │                            MM2S für in0/in1, S2MM für out
     ▼
   Shim-Tile (0,0)  ── ShimDMA, 16 Buffer-Descriptors
     │  @in0_L3L2_0 / @in1_L3L2_0  (L3→L2) ;  @out_L2L3_0  (L2→L3, zurück)
     ▼
   Mem-Tile (0,1, L2)  ── aie.objectfifo.link + dimensionsToStream
     │  relayout zu prmk / rqkn (Eingänge) ; pqmn→MN (Ausgang)
     │  @in0_L2L1_0 / @in1_L2L1_0  (L2→L1) ;  @out_L1L2_0_0  (L1→L2)
     ▼
   Compute-Tile (0,2, L1)  ── aie.objectfifo.acquire/release im aie.core
                              func.call @zero / @matmul

Die ObjectFIFO-Deklarationen (Typen, ``dimensionsToStream``, ``link``,
Tiefe 2) sind durch das vorgegebene Skeleton festgelegt und wurden
nicht verändert. Die Synchronisation läuft auf zwei Ebenen: im Host
über ``aiex.npu.dma_wait{@out_L2L3_0}`` (Buffer-Descriptor-Freigabe),
im Core über die Locks aus ``acquire``/``release``.

Task 3 — Implementierung
========================

Der Datenfluss verteilt sich auf zwei Stellen: die Core-Schleifen im
``aie.core`` (welche Tiles in welcher Reihenfolge konsumiert werden)
und die ``runtime_sequence`` (welche DMAs die Tiles bereitstellen).

Core-Schleifen
--------------

Das Compute-Tile arbeitet die ``128 = 16·8`` Ausgabe-Tiles ``(a,b)``
nacheinander ab. Pro Ausgabe-Tile wird das ``out``-Tile **einmal**
acquired und genullt, dann wird über die 16 ``c``-Blöcke akkumuliert:

.. code-block:: text

   scf.for %ab = %c0 to %c128 step %c1 {        // 16 a * 8 b Ausgabe-Tiles
     %out = ... acquire @out_L1L2_0_0(Produce, 1)
     func.call @zero(%out)                       // out-Tile nullen
     scf.for %c = %c0 to %c16 step %c1 {         // K-Reduktion (16 c-Blöcke)
       %in0 = ... acquire @in0_L2L1_0(Consume, 1)
       %in1 = ... acquire @in1_L2L1_0(Consume, 1)
       func.call @matmul(%in0, %in1, %out)       // out += in0_c @ in1_c
       ... release @in0_L2L1_0 / @in1_L2L1_0(Consume, 1)
     }
     ... release @out_L1L2_0_0(Produce, 1)
   }

Die Schleifenindizes werden im Core nicht ausgewertet — die
ObjectFIFOs liefern die Tiles in genau der konsumierten Reihenfolge
(``for a: for b: for c``).

.. _ch09_akkumulation:

Akkumulierender Kernel (Bezug zum A08-Feedback)
-----------------------------------------------

Da ``matmul`` pro ``(a,b)`` **16-mal** (über ``c``) auf dasselbe
``out``-Tile aufgerufen wird, muss der Kernel **akkumulieren**: Der
A08-Kernel überschrieb ``out`` (``mm.mul`` + ``store``); in A09 bliebe
damit nur der letzte ``c``-Block stehen. Der Kernel liest daher das
aktuelle ``out``-Tile, addiert den Beitrag des ``c``-Blocks und
schreibt zurück (Read-Modify-Write). Das ist exakt das vom
Assignment-08-Feedback geforderte „explizite OUT-Laden"
(:ref:`ch08_feedback`).

Geplant war ``mm.from_vector(...)`` zum Vorbelegen des Akkumulators.
Die in dieser ``mlir-aie``-Version (1.3.1) vorhandene
``aie::mmul<8,8,8,bf16,bf16,accfloat>`` bietet jedoch **kein**
``from_vector``. Stattdessen wird der Produkt-Beitrag des ``c``-Blocks
berechnet und auf das geladene ``out`` addiert:

.. code-block:: cpp

   MMUL mm;                                            // 8x8x8 BFP16-MAC
   mm.mul(load_v<64>(in0+p*512), load_v<64>(in1+q*64));
   for (unsigned r = 1; r < 8; ++r)                    // Reduktion über r
     mm.mac(load_v<64>(in0+p*512+r*64), load_v<64>(in1+r*128+q*64));
   auto prev = load_v<64>(out + p*128 + q*64);         // out bisher (bf16)
   auto prod = mm.to_vector<bfloat16>();               // A@B dieses c-Blocks
   store_v(out + p*128 + q*64, aie::add(prod, prev));  // out += A@B

Die Zwischensumme liegt damit zwischen den ``c``-Aufrufen als BF16 im
L1-``out``-Tile vor; das ist die Quelle des in :ref:`ch09_verifikation`
diskutierten Rundungsfehlers.

.. _ch09_zero:

Korrektur am ``zero``-Kernel
----------------------------

Das ``out``-L1-Tile ist ``2·2·8·8 = 256`` Elemente = **512 Byte**. Das
vorgegebene ``src/zero.s`` führte aber nur **vier** ``vst x0,[p0],#64``
aus = ``4·64 = 256`` Byte und nullte damit nur die ``p=0``-Hälfte. Die
``p=1``-Hälfte blieb stehen und akkumulierte auf alten Speicherinhalt
→ rund die Hälfte der Ergebniszeilen war falsch (``make run_matmul``:
45,5 % Abweichung, ausschließlich Zeilen mit ``row mod 16 ≥ 8``, also
``p=1``). Nach Erweiterung auf **acht** Stores (volle 512 Byte) sank
die Abweichung auf 0,02 % (siehe :ref:`ch09_verifikation`). Da
``vst x`` hier 64 Byte schreibt, sind genau 8 Stores nötig; sie stehen
vor dem ``ret lr`` (dessen 5 Delay-Slots als ``nop`` gefüllt sind).
Dies war eine bestätigte Fehlerkorrektur am Skeleton, kein Umbau.

Verifizierte Access-Patterns
----------------------------

Die ``a``-Schleife (16 M-Blöcke) liegt auf dem Host, ``b`` und ``c``
sind in das 4D-Zugriffsmuster jedes Buffer-Descriptors gefaltet
(Offsets/Sizes/Strides in **Elementen**, äußerste Dim links, innerste
Stride 1). Die L3-Strides sind ``in0`` 1024/Zeile, ``in1`` 128/Zeile,
``out`` 128/Zeile. Pro ``a``-Iteration (Offset im innersten Slot):

.. list-table::
   :header-rows: 1
   :widths: 14 18 20 24 24

   * - Tensor
     - offsets
     - sizes
     - strides
     - Bedeutung (außen→innen)
   * - ``out`` (S2MM)
     - ``[0,0,0,a*2048]``
     - ``[1,8,16,16]``
     - ``[0,16,128,1]``
     - filler · b · row(pitch 128) · col
   * - ``in0`` (MM2S)
     - ``[0,0,0,a*16384]``
     - ``[8,16,16,64]``
     - ``[0,64,1024,1]``
     - **b-Repeat (stride 0)** · c · row(pitch 1024) · col
   * - ``in1`` (MM2S)
     - ``[0,0,0,0]``
     - ``[8,16,64,16]``
     - ``[16,8192,128,1]``
     - b · c · row(pitch 128) · col

Zwei Besonderheiten: ``in0`` ist von ``b`` unabhängig, wird aber pro
``b`` neu konsumiert → es wird per ``stride = 0`` auf der äußersten Dim
8-fach **wiederholt** (Repeat-Technik der Folien, Stride 0 + Size ≥ 2
nur auf der äußersten Dim). ``in1`` ist von ``a`` unabhängig und wird
pro ``a`` mit Offset 0 erneut ausgestellt. Die Adressrechnung bestätigt
die Patterns (Element-Offsets): ``in0`` → ``a*16384 + c*64 + row*1024 +
col = in0[a*16+row, c*64+col]``; ``in1`` → ``b*16 + c*8192 + row*128 +
col = in1[c*64+row, b*16+col]``; ``out`` → ``a*2048 + b*16 + row*128 +
col = out[a*16+row, b*16+col]``.

Eine zweite Einschränkung erzwang das Ausrollen der ``a``-Schleife:
``aiex.npu.dma_memcpy_nd`` akzeptiert **nur konstante Offsets** („Only
constant offsets currently supported"). Ein ``scf.for`` mit einem aus
der Induktionsvariablen berechneten Offset lowert nicht. Die 16
``a``-Iterationen werden daher mit statischen Offsets
(``out: a*2048``, ``in0: a*16384``) ausgeschrieben.

.. _ch09_verifikation:

Verifikation
------------

``make run_matmul`` liefert:

.. code-block:: text

   Mismatched elements: 6 / 32768 (0.0%)
   Greatest absolute difference: 2.6722 at index (19, 39)  (up to 2 allowed)
   Greatest relative difference: 429.66 at index (17, 17)  (up to 0.5 allowed)

Das entspricht **99,98 %** korrekten Elementen. Die 6 verbleibenden
liegen knapp über ``atol = 2`` (absolute Differenz 2,32–2,67) und
**alle** an Stellen mit Referenzwert ≈ 0 (Near-Cancellation): Dort
explodiert die relative Differenz, und der absolute Restfehler des
BF16-Rauschens liegt minimal über 2.

Dass es sich um inhärentes BF16-Akkumulationsrauschen und nicht um
einen Logikfehler handelt, wurde durch einen Modellabgleich belegt:
Ein treues, in PyTorch nachgebildetes Akkumulationsmodell (pro
``c``-Block FP32-Produkt, Rundung auf BF16, BF16-Read-Modify-Write des
``out``-Tiles) erzeugt **dieselbe** mittlere absolute Differenz
gegenüber der FP32-Referenz wie die NPU (jeweils ``0,90`` bei einer
Ausgabe-Streuung von ``std ≈ 32``, Bereich ``±134``). Das NPU-Ergebnis
ist also so genau, wie es die BF16-Mathematik bei ``K = 1024`` und
einem BF16-``out``-Tile im L1 zulässt.

Ursache ist, dass die Zwischensumme zwischen den 16 ``c``-Blöcken als
BF16 im L1-``out``-Tile gespeichert wird (der durch das Skeleton
festgelegte ObjectFIFO-Typ ``@out_L1L2_0_0`` ist BF16). Jeder
``c``-Block rundet die laufende Summe erneut auf BF16; das ist die
dominante Fehlerquelle und kann **kernel-seitig nicht** beseitigt
werden — nur ein FP32-Akkumulator im L1 (größere Skeleton-Änderung)
würde sie vermeiden. Gemäß Vorgabe wird ``atol = 2`` / ``rtol = 0.5``
beibehalten und der Restfehler hier als BF16-Grenze dokumentiert.

Task 4 — Performance (non-blocking)
===================================

Die korrektheits-orientierte Task-3-Fassung wartete nach **jeder**
``a``-Iteration sofort auf deren ``out``-Transfer (``dma_wait`` direkt
hinter jeder Gruppe) — also vollständig blockierend. Die Aufgabe
verlangt, „dass es keinen blockierenden Wait gibt, d. h. es ist immer
eine Data-Movement-Operation ausstellbar (außer der letzten)".

Das Shim-Tile hat **16 Buffer-Descriptor-IDs**
(``slides/09_xdna_datamovement.pdf``). Sie werden in zwei Gruppen
aufgeteilt — ping ``(out 0, in0 1, in1 2)`` für gerade ``a``, pong
``(out 8, in0 9, in1 10)`` für ungerade ``a`` — sodass zwei
``a``-Iterationen gleichzeitig „in flight" sein können. Eine
BD-ID wird erst **zwei** Iterationen später wiederverwendet; da
``dma_wait{@out_L2L3_0}`` mit der **ältesten** offenen ``out``-
Übertragung dieses Symbols synchronisiert (in-order TCT), wird es
genau **vor** die Wiederverwendung gezogen:

.. code-block:: text

   a=0 (ping 0/1/2): issue out,in0,in1                 (kein wait)
   a=1 (pong 8/9/10): issue out,in0,in1                (kein wait)
   a=k>=2:  dma_wait @out  (drained a=k-2)  →  issue (ids von a-2 frei)
   …
   Ende:   2× dma_wait @out  (drainen a=14, a=15)

Damit ist außer den beiden finalen Drains immer mindestens ein
``out``-Transfer offen (Issue-ahead). Auf die Eingänge wird **nicht**
gewartet: Die ``in0``/``in1``-Tiles eines ``(a,b)`` werden vor dessen
``out`` konsumiert, ihre BDs sind also frei, sobald der zugehörige
``out``-Wait zurückkehrt (MM2S erzeugt ohne ``issue_token`` kein TCT).
Dieses Muster entspricht dem Referenz-Design
``programming_examples/basic/matrix_multiplication/single_core/single_core.py``
aus ``Xilinx/mlir-aie``.

Das Ergebnis ist **bit-identisch** zur blockierenden Fassung (dieselben
6 Randelemente, gleiche maximale Differenz) — die Umstellung ändert nur
die BD-IDs und den Wait-Zeitpunkt, nicht die Transfers selbst.

Durchsatz (FLOP/s)
------------------

„Performance" ist eine **Rate** (FLOP pro Sekunde). Die Matrixmultiplikation
umfasst

.. math::

   \text{FLOP} = 2 \cdot M \cdot N \cdot K
               = 2 \cdot 256 \cdot 128 \cdot 1024
               = 67{,}1 \cdot 10^{6}\ \text{FLOP}.

Auf der NPU gemessen (XRT, Mittel über 200 Dispatches nach 10
Warmup-Läufen, inkl. Host-Dispatch- und DMA-Overhead) ergibt sich eine
Zeit von ``t ≈ 1,91 ms`` pro vollständiger Matrixmultiplikation und
damit

.. math::

   \text{FLOPS} = \frac{\text{FLOP}}{t}
                = \frac{67{,}1 \cdot 10^{6}}{1{,}91 \cdot 10^{-3}\,\text{s}}
                \approx 35\ \text{GFLOP/s}.

Der Wert ist eine **End-to-End-Rate** für ein einzelnes Compute-Tile
und wird vom Per-Dispatch-Overhead und der Datenbewegung dominiert, ist
also keine Peak-Arithmetik-Angabe. Die non-blocking-Umstellung erhöht
die Überlappung von Datenbewegung und Rechnung; der absolute Gewinn ist
bei einem einzelnen Tile und der kleinen Problemgröße jedoch gering.

Task 5 — Buffer Placement (optional)
====================================

Diese optionale Aufgabe wurde **nicht** umgesetzt. 

Beiträge
========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Aufgabenstellung (RST), MLIR-/Driver-Gerüst, Tiling- und
       Access-Pattern-Analyse (Task 3), Datenfluss-Implementierung,
       non-blocking-Umstellung (Task 4), On-NPU-Build/-Verifikation,
       Report.
   * - Oliver Dietzel
     - MLIR-AIE-Operationsübersicht (Task 1), Datenfluss-Skizze
       (Task 2), Kernel-Akkumulation und ``zero``-Korrektur, Modell-
       abgleich des BF16-Befunds, Performance-Messung.
