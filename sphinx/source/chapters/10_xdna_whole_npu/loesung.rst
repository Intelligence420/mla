.. _ch10_loesung:

#############################
Lösung und Bearbeitung
#############################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Berechnet wird dieselbe Matrixmultiplikation ``out += in0 @ in1`` wie in
Assignment 09 (``M = 256``, ``N = 128``, ``K = 1024``, BF16), nun jedoch
auf der **gesamten** XDNA2-(AIE2P-)NPU statt auf einem einzelnen
Compute-Tile. Die NPU besteht aus einem 2D-Array: Zeile 0 sind die
**Shim-Tiles** (L3-Anbindung), Zeile 1 die **Mem-Tiles** (L2) und die
Zeilen 2–5 die **Compute-Tiles** (L1/Core). Mit 8 Spalten und 4
Compute-Zeilen ergeben sich **32 Compute-Tiles**.

Der Tensor-Kernel (``matmul.s``, ``zero.s``) und die L1-Tile-Layouts
(``in0: prmk`` = ``2x8x8x8``, ``in1: rqkn`` = ``8x2x8x8``,
``out: pqmn`` = ``2x2x8x8``) sind gegenüber Assignment 09 **unverändert**;
der gesamte Aufwand liegt im *MLIR-AIE*-Datenfluss. Die ``matmul.mlir``
wird durch den parametrierten Generator ``src/gen_matmul.py`` erzeugt
(8 Spalten × 4 Zeilen), damit die 32 Cores und die zugehörigen
ObjectFIFOs konsistent und fehlerfrei ausgeschrieben werden.

.. _ch10_tiling:

Tiling und räumliche Verteilung
-------------------------------

Die Aufgabe gibt die Zerlegung der drei Achsen fest vor (Reihenfolge
außen→innen):

.. list-table::
   :header-rows: 1
   :widths: 12 32 56

   * - Achse
     - Zerlegung
     - Index-Rekonstruktion
   * - ``M = 256``
     - ``a·x·p·m`` mit ``a=2, x=8, p=2, m=8``
     - ``M = a*128 + x*16 + p*8 + m``
   * - ``N = 128``
     - ``b·y·q·n`` mit ``b=2, y=4, q=2, n=8``
     - ``N = b*64 + y*16 + q*8 + n``
   * - ``K = 1024``
     - ``c·r·k`` mit ``c=16, r=8, k=8``
     - ``K = c*64 + r*8 + k``

Daraus ergeben sich die L3-Views ``in0: axpmcrk``, ``in1: crkbyqn`` und
``out: axpmbyqn``. Die Achsen ``a``, ``b`` und ``c`` werden weiterhin
**sequentiell** über Schleifen abgearbeitet, die Achsen ``x`` und ``y``
dagegen **räumlich** über das Array verteilt:

- ``x = 8`` → die **8 Spalten** (jede Spalte bearbeitet einen anderen
  ``M``-Streifen),
- ``y = 4`` → die **4 Compute-Zeilen** (jede Zeile einen anderen
  ``N``-Streifen).

Ein Compute-Tile ``(col, row)`` ist damit für den festen Index
``x = col`` und ``y = row-2`` zuständig und iteriert nur noch über die
``a·b = 4`` Ausgabe-Tiles, mit innerer ``c``-Reduktion (16 Blöcke).
Die Eingaben werden gebroadcastet: ``in0`` hängt nur von ``x`` ab
(nicht von ``N``) und wird daher entlang einer Spalte an alle 4 Zeilen
verteilt; ``in1`` hängt nur von ``y`` ab (nicht von ``M``) und wird
entlang einer Zeile an alle 8 Spalten verteilt.

Task 1 — Setup of the Whole NPU
===============================

**Tiles.** Für jede der 8 Spalten werden ein Shim-Tile ``aie.tile(c,0)``
und ein Mem-Tile ``aie.tile(c,1)`` deklariert, dazu die 32 Compute-Tiles
``aie.tile(c, r)`` mit ``c = 0..7`` und ``r = 2..5``.

**Fused ``ab``-Schleife.** Im Single-Tile-Skeleton lief die ``ab``-Schleife
über ``128 = 16·8`` Ausgabe-Tiles (gesamtes ``M×N`` auf einem Tile). Da
nun ``x`` und ``y`` räumlich verteilt sind, bleibt jedem Compute-Tile nur
noch ``a·b = 2·2 = 4``:

.. code-block:: text

   scf.for %i_ab = 0 to 4 step 1 {          // a*b = 4 Ausgabe-Tiles
     %out = acquire @out_L1L2_<col>_<ry>    // out-Tile (pqmn) holen
     func.call @zero(%out)
     scf.for %i_c = 0 to 16 step 1 {        // K-Reduktion (16 c-Blöcke)
       %in0 = acquire @in0_L2L1_<col>
       %in1 = acquire @in1_L2L1_<ry>
       func.call @matmul(%in0, %in1, %out)  // out += in0_c @ in1_c
       release @in0_L2L1_<col> / @in1_L2L1_<ry>
     }
     release @out_L1L2_<col>_<ry>
   }

**Core-Duplikation und FIFO-Suffixe.** Die Core-Funktion wird für jedes
der 32 Compute-Tiles dupliziert. Die Platzhalter-Suffixe ``_0`` bzw.
``_0_0`` des Skeletons werden durch die tatsächlichen Koordinaten ersetzt:
``in0_L2L1_<col>``, ``in1_L2L1_<ry>`` und ``out_L1L2_<col>_<ry>``, wobei
der Zeilenindex ``ry = row-2`` (nullbasiert ab der ersten Compute-Zeile)
ist. Für ``aie.tile(7,3)`` ergeben sich z. B. ``@in0_L2L1_7``,
``@in1_L2L1_1`` und ``@out_L1L2_7_1`` — exakt wie in der Aufgabe gefordert.

Task 2 — Broadcasting the Inputs
================================

Ein ObjectFIFO mit **mehreren Consumer-Tiles** realisiert einen Broadcast
über den Link-Punkt (Mem-Tile). Damit wird das Skeleton wie folgt
erweitert:

**``in0`` (Broadcast entlang der Spalte).** Pro Spalte ``c`` versorgt ein
Mem-Tile alle 4 Compute-Tiles dieser Spalte mit demselben ``in0``-Tile:

.. code-block:: text

   @in0_L3L2_c (shim_c -> {mem_c})                : memref<16x64xbf16>
   @in0_L2L1_c (mem_c  -> {tile_c_2..tile_c_5})   : memref<2x8x8x8xbf16>
   link [@in0_L3L2_c] -> [@in0_L2L1_c] ([] [])

Das vorhandene ``@in0_L2L1_0`` wird also um die Consumer ``tile_0_3..0_5``
ergänzt, und für die Spalten 1–7 werden je ein weiteres ``L3L2``- und
``L2L1``-Paar erzeugt (8 ``in0``-Spalten-FIFOs). Das ``dimensionsToStream``
(``prmk``-Reordering) ist je Spalte identisch zum Skeleton.

**``in1`` (Broadcast entlang der Zeile).** ``in1`` hängt nur von ``y`` ab,
deshalb genügen **vier** ``in1_L3L2``-FIFOs. Jedes versorgt über sein
Mem-Tile alle 8 Compute-Tiles der zugehörigen Zeile:

.. code-block:: text

   @in1_L3L2_ry (shim_ry -> {mem_ry})             : memref<64x16xbf16>
   @in1_L2L1_ry (mem_ry  -> {tile_0_r..tile_7_r}) : memref<8x2x8x8xbf16>
   link [@in1_L3L2_ry] -> [@in1_L2L1_ry] ([] [])

Die Platzierung der vier ``in1``-FIFOs ist frei wählbar; sie liegen hier
auf den Shim-/Mem-Tiles der Spalten 0–3 (``ry → Spalte ry``). Der
Broadcast von ``@in1_L2L1_ry`` reicht über alle 8 Spalten hinweg, die
Stream-Switch-Verschaltung wird vom ObjectFIFO-Lowering automatisch
geroutet.

**``dma_memcpy_nd`` und Buffer-Descriptors.** Die Eingabe-DMAs werden auf
die neuen FIFO-Queues abgebildet
(siehe :ref:`Zugriffsmuster <ch10_access>`).
Da jedes Shim-Tile **16 Buffer-Descriptors** besitzt und die FIFOs auf
*verschiedenen* Shim-Tiles liegen, dürfen BD-IDs zwischen Spalten
wiederverwendet werden. Pro Shim werden höchstens 8 IDs belegt
(``out`` 0–3, ``in0`` 4–5, ``in1`` 6–7), also deutlich unter dem Limit.

Task 3 — Writing the Output
===========================

**Join entlang der Spalte.** Die vier Compute-Tiles einer Spalte
(``y = 0..3``) erzeugen je ein ``out``-Tile (``pqmn``, 256 Elemente).
Diese werden über einen **join-Link** auf dem Mem-Tile zu einem
gemeinsamen L2-Puffer im Layout ``ypqmn`` zusammengeführt. Der Index
``y`` wird durch unterschiedliche **Schreib-Offsets** in den join-Puffer
realisiert (256 Elemente pro Zeile):

.. code-block:: text

   @out_L1L2_c_0..3 (tile_c_2..5 -> {mem_c})  : memref<2x2x8x8xbf16>
   @out_L2L3_c (mem_c -> {shim_c})            : memref<64x16xbf16>
   link [@out_L1L2_c_0, _1, _2, _3] -> [@out_L2L3_c] ([0, 256, 512, 768] [])

Bei einem join darf das ``dimensionsToStream`` des Ausgabe-FIFOs **nicht
über die Länge eines Quellsegments (256) hinaus** zugreifen (sonst meldet
das Lowering *„out of bounds access in join input"*). Das Umsortieren
findet daher **pro 256er-Segment** statt: Es reordert ``pqmn → pmqn``,
während die Iteration über die vier ``y``-Segmente implizit aus dem join
folgt. Beim Lesen aus L2 in den Stream entsteht so das Layout ``ypmqn``:

.. code-block:: text

   @out_L2L3_c dimensionsToStream
       [<size=2, stride=128>,   // p
        <size=8, stride=8>,     // m
        <size=2, stride=64>,    // q
        <size=8, stride=1>]     // n   (pqmn -> pmqn, je Segment)

.. _ch10_access:

**Angepasste ``dma_memcpy_nd``-Muster.** Alle Angaben in Elementen,
äußerste Dimension links, innerste Stride 1. Die ``a``-Schleife wird auf
dem Host ausgerollt (``dma_memcpy_nd`` akzeptiert nur konstante Offsets);
``b`` und ``c`` sind in das 4D-Zugriffsmuster gefaltet. Für Spalte ``x``,
Zeile ``y`` und ``a``-Iteration:

.. list-table::
   :header-rows: 1
   :widths: 12 14 22 22 30

   * - Tensor
     - FIFO
     - sizes
     - strides
     - offset / Bedeutung
   * - ``in0`` (MM2S)
     - ``in0_L3L2_x``
     - ``[2, 16, 16, 64]``
     - ``[0, 64, 1024, 1]``
     - ``a*131072 + x*16384`` · b-Repeat·c·(p·m)·(r·k)
   * - ``in1`` (MM2S)
     - ``in1_L3L2_y``
     - ``[2, 16, 64, 16]``
     - ``[64, 8192, 128, 1]``
     - ``y*16`` · b·c·(r·k)·(q·n)
   * - ``out`` (S2MM)
     - ``out_L2L3_x``
     - ``[1, 4, 16, 16]``
     - ``[0, 16, 128, 1]``
     - ``a*16384 + x*2048 + b*64`` · filler·y·(p·m)·(q·n)

``in0`` ist von ``b`` unabhängig und wird per ``stride = 0`` auf der
äußersten Dimension 2-fach wiederholt (eine Spalte erhält pro ``a`` immer
dasselbe ``in0``-Tile für beide ``b``). ``in1`` ist von ``a`` und ``x``
unabhängig; die zwei ``a``-Ausrollungen geben dasselbe Tile (Offset
``y*16``) erneut aus. Beim ``out`` erscheint ``x`` nicht im View (räumlich
über die Spalten), und ``y`` ist die äußerste echte Dimension des
join-Puffers (Stride 16 in ``N``); ``b`` und ``y`` werden — obwohl
zusammenführbar — getrennt gehalten.

Die Adressrechnung bestätigt die Muster (Element-Offsets):
``in0 → a*131072 + x*16384 + c*64 + (p*8+m)*1024 + (r*8+k) = in0[a*128+x*16+(p*8+m),\ c*64+(r*8+k)]``;
``in1 → y*16 + b*64 + c*8192 + (r*8+k)*128 + (q*8+n) = in1[c*64+(r*8+k),\ b*64+y*16+(q*8+n)]``;
``out → a*16384 + x*2048 + b*64 + y*16 + (p*8+m)*128 + (q*8+n) = out[a*128+x*16+(p*8+m),\ b*64+y*16+(q*8+n)]``.

.. _ch10_dma_wait:

Korrektur: vollständiges Drainen der Ausgabe
--------------------------------------------

Der erste Versuch wartete am Ende nur **einmal** je ``out_L2L3``-Symbol.
Die Verifikation zeigte ein scharfes Muster: pro Compute-Tile war
ausschließlich die **erste** ``ab``-Iteration ``(a=0, b=0)`` korrekt, alle
übrigen Blöcke waren 0 (72,7 % Fehler):

.. code-block:: text

   a0 x0:  .  .  .  .   26.7 25.6 25.0 24.2   (Spalten b0y0..b1y3)
   a1 x0: 25.3 25.4 ...   alle != 0-Block falsch

Ursache: Jedes ``out_L2L3_x`` hat **4** ausstehende Transfers (eine pro
``ab``). Ein ``aiex.npu.dma_wait`` drainiert genau den **ältesten**
offenen Transfer (in-order TCT, wie in Assignment 09 belegt). Ohne Wait
wird der zugehörige S2MM-Transfer **nicht** vor ``h.wait()`` nach L3
geflusht, und der Host liest den alten (Null-)Inhalt. Mit **vier**
``dma_wait`` je Spalte (32 insgesamt) werden alle Transfers gedrained;
seither sind **alle** 32·4 Blöcke korrekt (mittlerer Restfehler ``0,27``,
reines BF16-Rauschen).

Task 4 — Testing
================

``make run_matmul`` läuft erfolgreich durch:

.. code-block:: text

   [PASS] matmul verification passed.

Die im Treiber gesetzten Toleranzen (``atol = 1.2``, ``rtol = 0.05``)
werden eingehalten. Eine blockweise Auswertung (``src/debug.py``) bestätigt,
dass **alle** 32 Compute-Tiles für **alle** vier ``ab``-Iterationen
korrekte Ergebnisse liefern; der mittlere absolute Fehler über alle
32 768 Elemente beträgt ``0,27`` und stammt aus der BF16-Akkumulation
(Zwischensumme zwischen den ``c``-Blöcken liegt als BF16 vor, vgl.
:ref:`ch09_verifikation`).

Durchsatz
---------

Die Matrixmultiplikation umfasst

.. math::

   \text{FLOP} = 2 \cdot M \cdot N \cdot K
               = 2 \cdot 256 \cdot 128 \cdot 1024
               \approx 67{,}1 \cdot 10^{6}\ \text{FLOP}.

Gemessen (XRT, Mittel über 200 Dispatches nach 10 Warmup-Läufen,
``src/bench.py``) ergeben sich ``t ≈ 0,192\ \text{ms}`` und damit

.. math::

   \text{FLOPS} \approx \frac{67{,}1 \cdot 10^{6}}{0{,}192 \cdot 10^{-3}\,\text{s}}
                \approx 350\ \text{GFLOP/s}.

Gegenüber dem Single-Tile-Design aus Assignment 09 (``≈ 35`` GFLOP/s) ist
das rund **10-fach** schneller. Der Faktor liegt unter den theoretischen
32× der Tiles, weil bei dieser kleinen Problemgröße der Per-Dispatch- und
Datenbewegungs-Overhead (inkl. der vollständig blockierenden 32
``dma_wait`` am Ende) dominiert; der Wert ist eine End-to-End-Rate, keine
Peak-Arithmetik-Angabe.

Task 5 — Selection of Spatial Dimensions (optional)
===================================================

Vertauscht man die Rollen von ``x`` und ``y`` (``M → a y p m`` mit
``a=4, y=4`` und ``N → b x q n`` mit ``b=1, x=8``), so liegt nun die
**``M``-Teilachse ``y`` auf den Zeilen** und die **``N``-Teilachse ``x``
auf den Spalten** (physisch bleiben 8 Spalten × 4 Zeilen).

**Geänderte Broadcast-Richtungen.** ``in0`` hängt von ``M`` ab, ``in1``
von ``N``. Maßgeblich ist, welche Achse auf welcher physischen Richtung
liegt:

- Im **Original** liegt die ``M``-Achse (``x``) auf den **Spalten** →
  ``in0`` wird entlang der Spalten verteilt (jede Spalte ein anderes
  Tile, broadcast über die 4 Zeilen); die ``N``-Achse (``y``) liegt auf
  den **Zeilen** → ``in1`` wird **entlang der Zeilen** gebroadcastet
  (jede Zeile ein anderes Tile).
- Im **vertauschten** Fall liegt die ``M``-Achse (``y``) auf den
  **Zeilen** → jetzt wird **``in0`` entlang der Zeilen** gebroadcastet
  (jede Zeile ein anderes ``in0``-Tile, broadcast über die 8 Spalten);
  die ``N``-Achse (``x``) liegt auf den **Spalten** → ``in1`` wird
  **entlang der Spalten** verteilt.

Kurz: ``in1`` wird entlang der Zeilen gebroadcastet, wenn die
``N``-Teilachse auf den Zeilen liegt (Original); ``in0`` wird entlang der
Zeilen gebroadcastet, wenn die ``M``-Teilachse auf den Zeilen liegt
(vertauscht). Die Broadcast-Richtungen von ``in0`` und ``in1`` tauschen
also genau ihre Rollen.

**Anzahl der L3L2-Queues.** Da nun die Spalten-Achse 8 verschiedene
``N``-Streifen (``x=8``) und die Zeilen-Achse 4 verschiedene
``M``-Streifen (``y=4``) tragen, kehren sich die Stückzahlen um: statt
8× ``in0_L3L2`` + 4× ``in1_L3L2`` braucht man **4× ``in0_L3L2`` + 8×
``in1_L3L2``**. Die ``ab``-Schleife bleibt mit ``a·b = 4·1 = 4`` gleich
lang.

**Wann ist kein Performance-Unterschied zu erwarten?** Das verschobene
Datenvolumen ist in beiden Fällen identisch (dieselbe Multiplikation).
Ein Unterschied entsteht nur durch die **Asymmetrie des Arrays**
(8 Spalten ≠ 4 Zeilen): die Broadcast-Fanouts (über 4 Zeilen vs. über 8
Spalten) und die Anzahl gleichzeitiger L3L2-Ströme unterscheiden sich.
Wäre das Compute-Array **quadratisch** (gleich viele Spalten und Zeilen)
und das L1-Tile in ``M``- und ``N``-Richtung gleich groß
(``p·m = q·n``, hier je 16), so wären beide Konfigurationen vollständig
symmetrisch und es wäre **kein** Performance-Unterschied zu erwarten.

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Whole-NPU-Datenfluss in *MLIR-AIE* (Tiles, Broadcast-FIFOs für
       ``in0``/``in1``, join der Ausgabe), Generator ``gen_matmul.py``,
       Access-Pattern-Herleitung (Task 3), ``dma_wait``-Korrektur,
       On-NPU-Build/-Verifikation, Report.
   * - Oliver Dietzel
     - Tiling- und Broadcast-Analyse (Task 1/2), blockweise
       Fehlerdiagnose (``debug.py``), Durchsatzmessung (``bench.py``),
       Task-5-Analyse (x/y-Tausch), Projekt-Pitches, Report.
