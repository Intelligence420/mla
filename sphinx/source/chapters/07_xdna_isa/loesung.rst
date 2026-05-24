.. _ch07_loesung:

#######################################
Report: Inferring the VLIW ISA of XDNA2
#######################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Der XDNA2-Compute-Tile ist ein VLIW-In-Order-Prozessor **ohne
Hazard-Unit**: Slots, Reihenfolge und Operations-Latenzen müssen
explizit durch Compiler oder Programmierer verwaltet werden. Ziel des
Assignments ist es, diese Eigenschaften aus dem von Peano generierten
Assembly zurückzuschließen und das Wissen abschließend in einem
hand-scheduled Kernel zu nutzen.

Task 1: Vector-Add Kernel
==========================

Kernel-Implementierung
-----------------------

Im ``vadd_template`` in ``src/vadd.cpp`` ersetzt eine einzige Zeile den
TODO-Block:

.. code-block:: cpp

   v_out = aie::add(v_in0, v_in1);

Die ``aie::add``-Funktion ist die AIE-API-Operator-Overload für die
element-weise Addition zweier ``aie::vector``-Operanden; sie wird vom
Peano-Compiler in die entsprechende Vektor-Add-Instruktion übersetzt.

Driver-Verifikation
---------------------

``verify()`` in ``src/driver.py`` führt eine FP32-Referenzberechnung
durch und vergleicht das BF16-Ergebnis tolerant:

.. code-block:: python

   a = in0.to(torch.float32)
   b = in1.to(torch.float32)
   if kernel == "vadd":
       ref = (a + b).to(torch.bfloat16)
   elif kernel == "custom_vadd":
       ref = (a + b + b).to(torch.bfloat16)
   ...
   out_f32 = out.to(torch.float32)
   ref_f32 = ref.to(torch.float32)
   if not torch.allclose(out_f32, ref_f32, rtol=1.0 / 128.0, atol=0.0):
       max_err = (out_f32 - ref_f32).abs().max().item()
       raise AssertionError(f"{kernel}: mismatch (max_abs_err={max_err})")

Die FP32-Promotion vor dem Add bildet den Hardware-Pfad nach: die
Load-Instruktion ``vlda.conv.fp32.bf16`` konvertiert beim Laden nach
FP32 in den Akkumulator, die eigentliche Addition läuft im FP32-Akku,
und ``vst.conv.bf16.fp32`` rundet beim Speichern einmalig nach BF16.
Ein bit-exakter ``torch.equal``-Vergleich scheitert dennoch sporadisch
mit einem max. Fehler von ca. 0.03125, weil die BF16-Konvertierung der
AIE-Hardware in den Tie-Fällen anders rundet als PyTorchs
Round-to-Nearest-Even. Eine Toleranz von **einem BF16-ULP**
(``rtol = 2⁻⁷ = 1/128``) deckt diese Fälle ab; beide Kernels passieren
damit reproduzierbar:

.. code-block:: text

   [PASS] vadd verification passed.
   [PASS] custom_vadd verification passed.

BF16-Add-Mnemonic (Task-Frage)
-------------------------------

Das relevante Stück aus ``build/vadd.s`` lautet:

.. code-block:: asm

   mova    r0, #60
   vadd.f  dm0, dm0, dm1, r0

Die BF16-elementweise Addition läuft als **vadd.f** (vector add,
floating point) auf dem V-Slot. Quell- und Zielregister sind
``dm``-Akkumulatoren — die ``vlda.conv``-Loads haben die BF16-Eingaben
schon in den FP32-Akku konvertiert. ``r0 = 60`` dient als
Skalar-Modifier (Shift/Mode für die V-Unit).

Task 2: VLIW Slots
==================

Die ersten beiden VLIW-Instruktionen aus ``build/vadd.s`` legen die
Slot-Anordnung im Instruktionswort offen:

.. code-block:: asm

   vlda.conv.fp32.bf16 cml0, [p0, #0]; nopb; nops; nopxm; nopv
   vlda.conv.fp32.bf16 cmh0, [p0, #64]; nopx

Links nach rechts gelesen ergibt das erste Wort die Slot-Reihenfolge
**A, B, S, XM, V**. ``X`` und ``M`` teilen sich physisch einen
XM-Sub-Slot: sind beide idle, fasst ``nopxm`` beide in einem Eintrag
zusammen; ist nur eine Hälfte idle, wird ``nopx`` bzw. ``nopm``
einzeln geschrieben. Die zweite Instruktion zeigt diesen Fall — nur
``A`` ist belegt, ``X`` ist explizit ``nopx``, alle restlichen Slots
(``B``, ``S``, ``M``, ``V``) sind **leer** (im Wort nicht codiert).

.. list-table::
   :header-rows: 1
   :widths: 22 12 16 30

   * - Functional Unit
     - Slot
     - NOP-Mnemonic
     - Occupied in 2nd?
   * - Vector Unit
     - V
     - ``nopv``
     - nein
   * - Load Unit A
     - A
     - ``nopa``
     - **ja** (``vlda.conv.fp32.bf16``)
   * - Load Unit B
     - B
     - ``nopb``
     - nein
   * - Store Unit
     - S
     - ``nops``
     - nein
   * - Scalar/Control Unit
     - X (XM)
     - ``nopx``
     - nein
   * - Movement Unit
     - M (XM)
     - ``nopm``
     - nein

Task 3: Instruktionen und Register-Klassen pro Slot
====================================================

Slot pro Instruktion
---------------------

Die Slot-Zuordnung folgt entweder direkt aus dem Mnemonic-Suffix
(``.a``/``.b``/``.x``) oder — fehlt das Suffix — aus dem
Register-Präfix der Operanden in Verbindung mit der Slot-Funktion.

.. list-table::
   :header-rows: 1
   :widths: 38 10 52

   * - Instruktion
     - Slot
     - Begründung
   * - ``vlda.conv.fp32.bf16 cml0, [p0, #0]``
     - A
     - Suffix ``.a`` auf ``vld`` → Load-Unit A; lädt BF16, konvertiert
       in FP32 in die untere Hälfte des Akkumulators ``cml0``.
   * - ``movx r6, #1``
     - X
     - Suffix ``.x`` → Scalar/Control-Unit; skalarer Move-Immediate.
   * - ``vldb x1, [p1, #0]``
     - B
     - Suffix ``.b`` auf ``vld`` → Load-Unit B; lädt in
       Vektorregister ``x1``.
   * - ``vmov bmhl2, bmhh4``
     - M
     - Vektor-Move zwischen Akkumulator-Hälften (``bm``-Präfix).
       Kein ``.a``/``.b``/``.x``-Suffix; Bewegung zwischen
       Akku-Registern → Movement Unit.
   * - ``mova r0, #60``
     - A
     - Suffix ``.a`` → Load-Unit A (deren AGU kann auch skalare
       Move-Operationen ausführen).
   * - ``vadd.f dm0, dm0, dm1, r0``
     - V
     - Vektor-FP-Arithmetik auf Akkumulatoren → Vector Unit; ``r0``
       fungiert als Skalar-Modifier (z. B. Shift).
   * - ``ret lr``
     - X
     - Control-Flow-Instruktion (Return) → Scalar/Control-Unit.
   * - ``mov p1, p4``
     - X
     - Pointer-Register-Move ohne Slot-Suffix; Pointer-Arithmetik
       läuft im Scalar/Control-Unit-Halb des XM-Slots.
   * - ``vst.conv.bf16.fp32 cml0, [p2, #0]``
     - S
     - ``vst`` → Store-Unit; speichert ``cml0`` mit
       FP32→BF16-Konvertierung an ``[p2+0]``.

Register-Klassen pro Slot
--------------------------

Aus den Operanden der obigen Instruktionen sowie aus dem Aufbau des
Compute-Tiles (Folie 12: skalare ``r``-Register, vektorielle ``x/y``,
akkumulator ``dm/cm/bm``, Pointer ``p``):

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - Slot
     - Register-Klassen (dst / src)
     - Beispielregister
   * - V
     - dst = Akkumulator (``dm``/``cm``);
       src = Akkumulator + Skalar (``r``, als Modifier)
     - ``dm0, dm0, dm1, r0`` (``vadd.f``)
   * - A
     - dst = Akkumulator (``cm``) oder Skalar (``r``);
       src = Pointer (``p``) + Immediate
     - ``cml0, [p0, #0]`` (``vlda``); ``r0, #60`` (``mova``)
   * - B
     - dst = Vektor (``x``/``y``);
       src = Pointer (``p``) + Immediate
     - ``x1, [p1, #0]`` (``vldb``)
   * - S
     - src = Akkumulator (``cm``) + Pointer (``p``);
       dst = Memory (kein Register)
     - ``cml0, [p2, #0]`` (``vst.conv``)
   * - X
     - dst = Skalar (``r``) oder Pointer (``p``);
       src = Immediate / Skalar / Pointer / Linkregister (``lr``)
     - ``r6, #1`` (``movx``); ``p1, p4`` (``mov``); ``lr`` (``ret``)
   * - M
     - dst = Akkumulator (``bm``);
       src = Akkumulator (``bm``)
     - ``bmhl2, bmhh4`` (``vmov``)
   * - XM
     - Gemeinsamer Slot für X + M; Register-Klassen sind die
       Vereinigung der X- und M-Klassen. Eigenständig nutzbar nur
       als kombinierter NOP (``nopxm``) — sobald eine Hälfte aktiv
       ist, wird die andere einzeln per ``nopx``/``nopm`` codiert.
     - keine eigenständigen Operanden; siehe X und M

Task 4: Operations-Latenzen
============================

Die zeitliche Abfolge in ``build/vadd.s`` (jede Zeile = ein
VLIW-Zyklus; Zyklen ab 1 numeriert):

.. code-block:: asm

   c1:  vlda.conv.fp32.bf16  cml0, [p0, #0]
   c2:  vlda.conv.fp32.bf16  cmh0, [p0, #64]
   c3:  vlda.conv.fp32.bf16  cml1, [p1, #0]
   c4:  vlda.conv.fp32.bf16  cmh1, [p1, #64]
   c5:  nop
   c6:  nop
   c7:  mova    r0, #60
   c8:  vadd.f  dm0, dm0, dm1, r0
   c9:  nop
   c10: nop
   c11: ret lr
   c12: nop                              ; delay slot 5
   c13: nop                              ; delay slot 4
   c14: vst.conv.bf16.fp32  cml0, [p2, #0] ; delay slot 3
   c15: vst.conv.bf16.fp32  cmh0, [p2, #64] ; delay slot 2
   c16: nop                              ; delay slot 1

Daraus ergeben sich die Latenzen direkt nach der in der Aufgabe
genannten Zählregel (Produzent = Zyklus 1, erster abhängiger
Konsument exklusiv):

.. list-table::
   :header-rows: 1
   :widths: 15 20 30 15 15

   * - Instruktion
     - Output-Register
     - Erste abhängige Instruktion
     - Cycles apart
     - Latency
   * - ``mova``
     - ``r0``
     - ``vadd.f dm0, dm0, dm1, r0`` (c8) liest ``r0`` (gesetzt c7)
     - 1
     - **1**
   * - ``vadd.f``
     - ``dm0`` (≙ ``cml0`` + ``cmh0``)
     - ``vst.conv.bf16.fp32 cml0, [p2, #0]`` (c14) liest ``cml0``
       (geschrieben c8)
     - 6
     - **6**

Als Nebenerkenntnis liefert die Distanz vom letzten ``vlda`` (``cmh1``,
c4) zum ersten Konsumenten ``vadd.f`` (c8) eine **vlda-Latenz von 4**
— der Compiler füllt c5 und c6 mit NOPs, was exakt die fehlenden
Zyklen nach den vier sequenziellen Loads sind.

Task 5: Hand-scheduled BF16 Vector-Add (``C = A + B + B``)
===========================================================

Schedule
---------

Der Datenfluss ``vlda → vadd1 → vadd2 → vst`` definiert die kritische
Kette; mit Latenzen 4 / 6 / 6 ergibt sich der minimale Schedule:

.. code-block:: asm

   c1:  vlda.conv.fp32.bf16  cml0, [p0, #0]   ; A low  -> dm0.lo
   c2:  vlda.conv.fp32.bf16  cmh0, [p0, #64]  ; A high -> dm0.hi
   c3:  vlda.conv.fp32.bf16  cml1, [p1, #0]   ; B low  -> dm1.lo
   c4:  vlda.conv.fp32.bf16  cmh1, [p1, #64]  ; B high -> dm1.hi
   c5:  nop                                   ; vlda latency = 4
   c6:  nop
   c7:  mova    r0, #60                       ; modifier (mova latency = 1)
   c8:  vadd.f  dm0, dm0, dm1, r0             ; dm0 = A + B
   c9:  nop                                   ; vadd.f latency = 6
   c10: nop
   c11: nop
   c12: nop
   c13: nop
   c14: vadd.f  dm0, dm0, dm1, r0             ; dm0 = (A+B) + B
   c15: nop
   c16: ret lr                                ; 5 Delay-Slots folgen
   c17: nop                                   ; delay slot 5
   c18: nop                                   ; delay slot 4
   c19: nop                                   ; delay slot 3
   c20: vst.conv.bf16.fp32  cml0, [p2, #0]    ; delay slot 2  (c20−c14 = 6 ✓)
   c21: vst.conv.bf16.fp32  cmh0, [p2, #64]   ; delay slot 1

``mova r0, #60`` wird einmal gesetzt und behält den Wert bis ``vadd2``
(weder ``vadd1`` noch die NOPs schreiben ``r0``). Beide ``vst``
müssen sequenziell laufen, weil pro VLIW-Wort nur **ein** S-Slot zur
Verfügung steht. Damit ``vst cmh0`` noch *im* Delay-Slot 1 liegt, muss
``ret`` spätestens in c16 stehen; c15 bleibt dadurch zwangsweise als
NOP zwischen ``vadd2`` und ``ret``.

Anzahl Zyklen
--------------

Die Funktion belegt **21 VLIW-Zyklen** (c1–c21).

Ist das das Minimum?
---------------------

Ja, gegeben die Latenzen 4 / 6 / 6 und die Slot-Constraints:

* die 4 ``vlda`` sind serialisiert (1 A-Slot/VLIW) → mind. c1–c4,
* ``vadd1`` frühestens bei ``c4 + 4 = c8``,
* ``vadd2`` frühestens bei ``c8 + 6 = c14``,
* ``vst cml0`` frühestens bei ``c14 + 6 = c20``,
* ``vst cmh0`` frühestens bei ``c21`` (ein S-Slot/VLIW),
* ``ret`` mit 5 Delay-Slots muss so platziert sein, dass c21
  (= letzter Store) noch ein Delay-Slot ist → ``ret`` ≤ c16.

Damit liegt die kritische Pfadlänge bei ``c21``; weniger Zyklen sind
nur durch verkürzte Latenzen erreichbar (z. B. ein 128 B-Store
in einer VLIW-Instruktion oder Vorziehen einer ``vlda``-Pipeline-
Hälfte in einen anderen Slot). Beim verwendeten ISA-Subset führt
keine dieser Optionen zu kürzeren Zyklenzahlen.

``aiebu-asm`` akzeptiert die Datei ohne Fehler (``make obj_custom_vadd``),
``make run_custom_vadd`` liefert ``[PASS] custom_vadd verification passed``.

Task 6: MAC Kernel (optional)
==============================

Beide Targets übersetzen denselben ``aie::mmul<8,8,8, bfloat16,
bfloat16, accfloat>``-Aufruf — einmal in den nativen BF16-Pfad,
einmal mit ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` in den
BFP16-Pfad.

a) Instruktions-Zahlen
-----------------------

Gezählt als VLIW-Worte (Zeilen zwischen ``matmul:`` und
``.Lfunc_end0:``):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Modus
     - Datei
     - VLIW-Zyklen
   * - Normal (BF16-MMUL)
     - ``build/matmul_normal.s``
     - **43**
   * - BFP16-Emulation
     - ``build/matmul_bfp16.s``
     - **30**

Im Normal-Modus emittiert die AIE-API **16** ``vmac.f``-Instruktionen
(2 Akkumulator-Spalten × 8 K-Iterationen, je vier Vektor-Shuffles als
``vextbcst*``-Preprocessing pro K-Schritt). Im BFP16-Modus genügt **ein
einziger** ``vmac.f`` auf 128 B-BFP16-Operanden plus eine
vorbereitende ``vmul.f`` für die Konstante; die K-Schleife wird also
in einer einzigen nativen MAC-Operation zusammengezogen.

b) Wirkung des Flags
---------------------

``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` schaltet in der AIE-API
eine alternative Spezialisierung des ``aie::mmul<8,8,8, bfloat16, …>``
ein. Diese verwendet die ``vconv.bfp16ebs8.fp32``-Konversion, um die
beiden BF16-Operanden zur Laufzeit ins blockfloating BFP16-Format
(gemeinsamer 8-Bit-Exponent pro 8-Element-Block) zu falten, und
ruft dann den nativen *BFP16-MAC*-Befehl der Vector-Unit auf. Das
ersetzt 16 BF16-MACs + zugehörige Shuffles durch 1 BFP16-MAC
+ 2 Conversions.

c) Performance-Implikationen
-----------------------------

* **Durchsatz**: Der BFP16-Befehl erreicht die volle native
  MAC-Bandbreite der V-Unit; im Normal-Modus blockiert das
  Shuffle/Broadcast-Preprocessing den V-Slot in vielen Zyklen.
  Aus 43 vs. 30 Zyklen pro 8×8×8-Tile resultiert grob ein
  Speedup von ca. **1.4×**, bei größeren Akkumulationen
  (echte MAC-Schleifen, in denen das Conversion-Overhead über
  viele MACs amortisiert wird) deutlich mehr.
* **Speicher**: BFP16 belegt nur ca. 4.5 Bit/Wert effektiv (8 Bit
  Mantisse + geteilter 8-Bit-Exponent pro Block) gegenüber 16 Bit
  bei BF16 — Bandbreite und Cache-Footprint sinken.
* **Numerik**: BFP16 verliert pro Block einen Teil der dynamischen
  Reichweite, weil alle 8 Elemente denselben Exponenten teilen.
  Für Inferenz-Workloads (LLM-Aktivierungen, Conv-Features) ist das
  typischerweise akzeptabel; bei stark heterogenen Magnituden im
  selben Block (z. B. Aufmerksamkeit-Logits mit Outliern) sinkt die
  Genauigkeit messbar.
* **Programmiermodell**: Da die Conversion implizit von der AIE-API
  durchgeführt wird, ist der Kernel-Code (``aie::mmul``-Aufruf)
  identisch — die Wahl zwischen den Pfaden ist eine reine
  Compile-Time-Flag-Entscheidung. Das macht den BFP16-Modus zu einer
  fast risikolosen Optimierung, sobald die numerische Toleranz im
  Modell ausreichend ist.

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Aufgabenstellung als RST, ``vadd.cpp``, ``custom_vadd.s``
       (Skeleton), Driver-Boilerplate.
   * - Oliver Dietzel
     - ``verify()``-Implementierung (Tolerance-Vergleich), Hand-
       scheduled Schedule für ``custom_vadd.s``, Build- und
       Run-Verifikation auf der NPU, Report (Tasks 1–6).
