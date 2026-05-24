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
durch und vergleicht das BF16-Ergebnis bit-exakt:

.. code-block:: python

   a = in0.to(torch.float32)
   b = in1.to(torch.float32)
   if kernel == "vadd":
       ref = (a + b).to(torch.bfloat16)
   elif kernel == "custom_vadd":
       ref = (a + b + b).to(torch.bfloat16)
   ...
   if not torch.equal(out, ref):
       max_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
       raise AssertionError(f"{kernel}: mismatch (max_abs_err={max_err})")

Die FP32-Promotion vor dem Add bildet exakt den Hardware-Pfad nach:
die Load-Instruktion ``vlda.conv.fp32.bf16`` konvertiert beim Laden
nach FP32 in den Akkumulator, die eigentliche Addition läuft im
FP32-Akku, und ``vst.conv.bf16.fp32`` rundet beim Speichern einmalig
nach BF16. Ein einziger Round-Trip — daher reicht ``torch.equal``.

BF16-Add-Mnemonic (Task-Frage)
-------------------------------

*Antwort aus* ``build/vadd.s`` *einsetzen.*

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

*Latenzen für* ``mova`` *und* ``vadd.f`` *aus* ``build/vadd.s`` *.*

Task 5: Hand-scheduled BF16 Vector-Add (``C = A + B + B``)
===========================================================

*Implementierung in* ``custom_vadd.s`` *, Cycle-Zählung und
Diskussion (minimal möglich?).*

Task 6: MAC Kernel (optional)
==============================

*BF16-Matmul mit und ohne
``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``-Flag vergleichen.*

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - *Pending*
   * - Oliver Dietzel
     - *Pending*
