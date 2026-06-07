.. _ch08_feedback:

#####################
Feedback-Auswertung
#####################

.. admonition:: KI-generierte Schnell-Analyse (nachträglich)
   :class: warning

   Diese Seite ist eine **nachträgliche, KI-gestützte Schnell-Analyse** des
   Kontrolleur-Feedbacks gegen Code, Doku und Aufgabenstellung. Sie entstand
   *nach* der Abgabe und ist eine grobe Selbst-Einordnung – keine geprüfte
   Korrektur und nicht Teil der ursprünglichen Lösung. Dies ist das Assignment
   mit dem meisten Nacharbeitungsbedarf.

Feedback der Kontrolleure
=========================

   | Verspätet abgegeben (auch in git); bitte auf Abgabenstruktur achten (−2 P.).
   | Die Latenz von den Operationen ist fest, kein Bereich.
   | Manche Operationen besitzen forwarding/late forwarding.
   | ``VSHUFFLE`` und ``VCONV`` benötigen ausschließlich die Movement-Unit.
   | Kein konkretes Register-Blocking gewählt.
   | Stride-Tabellenüberschrift fehlerhaft.
   | Pointer-Updates für ``p2`` nicht beschrieben.
   | Es ist ein explizites Laden von ``OUT`` verlangt.
   | Kein selbst geschriebener Kernel.
   | Performance wird in FLOPs pro Sekunde (FLOPS) angegeben.

Organisatorisches (−2 P.)
=========================

Berechtigt. Verspätet (auch im git-Verlauf sichtbar) und Abgabestruktur nicht
eingehalten – reiner Abgabe-Fehler.

Task 2: Latenzen, Forwarding, Slots
===================================

Latenz ist fest, kein Bereich
-----------------------------

Berechtigt. Die Latenz-Tabelle gibt Bereiche an (``7–8``, ``2–3``, ``4–6``) und
bezeichnet sie als „sichere untere Schranken". Jede Operation hat aber eine
**feste** Latenz. Die beobachtete Streuung ist kein Latenz-Bereich, sondern
folgt aus Forwarding (s. u.). Korrekt wäre eine feste Zahl pro Instruktion
(z. B. ``vldb`` 7, ``vmul.f`` 6, ``vmac.f`` 6, ``vst.conv`` 6) statt eines
Intervalls.

Forwarding / late forwarding
----------------------------

Berechtigt und der eigentliche Grund für die „``7–8``". Manche
Producer→Consumer-Paare können das Ergebnis **früher** weiterreichen
(Forwarding) bzw. nur über einen späten Pfad (late forwarding). Die in
``loesung.rst`` als empirische Erkenntnis beschriebene Beobachtung
(„Ergebnis eines ``vlda.conv`` braucht ~7–8 Zyklen, bevor ``vconv`` es lesen
kann") ist also kein unscharfer Latenzwert, sondern genau ein Paar **ohne**
frühes Forwarding – die NOP-Distanz ergibt sich aus *fester Latenz minus
Forwarding-Pfad*, nicht aus einer Spanne. Das Modell „feste Latenz +
Forwarding-Tabelle" hätte die NOP-Abstände deterministisch hergeleitet, statt
sie zu vermessen.

Slot von ``vshuffle`` und ``vconv``
-----------------------------------

Berechtigt. Die Tabelle führt beide als ``V/M``. Richtig: ``vshuffle`` und
``vconv`` laufen **ausschließlich auf der Movement-Unit (M)**, nicht auf dem
V-Slot. Das ist auch für Task 6 relevant: weil sie den V-Slot **nicht**
belegen, können Shuffle/Konversion mit den ``vmac``-MACs (V-Slot)
überlappen – ein Argument, das die Pipelining-Diskussion stützt.

Task 3: Kein konkretes Register-Blocking
========================================

Berechtigt. Die Aufgabe verlangt „**Choose** a register blocking". Die Doku
stellt stattdessen **zwei** Optionen nebeneinander („``dm0``–``dm3`` … *oder*
``dm0``,``dm1``") und legt sich nicht fest – die Registertabelle bleibt damit
unbestimmt. Eine konkrete Wahl wäre z. B.:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Tensor
     - Register (konkret)
   * - ``out``
     - ``dm0, dm1, dm2, dm3`` — die vier 8×8-Tiles ``(p,q)`` resident
       (Single-Pass-Reduktion über ``r``, B nur einmal gestreamt).
   * - ``in0`` (A)
     - ``vlda.conv`` → ``cml0/cmh0`` (Staging) → ``vconv`` → ``ex2``.
   * - ``in1`` (B)
     - ``vldb`` → ``x2/x4`` → ``vshuffle`` → ``x6/x7`` → ``vmul.f`` →
       ``dm4`` (Staging) → ``vconv`` → ``ex4``.

Damit ist jedes Register einer Rolle zugeordnet und das Blocking ist
nachvollziehbar gewählt statt offengelassen.

Task 4: Strides, ``p2``-Updates, ``OUT``-Load
=============================================

Stride-Tabellenüberschrift fehlerhaft
-------------------------------------

Berechtigt. Die Spaltenköpfe fassen verschiedene Dimensionen zusammen
(„Stride p / r", „Stride k / m"), sodass die Zellen nicht zur Überschrift
passen – z. B. steht für ``in0`` der Wert ``r = 128 B`` unter „Stride k / m".
Die **Werte** stimmen, die Köpfe sind falsch zugeordnet. Sauberer ist eine
Spalte pro Dimension:

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16

   * - Tensor
     - äußerste
     - …
     - …
     - innerste
   * - ``in0`` (``prmk``)
     - p = 1024 B
     - r = 128 B
     - m = 16 B
     - k = 2 B
   * - ``in1`` (``rqkn``)
     - r = 256 B
     - q = 128 B
     - k = 16 B
     - n = 2 B
   * - ``out`` (``pqmn``)
     - p = 256 B
     - q = 128 B
     - m = 16 B
     - n = 2 B

Pointer-Updates für ``p2`` nicht beschrieben
--------------------------------------------

Berechtigt. ``p2`` (``out``) ist in der Doku als „statisch" geführt. Das ist
falsch: ``out`` hat vier ``(p,q)``-Blöcke à zwei Halb-Stores = **8 Stores** an
verschiedenen Offsets, ``p2`` muss also fortlaufen. Beschrieben gehört:
``+64 B`` zwischen den beiden Hälften eines Blocks, ``+128 B`` zum nächsten
``q``, ``+256 B`` zum nächsten ``p`` (jeweils als Post-Increment im Store).

Explizites Laden von ``OUT`` verlangt
-------------------------------------

Berechtigt. Die Kontraktion ist ``out += in0 @ in1`` – ``out`` muss vor der
Akkumulation **explizit in die Akkumulatoren geladen** werden (auch wenn der
Scratchpad null ist; genau dafür hat Task 3 eine ``out``-Registerzeile). Die
Doku argumentiert das Gegenteil („ein explizites Laden der Null-Ausgabe ist
nicht nötig", weil das erste ``vmul.f`` den Akku etabliert). Korrekt: ``out``
per ``vlda.conv`` in ``dm0``–``dm3`` laden, dann über **alle** ``r`` (inkl.
``r=0``) mit ``vmac.f`` akkumulieren, dann zurückschreiben.

Task 5: Kein selbst geschriebener Kernel
========================================

Berechtigt – und der **schwerwiegendste** Punkt. Die Kernaufgabe war:
„**Implement** the tensor kernel in ``src/matmul.s``. Do not use any
control-flow instruction other than the final ``ret lr``." Die abgegebene
``src/matmul.s`` ist aber der **Peano-Compiler-Output** von
``src/matmul_ref.cpp`` – belegbar direkt im Datei-Kopf und an der Struktur:

.. code-block:: text

   .file "matmul_ref.cpp"                 ; Compiler-Herkunft
   .LBB0_1: // %for.cond1.preheader  =>This Loop Header: Depth=1
   .LBB0_2: // %for.body4              Loop Header: Depth=2
   .LBB0_3: // %for.body13            Loop Header: Depth=3
   ...
   jnz r19, #.LBB0_2                   ; Schleifen-Branch
   jnz r18, #.LBB0_1                   ; Schleifen-Branch

Damit sind **zwei** Anforderungen verfehlt: (1) der Kernel wurde nicht selbst
geschrieben, und (2) er enthält Schleifen-Kontrollfluss (``jnz``, ``ls``-
Zero-Overhead-Loop), was die „nur ``ret lr``"-Vorgabe verletzt. Die Doku räumt
das selbst ein („verwendet daher das vom Peano-Compiler erzeugte
BFP16-Lowering").

Die **Analyse-Leistung** drumherum ist korrekt und wertvoll – die
A·Bᵀ-Semantik des MAC, die ``T16_8x8``-Transposition per ``vshuffle``, die
getrennten Staging-Akkus. Was fehlt, ist die eigentliche Umsetzung: ein
vollständig entrolltes, schleifenfreies Schedule. Der Bericht beschreibt einen
„nicht vollständig auflösbaren Pipeline-/Konversions-Hazard" als Grund für den
Rückgriff auf den Compiler – das ist der Punkt, der nachzuarbeiten ist (das
Hazard rührt vom BFP16-Block-Exponenten entlang der durch die Transposition
gewechselten Achse; auflösbar durch getrennte ``ex``-Register pro Operand und
korrekte NOP-Distanzen aus dem Forwarding-Modell oben).

Task 6: Performance in FLOPS, nicht FLOP
========================================

Berechtigt. „Performance" ist eine **Rate** (FLOP pro Sekunde), die Doku gibt
nur eine **Anzahl** an (``32768 FLOP``) plus Instruktionszahlen. Der FLOP-Wert
selbst ist richtig (``M·N·K = 16·16·64 = 16384`` MAC ``= 32768 FLOP``), es
fehlt die Division durch die Zeit:

.. math::

   \text{FLOPS} = \frac{\text{FLOP}}{t}
                = \frac{32768}{\text{Zyklen} / f_\text{clock}}

Aus der gezählten Instruktions-/Zyklenzahl und der AIE2P-Taktfrequenz wäre
also ein FLOP/s-Wert anzugeben (und sinnvollerweise gegen den Peak
``512\ \text{MAC} \times 2 \times f_\text{clock}`` einzuordnen). Erst das ist
die von Task 6 gefragte „performance".

.. note::

   Belege, die lokal nicht erneut verifiziert werden konnten (kein
   PDF-Rendering, kein torch/NPU): die exakten **festen** Latenzwerte und die
   Forwarding-Tabelle pro Slot (Folien Assignment 07/08) sowie die
   AIE2P-Taktfrequenz für die FLOPS-Zahl. Vor Korrektur in ``loesung.rst`` an
   Folien bzw. Datenblatt prüfen.
