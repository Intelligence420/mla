.. _ch09_feedback:

#####################
Feedback-Auswertung
#####################

.. admonition:: KI-generierte Schnell-Analyse (nachträglich)
   :class: warning

   Diese Seite ist eine **nachträgliche, KI-gestützte Schnell-Analyse** des
   Kontrolleur-Feedbacks gegen Code, Doku und Aufgabenstellung. Sie entstand
   *nach* der Abgabe und ist eine grobe Selbst-Einordnung – keine geprüfte
   Korrektur und nicht Teil der ursprünglichen Lösung.

Feedback der Kontrolleure
=========================

   | Bitte auf das Abgabeformat achten; nur die 4 Dateien abgeben; Datei mit
     aktuellem Git-Tag (``submission-09``) abgeben.
   | Den Fehler im Zero-Kernel habt ihr korrekt erkannt. Bei der Behebung fehlt
     noch ein ``NOP`` nach ``VBCAST`` (2 cycle latency).
   | Bitte Fehler in der Aufgabenstellung sofort mitteilen, damit diese
     schnellstmöglich für alle behoben werden können.
   | Auch die zu strenge Fehlertoleranz habt ihr richtig erkannt. Ich
     entschuldige mich für den entstandenen Mehraufwand.
   | Die Idee bei dem Tensor-Kernel war in ``bfp16ebs8`` zu rechnen, bei euch
     hat das entsprechende Flag beim Kompilieren gefehlt. Das macht bis auf
     Rundungsfehler bei dieser Aufgabe keinen inhaltlichen Unterschied.
   | Eine Referenzimplementierung für eine cpp-matmul gibt es im
     mlir-aie-Repository.
   | MLIR-Teil war einwandfrei.

Organisatorisches und Prozess
=============================

Betrifft ausschließlich Organisatorisches: Abgabeformat (nur die vier
vorgesehenen Dateien, Archiv mit dem aktuellen Git-Tag ``submission-09``) und
der Hinweis, Fehler in der Aufgabenstellung sofort zu melden, damit sie zentral
für alle behoben werden können. Reine Abgabe-/Prozess-Punkte, kein fachlicher
Mangel.

Korrekt erkannt (bestätigt)
===========================

Drei Punkte bestätigt der Kontrolleur ausdrücklich als richtig:

* **Zero-Kernel-Fehler** – das nur halb genullte ``out``-Tile (vier statt acht
  ``vst``, d. h. nur ``256`` der ``512`` B) wurde korrekt diagnostiziert und in
  ``src/zero.s`` auf acht Stores korrigiert. Die verbleibende Nacharbeit (ein
  fehlendes ``NOP``) siehe unten.
* **Zu strenge Fehlertoleranz** – die ursprünglich zu enge Toleranz wurde
  korrekt als Fehler der Aufgabenstellung erkannt; die abgegebene
  ``src/driver.py`` nutzt die korrigierten Werte ``atol=2``, ``rtol=0.5``
  (:ref:`Task 0 <ch09>`). Der Kontrolleur entschuldigt den dadurch entstandenen
  Mehraufwand.
* **MLIR-Teil einwandfrei** – die gesamte Datenbewegung (ObjectFIFOs,
  ``dimensionsToStream``, ``dma_memcpy_nd``, das non-blocking Ping-Pong-Schema)
  wurde nicht beanstandet; Task 1–4 sind fachlich in Ordnung.

Tensor-Kernel: fehlendes ``bfp16ebs8``-Flag (Task 0/3)
======================================================

Berechtigt und der inhaltlich interessanteste Punkt. Der Tensor-Kernel sollte
im nativen Block-Floating-Point-Format ``bfp16ebs8`` rechnen (8er-Blöcke mit
gemeinsamem Exponenten). Die abgegebene ``src/matmul.s`` wurde aber **ohne** das
Define ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` aus ``src/matmul_ref.cpp``
erzeugt und nutzt daher den gewöhnlichen BF16→FP32-Emulationspfad.

**Beleg 1 – Build-Flags.** Die ``KERNEL_FLAGS`` im ``Makefile`` enthalten das
Define nicht (und das Regenerierungs-Kommando für ``matmul.s`` ebenso wenig):

.. code-block:: make

   KERNEL_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -I $(AIE_INC)

**Beleg 2 – Assembly.** Im A09-Kernel laufen **alle** MACs im Modus ``#60``
(``mov r4, #60``; danach durchgängig ``vmul.f/vmac.f …, r4``) mit reinen
FP32-Konversionen (``vlda.conv.fp32.bf16``, ``vconv.bf16.fp32``,
``vst.conv.bf16.fp32``). Die Marker des nativen ``bfp16ebs8``-Pfads – Modus
``#780`` und ``vconv.bfp16ebs8.fp32`` (vgl. :ref:`ch08_loesung`, Task 2) –
fehlen vollständig. Der direkte Vergleich der beiden abgegebenen Kernel zeigt
den Unterschied unmittelbar:

.. list-table::
   :header-rows: 1
   :widths: 36 32 32

   * - Vorkommen in ``matmul.s``
     - A08 (mit Flag)
     - A09 (ohne Flag)
   * - ``bfp16``
     - 12 ×
     - 0 ×
   * - MAC-Modus ``#780``
     - 1 ×
     - 0 × (stattdessen ``#60``)

Der A08-Kernel rechnete also tatsächlich in ``bfp16ebs8``, der A09-Kernel nicht.
Wie der Kontrolleur anmerkt, ist das Ergebnis bis auf Rundung identisch; die
großzügigen Toleranzen (``atol=2``, ``rtol=0.5``) fangen den Unterschied ab,
weshalb ``make run_matmul`` trotzdem ``[PASS]`` liefert.

**Behebung:** Beim Erzeugen von ``matmul.s`` aus ``matmul_ref.cpp`` (dem
``.cpp``→``.s``-Schritt) das Define ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``
setzen und neu bauen. Der ``.s``→``.o``-Schritt im ``Makefile`` assembliert nur
die fertige ``matmul.s`` – dort wirkt das Define nicht mehr, es muss bei der
Regenerierung gesetzt sein. Alternativ die im Feedback verlinkte cpp-Referenz
verwenden:
`mm.cc, aie_kernels/aie2p <https://github.com/Xilinx/mlir-aie/blob/ad0f0686831bdea6868a7d6b4ac81b738c0ae6a1/aie_kernels/aie2p/mm.cc#L83>`_.

Zero-Kernel-Fix: fehlendes ``NOP`` nach ``VBCAST`` (Task 3)
============================================================

Berechtigt. Der korrigierte ``src/zero.s`` setzt das volle ``512``-B-Tile mit
acht ``vst`` auf null – das ist richtig. Direkt vor dem ersten Store steht
jedoch:

.. code-block:: asm

   vbcst.16 x0, r0        ; x0 := broadcast(0)          -- Latenz 2
   vst x0, [p0], #64      ; liest x0 SOFORT (Zyklus +1) -> Hazard

``vbcst.16`` hat eine **feste Latenz von 2 Zyklen**; das Ergebnis in ``x0`` ist
erst zwei Zyklen später gültig. Da der AIE2P-Compute-Tile **keine Hazard-Unit**
besitzt (vgl. :ref:`ch07_loesung`), liest der unmittelbar folgende ``vst`` einen
stale-Wert von ``x0`` – ohne Stall (derselbe „stille" stale-Read wie in
:ref:`ch08_loesung`, Task 2). Es fehlt **ein** ``NOP`` dazwischen:

.. code-block:: asm

   vbcst.16 x0, r0
   nop                    ; deckt die VBCAST-Latenz ab
   vst x0, [p0], #64
   ...

Dass ``make run_matmul`` dennoch ``[PASS]`` lieferte, ist plausibel kein reiner
Zufall, aber fragil: ``x0`` enthält beim ersten Aufruf nach dem Zero-Init des
Cores bereits null und wird von jedem ``vbcst`` erneut auf null gesetzt – der
stale-Wert ist hier also selbst null. Korrekt ist die Stelle erst mit dem
``NOP``; das Resultat darf nicht von einem latenz-verletzenden stale-Read
abhängen.

.. note::

   Lokal nicht erneut verifizierbar (kein torch/NPU unter WSL): das ``[PASS]``
   und das Laufzeitverhalten stammen aus dem ursprünglichen On-NPU-Lauf. Die
   ``bfp16``/``#780``-Befunde sind direkt aus den abgegebenen ``matmul.s``
   ausgezählt; die feste ``VBCAST``-Latenz (2) und die Modus-Semantik
   ``#60``/``#780`` stammen aus :ref:`ch07_loesung`/:ref:`ch08_loesung`. Vor
   Korrektur an Folien/Datenblatt gegenprüfen.
