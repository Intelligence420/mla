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

*Implementierung von* ``vadd.cpp`` *und* ``driver.py::verify`` *,
gefolgt von* ``make run_vadd`` *.*

BF16-Add-Mnemonic
-----------------

*Antwort aus* ``build/vadd.s`` *einsetzen.*

Task 2: VLIW Slots
==================

*Slot-Tabelle (V/A/B/S/X/M) ausfüllen.*

Task 3: Instruktionen und Register-Klassen pro Slot
====================================================

*Instruktions- und Register-Klassen-Tabelle.*

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
