.. _ch05_loesung:

##################################################
Report: Contraction Interface and L2 Optimization
##################################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Dieses Kapitel dokumentiert unsere Lösung des fünften Assignments:
*Contraction Interface and L2 Optimization*. Aufbauend auf den
cuTile-Kernels aus Assignment 04 entwickeln wir ein
Konfigurations-Interface für Tensor-Kontraktionen, einen Optimizer,
der diese Configs transformiert, und nutzen das Ganze, um einen
L2-optimierten cuTile-Kernel für eine batched Matrix-Multiplikation
``cmk, ckn -> cmn`` abzuleiten.

Task 1: Config Class
=====================

Aufgabenstellung
-----------------

Ein deklaratives Datenmodell für Tensor-Kontraktionen: Enums für
Dimension-Typen, Execution-Strategien, Primitive- und Datentypen sowie
ein ``Config``-Dataclass, der eine konkrete Kontraktion vollständig
beschreibt.


Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - TODO
   * - Oliver Dietzel
     - TODO
