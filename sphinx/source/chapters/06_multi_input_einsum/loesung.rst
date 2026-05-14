.. _ch06_loesung:

########################################
Report: Multi-Input Einsum Contraction
########################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Dieses Kapitel dokumentiert unsere Lösung des sechsten Assignments:
*Multi-Input Einsum Contraction*. Aufbauend auf dem
``Config``/``Optimizer``-Interface aus Assignment 05 kontrahieren wir
zwei intermediäre Tensoren einer Light-Field-Tensor-Ring-Zerlegung
(``acspx, bspy -> abcyx``), erst mit ``torch.einsum`` als Referenz und
anschließend mit einem cuTile-Kernel, der aus einer optimierten
``Config`` abgeleitet wird.

*Die Bearbeitung folgt — dieses Dokument wird parallel zur
Implementierung gefüllt.*

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - *folgt*
   * - Oliver Dietzel
     - *folgt*
