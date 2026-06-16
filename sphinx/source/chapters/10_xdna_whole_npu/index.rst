.. _ch10:

###################
Using the whole NPU
###################

In diesem Assignment wird dieselbe Matrixmultiplikation
``out += in0 @ in1`` (``M = 256``, ``N = 128``, ``K = 1024``, BF16) auf
der **gesamten** XDNA2-(AIE2P-)NPU ausgeführt. Aufbauend auf
Assignment 09 wird der Datenbewegungs-Code so erweitert, dass der
XDNA-Tensor-Kernel auf **allen** Compute-Tiles läuft: Die Dimensionen
``x`` und ``y`` werden **räumlich** über die Spalten bzw. Zeilen des
Compute-Tile-Arrays verteilt, während ``a``, ``b`` und ``c`` weiterhin
sequentiell über Schleifen abgearbeitet werden. Die ``in0``-Tiles
werden dabei entlang der Spalten und die ``in1``-Tiles entlang der
Zeilen gebroadcastet.

.. toctree::
   :maxdepth: 1

   aufgabe
   loesung
