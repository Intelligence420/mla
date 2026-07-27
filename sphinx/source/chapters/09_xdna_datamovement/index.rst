.. _ch09:

####################
XDNA Data Movement
####################

In diesem Assignment wird eine größere Matrixmultiplikation
``out += in0 @ in1`` mit ``M = 256``, ``N = 128`` und ``K = 1024``
(BF16) auf der XDNA2-(AIE2P-)NPU ausgeführt. Anders als in
Assignment 08 liegt der Schwerpunkt nicht auf dem Tensor-Kernel,
sondern auf dem **Datenfluss**: Die Matrizen liegen im Hauptspeicher
(L3) als Row-Major-Matrizen und müssen getilt und umsortiert über
Shim- und Memory-Tile in das L1-Scratchpad des Compute-Tiles bewegt
werden. Dazu wird Datenbewegungs-Code im *MLIR-AIE*-Dialekt
geschrieben und es werden ``a``/``b``/``c``-Schleifen um den
XDNA-Tensor-Kernel gelegt.

.. toctree::
   :maxdepth: 1

   aufgabe
   loesung
