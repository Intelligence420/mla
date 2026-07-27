.. _ch08:

##################
XDNA GEMM Kernel
##################

In diesem Assignment wird ein hand-geschriebener Tensor-Kernel für die
XDNA2 (AIE2P) Compute-Tile implementiert, der eine Matrixmultiplikation
``out += in0 @ in1`` mit ``M = N = 16`` und ``K = 64`` auf der NPU
ausführt. Die Eingaben liegen im L1-Scratchpad bereits in einem
getilten Layout (``in0: prmk``, ``in1: rqkn``, ``out: pqmn``) vor;
der Kernel nutzt den nativen BFP16-MAC der Vektor-Einheit und muss
ohne Schleifen-Kontrollfluss (außer dem finalen ``ret lr``) auskommen.

.. toctree::
   :maxdepth: 1

   aufgabe
   loesung
