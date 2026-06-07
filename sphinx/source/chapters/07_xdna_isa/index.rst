.. _ch07:

#################################
Inferring the VLIW ISA of XDNA2
#################################

In diesem Assignment werden zentrale Eigenschaften der VLIW-
Instruction-Set-Architecture (ISA) des XDNA2- (AIE2P-) Compute-Tiles
durch Reverse-Engineering ermittelt. Zwei einfache AIE-API-Kernels
werden mit dem Peano-Compiler übersetzt; aus dem generierten Assembly
werden Slots, Register-Klassen und Operations-Latenzen abgeleitet und
anschließend ein hand-scheduled BF16-Vektor-Add-Kernel
(``C = A + B + B``) geschrieben.

.. toctree::
   :maxdepth: 1

   aufgabe
   loesung
   feedback
