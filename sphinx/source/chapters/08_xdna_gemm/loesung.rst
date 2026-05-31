.. _ch08_loesung:

#############################
Lösung und Bearbeitung
#############################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========



Task 1: Verify-Funktion
=======================



Task 2: Instruktionen und Latenzen
==================================


Task 3: Register-Blocking
=========================


Task 4: Data-Layouts und Pointer-Updates
========================================

L1-Speicher-Layout
-------------------

Layout und Tile-Größen sind in der Aufgabenstellung (A08, Abschnitt
*Data Layout and Data Movement*) festgelegt: ``in0`` wird im L1 als
``prmk``, ``in1`` als ``rqkn`` und ``out`` als ``pqmn`` abgelegt
(jeweils mit ``p=q=2``, ``m=n=k=8``, ``r=8``, dtype BF16, 2 Byte pro
Element, Row-Major). Daraus berechnen sich die Strides direkt:

.. list-table::
   :header-rows: 1
   :widths: 12 18 18 18 18 16

   * - Tensor
     - Stride p / r
     - Stride q
     - Stride r / k
     - Stride m / k
     - Gesamtgröße
   * - ``in0`` (``prmk``)
     - p = 1024 B
     - —
     - r = 128 B
     - m = 16 B, k = 2 B
     - 2048 B
   * - ``in1`` (``rqkn``)
     - r = 256 B
     - q = 128 B
     - k = 16 B
     - n = 2 B
     - 2048 B
   * - ``out`` (``pqmn``)
     - p = 256 B
     - q = 128 B
     - —
     - m = 16 B, n = 2 B
     - 512 B

Ein 8×8-Block hat 64 BF16-Elemente = 128 Byte und füllt — als FP32
akkumuliert — genau einen vollen DM-Akku, der laut Folie 3 von
``08_xdna_isa_and_gemm_kernels.pdf`` 2048 bit groß ist (64 × 32 bit).
Die Lade-Instruktion ``vlda.conv.fp32.bf16`` schreibt jedoch immer
nur eine CM-Hälfte (1024 bit = 32 FP32 = 32 BF16 Eingang = 64 Byte
im L1) — das zeigt das Schedule aus Assignment 07
(``custom_vadd.s`` Z. 9–10):

.. code-block:: asm

   vlda.conv.fp32.bf16  cml0, [p0, #0]      ; A07: low half  -> dm0.lo
   vlda.conv.fp32.bf16  cmh0, [p0, #64]     ; A07: high half -> dm0.hi

Daher benötigt jeder 8×8-Block **zwei** Halb-Loads an Offsets ``#0``
und ``#64``.

Loop-Reihenfolge (vollständig entrollt)
----------------------------------------

Die Aufgabenstellung (A08 Task 5, Punkt 1) verbietet jede
Control-Flow-Instruktion außer dem finalen ``ret lr`` — der Kernel
ist daher vollständig entrollt. Die konzeptionelle Schleifenstruktur
lautet:

.. code-block::

   for r = 0 .. 7:
       load  in0[p=0, r]  ->  ex0          (via dm4 staging)
       load  in0[p=1, r]  ->  ex1          (via dm4 staging)
       load  in1[r, q=0]  ->  ex2          (via dm4 staging)
       load  in1[r, q=1]  ->  ex3          (via dm4 staging)
       vmac  dm0 += ex0 · ex2              (out[0,0])
       vmac  dm1 += ex0 · ex3              (out[0,1])
       vmac  dm2 += ex1 · ex2              (out[1,0])
       vmac  dm3 += ex1 · ex3              (out[1,1])

In dieser Reihenfolge wird jeder geladene BFP16-Block für **zwei**
MACs verwendet: ex0 (= ``in0[0,r]``) speist die MACs für ``out[0,0]``
*und* ``out[0,1]``; symmetrisch ex2 (= ``in1[r,0]``) speist
``out[0,0]`` *und* ``out[1,0]``. Das folgt direkt aus der GEMM-
Definition ``out[p,q] = Σ_r in0[p,r] · in1[r,q]`` — bei zwei p- und
zwei q-Werten teilt sich jeder Operand mit zwei MACs.

Pointer-Belegung und -Updates
------------------------------

Wir benutzen vier der acht Pointer-Register ``P0–P7`` (Folie 3 von
``08_xdna_isa_and_gemm_kernels.pdf``); alle Updates laufen als
Post-Increment im jeweils letzten Halb-Load eines Blocks — die
Post-Increment-Syntax ``vlda.conv.fp32.bf16 cmh2, [p0], #256`` ist
in Folie 6 derselben Datei dokumentiert.

.. list-table::
   :header-rows: 1
   :widths: 15 30 35 20

   * - Pointer
     - Tensor / Slice
     - Startadresse
     - Inkrement pro r-Schritt
   * - ``p0``
     - ``in0[p=0, *]``
     - ``&in0`` + 0
     - **+128 B** (ein r-Block)
   * - ``p1``
     - ``in1`` (alle r, beide q)
     - ``&in1`` + 0
     - **+256 B** (ein r-Block × 2 q)
   * - ``p2``
     - ``out``
     - ``&out`` + 0
     - statisch; nur Init-Loads und finale Stores
   * - ``p3``
     - ``in0[p=1, *]``
     - ``&in0`` + 1024
     - **+128 B** (ein r-Block)

Pro r-Schritt sind dies die acht Halb-Loads (Schema, ohne Schedule):

.. code-block:: asm

   ; in0[0,r] -> dm4 -> ex0
   vlda.conv.fp32.bf16  cml4, [p0, #0]
   vlda.conv.fp32.bf16  cmh4, [p0], #128       ; post-inc +128
   vconv.bfp16ebs8.fp32 ex0, dm4

   ; in0[1,r] -> dm4 -> ex1
   vlda.conv.fp32.bf16  cml4, [p3, #0]
   vlda.conv.fp32.bf16  cmh4, [p3], #128
   vconv.bfp16ebs8.fp32 ex1, dm4

   ; in1[r,0] -> dm4 -> ex2
   vlda.conv.fp32.bf16  cml4, [p1, #0]
   vlda.conv.fp32.bf16  cmh4, [p1, #64]        ; bleibt innerhalb r-Block
   vconv.bfp16ebs8.fp32 ex2, dm4

   ; in1[r,1] -> dm4 -> ex3 (mit Block-Increment)
   vlda.conv.fp32.bf16  cml4, [p1, #128]
   vlda.conv.fp32.bf16  cmh4, [p1], #256       ; post-inc +256 (ganzer r-Slab)
   vconv.bfp16ebs8.fp32 ex3, dm4

Damit erreicht jeder Pointer am Ende jedes r-Schritts den Anfang des
nächsten r-Blocks ohne separate Pointer-Arithmetik im ``X``-Slot.

Out-Initialisierung und finaler Store
---------------------------------------

Die Akkus ``dm0..dm3`` werden über den GEMM-Pfad aus Folie 11
(``08_xdna_isa_and_gemm_kernels.pdf``, Schritt 1: *Load BF16 output
and convert to FP32*) initialisiert: Der L1-Scratchpad ist laut
Aufgabenstellung (A08, *Data Layout and Data Movement*, letzter
Satz) bei NPU-Setup auf null initialisiert; 8 ``vlda.conv``-Halves
(je 2 pro Out-Tile) laden FP32-Nullen in die vier Akku-Register,
bevor die r-Schleife beginnt.

Am Ende speichern 8 ``vst.conv.bf16.fp32``-Halves die vier Ergebnis-
Tiles zurück nach L1 (Offsets ``#0/#64/#128/#192/#256/#320/#384/#448``
ab ``p2``).

Task 5: Implementierung
=======================


Task 6: Performance
===================


Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     -
   * - Oliver Dietzel
     -
