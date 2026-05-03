.. _ch04_loesung:

############################################
Report: Tensor Contractions on GPUs
############################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Dieses Kapitel dokumentiert unsere Lösung des vierten Assignments:
*Tensor Contractions on GPUs*. Im Mittelpunkt stehen cuTile-Kernels für
allgemeine Tensor-Kontraktionen mit FP16-Inputs und FP32-Akkumulator –
mit Fokus auf Parallelisierungs-Strategien, Primitive-Size-Merging
und Kernel-Fusion.

Task 1: Tiled Contraction Kernel Variants
==========================================

Aufgabenstellung
-----------------

Implementiert werden vier Varianten eines cuTile-Kernels für die
Kontraktion

.. math::

   C_{eabcxz} = \sum_{k,l,y} A_{eabklxy} \cdot B_{ecklyz}

mit FP16-Inputs/Outputs und FP32-Akkumulator. Variiert wird, welche
Dimensionen als GEMM-Dimensionen (GEneral Matrix-Matrix multiplication) ans ``ct.mma`` gehen und welche
sequentialisiert bzw. parallelisiert werden. Verifikation gegen
``torch.einsum``; Benchmark mit ``triton.testing.do_bench``.

Die Aufgabe ist genau die *Tiled Contraction* aus den Vorlesungsfolien: statt das Tensor-Produkt per
*Transpose-Transpose-GEMM-Transpose* erst in eine
2D-Matmul zu reshapen, identifizieren wir Tiles direkt im
mehr-dimensionalen Tensor und lassen einen einzigen Kernel über die
restlichen Dimensionen schleifen oder das Grid aufspannen. Das spart
die TTGT-Permutationen im globalen Speicher (Folie 12, Cons:
*„Permutation in memory is expensive"*); im Gegenzug muss der Kernel
selbst die richtige Tile-Geometrie wählen – genau das ist der
Spielraum von b)/c)/d)/e).

Task 1a: Klassifikation der Dimensionen
----------------------------------------

Die einsum-Signatur ``eabklxy, ecklyz -> eabcxz`` lässt sich
nach den klassischen GEMM-/Tensor-Kontraktions-Rollen einordnen
(Folie 8, *„Index Types in Einsum Expressions"*: M = frei in A,
N = frei in B, K = kontrahiert, C/B = Batch in beiden Inputs und
im Output):

.. list-table::
   :header-rows: 1
   :widths: 20 15 30 35

   * - Index
     - Typ
     - Vorkommen
     - Rolle
   * - ``e``
     - **B (Batch)**
     - A, B, C
     - Externe/Batch-Dimension – in jedem Tensor identisch
   * - ``a``, ``b``, ``x``
     - **M**
     - A, C (nicht B)
     - Output-Zeilen-Dim, frei für Parallelisierung oder Tiling
   * - ``c``, ``z``
     - **N**
     - B, C (nicht A)
     - Output-Spalten-Dim, frei für Parallelisierung oder Tiling
   * - ``k``, ``l``, ``y``
     - **K**
     - A, B (nicht C)
     - Kontrahierte Dim – wird im Kernel akkumuliert

Damit gibt es **eine** Batch-Dim, **drei** M-Dims, **zwei** N-Dims und
**drei** K-Dims. Die folgenden Varianten unterscheiden sich darin,
welche dieser K- und M-Dims auf das ``ct.mma`` abgebildet werden und
welche im Kernel als Schleife laufen.

Task 1b: GEMM = (x, y, z), parallel (e, a, b, c)
-------------------------------------------------

*erste* Skizze: ``Decompose pid -> (abc)``,
äußere Schleifen ``for k, for l``, mma-Shape ``(x, z, y)`` – plus die
zusätzliche Batch-Dim ``e``, die wir mit in das Grid falten.

**Mapping**

* GEMM-Dimensionen: ``x`` (M), ``y`` (K), ``z`` (N)
* Sequentialisiert (Schleifen im Kernel): ``k``, ``l`` und Tiles
  entlang ``y``
* Parallelisiert (über das Grid): ``e``, ``a``, ``b``, ``c`` sowie
  Tiles entlang ``x`` und ``z``

Das Grid ist 3D:
``(E·A, B·C, ⌈X/tx⌉·⌈Z/tz⌉)``. Innerhalb des Kernels werden die
Block-IDs in die jeweiligen Indizes zerlegt:

.. code-block:: python

   bid_eA = ct.bid(0)
   bid_BC = ct.bid(1)
   bid_xz = ct.bid(2)

   pid_e = bid_eA // Ad
   pid_a = bid_eA %  Ad
   pid_b = bid_BC // Cd
   pid_c = bid_BC %  Cd
   pid_x = bid_xz // num_tiles_z
   pid_z = bid_xz %  num_tiles_z

**Kernel-Kern**

.. code-block:: python

   acc = ct.full((tx, tz), 0, dtype=ct.float32)
   for kk in range(K):
       for ll in range(L):
           for yy in range(num_tiles_y):
               a_tile = ct.load(A,
                   index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                   shape=(1, 1, 1, 1, 1, tx, ty),
                   padding_mode=ct.PaddingMode.ZERO)
               b_tile = ct.load(B,
                   index=(pid_e, pid_c, kk, ll, yy, pid_z),
                   shape=(1, 1, 1, 1, ty, tz),
                   padding_mode=ct.PaddingMode.ZERO)
               a2d = ct.reshape(a_tile, (tx, ty))
               b2d = ct.reshape(b_tile, (ty, tz))
               acc = ct.mma(a2d, b2d, acc)

   out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
   ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, pid_x, pid_z), tile=out)

Die Singleton-Dimensionen werden per ``ct.reshape`` weggedrückt; das
``ct.mma`` arbeitet auf reinen 2D-Tiles ``(tx, ty) × (ty, tz) → (tx, tz)``.
Out-of-bounds-Anteile (z. B. wenn ``Y`` kein Vielfaches von ``ty`` ist)
sind durch ``PaddingMode.ZERO`` neutral für die Akkumulation.

Task 1c: zusätzlich b sequentialisiert
---------------------------------------

*zweite* Skizze: ``Decompose pid -> (bc)``,
zusätzlich eine ``for a_it``-Schleife im Kernel. In unserer Notation
heißt die sequentialisierte Achse ``b`` statt ``a``, das Prinzip ist
dasselbe – eine M-Dim wandert vom Grid in eine innere Schleife.

**Mapping**

* GEMM-Dimensionen: ``x``, ``y``, ``z`` (wie b)
* Sequentialisiert: ``k``, ``l``, Tiles entlang ``y``, **zusätzlich
  ``b``**
* Parallelisiert: ``e``, ``a``, ``c`` sowie Tiles entlang ``x``, ``z``

Das Grid schrumpft um den Faktor ``|b|``: ``(E·A, C, ⌈X/tx⌉·⌈Z/tz⌉)``.
Jeder Block produziert nun ``|b|`` Output-Tiles (entlang ``b``)
sequentiell. Der Trick: das B-Tile ``B[e,c,k,l,y,z]`` hängt **nicht**
von ``b`` ab und wird daher zwischen den ``b``-Iterationen im L2
wiederverwendet.

**Kernel-Kern**

.. code-block:: python

   for bb in range(Bd):
       acc = ct.full((tx, tz), 0, dtype=ct.float32)
       for kk in range(K):
           for ll in range(L):
               for yy in range(num_tiles_y):
                   a_tile = ct.load(A,
                       index=(pid_e, pid_a, bb, kk, ll, pid_x, yy),
                       shape=(1, 1, 1, 1, 1, tx, ty),
                       padding_mode=ct.PaddingMode.ZERO)
                   b_tile = ct.load(B,
                       index=(pid_e, pid_c, kk, ll, yy, pid_z),
                       shape=(1, 1, 1, 1, ty, tz),
                       padding_mode=ct.PaddingMode.ZERO)
                   acc = ct.mma(ct.reshape(a_tile, (tx, ty)),
                                ct.reshape(b_tile, (ty, tz)), acc)
       out = ct.reshape(ct.astype(acc, C.dtype), (1, 1, 1, 1, tx, tz))
       ct.store(C, index=(pid_e, pid_a, bb, pid_c, pid_x, pid_z), tile=out)

**Vermutung & Tests**

Trade-off: c) gewinnt B-Tile-Reuse über die b-Schleife, verliert
aber den b-Faktor im Grid – c) vorne erwartet bei großem ``|b|``,
b) vorne bei kleinem ``|b|``. Getestet:

* ``|b|=2``, ``|a|·|c|=64`` → Erwartung b) vorne.
* ``|b|=16``, ``|a|·|c|=4`` → Erwartung c) vorne.

Task 1d: GEMM = (x, y·l, z), l und y gemerged
----------------------------------------------

*dritte* Skizze: ``# Matmul shape = (x, z, y * l)``
mit nur noch einer äußeren ``for k``-Schleife. Wir greifen das volle Merging
(Variante 4) hier nicht auf ``(x, z, y·l·k)``.

**Mapping**

* GEMM-Dimensionen: ``x`` (M), ``y·l`` (gemergte K), ``z`` (N)
* Sequentialisiert: ``k``, Tiles entlang ``y``
* Parallelisiert: ``e``, ``a``, ``b``, ``c`` sowie Tiles entlang ``x``, ``z``

Statt ``l`` als Schleife laufen zu lassen, packen wir alle ``L``
Slices direkt in einen einzigen ``ct.mma`` mit GEMM-K-Dim ``L·ty``.
Das hebt die *arithmetic intensity* pro mma-Aufruf, weil pro Mma-Issue
``L`` Mal mehr K-Werte verrechnet werden.

**Permute-Trick**

Die Crux: ``A`` hat Layout ``[..., L, X, Y]``, also liegt ``X`` *zwischen*
``L`` und ``Y``. Ein simples Reshape würde ``(L, X, Y)`` in ``(L·X·Y)``
flachklopfen – wir wollen aber ``(X, L·Y)``. Daher zuerst eine Permutation
``(L, X, Y) → (X, L, Y)``, dann Reshape:

.. code-block:: python

   tly = L * ty   # gemergte K-Dim Groesse pro mma

   for kk in range(K):
       for yy in range(num_tiles_y):
           # A: load (..., L, tx, ty) -> permute (tx, L, ty) -> reshape (tx, L*ty)
           a_tile = ct.load(A,
               index=(pid_e, pid_a, pid_b, kk, 0, pid_x, yy),
               shape=(1, 1, 1, 1, L, tx, ty),
               padding_mode=ct.PaddingMode.ZERO)
           a_tile = ct.reshape(a_tile, (L, tx, ty))
           a_tile = ct.permute(a_tile, (1, 0, 2))
           a_tile = ct.reshape(a_tile, (tx, tly))

           # B: hat schon Layout (..., L, Y, Z) -> direkt (L*ty, tz) reshapen
           b_tile = ct.load(B,
               index=(pid_e, pid_c, kk, 0, yy, pid_z),
               shape=(1, 1, 1, L, ty, tz),
               padding_mode=ct.PaddingMode.ZERO)
           b_tile = ct.reshape(b_tile, (tly, tz))

           acc = ct.mma(a_tile, b_tile, acc)

Auf B-Seite ist kein Permute nötig, weil die Reihenfolge ``L, Y, Z``
bereits zur gewünschten Mergung ``L·Y`` passt.

**Vermutung & Tests**

Trade-off: d) bündelt ``L`` Slices pro mma (mehr Arbeit pro Issue,
weniger Loop-Overhead), zahlt aber den Permute. d) vorne erwartet bei
großem ``|l|`` (vor allem mit kleinem ``|y|``), b) vorne bei
``|l|=1`` (Permute wird reiner Overhead). Getestet:

* ``|l|=8``, ``|y|=32`` → Erwartung d) vorne.
* ``|l|=1``, ``|y|=64`` → Erwartung b) vorne.

Task 1e: GEMM = (e, x, y, z) als 3D-mma
----------------------------------------

Statt ``e`` (die einzige
Batch-/C-Index-Dim) als Grid-Achse zu nutzen, wandert
sie in den ``ct.mma`` selbst. Konzeptuell entspricht das einer
*Batched Matrix Multiplication* (Folie 3–5): mehrere kleine GEMMs
laufen entlang ``e`` parallel, jetzt aber innerhalb eines einzelnen
mma-Aufrufs statt über das Grid verteilt.

**Mapping**

* GEMM-Dimensionen: ``e`` (Batch des mma), ``x``, ``y``, ``z``
* Sequentialisiert: ``k``, ``l``, Tiles entlang ``y``
* Parallelisiert: ``a``, ``b``, ``c`` sowie Tiles entlang ``e``, ``x``, ``z``

Batch-Dim direkt über mehrere SM-Lanes / Mma-Instructions abgedeckt, statt ``e``
in den Grid-Index zu falten. Akkumulator ist 3D:

.. code-block:: python

   acc = ct.full((te, tx, tz), 0, dtype=ct.float32)
   for kk in range(K):
       for ll in range(L):
           for yy in range(num_tiles_y):
               a_tile = ct.load(A,
                   index=(pid_e, pid_a, pid_b, kk, ll, pid_x, yy),
                   shape=(te, 1, 1, 1, 1, tx, ty),
                   padding_mode=ct.PaddingMode.ZERO)
               b_tile = ct.load(B,
                   index=(pid_e, pid_c, kk, ll, yy, pid_z),
                   shape=(te, 1, 1, 1, ty, tz),
                   padding_mode=ct.PaddingMode.ZERO)
               acc = ct.mma(ct.reshape(a_tile, (te, tx, ty)),
                            ct.reshape(b_tile, (te, ty, tz)), acc)

Bei ``te = |e|`` deckt ein einziger Block die gesamte Batch-Dim ab;
das Grid wird entsprechend kürzer.

Verifikation
------------

Alle vier Varianten werden gegen ``torch.einsum`` mit FP32-Promotion
und Rückgabe in FP16 verglichen (``atol=2e-1, rtol=2e-2``):

.. code-block:: text

   Task 1 b)/c)/d)/e): Verifikation gegen torch.einsum
     Shapes: A=(2, 2, 2, 2, 2, 64, 64), B=(2, 2, 2, 2, 64, 64), ref=(2, 2, 2, 2, 64, 64)
     kernel_b    allclose=True   max_abs_err=0.0312
     kernel_c    allclose=True   max_abs_err=0.0312
     kernel_d    allclose=True   max_abs_err=0.0312
     kernel_e    allclose=True   max_abs_err=0.0312

Alle vier Kernels liefern denselben ``max_abs_err`` (FP16-Quantisierungs-
rauschen), also numerisch äquivalente Ergebnisse.

Benchmark-Ergebnisse
--------------------

Gemessen mit ``triton.testing.do_bench`` auf der DGX Spark (GB10), FP16-Inputs,
FP32-Akkumulator, GEMM-Tile ``(32, 32, 32)``:

.. list-table:: b) vs c)
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Konfiguration
     - Variante
     - ms
     - TFLOPS
     - Schneller
   * - ``|b|=2, |a|·|c|=64, X=Y=Z=128`` (FLOPs ≈ 2.15·10⁹)
     - b)
     - 0.524
     - **4.10**
     - **b) (2.0×)**
   * -
     - c)
     - 1.046
     - 2.05
     -
   * - ``|b|=16, |a|·|c|=4, X=Y=Z=128`` (FLOPs ≈ 5.37·10⁸)
     - b)
     - 0.161
     - **3.35**
     - **b) (2.3×)**
   * -
     - c)
     - 0.362
     - 1.48
     -

.. list-table:: b) vs d)
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Konfiguration
     - Variante
     - ms
     - TFLOPS
     - Schneller
   * - ``|l|=8, |y|=32, X=Z=128`` (FLOPs ≈ 1.07·10⁹)
     - b)
     - 0.500
     - 2.15
     - **d) (1.30×)**
   * -
     - d)
     - 0.386
     - **2.78**
     -
   * - ``|l|=1, |y|=64, X=Z=128`` (FLOPs ≈ 1.07·10⁹)
     - b)
     - 0.528
     - **2.03**
     - b) (≈)
   * -
     - d)
     - 0.538
     - 2.00
     -

.. list-table:: Quervergleich b) / d) / e), ``|e|=4``
   :header-rows: 1
   :widths: 25 15 15 15

   * - Variante
     - ms
     - TFLOPS
     - vs b)
   * - b)
     - 0.096
     - 2.79
     - 1.00×
   * - d)
     - 0.082
     - **3.27**
     - 1.17×
   * - e)
     - 0.203
     - 1.32
     - 0.47×

.. figure:: ../../../../assignments/04_assignment/src/task01_bc_vs_bd.png
   :align: center
   :alt: Vergleich b) vs c) und b) vs d), TFLOPS pro Konfiguration
   :width: 100%

   Vier Panels: links die zwei b)-vs-c)-Settings, rechts die zwei
   b)-vs-d)-Settings. Pro Konfiguration zeigt der höhere Balken die
   schnellere Variante.

.. figure:: ../../../../assignments/04_assignment/src/task01_e_compare.png
   :align: center
   :alt: Quervergleich b)/d)/e) — Laufzeit und Durchsatz
   :width: 90%

   Quervergleich der drei Varianten b), d) und e) auf einer mittleren
   Konfiguration mit ``|e| = 4``.

Beobachtungen und Vermutungen
------------------------------

* **b) vs c)**: b) gewinnt in *beiden* Konfigurationen – bei
  ``|b|=16`` läuft das Gegenteil der Erwartung (3.35 vs 1.48 TFLOPS).
  Vermutung: der L2-Reuse-Gewinn wird vom Occupancy-Verlust
  überkompensiert. 
* **b) vs d)**: wie erwartet. d) mit ~30 % vorne bei ``|l|=8``,
  Vorteil verschwindet bei ``|l|=1``. So wie wir es in VL angesprochen hatten: mehr
  K-Merging → größere innere GEMM-K → bessere Tensor-Core-Auslastung,
  aber nur wenn die gemergte Dim ``>1`` ist.
* **b/d/e**: e) ist ~2× langsamer als d). Vermutung: ``te=2`` mit
  ``|e|=4`` halbiert das Grid und bläht den Akkumulator auf
  ``(te, tx, tz)`` – Occupancy sinkt

Task 2: Kernel Fusion
======================

Aufgabenstellung
-----------------


Task 3: GEMM Dimension Size Sweep
==================================

Aufgabenstellung
-----------------


Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Implementierung Task 1 (Dimensions-Klassifikation, vier Kernel-Varianten
       mit Verifikation gegen ``torch.einsum`` und Vergleichs-Benchmarks
       b/c bzw. b/d), Sphinx-Report-Abschnitt zu Task 1
   * - Oliver Dietzel
     - TODO
