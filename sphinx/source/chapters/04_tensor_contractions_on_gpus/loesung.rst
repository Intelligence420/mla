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
Dimensionen als GEMM-Dimensionen ans ``ct.mma`` gehen und welche
sequentialisiert bzw. parallelisiert werden. Verifikation gegen
``torch.einsum``; Benchmark mit ``triton.testing.do_bench``.

Task 1a: Klassifikation der Dimensionen
----------------------------------------

Die einsum-Signatur ``eabklxy, ecklyz -> eabcxz`` lässt sich
nach den klassischen GEMM-/Tensor-Kontraktions-Rollen einordnen:

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

**Wann ist b) besser, wann c)?**

* ``b)`` gewinnt, wenn ``|b|`` klein ist (kaum Reuse-Potenzial) und
  ``|e|·|a|·|c|`` allein nicht reicht, um die SMs zu sättigen –
  hier hilft die zusätzliche b-Parallelität in ``b)`` direkt.
* ``c)`` gewinnt, wenn ``|b|`` groß ist (viel Reuse) und ``|e|·|a|·|c|``
  bereits genug Grid-Blöcke liefert – die b-Schleife im Kernel
  amortisiert dann das ein-mal-laden des B-Tiles über mehrere
  Output-Tiles und reduziert den globalen Speicherverkehr.

Konkret im Benchmark:

* **``b)`` vorne**: ``|b|=2``, ``|a|·|c|=64``, GEMM-Dims groß.
* **``c)`` vorne**: ``|b|=16``, ``|a|·|c|=4`` – c) hat hier mehr
  Reuse pro Block und das Grid in b) ist ohnehin groß genug, sodass
  die kleinere Grid-Größe von c) keine Occupancy-Probleme erzeugt.

Task 1d: GEMM = (x, y·l, z), l und y gemerged
----------------------------------------------

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

**Wann ist b) besser, wann d)?**

* ``d)`` gewinnt, wenn ``|l|`` groß ist – jeder mma macht dann
  ``L``-mal so viel Arbeit, ohne dass ``L`` zusätzliche Loop-Iterationen
  und ``ct.load``-Aufrufe nötig wären. Insbesondere bei kleinem ``|y|``
  (sonst dominiert die y-Schleife schon) zahlt sich der Merge aus.
* ``b)`` gewinnt, wenn ``|l| = 1`` (oder sehr klein): der Permute kostet
  Register-Bewegung, der gemergte mma ist nicht größer als der
  ungemergte, und b) spart sich den Reshape/Permute komplett.

Konkret im Benchmark:

* **``d)`` vorne**: ``|l|=8``, ``|y|=32`` – d) bündelt 8 Slices in
  einen mma; bei kleiner y-Tile-Größe hat b) sehr viele kurze
  K-Iterationen.
* **``b)`` vorne**: ``|l|=1``, ``|y|=64`` – kein Mergung möglich, der
  Permute in d) ist reiner Overhead.

Task 1e: GEMM = (e, x, y, z) als 3D-mma
----------------------------------------

**Mapping**

* GEMM-Dimensionen: ``e`` (Batch des mma), ``x``, ``y``, ``z``
* Sequentialisiert: ``k``, ``l``, Tiles entlang ``y``
* Parallelisiert: ``a``, ``b``, ``c`` sowie Tiles entlang ``e``, ``x``, ``z``

cuTile-``mma`` unterstützt 3D-Operanden, sodass die Batch-Dim direkt
über mehrere SM-Lanes / Mma-Instructions abgedeckt wird, statt ``e``
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

   Task 1a: Dimension-Klassifikation
     e            -> Batch (B-Typ)
     a, b, x      -> M-Typ
     c, z         -> N-Typ
     k, l, y      -> K-Typ (kontrahiert)

   Task 1 b)/c)/d)/e): Verifikation gegen torch.einsum
     Shapes: A=(2,2,2,2,2,64,64), B=(2,2,2,2,64,64), ref=(2,2,2,2,64,64)
     kernel_b   allclose=True   max_abs_err=...
     kernel_c   allclose=True   max_abs_err=...
     kernel_d   allclose=True   max_abs_err=...
     kernel_e   allclose=True   max_abs_err=...

Benchmark-Ergebnisse
--------------------

Die folgenden zwei Plots vergleichen die Varianten in den jeweils
gewählten Konfigurationen:

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

Erkenntnisse
------------

* **GEMM-Wahl folgt Datenfluss, nicht Buchstaben**: ``x`` ist nominell
  eine "frei wählbare" M-Dim, sitzt aber innen in A's Layout
  ``[..., L, X, Y]`` und ist daher ideal als M-Dim des mma. Eine
  alternative Wahl wie GEMM=(a, …, c) würde stride-feindlich laden
  und die Tile-Loads zerschneiden.
* **Sequentialisierte K-Dims kosten Loop-Overhead, nicht Bandwidth**:
  jede zusätzliche sequentielle K-Schleife (Variante b: drei verschachtelte
  Schleifen über ``k, l, y_tiles``) bedeutet pro Iteration einen ``ct.load``
  von A und B und ein ``ct.mma``. Bei kleinem GEMM-Tile dominiert dieser
  Loop-Overhead schnell – der Trick aus d) (``l`` in den mma falten) ist
  genau die Antwort darauf.
* **Sequentialisierung von M-Dims (Variante c) zahlt sich nur aus, wenn
  reused Tiles nicht schon über das Grid und L2 abgedeckt sind**: c)
  drückt das Grid kleiner und macht jeden Block länger, was nur dann
  hilft, wenn ``|b|`` groß genug ist, um die ein B-Tile-Lade-Investition
  über mehrere Output-Tiles zu amortisieren.
* **3D-mma in e) ist kein Gratis-Speedup**: der zusätzliche Output-Slot
  ``te`` braucht Akkumulator-Register, was die Occupancy senken kann.
  Sinnvoll, wenn ``|e|`` so klein ist, dass es als Grid-Achse wenig
  Auslastung bringt – dann lohnt sich, ``e`` als mma-Batch
  einzubauen statt in ``ct.bid``.

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
