.. _ch03_loesung:

#############################################
Report: Matrix Multiplication with cuTile
#############################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Dieses Kapitel dokumentiert unsere Lösung des dritten Assignments:
*Matrix Multiplication with cuTile*. Aufbauend auf einem einfachen
``ct.mma``-basierten Matmul-Kernel werden numerische Präzision,
Tile-Größen und das Block-Mapping (row-major vs. swizzled) variiert,
um deren Einfluss auf die Performance zu untersuchen.

Task 1: FP32 vs FP16 Performance
================================

Aufgabenstellung
----------------

Implementierung von zwei cuTile-Kernels, die je eine Matrixmultiplikation
:math:`A \times B = C` mit ``shape(A) = (64, 4096)``, ``shape(B) = (4096, 64)``
und ``shape(C) = (64, 64)`` berechnen. Tile-Shape fixiert auf
``(m_tile=64, n_tile=64, k_tile=64)``, ein einzelner CTA pro Launch.
Variante 1 verwendet FP16-Inputs mit FP32-Akkumulator,
Variante 2 reine FP32. Verifikation gegen ``torch.matmul``,
Benchmark via ``triton.testing.do_bench``.

.. note::

   Implementierung und Auswertung erfolgen durch Moritz Martin – dieser
   Abschnitt wird im Zuge des Reports von ihm ergänzt.

Task 2: Simple Matrix Multiplication Kernel
===========================================

Aufgabenstellung
----------------

Allgemeiner cuTile-Matmul-Kernel für beliebige ``(M, K) @ (K, N)`` mit
nutzerdefinierten Tile-Größen. Das Block-Mapping erfolgt row-major:
BID 0 erzeugt das Tile oben links, BID 1 das Tile rechts daneben usw.
Der Kernel muss auch Shapes unterstützen, die keine Zweierpotenzen sind.
``ct.mma`` ist für die innere Akkumulation zu verwenden, Verifikation
gegen ``torch.matmul``.

.. note::

   Implementierung und Auswertung erfolgen durch Moritz Martin – dieser
   Abschnitt wird im Zuge des Reports von ihm ergänzt.

Task 3: Benchmarking the Matrix Multiplication Kernel
======================================================

Aufgabenstellung
----------------

Benchmark des Matmul-Kernels aus Task 2. Performance wird in TFLOPS angegeben
mit

.. math::

   \text{TFLOPS} = \frac{2 \cdot M \cdot N \cdot K}{t_s \cdot 10^{12}}.

Teil **a)** misst den Kernel mit Tile ``(64, 64, 64)`` für quadratische
Matmuls :math:`M = N = K \in \{256, 512, 1024, 2048, 4096, 8192\}` und
plottet die TFLOPS. Teil **b)** fixiert die Matrixgröße auf
:math:`2048^3` und :math:`512^3` und sweept alle 27 Kombinationen
:math:`m_{tile}, n_{tile}, k_{tile} \in \{32, 64, 128\}`. Visualisiert
wird als Heatmap mit fixiertem ``k_tile = 64``, die beste Kombination wird
berichtet.

Implementierung
---------------

**Matmul-Kernel (row-major BID-Mapping)**

Jedes Programm berechnet ein Output-Tile :math:`(t_m, t_n)`. Aus der 1D-``bid``
ergeben sich die 2D-Tile-Indizes über Division/Modulo nach
``num_bid_n``. In der K-Schleife wird je ein A-Tile :math:`(t_m, t_k)`
und ein B-Tile :math:`(t_k, t_n)` mit ``ct.PaddingMode.ZERO`` geladen
(deshalb funktionieren auch nicht-Zweierpotenz-Shapes), per ``ct.mma``
auf einen FP32-Akkumulator akkumuliert, am Ende auf den Output-Dtype
gecastet und zurückgeschrieben.

.. code-block:: python

   @ct.kernel
   def matmul_kernel(A, B, C,
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):
       bid = ct.bid(0)

       M = A.shape[0]
       N = B.shape[1]

       # row-major Mapping aus 1D-bid
       num_bid_n = ct.cdiv(N, tn)
       bidx = bid // num_bid_n          # Zeilen-Index des Tiles
       bidy = bid % num_bid_n           # Spalten-Index des Tiles

       # Anzahl K-Tiles entlang axis=1 von A
       num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

       # FP32-Akkumulator, FP32-Inputs als tf32 für Tensor Cores
       accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
       zero_pad = ct.PaddingMode.ZERO
       dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

       for k in range(num_tiles_k):
           a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                       padding_mode=zero_pad).astype(dtype)
           b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                       padding_mode=zero_pad).astype(dtype)
           accumulator = ct.mma(a, b, accumulator)

       accumulator = ct.astype(accumulator, C.dtype)
       ct.store(C, index=(bidx, bidy), tile=accumulator)

**Host-Funktion**

.. code-block:: python

   def cutile_matmul(A, B, tm, tn, tk):
       M, K = A.shape
       _, N = B.shape
       C = torch.empty((M, N), device=A.device, dtype=A.dtype)
       grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, matmul_kernel, (A, B, C, tm, tn, tk))
       return C

**TFLOPS-Helfer**

.. code-block:: python

   def tflops(M, N, K, time_ms):
       return (2.0 * M * N * K) / (time_ms / 1000.0 * 1e12)

Verifikation
------------

Korrektheit gegen ``torch.matmul`` mit absichtlich nicht-Zweierpotenz-Shapes:

.. code-block:: python

   M, K, N = 257, 513, 129
   A = torch.randn(M, K, dtype=torch.float16, device="cuda")
   B = torch.randn(K, N, dtype=torch.float16, device="cuda")
   C = cutile_matmul(A, B, tm=64, tn=64, tk=64)
   assert torch.allclose(C, torch.matmul(A, B), atol=1e-1, rtol=1e-1)

Output:

.. code-block:: text

   Task 3: Verifikation
     Matmul-Kernel verifiziert.

Erkenntnisse: Task 3a – Quadratischer Sweep
--------------------------------------------

Tile ``(64, 64, 64)``, FP16-Inputs, FP32-Akkumulator:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - M = N = K
     - Laufzeit (ms)
     - TFLOPS
   * - 256
     - 0,0111
     - 3,02
   * - 512
     - 0,0275
     - 9,75
   * - 1024
     - 0,0746
     - 28,79
   * - 2048
     - 0,3687
     - 46,59
   * - 4096
     - 9,2538
     - 14,85
   * - 8192
     - 76,1466
     - 14,44

.. figure:: ../../../../assignments/03_assignment/task03a_tflops.png
   :align: center
   :alt: TFLOPS-Verlauf für quadratische Matmuls

   Erreichte TFLOPS des row-major Kernels für quadratische Matmuls
   mit Tile ``(64, 64, 64)``.

**Beobachtungen:**

* Bis :math:`N = 2048` skaliert die Performance erwartungsgemäß: kleinere
  Größen (256, 512) sind launch- bzw. latenzgebunden, mit wachsendem
  :math:`N` werden die ~108 SMs der DGX Spark immer besser ausgelastet.
* Maximum bei :math:`N = 2048` mit ≈ 47 TFLOPS – hier passen die
  wiederverwendeten A/B-Tiles noch gut in den 24 MB L2-Cache.
* Ab :math:`N = 4096` bricht die Performance auf ≈ 15 TFLOPS ein: der
  Working-Set übersteigt den L2 deutlich, und das row-major BID-Mapping
  produziert nun viele L2-Misses, da nebeneinander laufende Blöcke
  unterschiedliche A-Tile-Streifen anfordern. Genau dieses Problem
  adressiert Task 4 mit Block-Swizzling.

Erkenntnisse: Task 3b – Tile-Shape-Sweep
-----------------------------------------

Auszüge aus dem 27-Kombinationen-Sweep:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 25 30

   * - tm
     - tn
     - tk
     - Laufzeit (ms) :math:`2048^3`
     - TFLOPS
   * - 128
     - 128
     - 64
     - 0,3050
     - **56,33**
   * - 128
     - 64
     - 128
     - 0,3405
     - 50,46
   * - 32
     - 128
     - 32
     - 0,4142
     - 41,47
   * - 64
     - 32
     - 32
     - 2,0745
     - 8,28

.. figure:: ../../../../assignments/03_assignment/task03b_heatmap_2048.png
   :align: center
   :alt: Heatmap Tile-Shapes 2048^3

   Heatmap der TFLOPS für :math:`2048^3` (k_tile = 64, row-major).

.. figure:: ../../../../assignments/03_assignment/task03b_heatmap_512.png
   :align: center
   :alt: Heatmap Tile-Shapes 512^3

   Heatmap der TFLOPS für :math:`512^3` (k_tile = 64, row-major).

**Beste Kombinationen:**

* :math:`2048^3`: ``tm=128, tn=128, tk=64`` mit **56,33 TFLOPS**
* :math:`512^3` : ``tm=128, tn=64,  tk=32`` mit **11,49 TFLOPS**

**Beobachtungen:**

* Größere Tiles erhöhen den Anteil der Rechenzeit gegenüber Speicherzugriffen
  und nutzen Tensor Cores besser aus – allerdings nur bis Shared Memory
  und Register das mitmachen. ``(128, 128, 128)`` fällt auf :math:`2048^3`
  bereits wieder auf 21 TFLOPS zurück, was auf erhöhten Register-Druck
  bzw. weniger Occupancy hindeutet.
* Bei :math:`512^3` dominieren Launch-Overhead und niedrige Block-Anzahl;
  hier ist das beste Tile kleiner.
* Asymmetrische Kombinationen wie ``(64, 32, *)`` schneiden durchgehend
  schlecht ab, weil nur sehr schmale B-Tiles geladen werden und der
  Tensor-Core-Durchsatz nicht ausgeschöpft wird.

Task 4: L2 Cache Optimization via Block Swizzling
==================================================

Aufgabenstellung
----------------

Wie Task 2, aber das BID-Mapping wird so umgestellt, dass benachbarte
Blöcke räumlich naheliegende A/B-Tiles laden – das maximiert die
L2-Wiederverwendung. Es darf eine Kontraktionsdimension von
:math:`K = 4096` angenommen werden. In Teil **b)** wird derselbe
Tile-Shape-Sweep wie in Task 3b durchgeführt; abschließend werden
beide Kernels (row-major und swizzled) für eine Matmul der Form
:math:`8192 \times 8192 \times 4096` direkt verglichen.

Mapping: Super-Grouping
------------------------

Das gewählte Mapping ist eine Super-Grouping-Variante:

* Blöcke werden in Gruppen der Höhe ``GROUP_SIZE_M = 8`` Tile-Zeilen
  organisiert. Innerhalb einer Gruppe läuft der lineare ``bid``
  *column-major*: zuerst alle 8 Zeilen der ersten Spalte, dann der
  zweiten, usw.
* **Effekt:** ``GROUP_SIZE_M`` nebeneinander aktive Blöcke teilen sich
  denselben B-Tile-Streifen entlang K. Beim Wechsel auf die nächste
  Spalte sind die zuletzt benutzten A-Tile-Streifen noch im L2.
* Bei :math:`K = 4096` durchläuft jeder Block viele K-Iterationen –
  jede zusätzliche L2-Hit-Rate wirkt sich daher direkt auf die
  Laufzeit aus.
* Randstaendige Gruppen mit weniger als 8 Zeilen werden über
  ``min(num_bid_m - first_bid_m, GROUP_SIZE_M)`` korrekt behandelt.

Implementierung
---------------

**Swizzled Matmul-Kernel**

.. code-block:: python

   GROUP_SIZE_M = 8

   @ct.kernel
   def matmul_swizzled_kernel(A, B, C,
                              tm: ct.Constant[int],
                              tn: ct.Constant[int],
                              tk: ct.Constant[int],
                              group_size_m: ct.Constant[int]):
       bid = ct.bid(0)

       M = A.shape[0]
       N = B.shape[1]

       num_bid_m = ct.cdiv(M, tm)
       num_bid_n = ct.cdiv(N, tn)

       # Super-Grouping: alle Blöcke einer Gruppe teilen sich einen A-Streifen
       num_bid_in_group = group_size_m * num_bid_n
       group_id = bid // num_bid_in_group
       first_bid_m = group_id * group_size_m
       cur_group_size_m = min(num_bid_m - first_bid_m, group_size_m)

       bidx = first_bid_m + (bid % cur_group_size_m)
       bidy = (bid % num_bid_in_group) // cur_group_size_m

       num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
       accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
       zero_pad = ct.PaddingMode.ZERO
       dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

       for k in range(num_tiles_k):
           a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                       padding_mode=zero_pad).astype(dtype)
           b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                       padding_mode=zero_pad).astype(dtype)
           accumulator = ct.mma(a, b, accumulator)

       accumulator = ct.astype(accumulator, C.dtype)
       ct.store(C, index=(bidx, bidy), tile=accumulator)

**Host-Funktion**

.. code-block:: python

   def cutile_matmul_swizzled(A, B, tm, tn, tk, group_size_m=GROUP_SIZE_M):
       M, K = A.shape
       _, N = B.shape
       C = torch.empty((M, N), device=A.device, dtype=A.dtype)
       grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, matmul_swizzled_kernel,
                 (A, B, C, tm, tn, tk, group_size_m))
       return C

Verifikation
------------

.. code-block:: python

   M, K, N = 257, 513, 129
   A = torch.randn(M, K, dtype=torch.float16, device="cuda")
   B = torch.randn(K, N, dtype=torch.float16, device="cuda")
   C = cutile_matmul_swizzled(A, B, tm=64, tn=64, tk=64)
   assert torch.allclose(C, torch.matmul(A, B), atol=1e-1, rtol=1e-1)

Output:

.. code-block:: text

   Task 4a: Verifikation des Swizzled-Kernels
     Swizzled-Kernel verifiziert (auch nicht-Zweierpotenz).

Erkenntnisse: Task 4b – Tile-Shape-Sweep
-----------------------------------------

Auszüge aus dem 27-Kombinationen-Sweep mit dem swizzled Kernel:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 25 30

   * - tm
     - tn
     - tk
     - Laufzeit (ms) :math:`2048^3`
     - TFLOPS
   * - 128
     - 128
     - 64
     - 0,3031
     - **56,68**
   * - 128
     - 64
     - 128
     - 0,3178
     - 54,05
   * - 64
     - 128
     - 64
     - 0,3231
     - 53,17
   * - 64
     - 128
     - 32
     - 0,3264
     - 52,63

.. figure:: ../../../../assignments/03_assignment/task04b_heatmap_2048.png
   :align: center
   :alt: Heatmap Swizzled Tile-Shapes 2048^3

   Heatmap der TFLOPS für den swizzled Kernel bei :math:`2048^3`
   (k_tile = 64).

.. figure:: ../../../../assignments/03_assignment/task04b_heatmap_512.png
   :align: center
   :alt: Heatmap Swizzled Tile-Shapes 512^3

   Heatmap der TFLOPS für den swizzled Kernel bei :math:`512^3`
   (k_tile = 64).

**Beste Kombinationen (swizzled):**

* :math:`2048^3`: ``tm=128, tn=128, tk=64`` mit **56,68 TFLOPS**
* :math:`512^3` : ``tm=64,  tn=128, tk=64`` mit **13,11 TFLOPS**

Vergleich row-major vs. swizzled (8192 × 8192 × 4096)
------------------------------------------------------

Für die in der Aufgabe geforderte Größe :math:`M = N = 8192`,
:math:`K = 4096` mit Tile ``(128, 128, 64)``, FP16:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Variante
     - Laufzeit (ms)
     - TFLOPS
     - Speedup
   * - row-major (Task 2)
     - 19,064
     - 28,84
     - 1,00×
   * - swizzled (Task 4)
     - 7,317
     - **75,13**
     - **2,61×**

**Diskussion:**

* Der swizzled Kernel ist bei :math:`8192 \times 8192 \times 4096`
  rund **2,6× schneller** als der row-major Kernel und erreicht über
  75 TFLOPS – mehr als die beste row-major Konfiguration im
  :math:`2048^3`-Sweep.
* Bei kleinen Matrizen (:math:`512^3`, :math:`2048^3`) liegen beide
  Varianten dicht beieinander, weil dort der gesamte Working-Set bereits
  in den L2 passt und das Mapping kaum eine Rolle spielt.
* Erst wenn die A/B-Matrizen den L2 überschreiten – wie bei
  :math:`8192 \times 8192 \times 4096` – schlägt das Super-Grouping
  voll durch, weil die wiederverwendeten Tile-Streifen länger im L2
  bleiben.
* Konsistent dazu kollabiert die Performance des row-major Kernels in
  Task 3a ab :math:`N = 4096`, während der swizzled Kernel im großen
  Vergleich genau diese Grenze überspringt.

Verifikation – Gesamtübersicht
===============================

.. list-table::
   :header-rows: 1
   :widths: 15 40 20 25

   * - Task
     - Test
     - Referenz
     - Ergebnis
   * - 1
     - FP16/FP32 Matmul, :math:`64 \times 4096 \times 64`
     - ``torch.matmul`` + ``do_bench``
     - (Moritz)
   * - 2
     - Generischer Matmul-Kernel, beliebige Shapes
     - ``torch.matmul``
     - (Moritz)
   * - 3a
     - Quadratischer TFLOPS-Sweep (Task-2 Kernel)
     - —
     - 3 – 47 TFLOPS, Einbruch ab 4096
   * - 3b
     - 27 Tile-Kombinationen für :math:`2048^3` und :math:`512^3`
     - —
     - Beste: ``(128, 128, 64)`` @ 56,33 TFLOPS
   * - 4a
     - Swizzled Matmul, ``(257, 513, 129)``
     - ``torch.matmul``
     - ✓ allclose
   * - 4b
     - 27 Tile-Kombinationen swizzled + Vergleich :math:`8192 \times 8192 \times 4096`
     - row-major Kernel
     - ✓ 2,61× Speedup, 75,13 TFLOPS

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Implementierung Task 1 und Task 2 (FP16/FP32-Matmul, generischer
       Matmul-Kernel), zugehörige Verifikationen und Bench-Auswertung,
       Report-Abschnitte zu Task 1 und 2
   * - Oliver Dietzel
     - Implementierung Task 3 und Task 4 (TFLOPS-Sweep, Tile-Shape-Heatmaps,
       Block-Swizzling und Vergleich), Sphinx-Dokumentation (Report)
