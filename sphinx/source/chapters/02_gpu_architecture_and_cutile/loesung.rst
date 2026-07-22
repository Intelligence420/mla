.. _ch02_loesung:

#######################################
Report: GPU Architecture and cuTile
#######################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Dieses Kapitel dokumentiert unsere Lösung des zweiten Assignments:
*GPU Architecture and cuTile*. Ziel ist die Untersuchung von
GPU-Hardware-Eigenschaften sowie die Implementierung tile-basierter
Kernels mit `cuTile <https://github.com/nvidia/cutile-python>`_ –
von Reduktion über elementweise Addition bis zu Bandbreiten-Benchmarks.
Alle Kernels verwenden Tensoren mit Datentyp FP16.

Task 1: GPU Device Properties
=============================

Aufgabenstellung
----------------

Über ``cp.cuda.Device().attributes.items()`` sollen die Werte für
``L2CacheSize``, ``MaxSharedMemoryPerMultiprocessor`` und ``ClockRate``
auf dem DGX Spark ausgelesen und berichtet werden.

Implementierung
---------------

CuPys ``Device().attributes`` gibt ein Dictionary mit **allen** CUDA-Attributen
zurück. Da ``.items()`` immer alle Schlüssel-Wert-Paare liefert, wird während der
Iteration auf die gewünschte Teilmenge gefiltert:

.. code-block:: python

   def report_device_properties():
       keys_of_interest = {
           "L2CacheSize",
           "MaxSharedMemoryPerMultiprocessor",
           "ClockRate",
       }
       for key, value in cp.cuda.Device().attributes.items():
           if key in keys_of_interest:
               print(f"{key}: {value}")

Erkenntnisse
------------

Gemessene Werte auf dem DGX Spark:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Attribut
     - Wert
     - Einheit
   * - ``ClockRate``
     - 2 418 000
     - kHz (≈ 2,42 GHz)
   * - ``L2CacheSize``
     - 25 165 824
     - Bytes (≈ 24 MB)
   * - ``MaxSharedMemoryPerMultiprocessor``
     - 102 400
     - Bytes (100 KB)

Der L2-Cache von 24 MB ist groß genug, um häufig genutzte Tiles zwischen
aufeinanderfolgenden Kernel-Aufrufen zu halten. Der Shared Memory von 100 KB
pro SM ist eine harte Ressourcenschranke, die die maximale Tile-Größe pro Block
begrenzt.

Task 2: Matrix Reduction Kernel
===============================

Aufgabenstellung
----------------

cuTile-Kernel, der eine 2D-Eingabematrix der Form ``(M, K)`` entlang der
letzten Dimension ``K`` zu einem Vektor der Form ``(M,)`` reduziert
(Zeilensumme). Parallelisierung erfolgt über ``M`` via ``grid``; für
Tiles mit Größen jenseits der nächsten Zweierpotenz ist Zero-Padding
innerhalb des Kernels nötig.

Implementierung
---------------

**Kernel**

Jeder Block erhält über ``ct.bid(0)`` seine Zeilen-ID. Das Tile umfasst
die gesamte Zeile ``(1, tile_k)``, wobei ``tile_k`` die nächste
Zweierpotenz ≥ K ist. Überstehende Elemente werden per
``PaddingMode.ZERO`` aufgefüllt, damit die Summe korrekt bleibt:

.. code-block:: python

   @ct.kernel
   def row_sum_kernel(mat, output, tile_k: ct.Constant[int]):
       pid = ct.bid(0)                               # Zeilen-Index dieses Blocks
       tile = ct.load(mat,
                      index=(pid, 0),
                      shape=(1, tile_k),
                      padding_mode=ct.PaddingMode.ZERO)
       row_sum = ct.sum(tile, axis=1)               # (1,)
       ct.store(output, index=(pid,), tile=row_sum)

**Host-Funktion**

.. code-block:: python

   def row_sum(mat: torch.Tensor) -> torch.Tensor:
       M, K = mat.shape
       output = torch.empty(M, dtype=mat.dtype, device=mat.device)
       tile_k = 1
       while tile_k < K:
           tile_k *= 2
       grid = (M, 1, 1)
       ct.launch(torch.cuda.current_stream().cuda_stream,
                grid, row_sum_kernel, (mat, output, tile_k))
       return output

Verifikation
------------

Alle Shapes – auch Nicht-Zweierpotenzen für K – bestehen ``torch.allclose``
(``atol=1e-2``, ``rtol=1e-2`` für FP16-Präzision):

.. code-block:: text

   Task 2: Matrix Reduction Kernel
     M=64,  K=128 → allclose=True
     M=128, K=100 → allclose=True
     M=256, K=37  → allclose=True

Erkenntnisse: Einfluss von M und K
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Dimension
     - Effekt
   * - **M ↑**
     - Mehr Blöcke starten parallel → bessere GPU-Auslastung. Bei sehr
       kleinem M bleiben viele Streaming-Multiprozessoren (SMs) idle.
   * - **M ↓**
     - Weniger Parallelismus; GPU unterausgelastet. Der Overhead für
       Kernel-Start dominiert bei winzigem M.
   * - **K ↑**
     - Mehr Arbeit pro Block (höherer pro-Block-Load). Gleichzeitig
       wächst die Tile-Größe (nächste Zweierpotenz), was mehr Shared
       Memory belegt.
   * - **K ↓**
     - Blöcke sind schnell fertig; GPU-Utilization sinkt, weil die
       Rechenzeit je Block gering ist und Latenz dominiert.

Task 3: 4D Tensor Elementwise Addition
=======================================

Aufgabenstellung
----------------

cuTile-Kernel zur elementweisen Addition zweier 4D-Tensoren ``A`` und
``B`` der Form ``(M, N, K, L)``. Zwei Varianten:

1. Output-Tile deckt die Dimensionen ``K`` und ``L`` ab, parallelisiert
   wird über ``M`` und ``N``.
2. Output-Tile deckt die Dimensionen ``M`` und ``N`` ab, parallelisiert
   wird über ``K`` und ``L``.

Benchmark mit :math:`|M| = 16`, :math:`|N| = 128`, :math:`|K| = 16`,
:math:`|L| = 128` via ``triton.testing.do_bench``.

Implementierung: Variante 1 – Tile (K, L), Grid (M, N)
-------------------------------------------------------

Jeder Block ist über ``(pid_m, pid_n)`` eindeutig einer ``(m, n)``-Position
zugeordnet und bearbeitet den gesamten ``(K, L)``-Slice dieser Position:

.. code-block:: python

   @ct.kernel
   def add_4d_tile_KL(A, B, C, tile_k: ct.Constant[int], tile_l: ct.Constant[int]):
       pid_m = ct.bid(0)
       pid_n = ct.bid(1)
       a_tile = ct.load(A, index=(pid_m, pid_n, 0, 0),
                        shape=(1, 1, tile_k, tile_l),
                        padding_mode=ct.PaddingMode.ZERO)
       b_tile = ct.load(B, index=(pid_m, pid_n, 0, 0),
                        shape=(1, 1, tile_k, tile_l),
                        padding_mode=ct.PaddingMode.ZERO)
       ct.store(C, index=(pid_m, pid_n, 0, 0), tile=a_tile + b_tile)

   def add_4d_variant1(A, B):
       M, N, K, L = A.shape
       C = torch.empty_like(A)
       grid = (M, N, 1)                  # 16 × 128 = 2048 Blöcke
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, add_4d_tile_KL, (A, B, C, K, L))
       return C

Implementierung: Variante 2 – Tile (M, N), Grid (K, L)
-------------------------------------------------------

Jeder Block ist über ``(pid_k, pid_l)`` eindeutig einer ``(k, l)``-Position
zugeordnet und bearbeitet den gesamten ``(M, N)``-Slice dieser Position:

.. code-block:: python

   @ct.kernel
   def add_4d_tile_MN(A, B, C, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
       pid_k = ct.bid(0)
       pid_l = ct.bid(1)
       a_tile = ct.load(A, index=(0, 0, pid_k, pid_l),
                        shape=(tile_m, tile_n, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO)
       b_tile = ct.load(B, index=(0, 0, pid_k, pid_l),
                        shape=(tile_m, tile_n, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO)
       ct.store(C, index=(0, 0, pid_k, pid_l), tile=a_tile + b_tile)

   def add_4d_variant2(A, B):
       M, N, K, L = A.shape
       C = torch.empty_like(A)
       grid = (K, L, 1)                  # 16 × 128 = 2048 Blöcke
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, add_4d_tile_MN, (A, B, C, M, N))
       return C

Verifikation
------------

Beide Varianten bestehen ``torch.allclose`` gegen PyTorchs natives ``A + B``:

.. code-block:: text

   Task 3a: 4D Elementwise Addition — Verifikation
     Variante 1 (tile KL, grid MN) passed
     Variante 2 (tile MN, grid KL) passed

Erkenntnisse: Benchmark und Erklärung
--------------------------------------

.. code-block:: text

   Task 3b: Benchmark
     Variante 1 (tile KL): 0.1393 ms
     Variante 2 (tile MN): 0.4958 ms

**Variante 1 ist ≈ 3,6× schneller als Variante 2.**

Der Grund liegt in der Speicherzugriffslokalität. PyTorch speichert Tensoren
im Row-Major-Format (C-contiguous), d. h. die *letzten* Dimensionen sind im
Speicher zusammenhängend. Bei einem Tensor der Form ``(M, N, K, L)`` liegen
aufeinanderfolgende ``L``-Elemente direkt nebeneinander; dann kommen die
``K``-Elemente, und so weiter.

* **Variante 1** tiled über ``(K, L)`` – also über die letzten beiden,
  zusammenhängenden Dimensionen. Jeder Block greift auf einen einzigen
  contiguous Speicherbereich von ``K × L = 16 × 128 = 2048`` Elementen zu.
  Das ermöglicht coalesced Memory Accesses mit hoher effektiver Bandbreite.

* **Variante 2** tiled über ``(M, N)`` – die äußeren Dimensionen. Für eine
  feste ``(k, l)``-Position sind die ``M × N``-Elemente im Speicher mit
  einem Stride von ``K × L`` verteilt. Diese nicht-zusammenhängenden Zugriffe
  kann die Hardware nicht zu wenigen breiten Speichertransaktionen
  **zusammenfassen** (kein Coalescing): Die Zugriffe laufen zwar weiterhin über
  den L1-/L2-Cache, ein Warp löst aber statt weniger breiter viele kleine
  Transaktionen aus. Das senkt die Bus-Auslastung und damit den Durchsatz
  erheblich (uncoalesced accesses).

Task 4: Benchmarking Bandwidth
==============================

Aufgabenstellung
----------------

cuTile-Kernel, der eine 2D-Matrix der Form ``(M, N)`` kopiert
(Tile-Größe ``(tile_M, tile_N)``). Für ``M = 2048`` und
``N`` zwischen 16 und 128 wird bei ``tile_M = 64`` und ``tile_N = N``
die effektive Speicherbandbreite gemessen:

.. math::

   \text{bandwidth (GB/s)} = \frac{2 \cdot M \cdot N \cdot \text{sizeof(element)}}{t_s \cdot 10^9}

Der Faktor 2 berücksichtigt je einen Lese- und einen Schreibzugriff.

Implementierung
---------------

**Kernel**

.. code-block:: python

   @ct.kernel
   def copy_kernel(src, dst, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
       pid_m = ct.bid(0)
       pid_n = ct.bid(1)
       tile = ct.load(src, index=(pid_m, pid_n), shape=(tile_m, tile_n),
                      padding_mode=ct.PaddingMode.ZERO)
       ct.store(dst, index=(pid_m, pid_n), tile=tile)

**Host-Funktion**

Tile-Dimensionen werden auf die nächste Zweierpotenz aufgerundet. Das Grid
wird mit ``ct.cdiv`` so berechnet, dass die gesamte Matrix abgedeckt ist:

.. code-block:: python

   def copy_matrix(src, tile_m, tile_n):
       M, N = src.shape
       dst = torch.empty_like(src)
       tile_m_pow2 = next_power_of_2(tile_m)
       tile_n_pow2 = next_power_of_2(tile_n)
       grid = (ct.cdiv(M, tile_m_pow2), ct.cdiv(N, tile_n_pow2), 1)
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, copy_kernel, (src, dst, tile_m_pow2, tile_n_pow2))
       return dst

Verifikation
------------

.. code-block:: text

   Task 4a: Copy Kernel — Verification
     Copy kernel verified.

Erkenntnisse: Bandwidth-Messungen
-----------------------------------

Gemessen wurde die **volle Range** ``N = 16 … 128`` (Schrittweite 1, also 113
Messpunkte; M=2048, tile_M=64, tile_N=N, FP16). Die folgende Tabelle zeigt
repräsentative Punkte: jeweils die Zweierpotenzen (lokaler Peak, kein Padding)
und den direkt darauffolgenden Wert (Einbruch, weil sich ``tile_N`` verdoppelt):

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 20 30

   * - N
     - tile_N (2er-Potenz)
     - Laufzeit (ms)
     - Bandbreite (GB/s)
     - Charakter
   * - 16
     - 16
     - 0,0047
     - 27,9
     - Zweierpotenz, kein Padding (Peak)
   * - 17
     - 32
     - 0,0065
     - 21,5
     - ``tile_N`` verdoppelt → ~½ Padding (Einbruch)
   * - 32
     - 32
     - 0,0069
     - 38,2
     - Zweierpotenz (Peak)
   * - 33
     - 64
     - 0,0085
     - 31,7
     - ``tile_N`` verdoppelt → Einbruch
   * - 64
     - 64
     - 0,0082
     - 64,1
     - Zweierpotenz (Peak)
   * - 65
     - 128
     - 0,0116
     - 45,8
     - ``tile_N`` verdoppelt → Einbruch
   * - 128
     - 128
     - 0,0119
     - 88,2
     - Zweierpotenz, kein Padding (Peak)

.. figure:: ../../../../assignments/02_assignment/bandwidth_plot.png
   :align: center
   :alt: Effective Bandwidth vs. N über die volle Range N=16…128

   Effektive Speicherbandbreite des Copy-Kernels über die volle Range
   ``N = 16 … 128`` (Schrittweite 1). Deutlich sichtbar ist das Sägezahn-Muster.

**Beobachtungen:**

* **Grundtrend:** Über die gesamte Range steigt die effektive Bandbreite mit
  wachsendem N (von ≈ 22 GB/s am unteren Ende auf ≈ 88 GB/s bei N=128), weil
  breitere zusammenhängende Transfers den Overhead pro Speichertransaktion
  besser amortisieren und den Speicherbus höher auslasten.
* **Sägezahn durch Padding:** Der Kernel rundet ``tile_N`` auf die nächste
  Zweierpotenz auf. Innerhalb eines Bandes (z. B. ``33 ≤ N ≤ 64`` mit
  ``tile_N = 64``) wächst der Nutzanteil ``N / tile_N`` mit N, sodass die
  effektive Bandbreite ansteigt und an der Zweierpotenz ihr lokales Maximum
  erreicht (kein Padding). Beim nächsten Schritt (``N → N+1``) verdoppelt sich
  ``tile_N``, und fast die Hälfte des geladenen und gespeicherten Tiles ist
  Padding, das nicht zur Nutzdatenmenge zählt. Genau dort bricht die effektive
  Bandbreite scharf ein – am deutlichsten bei ``N = 17, 33, 65`` (direkt über
  den Zweierpotenzen 16, 32, 64). Erst die volle Range macht dieses Muster
  sichtbar; eine grobe 16er-Abtastung würde daran vorbeimessen.
* **Sekundär-Spitzen an Vielfachen von 16:** Dem Grundtrend überlagert zeigt
  der Plot zusätzlich regelmäßige Spitzen bei Vielfachen von 16 (``N = 48, 80,
  96, 112``); bei ``N = 48`` wird z. B. fast derselbe Wert wie am Band-Peak
  ``N = 64`` erreicht. Ursache ist die Ausrichtung an der Transaktions-
  Granularität: Bei FP16 (2 Byte) entspricht ein Vielfaches von 16 Elementen
  genau einer 32-Byte-Grenze, sodass jede Zeile sauber am Speicher-Sektor
  ausgerichtet beginnt und weniger Teil-Transaktionen anfallen. Zwischen diesen
  Vielfachen driften die Zeilenanfänge aus der Ausrichtung, was ein paar
  Prozent Bandbreite kostet.
* **Absolutwerte:** Die höchste gemessene Bandbreite (≈ 88 GB/s bei N=128) ist
  nur ein Bruchteil der theoretischen Spitzenbandbreite des DGX Spark – ein
  einfacher Copy-Workload mit schmalen Tiles sättigt den Speicher also noch
  nicht.

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
     - Attributabfrage über CuPy
     - CUDA Device API
     - ✓ korrekte Werte
   * - 2
     - ``row_sum`` für (64,128), (128,100), (256,37)
     - ``torch.sum(mat, dim=1)``
     - ✓ allclose (alle Shapes)
   * - 3a
     - ``add_4d_variant1`` und ``add_4d_variant2``
     - ``A + B`` (PyTorch)
     - ✓ allclose (beide Varianten)
   * - 3b
     - Benchmark
     - —
     - Variante 1 ≈ 3,6× schneller
   * - 4a
     - ``copy_matrix`` für (2048, 64)
     - ``torch.equal``
     - ✓ exakte Kopie
   * - 4b
     - Bandwidth-Sweep N=16…128 (Schritt 1, 113 Punkte)
     - —
     - ≈ 22–88 GB/s, steigend (Sägezahn durch Padding)

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Oliver Dietzel
     - Implementierung aller cuTile-Kernels (Task 1–4),
       Verifikation und Benchmarks, Lösungsdokumentation
   * - Moritz Martin
     - Projektstruktur, Sphinx-Dokumentation, Build-Setup
