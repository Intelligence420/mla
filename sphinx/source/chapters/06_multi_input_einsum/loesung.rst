.. _ch06_loesung:

########################################
Report: Multi-Input Einsum Contraction
########################################

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Einleitung
==========

Kontraktion zweier intermediärer Tensoren einer Light-Field-Tensor-Ring-
Zerlegung (``acspx, bspy -> abcyx``) — einmal als ``torch.einsum``-
Referenz, einmal als cuTile-Kernel mit dem ``Config``/``Optimizer``-
Interface aus Assignment 05.

``config.py`` und ``optimizer.py`` sind **Kopien** aus A05 (statt
Imports), weil A05 als ``submission-05`` getagt und abgegeben ist —
der dortige Stand bleibt eingefroren.

Geladene Shapes aus ``data/lf_tr_64_intermediate.npz``:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Tensor
     - Indizes
     - Shape
   * - ``tensor_acspx``
     - ``(a, c, s, p, x)``
     - ``(4, 3, 64, 64, 1536)``
   * - ``tensor_bspy``
     - ``(b, s, p, y)``
     - ``(4, 64, 64, 1152)``
   * - Output ``tensor_abcyx``
     - ``(a, b, c, y, x)``
     - ``(4, 4, 3, 1152, 1536)``

Achsen-Größen weichen leicht von Slide 14 ab (``x=1536`` statt 1280,
``y=1152`` statt 1536). Die Code-Pipeline leitet die Split-Faktoren
aus den Tensor-Shapes ab, läuft also auch für die anderen
``lf_*_intermediate.npz``-Datasets.

Task 1: PyTorch Reference Contraction
======================================

Task 1a: Index-Klassifikation
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 14 18 18 18 20

   * - Index
     - Größe
     - in ``acspx``
     - in ``bspy``
     - im Output
     - Typ
   * - ``a``
     - 4
     - ✓
     - —
     - ✓
     - **M**
   * - ``c``
     - 3
     - ✓
     - —
     - ✓
     - **M**
   * - ``x``
     - 1536
     - ✓
     - —
     - ✓
     - **M**
   * - ``b``
     - 4
     - —
     - ✓
     - ✓
     - **N**
   * - ``y``
     - 1152
     - —
     - ✓
     - ✓
     - **N**
   * - ``s``
     - 64
     - ✓
     - ✓
     - —
     - **K**
   * - ``p``
     - 64
     - ✓
     - ✓
     - —
     - **K**

3× M, 2× N, 2× K, keine C-Dim.
FLOPs :math:`\;= 2 \cdot |M| \cdot |N| \cdot |K| \approx 6{,}96 \cdot 10^{11}`.

Task 1b: Einsum-String
-----------------------

.. code-block:: text

   acspx, bspy -> abcyx

Implementierung minimal-invasiv gegen das Kurs-Boilerplate: ``.cuda()``
an die ``torch.tensor``-Calls anhängen und den ``# TODO`` durch zwei
``torch.einsum``-Aufrufe (FP32 + FP16) ersetzen. Der FP16-Output wird
vor dem Plot nach FP32 zurückgecastet (``plot_tensor`` verwendet
``.numpy()`` + ``*= 255``).

Task 1c: FP16 vs. FP32
-----------------------

.. figure:: ../../../../assignments/06_assignment/results/torch_32.png
   :align: center
   :width: 90%

   ``results/torch_32.png`` — FP32.

.. figure:: ../../../../assignments/06_assignment/results/torch_16.png
   :align: center
   :width: 90%

   ``results/torch_16.png`` — FP16.

Visuell nicht unterscheidbar. Jedoch nach kurzen Pixel vergleich: es gibt unterschiede!
Man könnte vielleicht in Zukunft einen "Standartisierten" Pixelvergleich / Qualitätsanalyse durchführen.

Task 2: Basic Config
====================

.. code-block:: python

   cfg = generate_config(
       'acspx,bspy->abcyx',
       [tuple(tensor_acspx.shape), tuple(tensor_bspy.shape)],
   )

.. code-block:: text

   pos name    type  exec      size   stride_A   stride_B   stride_C
   ------------------------------------------------------------------
   0   a       M     SEQ          4   18874368          0   21233664
   1   c       M     SEQ          3    6291456          0    1769472
   2   s       K     SEQ         64      98304      73728          0
   3   p       K     SEQ         64       1536       1152          0
   4   x       M     SEQ       1536          1          0          1
   5   b       N     SEQ          4          0    4718592    5308416
   6   y       N     SEQ       1152          0          1       1536

   data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

**Layout-Beobachtung.** In A hat die Stride-1-Dim M-Typ (``x``), in B
N-Typ (``y``). **Keine** K-Dim hat Stride 1 in A oder B — Unterschied
zu A05. Genau die Konstellation, für die Lecture 6 das Pre-Loading
mit ``ct.extract`` (Slide 11) und Multi-PRIM (Slides 6–9) einführt.

Task 3: Optimized Config
=========================

Strategie
----------

**Plan C**: Task-3-Config bleibt Single-PRIM mit Slide-5-L2-Pattern
(klar, deklarativ); Task-4a-Kernel nutzt **Pre-Loading via
``ct.extract``** (Slide 11) — direkt motiviert durch die fehlende
Stride-1-K-Dim. Multi-PRIM-K (Slide 7) bleibt für den optionalen Task.

PRIM-Tile-Größen aus A05 übernommen (auf GB10/FP16 die belegt-besten
``ct.mma``-Footprints): :math:`|x_{\text{prim}}| = |y_{\text{prim}}| = 64,
\; |s\!p_{\text{prim}}| = 32`.

Pipeline
--------

.. code-block:: python

   opt = Optimizer(cfg)
   opt.fuse_dims(2, 3)                           # s,p -> sp
   opt.split_dim(2, k_total // PRIM_K, PRIM_K)   # sp -> sp_seq, sp_prim
   opt.split_dim(4, x_sz   // PRIM_M, PRIM_M)    # x  -> x_seq,  x_prim
   opt.split_dim(7, y_sz   // PRIM_N, PRIM_N)    # y  -> y_seq,  y_prim
   opt.permute_dims([0, 1, 4, 6, 7, 2, 5, 8, 3]) # ins Slide-5-Layout
   opt.make_executable()

* Nur ``s``/``p`` lassen sich fusen — in A und B beide adjazent
  (``p`` innerer, ``s`` outer in beiden). Andere Paare (``a``+``c``,
  ``b``+``y``, ``x``+``y``) scheitern an der Adjazenz im Output.
* Explizites ``permute_dims`` vor ``make_executable`` erzwingt
  PRIM-Block ``[M, N, K]`` (sonst landet ``sp_prim`` links von
  ``x_prim``/``y_prim`` durch die stabile Default-Sortierung).
* ``make_executable`` setzt PAR/SEQ/PRIM und schließt mit ``verify()``.

Ergebnis
--------

.. code-block:: text

   pos name    type  exec      size   stride_A   stride_B   stride_C
   ------------------------------------------------------------------
   0   a       M     PAR          4   18874368          0   21233664
   1   c       M     PAR          3    6291456          0    1769472
   2   x_seq   M     PAR         24         64          0         64
   3   b       N     PAR          4          0    4718592    5308416
   4   y_seq   N     PAR         18          0         64      98304
   5   sp_seq  K     SEQ        128      49152      36864          0
   6   x_prim  M     PRIM        64          1          0          1
   7   y_prim  N     PRIM        64          0          1       1536
   8   sp_prim K     PRIM        32       1536       1152          0

   data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

Layout: 5× PAR (3M, 2N) | 1× SEQ-K | PRIM-Block ``[M, N, K]``.

**Tile-Geometrie im PRIM-Block.** A: ``x_prim`` Stride 1, ``sp_prim``
Stride 1536 → A-Tile ist memory-(K, M) statt (M, K). B: ``y_prim``
Stride 1, ``sp_prim`` Stride 1152 → klassisches (K, N)-Layout. Genau
diese A-Asymmetrie motiviert das Pre-Loading in Task 4a.

L2-Working-Set (FP16, GB10 ~30 MB L2):
:math:`m_{l2} = a \cdot c \cdot x_{\text{seq}} = 288,
\; n_{l2} = b \cdot y_{\text{seq}} = 72`.
Volle K-Schiene: A ≈ 2,3 MB, B ≈ 0,6 MB — passt locker.

Task 4: cuTile Kernel
======================

*folgt.*

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
