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

Visuell sind FP16 und FP32 nicht zu unterscheiden. Ein genauer Pixelvergleich zeigt
jedoch geringe Abweichungen. Eine standardisierte Pixel- bzw. Qualitätsanalyse
(z. B. PSNR/SSIM) wäre ein sinnvoller nächster Schritt.

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

Die L2-Optimierung wird **datengetrieben über die Config** ausgedrückt –
nicht per Hand im Kernel. Wie in Assignment 05 (Task 4) entsteht ein
2D-Super-Tile durch eine **zusätzliche Split-Ebene** und eine Permutation, die
die parallelen **M- und N-PAR-Achsen verschachtelt** (Gruppen-Achsen innen).
Der Kernel bleibt dadurch generisch (Grid über die PAR-Achsen, GEMM über die
PRIM-Achsen) und braucht keine eigene Swizzle-Arithmetik.

PRIM-Tile-Größen aus A05 übernommen (auf GB10/FP16 die belegt-besten
``ct.mma``-Footprints): :math:`|x_{\text{prim}}| = |y_{\text{prim}}| = 64,
\; |s\!p_{\text{prim}}| = 32`.

Pipeline (``build_optimized_config``)
-------------------------------------

.. code-block:: python

   opt = Optimizer(cfg)
   # 1) K fusen + mma-Tile abspalten
   opt.fuse_dims(2, 3)                     # s,p -> sp
   opt.split_dim(2, k_total // 32, 32)     # sp -> sp_seq, sp_prim
   opt.split_dim(4, x_seq, 64)             # x  -> x_seq,  x_prim
   opt.split_dim(7, y_seq, 64)             # y  -> y_seq,  y_prim
   # 2) Super-Tile abspalten (Gruppengröße GX, GY)
   opt.split_dim(4, x_seq // GX, GX)       # x_seq -> x_super, x_group
   opt.split_dim(8, y_seq // GY, GY)       # y_seq -> y_super, y_group
   # 3) M-/N-PAR-Achsen verschachteln, Gruppen-Achsen innen
   opt.permute_dims([0, 1, 7, 4, 8, 5, 9, 2, 6, 10, 3])
   opt.make_executable()

* Nur ``s``/``p`` lassen sich fusen — in A und B beide adjazent **und in
  gleicher relativer Reihenfolge** (``p`` innen, ``s`` außen). Andere Paare
  scheitern an der Adjazenz/Reihenfolge im Output.
* Die Permutation zieht ``x_group`` und ``y_group`` nach innen, sodass die
  Grid-Enumeration ein ``GX × GY``-Super-Tile abläuft, bevor sie zum nächsten
  Super-Block springt — das *ist* der Swizzle, rein deklarativ in der Config.
* ``make_executable`` setzt PAR/SEQ/PRIM und schließt mit ``verify()``.

Ergebnis
--------

Optimierte Config (``GX = 4, GY = 3``):

.. code-block:: text

   pos name     type  exec      size   stride_A   stride_B   stride_C
   -------------------------------------------------------------------
   0   a        M     PAR          4   18874368          0   21233664
   1   c        M     PAR          3    6291456          0    1769472
   2   b        N     PAR          4          0    4718592    5308416
   3   x_super  M     PAR          6        256          0        256
   4   y_super  N     PAR          6          0        192     294912
   5   x_group  M     PAR          4         64          0         64
   6   y_group  N     PAR          3          0         64      98304
   7   sp_seq   K     SEQ        128      49152      36864          0
   8   x_prim   M     PRIM        64          1          0          1
   9   y_prim   N     PRIM        64          0          1       1536
   10  sp_prim  K     PRIM        32       1536       1152          0

   data_type=FLOAT16  prim_main=GEMM  prim_last=NONE  prim_first=ZERO

Layout: 7× PAR mit **verschachtelten** M-/N-Achsen (``x_group``, ``y_group``
innen) | 1× SEQ-K | PRIM-Block ``[M, N, K]``. Das komplette Super-Tiling steckt
in der Dimensionsstruktur; der Kernel liest sie nur aus. ``GX = x_{seq}`` bzw.
``GY = y_{seq}`` (super = 1) ergäbe wieder die natürliche Enumeration.

Task 4: cuTile Kernel
======================

Task 4a: Kernel-Design
-----------------------

Alle Nicht-Referenz-Kernel folgen der Config: ein 3D-Grid über die PAR-Achsen
(auf drei ``ct.bid`` gefaltet, da cuTile nur drei Block-IDs hat), eine innere
``for sp_seq``-Schleife über die einzige SEQ-K-Achse und im Rumpf genau ein
``ct.mma`` auf dem PRIM-Block ``(x_prim, y_prim, sp_prim)``.

Generischer, config-getriebener Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``kernel_generic`` verlagert die L2-Lokalität vollständig in die Config: sie
entsteht aus der **Dimensionsstruktur der Config**, nicht aus einer
Swizzle-Formel im Kernel. Grid:
``(a·c, b, x_super·y_super·x_group·y_group)``. Die ``bid(2)``-Achse wird per
verschachteltem divmod über die Super-/Group-Größen (aus ``_extract_par(cfg)``
gelesen) dekodiert – mit den **Gruppen-Achsen innen**, sodass aufeinander
folgende BIDs ein ``GX × GY``-Super-Tile abdecken:

.. code-block:: python

   g = ct.bid(2)
   y_grp = g %  GY;  g = g // GY
   x_grp = g %  GX;  g = g // GX
   y_sup = g %  YS;  x_sup = g // YS
   pid_x = x_sup * GX + x_grp     # x_seq-Tile-Index
   pid_y = y_sup * GY + y_grp     # y_seq-Tile-Index

Keine ``// blocks_per_group``-Arithmetik mehr. Ändert man die Split-/Permute-
Pipeline in ``build_optimized_config``, passt sich das Launch-Layout über
``_extract_par`` automatisch an – der Kernel bleibt unverändert.

mma-Reihenfolge und der B-Permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Output-Tile-Form ist :math:`(y_{\text{prim}}, x_{\text{prim}})` —
``y`` ist die äußere, ``x`` die innere (stride-1) Achse von ``tensor_abcyx``.
``ct.mma`` ist ``(M, K) @ (K, N) → (M, N)``, also
:math:`M = y_{\text{prim}}, N = x_{\text{prim}}, K = sp_{\text{prim}}`:

.. code-block:: python

   acc = ct.full((ty, tx), 0, dtype=ct.float32)
   for sp_seq in range(SPSEQ):
       a_tile = ct.load(A, index=(pid_a, pid_c, sp_seq, 0, pid_x, 0),
                        shape=(1, 1, 1, tk, 1, tx), padding_mode=ct.PaddingMode.ZERO)
       b_tile = ct.load(B, index=(pid_b, sp_seq, 0, pid_y, 0),
                        shape=(1, 1, tk, 1, ty), padding_mode=ct.PaddingMode.ZERO)
       a_kx = ct.reshape(a_tile, (tk, tx))                       # (K, N)
       b_yk = ct.permute(ct.reshape(b_tile, (tk, ty)), (1, 0))   # (M, K)
       acc = ct.mma(b_yk, a_kx, acc)                             # (M, N)=(y_prim,x_prim)

Reshape-Views (gratis)
^^^^^^^^^^^^^^^^^^^^^^^

Die Eingabe-Tensoren werden host-seitig als **Views** in die Kernel-Form
gebracht (``.view`` ist O(1), keine Kopie) — möglich, weil ``s, p`` adjazent
im Speicher liegen (genau der ``fuse_dims``-Check aus Assignment 05):

.. code-block:: python

   A = tensor_acspx.contiguous().view(Ad, Cd, SPSEQ, PRIM_K, XSEQ, PRIM_M)
   B = tensor_bspy.contiguous().view(Bd,         SPSEQ, PRIM_K, YSEQ, PRIM_N)

Vergleichs-Kernel
^^^^^^^^^^^^^^^^^^

* ``kernel_baseline`` — dieselbe Rechnung mit dem einfachen 3D-Grid
  ``(a·c, b, x_seq·y_seq)`` (natürliche Enumeration, kein Super-Tile).
* ``kernel_big`` — Baseline-Layout mit größeren PRIM-Tiles ``128 × 128``.

Task 4b: Verifikation
----------------------

Gegen ``torch.einsum("acspx,bspy->abcyx", a16, b16)`` mit FP32-Promotion
und FP16-Rückgabe, ``atol=2e-1, rtol=2e-2``:

.. code-block:: text

   baseline      allclose=True   max_abs_err=0.0010
   generic(ilv)  allclose=True   max_abs_err=0.0010
   generic(128)  allclose=True   max_abs_err=0.0010
   big-prim 128  allclose=True   max_abs_err=0.0010

Alle vier Varianten liegen im FP16-Quantisierungsrauschen. Auf den echten
Light-Field-Daten (kleine Wertebereiche) beträgt der maximale Absolutfehler
``0.0010``; auf synthetischen ``randn``-Tensoren gleicher Form ist er
erwartungsgemäß größer (``≈ 0.25``), weil die K-Dim 4096 Terme tief ist und die
dadurch große Summenmagnitude bei FP16-Ein- und -Ausgabe zu größeren
Quantisierungsfehlern führt (akkumuliert wird in FP32).

Task 4c: Benchmark
-------------------

``triton.testing.do_bench(warmup=10, rep=50)``, DGX Spark (GB10), FP16.
FLOPs der Kontraktion:

.. math::

   2 \cdot |a| \cdot |b| \cdot |c| \cdot |s| \cdot |p| \cdot |x| \cdot |y|
   \;=\; 2 \cdot 4 \cdot 4 \cdot 3 \cdot 64 \cdot 64 \cdot 1536 \cdot 1152
   \;\approx\; 6{,}96 \cdot 10^{11}

Alle Custom-Varianten laufen über den generischen Kernel bzw. das
Baseline-Layout; nur die **Config** (Tile-Größe, Super-Tile) unterscheidet sie:

.. list-table::
   :header-rows: 1
   :widths: 38 16 14 14 18

   * - Variante
     - Tile (M,N,K)
     - ms
     - TFLOPS
     - vs. ``torch``
   * - ``torch.einsum`` (Referenz)
     - —
     - 12,07
     - **57,67**
     - 1,00×
   * - ``baseline`` 3D
     - (64, 64, 32)
     - 43,55
     - 15,98
     - 0,28×
   * - ``generic`` natural
     - (64, 64, 32)
     - 44,26
     - 15,72
     - 0,27×
   * - ``generic`` natural
     - (128, 128, 32)
     - **29,40**
     - **23,66**
     - **0,41×**
   * - ``big-prim``
     - (128, 128, 32)
     - 29,51
     - 23,58
     - 0,41×
   * - ``generic`` interleaved GX,GY=(4,3)
     - (64, 64, 32)
     - 57,76
     - 12,05
     - 0,21×
   * - ``generic`` interleaved GX,GY=(12,9)
     - (64, 64, 32)
     - 46,00
     - 15,13
     - 0,26×

**Beobachtungen.**

* **Der generische, config-getriebene Kernel hat keinen Overhead.** In
  natürlicher Ordnung erreicht er exakt die Baseline (15,7 vs. 16,0 TFLOPS bei
  ``64²``; 23,7 vs. 23,6 bei ``128²``) — der Ansatz *Grid über PAR / GEMM über
  PRIM* kostet nichts, das gesamte Mapping steckt in der Config.

* **Größere PRIM-Tiles sind der eigentliche Hebel.** ``128 × 128`` statt
  ``64 × 64`` bringt ~1,5× (15,7 → 23,7 TFLOPS) — konsistent mit der Heatmap aus
  Assignment 03: bei großen GEMM-Dims (``x = 1536, y = 1152``) amortisieren
  große Tiles den Per-Block-Overhead und lasten die Tensor Cores besser aus.

* **Der 2D-Super-Tile-Swizzle hilft hier NICHT** (interleaved 12–15 TFLOPS,
  unter der Baseline). Grund — dieselbe Lehre wie in Assignment 04/05: Das
  B-Tensor passt **pro Batch** in den L2
  (:math:`|s\,p\,y| \cdot 2\,\text{B} \approx 9` MB < 24 MB), sodass die
  Baseline ihr B ohnehin aus dem L2 zieht. Der Swizzle spart also keine
  DRAM-Bandbreite, bricht aber die *natürliche* A-Reuse (``YSEQ`` konsekutive
  BIDs teilen bei natürlicher Ordnung dasselbe ``x_seq``) und kostet dadurch
  Durchsatz. Anders als in Assignment 05, wo B den L2 sprengte und der
  Super-Tile ~4× brachte. Wichtig: Die Config drückt den Swizzle jetzt
  **korrekt deklarativ** aus — auf Shapes mit L2-sprengendem B würde er greifen.

Optionaler Task: Beat ``torch.einsum``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Der beste Custom-Kernel (``generic`` bzw. ``big-prim`` bei ``128²``) erreicht
**≈ 0,41×** von ``torch.einsum``. torch nutzt ``opt_einsum``-Routing plus cuBLAS
mit ausgereiften Tile-Cascades und Software-Pipelining, die ein
handgeschriebener cuTile-Kernel ohne erheblichen Aufwand nicht erreicht.
Offene Beat-Path-Ideen: Multi-PRIM-K (mehrere ``ct.mma`` pro Iteration),
Persistent Kernel mit Load/Compute-Overlap, oder die Kontraktion als zwei
batched GEMMs zerlegen (dann aber kein einzelner cuTile-Kernel mehr).

Stand: **Kernel verifiziert, 23,7 TFLOPS (best), vollständig config-getrieben**;
``torch.einsum`` nicht geschlagen — der optionale Task bleibt offen.

Beiträge
=========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Person
     - Beitrag
   * - Moritz Martin
     - Boilerplate-Anpassungen in ``main.py`` (CUDA-Casts, FP32-/FP16-
       ``torch.einsum``-Aufrufe), Index-Klassifikation (Task 1a),
       FP32-vs-FP16-Vergleich (Task 1c), Basis- und optimierte
       Config-Pipeline (Task 2, 3), Report-Abschnitte zu Task 1–3.
   * - Oliver Dietzel
     - cuTile-Kernel: ``kernel_baseline`` sowie der generische,
       config-getriebene ``kernel_generic`` (2D-Super-Tile aus der Config,
       kein Hand-Swizzle), Verifikation gegen ``torch.einsum``, Benchmark +
       Sweep über PRIM-Tile-Größen und Gruppengröße (Task 4a–c sowie
       optionaler Task); Report-Abschnitt zu Task 4.
