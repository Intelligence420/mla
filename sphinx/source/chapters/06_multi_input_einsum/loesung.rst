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

Task 4a: Kernel-Design
-----------------------

Der Baseline-Kernel ist die wörtliche Umsetzung der Task-3-Config: ein
3D-Grid über die fünf PAR-Achsen (zu drei ``ct.bid``-Achsen gefaltet,
weil cuTile nur drei Block-IDs hat), eine innere ``for sp_seq``-Schleife
über die einzige SEQ-K-Achse, und im Schleifenrumpf genau ein
``ct.mma`` auf dem PRIM-Block ``(x_prim, y_prim, sp_prim)``.

Grid-Faltung
^^^^^^^^^^^^

Insgesamt :math:`|a| \cdot |c| \cdot |x_{\text{seq}}| \cdot |b| \cdot
|y_{\text{seq}}| = 4 \cdot 3 \cdot 24 \cdot 4 \cdot 18 = 20\,736`
Blöcke. Auf drei ``ct.bid``-Achsen geklappt:

.. code-block:: text

   bid(0) = a · c            range  12   →  (pid_a, pid_c)
   bid(1) = b                range   4   →   pid_b
   bid(2) = x_seq · y_seq    range 432   →  (pid_x, pid_y)

Dadurch teilen ``YSEQ`` konsekutive BIDs (``bid(2) // YSEQ = x_seq``)
denselben ``x_seq``-Streifen — die A-Tile-Spalte wird über bis zu 18
Blöcke wiederverwendet, ohne dass es eine eigene Swizzle-Logik braucht.

mma-Reihenfolge und der B-Permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Output-Tile-Form ist :math:`(y_{\text{prim}}, x_{\text{prim}})` —
``y`` ist die äußere, ``x`` die innere (stride-1) Achse von
``tensor_abcyx``. ``ct.mma`` ist ``(M, K) @ (K, N) → (M, N)``, also
mappen wir :math:`M = y_{\text{prim}}, N = x_{\text{prim}},
K = sp_{\text{prim}}`. Daraus ergibt sich:

* A-Tile aus ``tensor_acspx``: ``shape=(1, 1, 1, tk, 1, tx)`` →
  Reshape auf :math:`(K, N)` direkt verwendbar – kein Permute.
* B-Tile aus ``tensor_bspy``: ``shape=(1, 1, tk, 1, ty)`` →
  Reshape auf :math:`(K, M)`, **muss** auf :math:`(M, K)` permutiert
  werden, bevor es als erstes mma-Argument geht. Genau einmal pro
  Schleifeniteration, also 128 ``ct.permute``-Calls pro Block. Die
  Permute kann der Compiler auf Tile-Fragment-Ebene mit der nächsten
  mma-Operation fusionieren – im Profil kein eigener Posten.

Kernel-Rumpf
^^^^^^^^^^^^

.. code-block:: python

   acc = ct.full((ty, tx), 0, dtype=ct.float32)
   for sp_seq in range(SPSEQ):
       a_tile = ct.load(A, index=(pid_a, pid_c, sp_seq, 0, pid_x, 0),
                        shape=(1, 1, 1, tk, 1, tx),
                        padding_mode=ct.PaddingMode.ZERO)
       b_tile = ct.load(B, index=(pid_b, sp_seq, 0, pid_y, 0),
                        shape=(1, 1, tk, 1, ty),
                        padding_mode=ct.PaddingMode.ZERO)
       a_kx = ct.reshape(a_tile, (tk, tx))      # (sp_prim, x_prim) = (K, N)
       b_ky = ct.reshape(b_tile, (tk, ty))      # (sp_prim, y_prim) = (K, M)
       b_yk = ct.permute(b_ky, (1, 0))          # (y_prim, sp_prim) = (M, K)
       acc = ct.mma(b_yk, a_kx, acc)            # (M, N) = (y_prim, x_prim)

Reshape vs. Kernel-Indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Eingabe-Tensoren werden auf Host-Seite **als Views** in die
Kernel-erwartete Form gebracht (``.view`` ist O(1), keine Kopie):

.. code-block:: python

   A = tensor_acspx.contiguous().view(Ad, Cd, SPSEQ, PRIM_K, XSEQ, PRIM_M)
   B = tensor_bspy.contiguous().view(Bd,         SPSEQ, PRIM_K, YSEQ, PRIM_N)
   C_view = C.view(Ad, Bd, Cd, YSEQ, PRIM_N, XSEQ, PRIM_M)

Diese Reshapes sind *gratis*, weil ``s,p`` adjazent im Speicher liegen
(``stride_s = stride_p · |p|``) — genau der ``fuse_dims``-Check aus
Assignment 05, der hier nicht mehr explizit gebraucht wird.

Pre-Loading via ``ct.extract``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Task 3 hatten wir das Pre-Loading-Pattern (Slide 11) als Plan
notiert: einen größeren A-Tile global laden und per ``ct.extract``
im Schleifenrumpf in mma-Größe schneiden. Im Profiling-Test brachte
das auf den vorliegenden Shapes **keinen** Gewinn — der Engpass war
nicht die A-Bandbreite, sondern die per-Block-Arbeit (siehe Task 4c:
große PRIM-Tiles helfen, Pre-Loading nicht). Daher zeigt der eingereichte
Kernel die schlichte Variante ohne ``ct.extract`` – Slide 11 wäre
relevant bei einem K-dominanten Workload mit kleinen PRIM-M/N.

Task 4b: Verifikation
----------------------

Gegen ``torch.einsum("acspx,bspy->abcyx", a16, b16)`` mit FP32-Promotion
und FP16-Rückgabe, ``atol=2e-1, rtol=2e-2``:

.. code-block:: text

   baseline      allclose=True   max_abs_err=0.2500
   l2-swizzle    allclose=True   max_abs_err=0.2500
   big-prim 128  allclose=True   max_abs_err=0.2500

Alle Varianten liegen im FP16-Quantisierungsrauschen (max 0,25 absolute
Abweichung — die K-Dim ist mit 4096 FP16-Akkumulationen pro Output-Element
deutlich tiefer als in den vorherigen Assignments, daher der etwas
größere Fehler als die üblichen ``0.0078 = 2⁻⁷``).

Task 4c: Benchmark
-------------------

``triton.testing.do_bench(warmup=10, rep=50)``, DGX Spark (GB10), FP16.
FLOPs der Kontraktion:

.. math::

   2 \cdot |a| \cdot |b| \cdot |c| \cdot |s| \cdot |p| \cdot |x| \cdot |y|
   \;=\; 2 \cdot 4 \cdot 4 \cdot 3 \cdot 64 \cdot 64 \cdot 1536 \cdot 1152
   \;\approx\; 6{,}96 \cdot 10^{11}

.. list-table::
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - Variante
     - Tile (M,N,K)
     - ms
     - TFLOPS
     - vs. ``torch``
   * - ``torch.einsum`` (Referenz)
     - —
     - 11,36
     - **61,25**
     - 1,00×
   * - ``baseline`` (Task 4a)
     - (64, 64, 32)
     - 42,47
     - 16,39
     - 0,27×
   * - ``big-prim``
     - (128, 128, 32)
     - **24,51**
     - **28,38**
     - **0,46×**
   * - ``big-prim`` (PRIM_K=64)
     - (128, 128, 64)
     - 25,54
     - 27,24
     - 0,44×
   * - ``l2-swizzle`` GY=9
     - (64, 64, 32)
     - 45,10
     - 15,43
     - 0,25×

**Beobachtungen.**

* Der **Baseline-Kernel mit der unveränderten Task-3-Config** erreicht
  **16,4 TFLOPS** — eine Größenordnung über dem, was naive
  Python-Schleifen erlauben würden, aber klar unter den ~75 TFLOPS aus
  Assignment 05. Begründung: die PRIM-Tiles aus Task 3 sind mit
  ``(64, 64, 32)`` für *kleine* GEMM-Dims optimiert (Assignment 04
  Task 3 Sweep) — bei :math:`x = 1536, y = 1152` macht ein einzelnes
  Output-Tile von ``64×64`` nur ``1/(24·18) = 0{,}23 %`` des Outputs
  aus, und der Per-Block-Overhead wird sichtbar.

* **``big-prim`` mit (128, 128, 32) erreicht 28,4 TFLOPS** — Faktor
  1,73× gegenüber Baseline ohne sonstige Änderungen. Hardware-Begründung
  identisch zur Heatmap aus Assignment 03 Task 3b: bei großen Problemen
  liegt der Sweet Spot bei ``128×128`` (mehr Arbeit pro Issue, bessere
  Tensor-Core-Auslastung). Größeres ``PRIM_K = 64`` bringt **nichts**
  zusätzlich, weil bereits ``PRIM_K = 32`` die K-Schleife auf 128
  Iterationen drückt — der Tensor-Core ist nicht K-issue-bound.

* **``l2-swizzle`` *verschlechtert* den Durchsatz konsistent**, je
  kleiner ``GY`` desto schlimmer. Erklärung: das natürliche Grid-Mapping
  ``pid_x = bid_xy // YSEQ`` lässt ``YSEQ = 18`` konsekutive BIDs
  denselben A-Streifen teilen — dort kommt der A-Reuse *gratis*.
  Jedes ``GY < 18`` bricht genau diesen impliziten Streifen auf,
  bevor B-Reuse einen Gewinn bringen könnte. Lehre: vor manuellem
  Swizzling immer das Default-Mapping nachrechnen.

* Erst bei ``GY = 9`` (also halbe y-Spalte pro Gruppe) nähern wir uns
  wieder dem Baseline-Durchsatz, ohne ihn zu erreichen. Bei ``GY = YSEQ
  = 18`` wäre der Kernel mathematisch äquivalent zum Baseline.

Optionaler Task: Beat ``torch.einsum``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Auch mit allen oben gezeigten Optimierungen bleiben wir bei
**0,46×** gegenüber ``torch.einsum``. Vermutlich passiert dort:

1. **opt_einsum**-Routing: das Modul ist explizit als
   ``import opt_einsum`` im Boilerplate enthalten — torch nutzt das
   für die optimale Kontraktions-Reihenfolge und/oder die Übersetzung
   in mehrere ``torch.matmul`` / cuBLAS-Calls.
2. **cuBLAS GEMM** mit ausgereiften Tile-Cascades, Software-Pipelining,
   warp-level Speculative Prefetch usw., die ein handgeschriebener
   cuTile-Kernel ohne signifikanten Aufwand nicht erreicht.

Beat-Path-Ideen, die im Rahmen dieses Assignments nicht mehr verfolgt
wurden:

* **Multi-PRIM-K** (Slide 7): zwei oder mehr ``ct.mma``-Calls pro
  Schleifeniteration, jeder mit unterschiedlichem K-Slice – mehr
  Tensor-Core-Issue-Rate.
* **Persistent Kernel + Producer/Consumer**: ein Block bedient mehrere
  Output-Tiles und überlappt Load mit Compute (Slide 12 / Cooperative
  Groups). cuTile bietet kein direktes Pipelining-Primitiv.
* **opt_einsum-äquivalentes Splitting**: die Kontraktion vorher als
  zwei batched GEMMs zerlegen (``a c (sp) x`` × ``b (sp) y`` →
  ``(a c) b (sp) y x`` Pfad). Dann ist das aber kein einzelner
  cuTile-Kernel mehr.

Stand der Abgabe: **Kernel verifiziert, 28,4 TFLOPS (best)**,
``torch.einsum`` nicht geschlagen — der optionale Task bleibt offen,
mit dokumentierten nächsten Schritten.

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
     - cuTile-Kernel ``kernel_baseline`` strikt nach Task-3-Config,
       Verifikation gegen ``torch.einsum``, Benchmark + Sweep über
       PRIM-Tile-Größen und Y-Swizzle (Task 4a–c sowie optionaler Task);
       Report-Abschnitt zu Task 4.
