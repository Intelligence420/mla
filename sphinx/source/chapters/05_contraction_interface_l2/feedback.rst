.. _ch05_feedback:

#####################
Feedback-Auswertung
#####################

.. admonition:: KI-generierte Schnell-Analyse (nachträglich)
   :class: warning

   Diese Seite ist eine **nachträgliche, KI-gestützte Schnell-Analyse** des
   Kontrolleur-Feedbacks gegen Code, Doku und Aufgabenstellung. Sie entstand
   *nach* der Abgabe und ist eine grobe Selbst-Einordnung – keine geprüfte
   Korrektur und nicht Teil der ursprünglichen Lösung.

Feedback der Kontrolleure
=========================

   | Task 3b (−0,5): Es wird zwar überprüft, ob die Dimensionen benachbart in
   |   jedem Tensor sind, jedoch nicht, ob sie auch die gleiche relative
   |   Reihenfolge in jedem Tensor haben.
   |
   | Task 4 (−1 P.): Implementierung der Swizzling-Logik im Kernel, jedoch
   |   nicht direkt über die Kontraktions-Configuration: Ein „Super-Tile"
   |   kann durch Dimensionssplitting in der Config erreicht werden.

Task 3b: ``fuse_dims`` prüft Reihenfolge nicht
==============================================

Berechtigt (−0,5 P.). Die Adjazenz-Prüfung in ``src/optimizer.py`` (Zeile 63)
testet pro Tensor mit einem **ODER**:

.. code-block:: python

   adjacent = (stra == strb * size_b) or (stra * size_a == strb)

Der linke Zweig bedeutet „``a`` außen, ``b`` innen" (Reihenfolge ``a,b``), der
rechte „``b`` außen, ``a`` innen" (Reihenfolge ``b,a``). Weil jeder Tensor
unabhängig geprüft wird, akzeptiert die Schleife auch den Fall, dass Tensor X
die Reihenfolge ``a,b`` und Tensor Y die Reihenfolge ``b,a`` hat. Beide sind je
für sich „benachbart", lassen sich aber **nicht** zu einer einzigen
konsistenten Dimension verschmelzen – genau das fehlte.

Wie es richtig gewesen wäre
---------------------------

Die relative Reihenfolge muss über alle Tensoren, in denen beide Dims
auftauchen, **identisch** sein:

.. code-block:: python

   order = None  # "ab" = a außen, "ba" = b außen
   for t, strides in enumerate(cfg.strides):
       stra, strb = strides[dim_id_a], strides[dim_id_b]
       if stra == 0 or strb == 0:
           continue
       if   stra == strb * size_b:   this = "ab"
       elif stra * size_a == strb:   this = "ba"
       else:
           raise ValueError(f"dims not adjacent in tensor {t}")
       if order is None:
           order = this
       elif order != this:
           raise ValueError(
               f"dims adjacent but relative order differs in tensor {t} "
               f"({order} vs {this}) – fusion not well-defined")

Praktischer Schaden hier gering, weil die Task-4-Pipeline nur ``split_dim``
nutzt und ``fuse_dims`` nur in einem Round-Trip-Sanity-Check vorkommt – die
Funktion ist aber als allgemeiner Optimizer-Baustein spezifiziert und damit zu
permissiv.

Task 4: Super-Tile gehört in die Config, nicht in den Kernel
============================================================

Berechtigt (−1 P.). Die ``build_l2_config``-Pipeline splittet ``m`` und ``n`` in
``(l2, prim)`` und produziert das deklarative 6-Dim-Layout – aber der Kernel
``kernel_l2_optimized`` (``src/kernel.py``, Zeile 129) **ignoriert** dieses
Layout und re-implementiert die L2-Gruppierung von Hand als klassischen
Triton/CUTLASS-BID-Swizzle mit ``GROUP_M``/``GROUP_N`` (Zeile 47):

.. code-block:: python

   blocks_per_group = group_m * num_n_tiles
   group_id         = pid_id // blocks_per_group
   pid_m            = first_m_in_group + (in_group % group_size_m)
   pid_n            = in_group // group_size_m

Die Doku benennt den Bruch selbst: die Split-Config sei „nur die *deklarative*
Seite", die echte Optimierung komme „aus einem Super-Tile-Swizzle im Kernel".
Genau das war nicht gefragt – der ganze Sinn des ``Config``/``Optimizer``-
Interfaces ist, das Super-Tiling **datengetrieben** auszudrücken.

Wie es gemeint war
------------------

Das Super-Tile entsteht durch eine zusätzliche Split-Ebene: ``m`` (und ``n``)
nicht nur in ``(l2, prim)``, sondern die L2-Achse weiter in
``(super, group)``. Die Gruppen-Achsen werden so permutiert, dass die
Grid-Enumeration sie zusammenhängend abläuft – das *ist* der Swizzle, ganz ohne
Index-Arithmetik im Kernel:

.. code-block:: python

   # m, n = 4096; mma-Tile = 64 -> 64 Tiles je Achse; Gruppe 8x8
   opt.split_dim(m_id, 64, 64)        # m -> (m_l2=64, m_prim=64)
   opt.split_dim(m_l2_id, 8, 8)       # m_l2 -> (m_super=8, m_group=8)
   opt.split_dim(n_id, 64, 64)        # n -> (n_l2=64, n_prim=64)
   opt.split_dim(n_l2_id, 8, 8)       # n_l2 -> (n_super=8, n_group=8)
   # PAR-Layout: [c, m_super, n_super, m_group, n_group], PRIM: [m_prim, n_prim, k]
   opt.permute_dims([...])            # group-Achsen innen -> Super-Tile pro (super)-Block

Der Kernel bleibt dann **generisch**: Grid über die PAR-Dimensionen, GEMM über
die PRIM-Dimensionen ``(m_prim, n_prim, k)`` – die L2-Lokalität fällt aus der
Enumerationsreihenfolge der gesplitteten Achsen, nicht aus
``// blocks_per_group``. Funktional war das Ergebnis korrekt und schnell; der
Abzug betrifft, dass die Optimierung am vorgesehenen Interface vorbei
implementiert wurde.
