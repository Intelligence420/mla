.. _ch06_feedback:

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

   −1 P. (3a): Die Swizzling-Logik kann direkt in der Config abgebildet werden.
   Die Teilungsfaktoren in der optimierten Config sind grundsätzlich gut, jedoch
   ist die Reihenfolge der Dimensionen nicht optimal. Hier sollten die parallelen
   M- und N-Dimensionen verschachtelt sein.

Task 3a: Dimensionsreihenfolge + Swizzle in der Config
======================================================

Berechtigt (−1 P.), und es ist derselbe Befund wie in Assignment 05, Task 4
(:ref:`ch05_feedback`): die L2-Optimierung wurde am ``Config``-Interface vorbei
in den Kernel gelegt, statt sie über die Dimensionsstruktur auszudrücken.

Was stimmte
-----------

Die **Teilungsfaktoren** sind in Ordnung: ``sp``-Fusion plus Splits
``x → (x_seq, x_prim=64)``, ``y → (y_seq, y_prim=64)``,
``sp → (sp_seq, sp_prim=32)`` (``src/main.py``, Zeile 100–103) – PRIM-Tiles
wie in Assignment 05 vom GB10-Peak übernommen. Das hat der Kontrolleur explizit
bestätigt.

Was nicht optimal war
---------------------

Die PAR-Achsen stehen als **erst alle M, dann alle N** in der Config
(``loesung.rst``: „5× PAR (3M, 2N)"):

.. code-block:: text

   pos name    type exec   size
   0   a       M    PAR       4
   1   c       M    PAR       3
   2   x_seq   M    PAR      24     <- M-Block
   3   b       N    PAR       4
   4   y_seq   N    PAR      18     <- N-Block

cuTile enumeriert die PAR-Achsen mit der rechtesten als innerster. Damit fegt
eine Wave bei festem ``(a, c, x_seq)`` über **alle** ``y_seq`` – die A-Tiles
(M-Seite) werden wiederverwendet, aber jeder Block lädt ein neues B-Tile
(N-Seite). Genau dieses fehlende B-Reuse hat der abgegebene Kernel dann per
**manuellem BID-Swizzle** nachgerüstet (``src/kernel.py``, Zeile 97 ff.,
``blocks_per_group = GY * XSEQ``) – also außerhalb der Config.

Wie es gemeint war
------------------

Die M- und N-PAR-Achsen **verschachteln**, sodass benachbarte BIDs ein
2D-Super-Tile (ein paar M- *und* ein paar N-Tiles) abdecken → A *und* B liegen
gemeinsam im L2. Sauber ausgedrückt durch eine zusätzliche Split-Ebene und eine
Permutation, die je eine M- und N-Gruppe nach innen zieht:

.. code-block:: python

   # x_seq=24, y_seq=18 weiter in (super, group) splitten
   opt.split_dim(x_seq_id, 24 // GX, GX)     # -> x_super, x_group
   opt.split_dim(y_seq_id, 18 // GY, GY)     # -> y_super, y_group
   # PAR-Order: [a, c, x_super, y_super, x_group, y_group]  (group-Achsen innen)
   opt.permute_dims([...])

Bei festem ``(a, c, x_super, y_super)`` läuft die Wave dann über das
``x_group × y_group``-Rechteck – das *ist* der Super-Tile-Swizzle, rein
deklarativ. Der Kernel wird damit generisch (Grid über PAR, GEMM über PRIM) und
braucht weder ``GY`` noch ``// blocks_per_group``. Funktional war die Abgabe
korrekt und der GY-Sweep belegt den Effekt – der Abzug betrifft, dass das
Interface der Aufgabe nicht genutzt wurde.
