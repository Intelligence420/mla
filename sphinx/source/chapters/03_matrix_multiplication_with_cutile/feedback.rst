.. _ch03_feedback:

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

   | Anmerkung: Die GB10 hat 48 SMs, nicht 108.
   |
   | 4a (−0,5 P.): Begründung fehlt, warum ``GROUP_SIZE_M = 8`` gewählt
   |   wurde (vielleicht ist es besser, verschiedene ``GROUP_SIZE_M`` je
   |   nach Matrizengröße zu wählen?).

Anmerkung: SM-Zahl
==================

Berechtigt. Die GB10 (DGX Spark) hat **48 SMs**; die abgegebene Fassung nannte
108. Die Doku verwendet inzwischen durchgängig den korrekten Wert (vgl.
``loesung.rst``, Skalierungs-Beobachtung zu Task 3b).

Task 4a: Begründung für ``GROUP_SIZE_M = 8``
============================================

Berechtigt (−0,5 P.). Die Doku erklärt den **Mechanismus** der Super-Grouping
(``GROUP_SIZE_M`` benachbarte Blöcke teilen sich einen A-Tile-Streifen → mehr
L2-Wiederverwendung), begründet aber nie, **warum gerade 8** – der Wert in
``src/task_04.py`` (Zeile 37) ist gesetzt, nicht hergeleitet:

.. code-block:: python

   GROUP_SIZE_M = 8     # fest gewählt, ohne Sweep/Begründung

Was gefehlt hat
---------------

``GROUP_SIZE_M`` ist ein Trade-off: größere Gruppen erhöhen die A-Streifen-
Wiederverwendung, vergrößern aber den gleichzeitig „heißen" Working-Set
(B-Streifen × Gruppenhöhe) im 24 MB L2. Das Optimum hängt damit von Tile-Shape
**und Matrixform** ab – genau der Punkt des Kontrolleurs. Ein Wert wäre zu
belegen gewesen, nicht zu setzen.

Wie es besser gewesen wäre
--------------------------

Die Host-Funktion ``cutile_matmul_swizzled`` nimmt ``group_size_m`` bereits als
Parameter – ein kleiner Sweep pro Matrixgröße hätte den Wert empirisch belegt:

.. code-block:: python

   # Sweep statt fester 8: bestes GROUP_SIZE_M je Matrixgröße bestimmen
   import triton

   for (M, N, K) in [(512, 512, 4096), (2048, 2048, 4096), (8192, 8192, 4096)]:
       A = torch.randn(M, K, device="cuda", dtype=torch.float16)
       B = torch.randn(K, N, device="cuda", dtype=torch.float16)
       best = min(
           (1, 2, 4, 8, 16, 32),
           key=lambda g: triton.testing.do_bench(
               lambda: cutile_matmul_swizzled(A, B, 64, 64, 64, group_size_m=g)))
       print(f"{M}x{N}x{K}: bestes GROUP_SIZE_M = {best}")

Ergebnis wäre eine kleine Tabelle „Matrixgröße → bestes ``GROUP_SIZE_M``"
gewesen; 8 ist plausibel als Default, aber für kleine ``N`` (wenige
Block-Spalten) oft zu groß und für sehr große Matrizen ggf. zu klein.
