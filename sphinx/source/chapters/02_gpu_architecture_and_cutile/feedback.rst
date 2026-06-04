.. _ch02_feedback:

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

   | zu 3b: „Diese nicht-zusammenhängenden Zugriffe können nicht gecached
   |   werden" → „nicht zusammengefasst" wäre hier exakter.
   |
   | zu 4b: Gerne noch die ganze Range plotten :)

Task 3b: „gecached" vs. „zusammengefasst"
=========================================

Berechtigt. Die Formulierung in ``loesung.rst`` war konzeptionell unsauber:

   „Diese nicht-zusammenhängenden Zugriffe **können nicht gecached werden**
   und führen zu vielen separaten Speichertransaktionen (uncoalesced
   accesses)."

Strided Global-Memory-Zugriffe gehen sehr wohl durch den L1-/L2-Cache – das
Cachen wird nicht abgeschaltet. Was *nicht* passiert, ist das **Coalescing**:
Die Hardware kann die Zugriffe eines Warps bei großem Stride nicht zu wenigen
breiten Transaktionen **zusammenfassen**, daher viele kleine Transaktionen und
schlechte Bus-Auslastung. „Nicht zusammengefasst (uncoalesced)" trifft die
Ursache, „nicht gecached" verfehlt sie. Der eigentliche Befund (Variante 2 ist
≈ 3,6× langsamer wegen strided Zugriffen) bleibt korrekt.

Task 4b: Plot-Auflösung
=======================

Keine fachliche Korrektur, nur eine Sampling-Anregung: statt der diskreten
16er-Schritte (``N = 16, 32, …, 128``) die volle Range dichter abtasten, um
auch die Zwischenwerte (z. B. Nicht-Zweierpotenzen) im Kurvenverlauf zu zeigen.

In ``src/task_04.py`` (Zeile 102) wurde nur in 16er-Schritten gemessen:

.. code-block:: python

   ns = list(range(16, 129, 16))             # [16, 32, 48, …, 128] – zu grob

Besser wäre die volle Range mit Schrittweite 1 gewesen, damit auch die
Einbrüche an Nicht-Zweierpotenzen (``tile_N`` wird hochgerundet) sichtbar
werden:

.. code-block:: python

   ns = list(range(16, 129))                 # [16, 17, 18, …, 128] – ganze Range

Der restliche Mess- und Plot-Code (``bandwidth_benchmark()`` in
``src/task_04.py``) bleibt unverändert; lediglich die Wertemenge ``ns`` wird
dichter.
