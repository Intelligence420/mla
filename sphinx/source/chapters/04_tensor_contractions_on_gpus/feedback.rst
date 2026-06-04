.. _ch04_feedback:

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

   −0,5 P. (1c): Kernel aus 1c nie schneller als 1b.

Task 1c: c) nie schneller als b)
================================

Berechtigt (−0,5 P.). Die Aufgabe verlangt für 1c **explizit** eine
Konfiguration, „where your new kernel from c) performs better" als b). Diese
Konfiguration wurde nicht gefunden – die Doku berichtet selbst, dass b) in
*beiden* Settings gewinnt (auch bei ``|b|=16`` lief das Gegenteil der Erwartung,
3,35 vs. 1,48 TFLOPS). Die Aufgabenanforderung blieb damit unerfüllt.

Warum c) verlor
---------------

Der Grund liegt nicht (nur) im Kernel, sondern in der **Wahl der
Vergleichs-Config**. Die c-freundliche Konfiguration in ``src/task_01.py``
(Zeile 441) erhöht zwar ``|b|``, kollabiert dafür aber alle übrigen
Parallel-Dimensionen:

.. code-block:: python

   cfg_bc_c = dict(E=1, A=2, B=16, C=2, K=2, L=2, X=128, Y=64, Z=128)

c) verschiebt ``b`` vom Grid in eine innere Schleife – das Grid schrumpft um
den Faktor ``|b|``. Mit ``E=1, A=2, C=2`` und Tile ``32`` bleibt c) nur ein
Grid von ``(E·A, C, ⌈X/tx⌉·⌈Z/tz⌉) = (2, 2, 16) = 64`` Blöcken übrig –
für 48 SMs occupancy-hungrig und ohne Latenz-Überdeckung. b) startet auf
derselben Form ``(2, 32, 16) = 1024`` Blöcke. Der mögliche L2-Reuse-Gewinn von
c) (B-Tile hängt nicht von ``b`` ab) kann diesen 16-fachen Occupancy-Verlust
nicht kompensieren. Die Doku vermutet das korrekt, zog daraus aber nicht den
Schluss, die Config anders zu wählen.

Wie es besser gewesen wäre
--------------------------

Großes ``|b|`` für Reuse **und** genug Restparallelität, damit c)s Grid die 48
SMs noch sättigt – große ``E``/``A`` und mehr X·Z-Tiles statt geschrumpfter
Dimensionen:

.. code-block:: python

   # |b| groß für B-Tile-Reuse, ABER Grid bleibt groß (Occupancy erhalten)
   cfg_bc_c = dict(E=8, A=4, B=8, C=4, K=2, L=2, X=256, Y=64, Z=256)
   # c)-Grid: (E·A, C, num_x·num_z) = (32, 4, 64) = 8192 Blöcke

Ehrlich bleibt: c) macht exakt dieselbe MMA-Arbeit wie b) und gewinnt
ausschließlich durch B-Tile-Reuse. Auf der GB10 kann b) dieselben B-Tiles
ebenfalls aus dem L2 ziehen, sodass c)s Vorteil auch bei fairer Config klein
und nicht garantiert ist. Der Punktabzug ist dennoch korrekt: gefordert war
ein Nachweis, und dafür hätte mindestens eine occupancy-erhaltende Config mit
spürbarer L2-Last getestet werden müssen.
