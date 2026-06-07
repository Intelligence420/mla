.. _ch07_feedback:

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

   | Abgabestruktur nicht eingehalten und alte Dateien im Ordner ``tar``
   |   abgegeben (−2 P.).
   |
   | „``mov p1, p4``" gehört zum M-Slot.
   |
   | Hinweis: Register-Klassen pro Slot sind nicht vollständig, siehe Folien.

Organisatorisches (−2 P.)
=========================

Berechtigt. Abgabestruktur nicht eingehalten und veraltete Dateien im
``tar``-Ordner mitgegeben – reiner Abgabe-Fehler, keine fachliche Frage.

Task 3: ``mov p1, p4`` gehört zum M-Slot
========================================

Berechtigt. In ``loesung.rst`` wurde ``mov p1, p4`` dem **X-Slot** zugeordnet
(„Pointer-Arithmetik läuft im Scalar/Control-Halb des XM-Slots"). Richtig ist
der **M-Slot**:

Der XM-Slot vereint zwei Einheiten – **X** (Scalar/Control: skalare ALU,
Control-Flow, Immediate-Moves nach ``r``) und **M** (Move-Unit:
Register-zu-Register-Moves). ``mov p1, p4`` ist ein reiner Register-Move ohne
Arithmetik → das ist Aufgabe der **Move-Unit**, also M. Die ursprüngliche
Begründung („Pointer-Arithmetik") trägt nicht, weil hier nichts gerechnet,
sondern nur kopiert wird.

Das war auch **inkonsistent**: ``vmov bmhl2, bmhh4`` wurde korrekt dem M-Slot
zugeordnet (Akkumulator-Move). Ein Pointer-Move gehört nach derselben Logik
ebenfalls in den M-Slot – beide sind Register-Moves, nur andere Register-Klasse.

Folgefehler in der Register-Klassen-Tabelle
-------------------------------------------

Aus der falschen Slot-Zuordnung folgt direkt:

- Die **M-Zeile** listet nur ``dst/src = Akkumulator (bm)``. Sie muss zusätzlich
  die **Pointer-Klasse** (``p``) enthalten – Beispiel ``p1, p4`` (``mov``).
- Die **X-Zeile** führt ``p1, p4`` fälschlich als X-Beispiel; das gehört in die
  M-Zeile.

Task 3: Register-Klassen pro Slot unvollständig
===============================================

Berechtigt (Hinweis). Die Tabelle wurde **nur aus den Operanden der sieben
Beispiel-Instruktionen** abgeleitet (so auch in der Doku vermerkt). Damit
listet sie pro Slot nur die zufällig im Beispiel vorkommenden Register-Klassen
und nicht den vollständigen, in den Folien definierten Satz – z. B. fehlt im
M-Slot die Pointer-Klasse (siehe oben), und auch die übrigen Slots sind so nur
ausschnittweise belegt. Korrekt wäre, die vollständigen Klassen pro Slot aus
den Folien zur Slot-/Register-Definition zu übernehmen statt sie aus den
Beispielen zu rekonstruieren.

.. note::

   Die Folien-Belege konnten lokal nicht erneut geprüft werden (PDF-Rendering
   nicht verfügbar); diese Einordnung folgt der Kontrolleur-Korrektur und der
   Slot-Logik. Vor einer Korrektur in ``loesung.rst`` die genauen
   Register-Klassen an der ISA-/Compute-Tile-Folie verifizieren.
