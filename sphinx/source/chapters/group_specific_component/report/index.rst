.. _gsc_report:

##############
Project Report
##############

Finale Abgabe am **27.07.2026**. Ein Projektbericht.

Das **cuTile Performance Lab** ist ein interaktiver einsum-/GEMM-Explorer für die
NVIDIA **GB10** (Grace-Blackwell, ``sm_121``): Man gibt einen einsum-Ausdruck ein
(z. B. ``ik,kj->ij``), wählt Zahlenformat, Kachelung und Cache-Umordnung — das Tool
**erzeugt daraus automatisch einen cuTile-Kernel**, verifiziert ihn gegen eine
fp32-Referenz, misst ihn auf der GPU und stellt Durchsatz, Genauigkeit und die
Roofline live gegenüber. Es beantwortet damit die Leitfrage jeder GPU-Kontraktion:
*Wie schnell, wie genau, und wie nah am Hardware-Limit?*

Die grafische Oberfläche ist nur die
Visualisierungs-Schicht über einer vollständigen, gegen fp32 verifizierten
Pipeline. Alle Zahlen und Figuren dieses Berichts stammen aus einem
reproduzierbaren Batch-Lauf des Werkzeugs selbst und es fließt kein Ergebnis ein, das die
fp32-Verifikation nicht bestanden hat (*verify-before-trust*).

Wie dieser Bericht zu lesen ist
===============================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Teil
     - Beantwortet die Frage
   * - :ref:`1 — Grundlagen <gsc_report_grundlagen>`
     - Auf welcher Maschine läuft das, wo ist ihr Limit, und mit welchem Modell
       wird das sichtbar? Was ist cuTile, und welche Risiken hat generierter
       Kernel-Code?
   * - :ref:`2 — Architektur <gsc_report_architektur>`
     - Wie ist das System geschnitten? Was ist der Vertrag zwischen Kern und
       Oberfläche?
   * - :ref:`3 — Die Pipeline im Detail <gsc_report_pipeline>`
     - Wie wird aus einem einsum-String ein lauffähiger, verifizierter,
       gemessener GPU-Kernel — Stufe für Stufe
   * - :ref:`4 — Das Frontend <gsc_report_frontend>`
     - Warum Plotly Dash? Wie wird der mehrsekündigen
       GPU-Job gehandhabt. Was gibt es für Bedienung?
   * - :ref:`5 — Beispiel Analyse <gsc_report_ergebnisse>`
     - Was kommt dabei heraus — ein vollständig dokumentierter Einzellauf, dann
       die Messreihen, ihre Streuung und die Gültigkeitsgrenzen.
   * - :ref:`6 — Starten und Benutzen <gsc_report_bedienung>`
     - Wie starte und bediene ich das Werkzeug konkret? Alle Befehle, die
       Oberfläche Schritt für Schritt, die CLI-Schalter und die Fehlerbilder.
   * - :ref:`7 — Anhang <gsc_report_anhang>`
     - Was steckt im Sweep, wie sehen die Daten aus, was decken die Tests ab?

.. toctree::
   :maxdepth: 2

   01_grundlagen
   02_architektur
   03_pipeline
   04_frontend
   05_bsp_analyse
   06_bedienung
   07_anhang
