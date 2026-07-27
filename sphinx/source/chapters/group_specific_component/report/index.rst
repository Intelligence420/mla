.. _gsc_report:

##############
Project Report
##############

Finale Abgabe am **27.07.2026**. Die Group-Specific Component ist nicht bloß
Dokumentation, sondern ein **eigenständiger Projektbericht**.

Das **cuTile Performance Lab** ist ein interaktiver einsum-/GEMM-Explorer für die
NVIDIA **GB10** (Grace-Blackwell, ``sm_121``): Man gibt einen einsum-Ausdruck ein
(z. B. ``ik,kj->ij``), wählt Zahlenformat, Kachelung und Cache-Umordnung — das Tool
**erzeugt daraus automatisch einen cuTile-Kernel**, verifiziert ihn gegen eine
fp32-Referenz, misst ihn auf der GPU und stellt Durchsatz, Genauigkeit und die
Roofline live gegenüber. Es beantwortet damit die Leitfrage jeder GPU-Kontraktion:
*Wie schnell, wie genau, und wie nah am Hardware-Limit?*

Der Kern ist bewusst **kein Mockup**: Die grafische Oberfläche ist nur die
Visualisierungs-Schicht über einer vollständigen, gegen fp32 verifizierten
Pipeline. Alle Zahlen und Figuren dieses Berichts stammen aus **einem**
reproduzierbaren Batch-Lauf des Werkzeugs selbst (``python -m tool_pipeline.cli
--sweep``, 33 Konfigurationen), und es fließt **kein** Ergebnis ein, das die
fp32-Verifikation nicht bestanden hat (*verify-before-trust*).

Wie dieser Bericht zu lesen ist
===============================

Der Bericht ist so geschrieben, dass er **von vorne nach hinten** gelesen werden
kann und dabei nichts voraussetzt außer GPU-Grundlagen: Was die Maschine kann,
warum das Roofline-Modell die richtige Brille ist und was cuTile überhaupt ist,
wird in Teil 1 hergeleitet — erst danach kommt die Umsetzung.

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
       Oberfläche — und warum genau **einer**?
   * - :ref:`3 — Die Pipeline im Detail <gsc_report_pipeline>`
     - Wie wird aus einem einsum-String ein lauffähiger, verifizierter,
       gemessener GPU-Kernel — Stufe für Stufe, mit Begründung je Entscheidung?
   * - :ref:`4 — Das Frontend <gsc_report_frontend>`
     - Warum Plotly Dash, wie übersteht eine Weboberfläche einen mehrsekündigen
       GPU-Job, und wie ist die Bedienung aufgebaut?
   * - :ref:`5 — Beispiel Analyse <gsc_report_ergebnisse>`
     - Was kommt dabei heraus — ein vollständig dokumentierter Einzellauf, dann
       die Messreihen, ihre Streuung und die Gültigkeitsgrenzen.
   * - :ref:`6 — Starten und Benutzen <gsc_report_bedienung>`
     - Wie starte und bediene ich das Werkzeug konkret? Alle Befehle, die
       Oberfläche Schritt für Schritt, die CLI-Schalter und die Fehlerbilder.
   * - :ref:`7 — Anhang <gsc_report_anhang>`
     - Was steckt im Sweep, wie sehen die Daten aus, was decken die Tests ab?

Die Kernaussagen in fünf Sätzen
===============================

Jede ist in Teil 5 mit Messwerten belegt:

* **Die GB10 ist eine memory-bound Maschine.** Ihr Ridge-Point liegt bei
  ≈ 780 FLOP/Byte; die praktisch erreichbare Bandbreite ist mit **223 GB/s**
  (an reiner Datenbewegung gemessen) 82 % des theoretischen Werts. Elementweise
  Operationen erreichen 80–82 % — sie laufen also am Anschlag der Maschine.
* **Ein einfacher f-String-Codegen kommt auf 73 % von cuBLAS** (fp16/bf16 bei
  1024³) — ohne Autotuning.
* **Die Kachelwahl ist der stärkste Hebel** (Faktor 5 zwischen guter und
  schlechter Kachel bei identischem Ausdruck), und der **L2-Swizzle ist kein Feintuning**: auf einem
  großen Blockgitter (4096³) ist er **2,03×** wert, auf einem kleinen
  wirkungslos. Beides ist derselbe Kernel — nur die Block-Reihenfolge ändert sich.
* **Fusion zahlt sich genau in dem Maß aus, in dem eine Operation bandbreiten-
  und nicht rechenlimitiert ist** — von 2,71× (schmale Form) monoton fallend auf
  1,03× (tiefe Form).
* **Das Verify-Gate hat einen echten Codegen-Bug gefangen**, der aussah wie eine
  Format-Grenze: ein Akkumulator im falschen Format, 51 000× zu großer Fehler.

.. toctree::
   :maxdepth: 2

   01_grundlagen
   02_architektur
   03_pipeline
   04_frontend
   05_bsp_analyse
   06_bedienung
   07_anhang
