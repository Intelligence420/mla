.. _gsc_pitch:

#####
Pitch
#####

Am **17.06.2026** werden beide Projektideen vorgestellt (~5 min pro Idee). Der
Pitch wird **nicht benotet**, dient aber dem Feedback der Betreuer und der
Entscheidung, welche Idee tatsächlich umgesetzt wird. Pro Idee sind mindestens
zwei Slides vorgesehen, die jeweils die folgenden vier Punkte abdecken:
**Einführung**, **Problemformulierung**, **Lösungsansatz** und
**erwartete Ergebnisse / Erkenntnisse**.

.. _gsc_pitch_idee1:

Idee 1 — XDNA-NPU
=================

.. note:: TODO — Titel und Inhalt werden gemeinsam formuliert.

Einführung
----------

.. note:: TODO

Problemformulierung
-------------------

.. note:: TODO

Lösungsansatz
-------------

.. note:: TODO

Erwartete Ergebnisse / Erkenntnisse
-----------------------------------

.. note:: TODO

.. _gsc_pitch_idee2:

Idee 2 — GPU/cuTile: Interaktiver einsum/GEMM-Performance-Explorer
==================================================================

.. tip::

   Ein Tool, dem man eine Matrix-/Einsum-Operation gibt;
   es baut daraus automatisch ein GPU-Kernel, lässt Zahlenformat und
   Kachelgröße verstellen und zeigt live in Graphen, wie schnell und wie
   genau das Ergebnis ist.

.. note::

   Die GUI ist nur die Visualisierungs-Schicht — die eigentliche
   Substanz sind Kernel-Erzeugung, Autotuning und ehrliche Messung. 

Einführung
----------

Aus den bisherigen Assignments existieren bereits manuell gebaute
cuTile-Kernel für GEMM, Tensor-Kontraktionen, Multi-Input-Einsum und
Swizzling. Diese Idee bündelt das in einen **interaktiven
Performance-Explorer**: Aus einem einsum-/GEMM-Ausdruck wird ein
cuTile-Kernel generiert (Idee „Einsum→cuTile"), und dessen Leistung sowie
Genauigkeit werden bei verstellbarem Zahlenformat und Tiling sofort
visualisiert (Idee „Mixed-Precision/Roofline").

Problemformulierung
-------------------

Für jede Kontraktion gibt es viele Varianten von Tiling, Swizzling und
Zahlenformat — mit stark unterschiedlichem Durchsatz und unterschiedlicher
Genauigkeit. Die beste Konfiguration von Hand zu finden ist mühsam, und der
Zusammenhang zwischen Geschwindigkeit, Genauigkeit und Hardware-Peak bleibt
dabei unsichtbar.

Lösungsansatz
-------------

Pipeline: Ausdruck parsen → Kontraktionsreihenfolge via ``opt_einsum`` →
parametrierter cuTile-Codegen (Tile-Größe, Swizzle, ``dtype``) → Messung auf
der GPU → Visualisierung in einem Web-Frontend.

* Mindestens: Ausdruck eingeben → Kernel generieren → Format und Tile-Größe
  umstellbar → zwei Graphen: Durchsatz (vs. cuBLAS) und Genauigkeit
  (Fehler gegen fp32-Referenz).
* Optional/Zusätlich (wenn die Zeit reicht): Roofline-Plot, Heatmap der Tile-Autotuning-Landschaft,
  „Auto-Tune"-Button, Anzeige des generierten Kernel-Codes und optionales

Erwartete Ergebnisse / Erkenntnisse
-----------------------------------

* Der Accuracy-Throughput-Tradeoff der Formate (fp16/bf16/fp8) wird sichtbar.
* Form der Autotuning-Landschaft — welche Tile-Größen lohnen sich?

