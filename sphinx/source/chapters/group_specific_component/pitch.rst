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

Idee 1 — XDNA-NPU: Fusionierter Transformer-FFN-Block
=====================================================

.. tip::

   Die zwei großen Matrix-Multiplikationen einer Transformer-Schicht plus die
   Aktivierungsfunktion dazwischen zu *einem* Ablauf auf der NPU verschmelzen,
   statt das große Zwischenergebnis mehrfach durch den langsamen Hauptspeicher
   zu schicken.

Einführung
----------

Der Feed-Forward-Block (``FFN``, auch „MLP-Block") ist neben der Attention der
zweite Kernbaustein jeder Transformer-Schicht und wird auf jeden Token gleich
angewendet: ``y = W₂ · aktivierung(W₁ · x + b₁) + b₂``. Er besteht aus zwei
Linear-Schichten (= zwei GEMMs) mit einer elementweisen Aktivierung dazwischen
— und damit aus genau den Bausteinen, die bereits existieren: die
Whole-NPU-GEMM (A10), die elementweise Addition für den Bias (A07) und der
Tensor-Microkernel (A08). Das Projekt verkettet und *verschmilzt* diese
Bausteine zu einer kompletten FFN-Teilschicht.

Problemformulierung
-------------------

Die erste Linear-Schicht bläht die Daten auf (typisch ein rund 4× breiteres
Zwischenergebnis ``H``), die zweite staucht sie wieder zusammen. Führt man die
drei Schritte getrennt aus, wird dieses große Zwischenergebnis nach L3
geschrieben, für die Aktivierung wieder gelesen, erneut geschrieben und für das
zweite GEMM nochmals gelesen. Genau dieser Datenbewegungs- und
Dispatch-Overhead dominierte bereits in A10 bei kleinen Problemgrößen.

Lösungsansatz
-------------

Beide GEMMs und die Aktivierung werden zu einer Pipeline verschmolzen: Die
Ergebnis-Kachel des ersten GEMM wird **direkt auf dem Compute-Tile** mit Bias
versehen und aktiviert und sofort in das zweite GEMM weitergereicht — das
Zwischenergebnis bleibt L1/L2-resident und wird nie nach L3 zurückgeschrieben.
Wiederverwendet werden der Matmul-Microkernel (zweimal) und die elementweise
Addition; der eigentliche Aufwand liegt im MLIR-AIE-Datenfluss, der die beiden
GEMMs mit der Zwischenstufe verkettet.

* Mindestens: ein fusionierter FFN-Block mit **ReLU** (``max(0, x)``, trivial
  auf der Vektoreinheit), gegen eine PyTorch-Referenz verifiziert.
* Optional (wenn die Zeit reicht): **GELU** über eine Polynom-/LUT-Approximation
  auf der AIE2-Vektoreinheit.

Erwartete Ergebnisse / Erkenntnisse
-----------------------------------

* Speedup des fusionierten Blocks gegenüber drei separaten
  GEMM-/Aktivierungs-Dispatches.
* Ab welcher Problemgröße sich die Fusion lohnt (speicher- vs.
  rechengebunden).
* bf16-Genauigkeit; bei GELU zusätzlich die Güte der Approximation.

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

.. figure:: /_static/mla_gpu_mockup_1.png
   :width: 100%
   :alt: Mockup des interaktiven einsum/GEMM-Performance-Explorers

   KI generiert

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
  „Auto-Tune"-Button, Anzeige des generierten Kernel-Codes

Erwartete Ergebnisse / Erkenntnisse
-----------------------------------

* Der Accuracy-Throughput-Tradeoff der Formate (fp16/bf16/fp8) wird sichtbar.
* Form der Autotuning-Landschaft — welche Tile-Größen lohnen sich?

