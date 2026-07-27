.. _gsc_report_ergebnisse:

##########################
Teil 5 — Beispiel Analyse
##########################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Alle Kennwerte dieses Teils stammen aus einer einzigen verifizierten Sweep-Charge
(``CLI-Report-Sweep``, 33 Konfigurationen, Status durchgängig ``ok``) auf der
GB10. Sie sind damit unter identischen Hardware- und Softwarebedingungen erhoben
und nicht aus mehreren Läufen zusammengesetzt. Der Formatvergleich verwendet den
Ausdruck ``ik,kj->ij`` bei :math:`1024^3`. Die speichergebundenen Familien werden
bei :math:`4096^2` gemessen, die ``GROUP_M``-Reihe bei :math:`4096^3`.

Einzellauf über alle Pipeline-Stufen
====================================

Zur Nachvollziehbarkeit der folgenden Aggregate wird zunächst ein einzelner Lauf
vollständig dokumentiert. Angegeben sind die tatsächlich gespeicherten Werte. Der
Lauf entspricht der ersten Zeile der Formatvergleichstabelle weiter unten.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Stufe
     - Ergebnis
   * - **Eingabe**
     - ``ik,kj->ij`` · ``{i: 1024, k: 1024, j: 1024}`` · fp16 → fp32 ·
       Tile 128/128/64 · kein Swizzle · Baseline cuBLAS · 10 warmup / 30 iters
   * - **[1] parse**
     - ``ContractionIR`` mit M=[i], N=[j], K=[k], Batch=[]. Es handelt sich um
       die einfachste mögliche Klassifikation, erzeugt jedoch von demselben Code,
       der auch ``acspx,bspy->abcyx`` zerlegt.
   * - **[2] B1-Reshape**
     - ``Canonical(M=1024, N=1024, K=1024, B=1)``. Der Umbau ist die Identität
       (``transform_needed = False``), da ein zweidimensionales GEMM bereits
       kanonisch vorliegt. Die Batch-Achse wird mit Länge 1 ergänzt.
   * - **[3] Codegen**
     - 75 Zeilen cuTile-Quelltext mit den Literalen ``TM=128 TN=128 TK=64`` und
       dem Akkumulator ``ct.float32``. Cast-, Swizzle- und Epilog-Block
       entfallen, da fp16 nativ unterstützt wird und keine der beiden Optionen
       gewählt ist.
   * - **[4] compile**
     - Slug ``ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64``, abgelegt als
       ``results/kernels/ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py``. Das Grid
       umfasst :math:`(8, 8, 1)`, also 64 Blöcke, die K-Schleife 16 Kacheln.
   * - **[5] Kalt-Lauf**
     - ``compile_ms = 50,7``, gemessen als Wall-Clock-Zeit des host-seitigen
       cuTile-JIT.
   * - **[6] verify**
     - Referenz ``torch.einsum("ik,kj->ij", A.float(), B.float())``. Es ergeben
       sich ``max_abs_err = 3,20·10⁻⁴``, ``mean_abs_err = 3,19·10⁻⁵`` und
       ``rel_err = 1,29·10⁻⁶``. Das Ergebnis ist ein PASS bei ``atol = 0,2`` und
       ``rtol = 0,02``.
   * - **[7] Messung**
     - 30 getaktete Iterationen mit ``run_ms = 0,07582`` als Median sowie
       ``min = 0,07360``, ``p90 = 0,08307`` und ``σ = 0,00820``.
   * - **Kennzahlen**
     - Aus :math:`2 \cdot 1024^3 = 2{,}147` GFLOP und 0,07582 ms folgen
       28,32 TFLOP/s. Der Traffic von 8 MiB entspricht 110,6 GB/s bei einer
       arithmetischen Intensität von 256 FLOP/Byte. Das sind 13,3 % des
       fp16-Rechenpeaks und 40,5 % der theoretischen Bandbreite.
   * - **Baseline**
     - cuBLAS über ``torch.matmul`` mit gleichen Operanden und gleicher
       Messschleife benötigt 0,0553 ms und erreicht 38,84 TFLOP/s. Der
       generierte Kernel erreicht 72,9 % dieses Werts.
   * - **Provenienz**
     - GPU „NVIDIA GB10", SM-Takt 2418 MHz, 38 °C, 9,45 W, Auslastung 0 %. Die
       Abfrage erfolgt nach der Messschleife, die GPU befand sich zu diesem
       Zeitpunkt bereits im Idle-Zustand.
   * - **[8] Store**
     - Eine JSON-Zeile in ``results/results.jsonl`` mit der ``run_id`` der
       Charge.

Zwei Werte dieser Tabelle bedürfen einer Einordnung.

Der Durchsatz von 28,32 TFLOP/s entspricht 13 % des fp16-Rechenpeaks von
213 TFLOP/s. Bei einer arithmetischen Intensität von 256 FLOP/Byte ist der Peak
jedoch nicht die bindende Schranke. Bindend ist die Bandbreiten-Schräge mit
:math:`0{,}273 \cdot 256 = 69{,}9` TFLOP/s, und darauf bezogen beträgt die
Ausnutzung 41 %. cuBLAS erreicht mit 38,8 TFLOP/s 56 % derselben Schranke.

Der maximale absolute Fehler von :math:`3{,}2 \cdot 10^{-4}` entspricht der
Erwartung für eine Summe über 1024 fp16-Produkte bei fp32-Akkumulation.

Einordnung im Roofline-Modell
=============================

.. figure:: /_static/gsc/roofline.png
   :align: center
   :width: 100%
   :alt: Roofline-Diagramm der GB10 mit memory-bound- und compute-nahen Punkten

   Roofline der GB10 mit zwei Bandbreiten-Schrägen. Die obere entspricht den
   theoretischen 273 GB/s, die gestrichelte den im ``copy``-Lauf gemessenen
   223 GB/s. Die speichergebundenen Familien liegen bei einer arithmetischen
   Intensität von 0,08 bis 0,5, die Kontraktionen bei 171 bis 512. Die n-äre
   Kette ist als ein aggregierter Punkt eingetragen.

Der Ridge-Point der GB10 liegt bei etwa 780 FLOP/Byte, bezogen auf die gemessene
Bandbreite bei etwa 955 FLOP/Byte. Beide Werte liegen oberhalb der arithmetischen
Intensität auch großer GEMMs. Für alle Punkte des Formatvergleichs ist deshalb die
Bandbreiten-Schräge die bindende Schranke und nicht der Rechenpeak. Die
213 TFLOP/s sind bei diesen Intensitäten grundsätzlich nicht erreichbar.

Die zweite, gemessene Schräge dient als praktische Obergrenze. Eine elementweise
Addition erreicht 80,6 % der theoretischen und damit 99 % der gemessenen
Bandbreite. Optimierungspotenzial besteht folglich nur bei den Punkten, die
deutlich unter beiden Schrägen liegen.

.. note::

   Die ``copy``-Läufe sind in der Figur nicht als Punkte, sondern als
   gestrichelte Linie dargestellt. Mit null FLOP und damit einer arithmetischen
   Intensität von 0 besitzen sie auf einer logarithmischen Achse keine Position.
   Ihre Darstellung als Obergrenze ist deshalb die inhaltlich korrekte.

Ein einziger Punkt der Charge liegt rechts vom Ridge-Point, das
:math:`4096^3`-GEMM mit einer arithmetischen Intensität von 1024. Dort ist die
Rechenschranke bindend, und der Kernel erreicht mit 73,7 TFLOP/s 34,6 % des
fp16-Peaks, also mehr als das Doppelte des Anteils bei :math:`1024^3`. Der Punkt
ist nicht in der Roofline-Figur eingetragen, da er die Formatpunkte überdecken
würde. Er wird im Abschnitt zur Blockumordnung ausgewertet.

Formatvergleich: Durchsatz und Genauigkeit
==========================================

.. figure:: /_static/gsc/durchsatz_formate.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz je Zahlenformat, cuTile gegen cuBLAS

   Durchsatz der Kontraktion je Zahlenformat, generierter cuTile-Kernel gegen
   die cuBLAS-Obergrenze aus ``torch.matmul``. Für fp8 existiert kein direkter
   ``matmul``-Pfad, weshalb die Vergleichssäule entfällt.

.. figure:: /_static/gsc/genauigkeit_durchsatz.png
   :align: center
   :width: 90%
   :alt: Streudiagramm Genauigkeit gegen Durchsatz je Format

   Genauigkeit gegen Durchsatz. fp16 und bf16 erreichen die geringsten Fehler,
   fp8 den höchsten Durchsatz bei größtem Fehler. tf32 ist in beiden Dimensionen
   unterlegen.

.. list-table:: Kontraktion :math:`1024^3`, Tile 128/128/64 (verifiziert)
   :header-rows: 1
   :widths: 22 15 15 12 12 12 12

   * - Format
     - cuTile [TFLOP/s]
     - cuBLAS [TFLOP/s]
     - Anteil
     - max. abs. Fehler
     - rel. Fehler
     - GB/s
   * - fp16 → fp32
     - 28,3
     - 38,8
     - **73 %**
     - 3,2·10⁻⁴
     - 1,3·10⁻⁶
     - 111
   * - bf16 → fp32
     - 28,7
     - 38,5
     - **75 %**
     - 2,1·10⁻⁴
     - 1,0·10⁻⁶
     - 112
   * - tf32 → fp32
     - 8,3
     - 20,2
     - 41 %
     - 4,6·10⁻²
     - 2,9·10⁻⁴
     - 49
   * - fp8 e4m3 → fp16
     - **47,7**
     - —
     - —
     - 3,4·10⁻¹
     - 8,4·10⁻⁴
     - 93

Aus dem Vergleich ergeben sich vier Befunde.

* Der auf f-Strings basierende Codegen erreicht bei fp16 und bf16 73 bis 75 % des
  cuBLAS-Durchsatzes. Erreicht wird das ohne Autotuning, ohne
  Software-Pipelining und ohne handoptimierte Ladepfade. Für eine
  Template-Generierung dieses Umfangs ist der Abstand zur Bibliotheksimplementierung
  damit geringer als erwartet.
* fp8 liefert mit 47,7 TFLOP/s den höchsten Durchsatz bei zugleich größtem Fehler
  von 3,4·10⁻¹. Ein Teil des Vorsprungs ist auf die Bandbreite zurückzuführen, da
  ein Byte je Eingabeelement die arithmetische Intensität auf 512 verdoppelt.
* tf32 ist in dieser Messreihe dominiert. Es ist langsamer als fp16 und
  gleichzeitig um zwei Größenordnungen ungenauer. Der Rechenpeak liegt mit
  53 TFLOP/s deutlich unter den 213 TFLOP/s von fp16, und die Operanden belegen
  4 statt 2 Byte, was die arithmetische Intensität auf 171 senkt.
* Für fp8 liegt keine cuBLAS-Referenz vor. ``torch`` meldet ``"baddbmm_cuda" not
  implemented for 'Float8_e4m3fn'``. Die Baseline wird deshalb mit
  ``available: false`` und dem Grund gespeichert, statt ohne Angabe zu entfallen.

Kachelung als Tuning-Achse
==========================

.. figure:: /_static/gsc/tile_swizzle.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz über Tile-Größen und Swizzle-Gruppengrößen

   fp16-Tuning-Raum bei :math:`1024^3`. Zugrunde liegt derselbe verifizierte
   Kernel, variiert werden ausschließlich Kachelung und Blockumordnung.

.. list-table:: fp16, :math:`1024^3` — Kachelung und L2-Swizzle
   :header-rows: 1
   :widths: 34 16 16 34

   * - Konfiguration
     - TFLOP/s
     - run_ms
     - Anmerkung
   * - Tile 256/128/64
     - **5,5**
     - 0,392
     - Minimum der Reihe, Faktor 5,2 unter der besten Kachel
   * - Tile 64/64/32
     - 25,6
     - 0,084
     - Mehr Blöcke, geringere Wiederverwendung je Block
   * - Tile 128/128/64
     - 28,3
     - 0,076
     - Referenzkonfiguration aller übrigen Messungen
   * - + Swizzle G8
     - 29,1
     - 0,074
     - Identische Permutation wie G16 und G32, siehe folgender Abschnitt
   * - + Swizzle G16
     - 29,0
     - 0,074
     - Identische Permutation wie G8 und G32
   * - + Swizzle G32
     - 29,2
     - 0,074
     - Identische Permutation wie G8 und G16

Die Kachelwahl ist die wirksamste einzelne Konfigurationsachse des Werkzeugs. Die
Kachel 256/128/64 fällt auf 5,5 TFLOP/s ab, was einem Faktor 5,2 gegenüber
128/128/64 entspricht, und zwar bei identischem Ausdruck und identischem
Ergebnis. Als Ursache kommt die Blockzahl in Betracht. Bei :math:`M = 1024` und
:math:`TM = 256` entstehen nur :math:`4 \times 8 = 32` Blöcke für 48 SMs, sodass
ein Teil der Multiprozessoren unbeschäftigt bleibt. Zusätzlich erhöhen größere
Kacheln den Registerdruck. Ein Nachweis dieser Ursache erfordert Profiling und
wird mit den vorliegenden Daten nicht geführt. Für die Fragestellung des Werkzeugs
genügt, dass der Effekt reproduzierbar messbar ist.

Blockumordnung: Abhängigkeit von der Gittergröße
================================================

Die drei Swizzle-Konfigurationen der vorigen Tabelle liegen innerhalb von 0,7 %
beieinander. Dieser Befund ist strukturell bedingt und kein Ergebnis über
``GROUP_M``.

Der Grund liegt in der Gittergröße. :math:`1024^3` mit ``TM = TN = 128`` ergibt
ein :math:`8 \times 8`-Blockgitter. Die Rasterung begrenzt die Gruppengröße auf
``min(num_pid_m - first_pid_m, GROUP_M)``, hier also auf 8. Es entsteht eine
einzige Gruppe, und für jedes ``GROUP_M ≥ 8`` ist die Blockpermutation identisch.
Die drei Zeilen entsprechen damit demselben Kernel, ihre Streuung ist die
Messunsicherheit.

Messbar wird die Achse erst auf einem Gitter, das mehrere Gruppen zulässt.
:math:`4096^3` ergibt :math:`32 \times 32` Blöcke:

.. figure:: /_static/gsc/group_m.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz über GROUP_M bei 4096 hoch 3

   ``GROUP_M`` auf einem :math:`32 \times 32`-Blockgitter. Kernel, Zahlenformat
   und Kachelung sind identisch, variiert wird ausschließlich die Zuordnung von
   Blöcken zu Kacheln.

.. list-table:: fp16, :math:`4096^3`, Tile 128/128/64 — ``GROUP_M`` (verifiziert)
   :header-rows: 1
   :widths: 20 14 16 16 34

   * - Konfiguration
     - Gruppen
     - TFLOP/s
     - run_ms
     - relativ
   * - ohne Swizzle
     - —
     - 36,3
     - 3,790
     - 1,00× (Bezugspunkt)
   * - Swizzle G2
     - 16
     - 45,9
     - 2,997
     - 1,26×
   * - Swizzle G4
     - 8
     - 68,0
     - 2,022
     - 1,87×
   * - **Swizzle G8**
     - 4
     - **73,7**
     - **1,866**
     - **2,03×**
   * - Swizzle G16
     - 2
     - 61,8
     - 2,223
     - 1,70×
   * - Swizzle G32
     - 1
     - 38,8
     - 3,543
     - 1,07×

Die Reihe stützt drei Aussagen.

1. Die Blockumordnung ist kein Feinjustierungseffekt, sondern verändert den
   Durchsatz um den Faktor 2,03. Das Ergebnis bleibt dabei bit-identisch, da die
   Permutation bijektiv ist. Die Verifikation liefert für alle sechs Läufe
   denselben Fehler von 2,62·10⁻³. Verändert wird ausschließlich der Zeitpunkt,
   zu dem eine Kachel im L2 vorliegt.
2. ``GROUP_M`` besitzt ein Optimum und keine monotone Wirkungsrichtung. Kleine
   Gruppen wie G2 erzeugen wenig Wiederverwendung, große wie G32 überschreiten
   die Cache-Kapazität. Das Maximum liegt bei G8, also bei dem Wert, der in
   Triton-Beispielen als Konvention verwendet wird. Die Messung bestätigt diese
   Konvention für den vorliegenden Fall. Eine Implementierung, die ``GROUP_M``
   nicht als freien Parameter führt, kann den Verlauf nicht abbilden.
3. G32 fällt mit 1,07× nahezu auf den Bezugspunkt zurück. Das bestätigt die
   strukturelle Erklärung des vorigen Abschnitts. Ist ``GROUP_M`` gleich der
   Gitterhöhe, entsteht wieder eine einzige Gruppe, und die Permutation
   degeneriert zu einer Transposition der Durchlaufreihenfolge. Es handelt sich um
   denselben Mechanismus, der bei :math:`1024^3` alle Werte ab 8 gleichsetzt.

Methodisch folgt daraus, dass eine Tuning-Achse auf einer zu klein gewählten
Testform strukturell unsichtbar bleiben kann. Eine Messung ausschließlich bei
:math:`1024^3` hätte den Effekt der Blockumordnung mit 3 % angegeben. Dieser Wert
wäre nicht durch Messfehler falsch, sondern durch die Wahl der Problemform.

Speichergebundene Familien: Bandbreite als Primärmetrik
=======================================================

.. list-table:: Elementwise & Reduktion & Copy, :math:`4096^2` (verifiziert)
   :header-rows: 1
   :widths: 22 18 12 14 12 22

   * - Familie · Op
     - Format
     - GB/s
     - % th. BW
     - AI
     - max. abs. Fehler
   * - Elementwise · copy
     - fp16 → fp16
     - 209,7
     - 76,8 %
     - 0
     - 0 (exakt)
   * - Elementwise · copy
     - fp16 → fp32
     - **222,9**
     - **81,7 %**
     - 0
     - 0 (exakt)
   * - Elementwise · copy
     - fp32 → fp32
     - 222,2
     - 81,4 %
     - 0
     - 0 (exakt)
   * - Elementwise · add
     - fp16 → fp32
     - 219,9
     - 80,6 %
     - 0,12
     - 2,0·10⁻³
   * - Elementwise · add
     - bf16 → fp32
     - 220,3
     - 80,7 %
     - 0,12
     - 1,6·10⁻²
   * - Elementwise · add
     - fp32 → fp32
     - 224,8
     - 82,4 %
     - 0,08
     - 0 (exakt)
   * - Reduktion · sum
     - fp16 → fp32
     - 172,6
     - 63,2 %
     - 0,50
     - 3,1·10⁻⁵
   * - Reduktion · sum
     - bf16 → fp32
     - 172,5
     - 63,2 %
     - 0,50
     - 3,1·10⁻⁵
   * - Reduktion · sum
     - fp32 → fp32
     - 215,6
     - 79,0 %
     - 0,25
     - 3,8·10⁻⁵

* Die elementweise Addition ist bandbreitengesättigt. Die 80 bis 82 % der
  theoretischen Bandbreite entsprechen 98 bis 101 % der gemessenen
  ``copy``-Obergrenze von 222,9 GB/s. Ein Optimierungsspielraum besteht damit
  nicht mehr.
* Die Kopie dient als Bezugspunkt und zeigt ein zunächst kontraintuitives
  Verhalten. ``fp16 → fp16`` bewegt 4 Byte je Element und erreicht mit
  209,7 GB/s weniger als ``fp32 → fp32`` mit 8 Byte je Element und 222,2 GB/s.
  Erklärbar ist das über den Overhead. Bei gleicher Elementzahl bewegt der
  schmale Fall halb so viele Bytes, während die Kosten je Element für
  Adressierung, Kachelverwaltung und Launch konstant bleiben. Die Bandbreite ist
  dann nicht mehr der einzige begrenzende Faktor.
* Die Reduktion erreicht in fp16 und bf16 63 % und in fp32 79 % der theoretischen
  Bandbreite. Die Ursache ist dieselbe, da sie im schmalen Format halb so viele
  Bytes liest und deshalb relativ stärker von den übrigen Kosten begrenzt wird.
* Die Genauigkeit der Reduktion ist mit etwa 3·10⁻⁵ formatunabhängig, da
  unabhängig vom Eingabeformat im fp32-Akkumulator summiert wird. Diese
  Eigenschaft gilt erst nach der Korrektur eines Codegen-Defekts, der weiter
  unten dokumentiert ist.

N-äre Kette
===========

Die Kette ``ij,jk,kl->il`` bei :math:`256^4` wird in zwei paarweise GEMMs
zerlegt. Der geplante Pfad lautet ``ij,jk->ik`` und anschließend ``kl,ik->il``.
Verifiziert wird gegen ``torch.einsum`` über den vollständigen Ausdruck in fp32,
gemessen wird die Kette als ein aggregierter Punkt. Das Ergebnis sind
1,64 TFLOP/s bei einer arithmetischen Intensität von 64 FLOP/Byte.

Dass die Intensität unter der eines einzelnen GEMMs liegt, ist erwartungskonform.
Der Zwischentensor wird geschrieben und erneut gelesen, was in die aggregierten
Bytes eingeht. Der folgende Abschnitt untersucht dieselbe Kostenart systematisch.

.. note::

   Bei installiertem ``opt_einsum`` kann der Planer eine andere Zerlegung wählen
   als den Fold von links nach rechts, wodurch ein anderer Kernel-Pfad gemessen
   wird. Der geplante Pfad ist deshalb in jeder Zeile von ``results.jsonl`` unter
   ``provenance.sizes.path`` protokolliert.

Fusion des Epilogs auf dem Akkumulator
======================================

Werden eine Kontraktion und die anschließende elementweise Operation, also
:math:`C = A \cdot B` gefolgt von :math:`+D` oder :math:`\max(\cdot, 0)`, in zwei
getrennten Kerneln ausgeführt, wird das Zwischenergebnis nach DRAM geschrieben
und unmittelbar danach wieder gelesen. Auf einer speichergebundenen Maschine geht
dieser Umweg vollständig in die Laufzeit ein. Die Fusion vermeidet genau diesen
Roundtrip von :math:`2 \cdot 4 \cdot M \cdot N` Bytes.

Die eingesparte Datenmenge ist von :math:`K` unabhängig, die Kosten der
Kontraktion steigen dagegen mit :math:`K`. Der Sweep variiert deshalb die
arithmetische Intensität und nicht die Arbeitsmenge. Die schmale und die
quadratische Form weisen mit 2,15 GFLOP dieselbe FLOP-Zahl auf:

.. list-table:: Fusion vs. sequentiell (fp16 → fp32, gegen ``torch.einsum`` + Epilog verifiziert)
   :header-rows: 1
   :widths: 20 12 10 13 15 15 15

   * - Form (M·N·K)
     - Epilog
     - AI
     - fused [ms]
     - sequentiell [ms]
     - Speedup
     - gespart
   * - 4096·4096·64
     - bias
     - 21
     - 0,522
     - 1,108
     - **2,12×**
     - 128 MiB
   * - 4096·4096·64
     - relu
     - 32
     - 0,349
     - 0,945
     - **2,71×**
     - 128 MiB
   * - 1024·1024·1024
     - bias
     - 205
     - 0,087
     - 0,107
     - 1,22×
     - 8 MiB
   * - 1024·1024·1024
     - relu
     - 256
     - 0,076
     - 0,095
     - 1,25×
     - 8 MiB
   * - 1024·1024·8192
     - bias
     - 431
     - 0,367
     - 0,381
     - 1,04×
     - 8 MiB
   * - 1024·1024·8192
     - relu
     - 455
     - 0,361
     - 0,370
     - 1,03×
     - 8 MiB

.. figure:: /_static/gsc/fusion.png
   :align: center
   :width: 100%
   :alt: Speedup der Fusion über der arithmetischen Intensität, je Epilog eine Kurve

   Fusions-Speedup über der arithmetischen Intensität. Links die schmale Form
   4096·4096·64, in der Mitte :math:`1024^3`, rechts die tiefe Form
   1024·1024·8192. Beide Kurven fallen monoton gegen die Referenzlinie 1,0.

Der Verlauf ist ein monotoner Trend über drei Formen und kein Einzelbefund.

* Bei der schmalen, speichergebundenen Form liegt der Speedup bei 2,12× und
  2,71×, da der eingesparte Roundtrip von 128 MiB die Laufzeit dominiert. Der
  fusionierte Kernel erreicht dabei 195 GB/s, also 87 % der gemessenen
  Bandbreitenobergrenze, und liegt damit nahe am Maschinenlimit.
* Bei der tiefen, rechendominierten Form beträgt der Speedup 1,03 bis 1,04×. Die
  Kontraktion benötigt dort 0,36 ms, der eingesparte Roundtrip von 8 MiB
  rechnerisch 0,01 bis 0,02 ms, was in der Streuung der Rechenzeit aufgeht.
* ``relu`` erreicht durchgängig höhere Speedups als ``bias``. Das ist
  konsistent, da ``relu`` keinen zusätzlichen Operanden benötigt und der
  fusionierte Kernel damit nur A und B liest, während ``bias`` zusätzlich das
  Bias-Feld D lädt.

In dieselbe Systematik ordnet sich der Befund aus Assignment 04 ein, der dieses
Teilziel motiviert hat. Dort lag die Fusion mit 0,984× geringfügig unter dem
sequentiellen Pfad, bei einer Kontraktion von 12,83 ms gegenüber einem Epilog von
0,067 ms. Diese Form liegt bezüglich der arithmetischen Intensität oberhalb der
hier tiefsten Form. Jenseits des Punktes, an dem der eingesparte Speicherverkehr
messbar ist, verbleibt nur der Zusatzaufwand des größeren Kernels. Die zulässige
Aussage lautet daher nicht, dass Fusion schneller ist, sondern:

.. admonition:: Kernaussage

   Der Gewinn der Fusion skaliert mit dem Grad, in dem eine Operation
   bandbreiten- und nicht rechenlimitiert ist. Die GB10 besitzt mit einem
   Ridge-Point von etwa 780 FLOP/Byte einen großen bandbreitenlimitierten
   Bereich.

Zwei Eigenschaften der Umsetzung sind bewusst konservativ gewählt. Erstens ist
die Fusion rein additiv. Ohne gewählten Epilog erzeugt der Codegen
byte-identischen Quelltext und denselben Slug wie vor der Erweiterung, was ein
Textvergleich im Test absichert. Zweitens wird der sequentielle Vergleichspfad
selbst gemessen, als zweiter Kernel-Paar-Lauf im selben ``run()`` mit identischer
Messschleife, und ebenfalls gegen fp32 verifiziert. Der Speedup ist damit ein
Vergleich gegen eine gemessene und verifizierte Alternative und nicht gegen eine
Schätzung. Schlägt diese Zweitmessung fehl, entfällt nur der Vergleich, nicht das
Ergebnis des Laufs.

Verifikation: ein erkannter Codegen-Defekt
==========================================

Die dokumentierte Charge umfasst 33 Konfigurationen, die alle die
fp32-Verifikation bestehen. Dieser Stand ist selbst ein Resultat des Gates, da
eine vorangehende Charge einen Fehlschlag enthielt.

**Beobachtung.** Die bf16-Reduktion über :math:`4096^2` überschritt die Toleranz
deutlich, mit einem maximalen absoluten Fehler von 1,574 bei einem ``atol`` von
1,0. Der Lauf erhielt den Status ``verify_failed``, wurde ohne Durchsatzwert
gespeichert und ging in keine Figur ein. Naheliegend wäre die Deutung gewesen,
dass bf16 mit 8 Mantissenbits für eine Summe über 4096 Summanden zu grob
auflöst. Diese Deutung war falsch.

**Ursache.** Der Defekt lag im eigenen Codegen. Das Reduktions-Template besitzt
zwei Pfade. Der K-Loop-Fallback akkumulierte im angeforderten Akkumulatorformat,
der für alle Report-Größen gewählte single-shot-Pfad dagegen im Eingabeformat.
Dort stand ``ct.sum(tile, axis=1)`` anstelle von ``ct.sum(ct.astype(tile,
ct.float32), axis=1)``, sodass der Parameter ``acc_dtype`` auf diesem Pfad
wirkungslos blieb. Auffällig wurde der Defekt an einer Stelle, an der die
Erwartung entgegengesetzt ist. Dieselbe Eingabe über eine doppelt so lange Achse
reduziert, also jenseits von :math:`K = 16384` und damit über den Loop-Pfad, war
drei Größenordnungen genauer. Der Unterschied lag folglich nicht in der Länge der
Summe, sondern im Akkumulatorformat.

**Korrektur.** Die Korrektur besteht in einer Zeile, dem Cast vor ``ct.sum``
analog zum Loop-Pfad. Danach liefert dieselbe bf16-Reduktion einen Fehler von
3,05·10⁻⁵, also einen um den Faktor 51 000 kleineren Wert. Der Lauf besteht die
Verifikation und ist in der Tabelle der speichergebundenen Familien enthalten.
Die fp16-Reduktion lag zuvor mit einem Fehler von 0,22 innerhalb ihrer Toleranz
und war als ``ok`` gewertet worden. Sie verbesserte sich mit derselben Korrektur
auf 3,05·10⁻⁵.

**Bewertung.** Das Gate hat keine Eigenschaft eines Zahlenformats angezeigt,
sondern einen Defekt im generierten Kernel, und zwar bevor eine falsch
etikettierte Genauigkeitsangabe in diesen Bericht eingegangen ist. Ohne
fp32-Referenz wäre der Defekt nicht erkennbar gewesen, da ein Wert von 1,574 bei
einer Zeilensumme über 4096 bf16-Werte als Formatgrenze interpretierbar ist und
der fp16-Fall mit 0,22 unauffällig blieb. Genau gegen diese Klasse stiller
Falschergebnisse muss generierter Kernel-Code abgesichert werden.

Ein Toleranzfall bleibt daneben inhaltlich echt. Bei tieferen n-ären fp16-Ketten
summiert sich ab :math:`384^4` der Fehler beider GEMM-Schritte über die Toleranz.
Der Bericht zeigt die Kette deshalb bei :math:`256^4`. Solche Fälle werden als
Fehlschlag gemeldet und nicht mit einem Ergebniswert versehen.

Messstreuung und Compile-Zeit
=============================

Durchsatzangaben mit drei signifikanten Stellen erfordern eine Angabe zur
Stabilität. Da jede Messung als Verteilung über 30 Iterationen gespeichert wird,
ist diese Angabe aus derselben Charge ableitbar:

.. list-table:: Streuung über 30 getaktete Iterationen (dieselbe Charge)
   :header-rows: 1
   :widths: 30 16 16 16 22

   * - Lauf
     - Median [ms]
     - σ [ms]
     - σ / Median
     - p90 / Median
   * - Kontraktion 1024³ fp16
     - 0,0758
     - 0,0082
     - 11 %
     - 1,10
   * - Kontraktion 4096³ fp16, G8
     - 1,866
     - 0,184
     - 10 %
     - 1,23
   * - Elementwise add 4096² fp16
     - 0,610
     - 0,0065
     - 1,1 %
     - 1,01
   * - Reduktion 4096² fp16
     - 0,195
     - 0,0008
     - 0,4 %
     - 1,01
   * - Copy 4096² fp32
     - 0,604
     - 0,0027
     - 0,4 %
     - 1,01

Das Muster ist konsistent. Die speichergebundenen Läufe sind mit einem
relativen σ unter 1 % gut reproduzierbar, da sie durchgängig an der Bandbreite
anliegen und wenig Spielraum für Variation besteht. Die Kontraktionen streuen mit
etwa 10 % stärker, und ihr p90 liegt 10 bis 23 % über dem Median. Hier wirken
Takt- und Cache-Effekte, und der L2-Flush zwischen den Iterationen erzeugt
absichtlich für jede Iteration einen kalten Startzustand, dessen Kosten von der
Verdrängungsreihenfolge abhängen. Als Kennwert wird deshalb der Median
gespeichert, da ein arithmetisches Mittel von den p90-Ausreißern nach oben
verzerrt würde.

Für die Compile-Zeit gilt eine Einschränkung. ``compile_ms`` erfasst den ersten
Launch im laufenden Prozess. In dieser Charge liegen die Werte zwischen 9,8 und
50,7 ms, allerdings nur deshalb, weil viele Läufe einen Kernel-Slug verwenden,
der im selben Prozess bereits compiliert wurde. Der :math:`4096^3`-Sweep nutzt
beispielsweise dasselbe Artefakt wie der :math:`1024^3`-Sweep. Für einen erstmals
compilierten Kernel in einem frischen Prozess reichen die Werte im Store von rund
310 ms bis über 1,7 s, wobei der Maximalwert auf die tf32-Varianten der
Tensor-Kontraktion ``acspx,bspy->abcyx`` fällt, deren Kernel-Cast den JIT
zusätzlich belastet. Die Größe beschreibt damit keine Kerneleigenschaft, sondern
den Zustand des Prozesses. Sie wird deshalb getrennt von ``run_ms`` geführt und
geht in keine Kennzahl ein.

Gültigkeitsgrenzen
==================

Die folgenden Einschränkungen begrenzen die Reichweite der berichteten
Ergebnisse:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Grenze
     - Konsequenz
   * - GB/s sind eine Untergrenze
     - Die Byte-Zahlen entsprechen dem algorithmischen Mindest-Traffic ohne
       Tiling-Rereads. Der reale DRAM-Verkehr einer Kontraktion ist höher, die
       Angabe in Prozent der Bandbreite entsprechend konservativ. Für die
       speichergebundenen Familien ist der Wert scharf, da jedes Element genau
       einmal gelesen wird.
   * - Kein Autotuning
     - Alle Kachel- und Swizzle-Angaben sind gemessene Einzelkonfigurationen und
       keine gefundenen Optima. Dass G8 bei :math:`4096^3` das Maximum der
       geprüften Werte bildet, belegt keine globale Optimalität.
   * - Rechenpeaks aus Fremdmessung
     - Die Rechenpeaks stammen aus einem veröffentlichten Microbenchmark und
       nicht aus einem Whitepaper. Alle Angaben in Prozent des Peaks erben diese
       Unsicherheit. Die Bandbreitenschranke ist demgegenüber selbst gemessen.
   * - Nur zwei Operanden pro Kernel
     - N-äre Ausdrücke werden zerlegt. Diagonalen und Spuren wie ``ii->i`` sind
       nicht unterstützt und werden mit Fehlermeldung abgelehnt.
   * - Epilog nur bei 2-Operanden-Kontraktionen
     - Auf einer n-ären Kette bliebe der Epilog ohne Wirkung. Die Kombination
       wird deshalb explizit abgelehnt und nicht stillschweigend ignoriert.
   * - Eine Maschine, eine GPU
     - Alle Ergebnisse gelten für diese GB10. Es liegen keine Multi-GPU-Läufe und
       kein Vergleich gegen andere Hardware vor.
   * - Reduktions-Fallback nicht aus A02 belegt
     - Der K-Loop-Pfad des Reduktions-Templates ist gegen ``torch`` verifiziert,
       jedoch nicht durch eine Assignment-Vorlage abgesichert. Im generierten
       Code ist er entsprechend markiert.
