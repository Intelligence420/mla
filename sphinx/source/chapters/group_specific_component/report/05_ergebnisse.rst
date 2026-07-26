.. _gsc_report_ergebnisse:

##################################
Teil 5 — Ergebnisse & Erkenntnisse
##################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Alle Zahlen dieses Teils stammen aus **einer** verifizierten Sweep-Charge
(``CLI-Report-Sweep``, 33 Konfigurationen, alle ``ok``) auf der GB10. Die
Kontraktions-Läufe des Format-Vergleichs sind ``ik,kj->ij`` bei :math:`1024^3`, die
memory-bound-Läufe bei :math:`4096^2`, die ``GROUP_M``-Achse bei :math:`4096^3`.

Ein Lauf im Röntgenbild
=======================

Bevor die Aggregate kommen: **ein** Lauf, durch alle Stufen verfolgt, mit den
tatsächlich gespeicherten Werten. Das ist derselbe Lauf, der in der ersten Zeile
der Format-Tabelle weiter unten steht.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Stufe
     - Was dabei entstand
   * - **Eingabe**
     - ``ik,kj->ij`` · ``{i: 1024, k: 1024, j: 1024}`` · fp16 → fp32 ·
       Tile 128/128/64 · kein Swizzle · Baseline cuBLAS · 10 warmup / 30 iters
   * - **[1] parse**
     - ``ContractionIR``: M=[i], N=[j], K=[k], Batch=[] — die einfachste
       Klassifikation, aber derselbe Code, der auch ``acspx,bspy->abcyx`` zerlegt
   * - **[2] B1-Reshape**
     - ``Canonical(M=1024, N=1024, K=1024, B=1)``; der Umbau ist hier die
       Identität (``transform_needed = False``) — ein 2D-Plain-GEMM ist schon
       kanonisch, bekommt aber trotzdem die Batch-Achse der Länge 1
   * - **[3] Codegen**
     - 75 Zeilen cuTile-Quelltext; Literale ``TM=128 TN=128 TK=64``,
       Akkumulator ``ct.float32``, kein Cast-Block (fp16 ist nativ), kein
       Swizzle-Block, kein Epilog-Block
   * - **[4] compile**
     - Slug ``ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64`` →
       ``results/kernels/ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py``; Grid
       :math:`(8, 8, 1)` = 64 Blöcke, K-Schleife über 16 Kacheln
   * - **[5] Kalt-Lauf**
     - ``compile_ms = 50,7`` — der cuTile-JIT (host-seitig), gemessen per
       Wall-Clock
   * - **[6] verify**
     - gegen ``torch.einsum("ik,kj->ij", A.float(), B.float())``:
       ``max_abs_err = 3,20·10⁻⁴``, ``mean_abs_err = 3,19·10⁻⁵``,
       ``rel_err = 1,29·10⁻⁶`` — **PASS** bei ``atol=0,2``, ``rtol=0,02``
   * - **[7] Messung**
     - 30 getaktete Iterationen: ``run_ms = 0,07582`` (Median),
       ``min = 0,07360``, ``p90 = 0,08307``, ``σ = 0,00820``
   * - **Kennzahlen**
     - :math:`2 \cdot 1024^3 = 2{,}147` GFLOP / 0,07582 ms = **28,32 TFLOP/s**;
       8 MiB Traffic ⇒ **110,6 GB/s**; AI = 256 FLOP/Byte; 13,3 % des
       fp16-Rechen-Peaks, 40,5 % der theoretischen Bandbreite
   * - **Baseline**
     - cuBLAS (``torch.matmul``, gleiche Operanden, gleiche Schleife):
       0,0553 ms ⇒ 38,84 TFLOP/s ⇒ unser Kernel erreicht **72,9 %**
   * - **Provenienz**
     - GPU „NVIDIA GB10", SM-Takt 2418 MHz, 38 °C, 9,45 W, Auslastung 0 %
       (direkt **nach** der Messung abgefragt — die GPU war da bereits wieder
       idle)
   * - **[8] Store**
     - eine JSON-Zeile in ``results/results.jsonl``, mit ``run_id`` der Charge

Zwei Dinge lohnen einen zweiten Blick. **Erstens** die Einordnung der
28,32 TFLOP/s: Gegen den fp16-Peak (213) sind das 13 %, was nach wenig klingt.
Gegen die *operative* Decke bei AI = 256 — die Bandbreiten-Schräge mit
:math:`0{,}273 \cdot 256 = 69{,}9` TFLOP/s — sind es 41 %. Und cuBLAS selbst kommt
mit 38,8 TFLOP/s auch nur auf 56 % dieser Decke. Die Roofline ist eben eine
Schranke, nicht ein Versprechen. **Zweitens** die Genauigkeit: Der maximale
absolute Fehler von :math:`3{,}2 \cdot 10^{-4}` bei einer Summe über 1024
fp16-Produkte ist das, was ein fp32-Akkumulator leistet — genau deshalb schreibt
die Aufgabenstellung ihn vor.

Die Roofline: GB10 ist memory-bound
===================================

.. figure:: /_static/gsc/roofline.png
   :align: center
   :width: 100%
   :alt: Roofline-Diagramm der GB10 mit memory-bound- und compute-nahen Punkten

   Roofline (GB10). Zwei Schrägen: die theoretischen 273 GB/s und — gestrichelt
   darunter — die **gemessenen** 223 GB/s aus dem ``copy``-Lauf. Die
   memory-bound-Familien liegen weit links (AI 0,08–0,5), die Kontraktion rechts
   (AI 171–512), die n-äre Kette als ein aggregierter Punkt dazwischen.

Der Ridge-Point der GB10 liegt bei ≈ 780 FLOP/Byte (mit der gemessenen Bandbreite
sogar bei ≈ 955) — weit jenseits der arithmetischen Intensität selbst großer
GEMMs. **Für alle Punkte des Format-Vergleichs ist damit die Bandbreiten-Schräge
die operative Decke**, nicht der Rechen-Peak; die flachen 213 TFLOP/s werden nie
erreicht und *können* bei diesen Intensitäten auch nicht erreicht werden.

Die gemessene zweite Schräge macht die Aussage erst ehrlich: Wenn eine
elementweise Addition 80,6 % der *theoretischen* Bandbreite erreicht, sind das
99 % der *gemessenen* — es ist nichts mehr zu holen. Die Punkte, die im Bild weit
unter beiden Schrägen liegen, sind genau die, bei denen sich Tuning lohnt.

.. note::

   Eine Grenze des Bildes: Die ``copy``-Läufe selbst sind **keine Punkte** in der
   Figur, sondern ihre gestrichelte Linie. Mit :math:`AI = 0` (null FLOP) haben sie
   auf einer logarithmischen Achse keinen Ort. Genau deshalb sind sie als Decke
   eingezeichnet — dort, wo sie inhaltlich hingehören.

Der einzige Punkt der Charge, der **rechts** vom Ridge liegt, ist das
:math:`4096^3`-GEMM mit AI = 1024. Dort bindet tatsächlich die Rechen-Decke, und
der Kernel erreicht mit 73,7 TFLOP/s **34,6 % des fp16-Peaks** — mehr als das
Doppelte des Anteils bei :math:`1024^3`. Er steht bewusst nicht in der
Roofline-Figur (er würde die Format-Punkte verdecken), sondern trägt den
``GROUP_M``-Befund weiter unten.

Durchsatz und Genauigkeit je Format
===================================

.. figure:: /_static/gsc/durchsatz_formate.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz je Zahlenformat, cuTile gegen cuBLAS

   Kontraktion je Format: der generierte cuTile-Kernel gegen die
   cuBLAS-Obergrenze (``torch.matmul``). Für fp8 gibt es keinen direkten
   ``matmul``-Pfad — deshalb fehlt die Vergleichssäule.

.. figure:: /_static/gsc/genauigkeit_durchsatz.png
   :align: center
   :width: 90%
   :alt: Streudiagramm Genauigkeit gegen Durchsatz je Format

   Genauigkeit ↔ Durchsatz. fp16/bf16 sind genau, fp8 ist am schnellsten, aber am
   ungenauesten; tf32 ist hier der schlechteste Kompromiss — langsam **und**
   ungenau.

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

Was daraus zu lernen ist:

* **Der einfache f-String-Codegen erreicht bei fp16/bf16 drei Viertel von
  cuBLAS** (73–75 %) — ohne Autotuning, ohne Software-Pipelining, ohne
  handoptimierte Ladepfade. Für eine Kernel-Fabrik aus ein paar hundert Zeilen
  Template ist das der überraschendste Befund des Projekts.
* **fp8 ist mit 47,7 TFLOP/s klar am schnellsten** und zahlt es mit dem größten
  Fehler (3,4·10⁻¹). Der Grund für den Vorsprung ist zur Hälfte Bandbreite: 1 Byte
  je Eingabeelement verdoppelt die arithmetische Intensität auf 512.
* **tf32 ist der schlechteste Kompromiss** — langsamer als fp16 *und* um zwei
  Größenordnungen ungenauer. Auf dieser Maschine gibt es kaum einen Grund, es zu
  wählen: Sein Rechen-Peak liegt bei 53 statt 213 TFLOP/s, und seine Operanden
  belegen 4 Byte statt 2, was die Intensität auf 171 drückt.
* **Für fp8 gibt es keine cuBLAS-Zahl**, und das Tool sagt auch warum: ``torch``
  meldet ``"baddbmm_cuda" not implemented for 'Float8_e4m3fn'``. Die Baseline
  trägt ``available: false`` samt Grund — das ist der Unterschied zwischen einer
  fehlenden Säule und einer stillschweigend weggelassenen.

Tuning-Raum I: die Kachelung
============================

.. figure:: /_static/gsc/tile_swizzle.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz über Tile-Größen und Swizzle-Gruppengrößen

   fp16-Tuning-Raum bei :math:`1024^3`: **derselbe** verifizierte Kernel, nur
   Kachelung bzw. Block-Umordnung variiert.

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
     - der Einbruch — Faktor 5,2 gegenüber der besten Kachel
   * - Tile 64/64/32
     - 25,6
     - 0,084
     - kleine Kacheln: mehr Blöcke, weniger Wiederverwendung je Block
   * - Tile 128/128/64
     - 28,3
     - 0,076
     - der Referenzpunkt aller anderen Messungen
   * - + Swizzle G8
     - 29,1
     - 0,074
     - dieselbe Permutation wie G16/G32 — siehe unten
   * - + Swizzle G16
     - 29,0
     - 0,074
     - dieselbe Permutation wie G8/G32
   * - + Swizzle G32
     - 29,2
     - 0,074
     - dieselbe Permutation wie G8/G16

**Die Kachelwahl ist der stärkste einzelne Hebel des Werkzeugs.** Ein ungünstiges
Tile (256/128/64) bricht auf 5,5 TFLOP/s ein — Faktor 5,2 gegenüber 128/128/64,
bei identischem Ergebnis und identischem Ausdruck. Die plausible Erklärung: Bei
:math:`M = 1024` und :math:`TM = 256` bleiben nur noch :math:`4 \times 8 = 32`
Blöcke für 48 SMs — ein Drittel der Maschine bekommt nichts zu tun, und die
größeren Kacheln erhöhen zugleich den Registerdruck. Genau solche Effekte macht
das Werkzeug sichtbar, ohne dass man sie vorher kennen muss.

Tuning-Raum II: der L2-Swizzle hängt an der Gittergröße
=======================================================

Die drei Swizzle-Zeilen oben liegen innerhalb von 0,7 % beieinander. Das ist
**kein** Messergebnis über ``GROUP_M``, sondern Struktur — und diese Erklärung war
in einer früheren Fassung dieses Berichts eine unbelegte Behauptung. Jetzt ist sie
gemessen.

Zuerst die Struktur: :math:`1024^3` mit ``TM = TN = 128`` ergibt ein
:math:`8 \times 8`-Blockgitter. Die Rasterung begrenzt die Gruppe auf
``min(num_pid_m - first_pid_m, GROUP_M)``, also auf 8 — es entsteht **eine
einzige Gruppe**, und für jedes ``GROUP_M ≥ 8`` ist die Block-Permutation
*identisch*. Die drei Zeilen sind faktisch dreimal derselbe Kernel; ihre Streuung
ist die Messgenauigkeit.

Messbar wird die Achse erst auf einem Gitter, das mehrere Gruppen zulässt.
:math:`4096^3` ergibt :math:`32 \times 32` Blöcke:

.. figure:: /_static/gsc/group_m.png
   :align: center
   :width: 100%
   :alt: Balkendiagramm Durchsatz über GROUP_M bei 4096 hoch 3

   ``GROUP_M`` auf einem :math:`32 \times 32`-Blockgitter. Identischer
   verifizierter Kernel, identisches Format und Tile — variiert wird
   ausschließlich die Block→Kachel-Zuordnung.

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

Das ist der stärkste neue Befund dieses Berichts, und er hat drei Teile:

1. **Der L2-Swizzle ist kein Feintuning, sondern ein Faktor 2.** Auf einem
   hinreichend großen Gitter verdoppelt eine reine Umordnung der
   Block-Reihenfolge den Durchsatz — bei *bit-identischem* Ergebnis (die
   Permutation ist bijektiv, und die Verifikation liefert für alle sechs Läufe
   denselben Fehler von 2,62·10⁻³). Kein Byte mehr oder weniger wird gerechnet;
   die Daten liegen nur zum richtigen Zeitpunkt im L2.
2. **``GROUP_M`` hat ein Optimum, nicht eine Richtung.** Zu kleine Gruppen (G2)
   bringen kaum Wiederverwendung, zu große (G32) sprengen den Cache. Das Maximum
   liegt bei G8 — genau dem Wert, der als Triton-Konvention hart verdrahtet war,
   was diese Konvention nachträglich rechtfertigt. Ein Werkzeug, das nur „Swizzle
   an/aus" könnte, hätte diese Kurve nie gezeigt.
3. **G32 fällt fast auf den Bezugspunkt zurück** (1,07×) — und das ist die
   *Bestätigung* der Struktur-Erklärung von oben: Bei ``GROUP_M`` = Gitterhöhe
   entsteht wieder nur eine Gruppe, die Permutation degeneriert zu einer bloßen
   Transposition der Durchlaufreihenfolge. Derselbe Mechanismus, der bei
   :math:`1024^3` alle Werte ≥ 8 gleichmacht, macht hier G32 wirkungslos.

Die Lehre daraus ist allgemeiner als der Messwert: **Eine Tuning-Achse kann auf
einer zu kleinen Testform strukturell unsichtbar sein.** Hätten wir nur bei
:math:`1024^3` gemessen, wäre der Schluss „der Swizzle bringt 3 %" gewesen — er ist
falsch, und zwar nicht wegen eines Messfehlers, sondern weil die Form die Frage
nicht zulässt.

Memory-bound: Bandbreite als Primärmetrik
=========================================

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

* **Die elementweise Addition ist bandbreitengesättigt.** 80–82 % der
  theoretischen Bandbreite sind — gemessen an der ``copy``-Obergrenze von
  222,9 GB/s — **98–101 %** des praktisch Erreichbaren. Hier ist nichts mehr zu
  optimieren; der Kernel *ist* die Maschine.
* **Die Kopie ist der Bezugspunkt und zugleich ein Kuriosum:** ``fp16 → fp16``
  (4 Byte/Element) ist mit 209,7 GB/s **langsamer** als ``fp32 → fp32``
  (8 Byte/Element, 222,2 GB/s). Wer nur Bytes zählt, erwartet das Gegenteil. Die
  Erklärung: Bei gleicher Elementzahl bewegt der schmale Fall nur halb so viele
  Bytes, während der Pro-Element-Overhead (Adressierung, Kachel-Verwaltung,
  Launch) gleich bleibt — die Bandbreite ist dann nicht mehr der einzige
  Engpass. Dasselbe Muster erklärt die Reduktion.
* **Die Reduktion liegt in fp16/bf16 bei 63 %, in fp32 bei 79 %** — dieselbe
  Ursache: Sie liest im schmalen Format halb so viele Bytes und wird dadurch
  relativ stärker von allem anderen begrenzt.
* **Die Reduktion ist formatunabhängig genau** (≈ 3·10⁻⁵ in allen drei Formaten),
  weil sie unabhängig vom Eingabeformat im fp32-Akkumulator summiert. Dass dieser
  Satz stimmt, ist das Ergebnis eines gefangenen Bugs — siehe unten.

Die n-äre Kette als ein Punkt
=============================

Die Kette ``ij,jk,kl->il`` bei :math:`256^4` wird in zwei paarweise GEMMs zerlegt
(geplanter Pfad ``ij,jk->ik``, dann ``kl,ik->il``), gegen ``torch.einsum`` über den
**vollen** Ausdruck in fp32 verifiziert und als **ein** aggregierter Punkt
gemessen: **1,64 TFLOP/s bei AI = 64 FLOP/Byte**.

Dass die Intensität *unter* der eines einzelnen GEMMs liegt, ist erwartbar und
genau der Punkt: Der Zwischentensor wird geschrieben und wieder gelesen, was in
die aggregierten Bytes eingeht. Die Kette ist damit ein Beispiel für das, was das
Fusions-Kapitel als nächstes systematisch untersucht.

.. note::

   Ist ``opt_einsum`` installiert, kann der Planer eine **andere** Zerlegung wählen
   als der Links-nach-rechts-Fold — dann wird auch etwas anderes gemessen. Der
   geplante Pfad steht deshalb in jeder ``results.jsonl``-Zeile
   (``provenance.sizes.path``).

Fusion: wann lohnt ein Epilog auf dem Akkumulator?
==================================================

Wer eine Kontraktion und eine anschließende elementweise Operation
(:math:`C = A \cdot B`, dann :math:`+D` oder :math:`\max(\cdot, 0)`) als **zwei**
Kernel fährt, schreibt das Zwischenergebnis nach DRAM und liest es sofort wieder —
auf einer memory-bound Maschine bezahlt man diesen Umweg voll. Gespart wird durch
die Fusion genau dieser Roundtrip, :math:`2 \cdot 4 \cdot M \cdot N` Bytes.

Die entscheidende Beobachtung: **Die Ersparnis ist absolut konstant, die
Kontraktion selbst wird mit steigendem :math:`K` immer teurer.** Der Sweep variiert
deshalb die arithmetische Intensität und nicht die Arbeitsmenge — die schmale und
die quadratische Form haben mit 2,15 GFLOP dieselbe FLOP-Zahl:

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
   (4096·4096·64), in der Mitte :math:`1024^3`, rechts die tiefe Form
   (1024·1024·8192). Beide Kurven fallen monoton gegen die Referenzlinie 1,0.

Das Ergebnis ist ein **Trend**, kein Einzelbefund:

* Bei der **schmalen, memory-bound** Form ist der fusionierte Kernel mehr als
  doppelt so schnell (2,12× / 2,71×) — dort dominiert der gesparte 128-MiB-
  Roundtrip die Laufzeit. Der fused Kernel erreicht dabei 195 GB/s, also **87 %
  der gemessenen Bandbreiten-Obergrenze**: Er ist nicht nur schneller, er ist
  nahe am Maschinenlimit.
* Bei der **tiefen, compute-dominierten** Form schrumpft der Gewinn auf 1,03–1,04×:
  Die Kontraktion braucht dort 0,36 ms, der gesparte 8-MiB-Roundtrip nur etwa
  0,01–0,02 ms — er verschwindet im Rauschen der Rechenzeit.
* ``relu`` gewinnt durchweg mehr als ``bias``, und das ist konsistent:
  ``relu`` braucht keinen zusätzlichen Operanden, der fused Kernel liest also
  wirklich nur A und B, während ``bias`` das Bias-Feld D zusätzlich lesen muss.

Damit ordnet sich auch der Befund aus Assignment 04 ein, der dieses Teil-Ziel
motiviert hat: Dort war die Fusion mit **0,984×** minimal *langsamer* als der
sequentielle Pfad (Kontraktion 12,83 ms gegenüber einem Epilog von 0,067 ms). Diese
Form liegt noch weiter rechts als unsere tiefste — jenseits des Punktes, an dem der
gesparte Speicherverkehr überhaupt messbar ist, bleibt nur der zusätzliche Aufwand
des größeren Kernels übrig. Die ehrliche Aussage lautet deshalb nicht „Fusion ist
schneller", sondern:

.. admonition:: Kernaussage

   **Fusion zahlt sich in dem Maß aus, in dem die Operation bandbreiten- und
   nicht rechenlimitiert ist** — und die GB10 ist mit ihrem Ridge-Point von
   ≈ 780 FLOP/Byte eine Maschine, auf der dieser Bereich groß ist.

Zwei Eigenschaften der Umsetzung sind dabei bewusst konservativ. **Erstens** ist
die Fusion rein additiv: ohne gewählten Epilog erzeugt der Codegen byte-identischen
Quelltext und denselben Slug wie vorher — durch einen Textvergleich im Test
festgeschrieben. **Zweitens** misst das Tool den sequentiellen Vergleichspfad
**selbst** (zweiter Kernel-Paar-Lauf im selben ``run()``, gleiche Messschleife) und
verifiziert auch dessen Ergebnis gegen fp32. Der Speedup ist damit kein Vergleich
gegen eine Schätzung, sondern gegen eine gemessene, verifizierte Alternative.
Schlägt diese Zweitmessung fehl, verliert der Lauf nur den Vergleich, nicht sein
eigenes Ergebnis.

verify-before-trust in Aktion
=============================

Der abgebildete Sweep umfasst 33 Konfigurationen, und **alle 33** bestehen die
fp32-Verifikation. Dieser saubere Stand ist allerdings selbst das Ergebnis des
Prinzips — denn eine Charge vorher tat er es nicht, und der eine Fehlschlag war
lehrreich genug, um hier ausführlich zu stehen.

**Der Fund.** Die bf16-Reduktion über :math:`4096^2` überschritt die Toleranz
deutlich: maximaler absoluter Fehler **1,574** bei einem ``atol`` von 1,0. Der Lauf
bekam ``verify_failed``, **keine** Durchsatzzahl und erschien in keiner Figur. Die
naheliegende Erklärung wäre gewesen, dass bf16 mit seinen 8 Mantissenbits über 4096
Summanden schlicht zu grob ist — eine Format-Grenze, plausibel formuliert und bequem
zu glauben. Sie war falsch.

**Die Ursache lag im eigenen Codegen.** Das Reduktions-Template hat zwei Pfade. Der
K-Loop-Fallback akkumulierte korrekt im angeforderten Akku-Format, der
single-shot-Pfad — der für alle Report-Größen gewählt wird — dagegen **im
Eingabeformat**: ``ct.sum(tile, axis=1)`` statt ``ct.sum(ct.astype(tile,
ct.float32), axis=1)``. Der ``acc_dtype``-Regler war auf diesem Pfad wirkungslos.
Auffällig wurde das an einer Stelle, an der die Intuition genau anders herum zeigt:
Dieselbe Eingabe über eine **doppelt so lange** Achse reduziert (jenseits von
:math:`K = 16384`, also über den Loop-Pfad) war drei Größenordnungen *genauer*.
Nicht die Länge der Summe war der Unterschied, sondern der Akkumulator.

**Die Korrektur** ist eine Zeile — der Cast vor ``ct.sum``, exakt wie im Loop-Pfad.
Danach liefert dieselbe bf16-Reduktion einen Fehler von **3,05·10⁻⁵**, rund
**51 000×** kleiner, besteht die Verifikation und steht in der
memory-bound-Tabelle oben. Die fp16-Reduktion, die zuvor mit einem Fehler von 0,22
*innerhalb* ihrer Toleranz lag und deshalb als ``ok`` durchgegangen war, verbesserte
sich im selben Zug auf 3,05·10⁻⁵.

**Das ist der eigentliche Wert des Gates.** Es hat keine Eigenschaft eines
Zahlenformats gemeldet, sondern einen **echten Defekt im generierten Kernel** — und
zwar bevor eine falsch beschriftete Genauigkeitszahl in diesen Report gelangte. Ohne
die fp32-Referenz wäre der Fehler unsichtbar geblieben: 1,574 auf einer Zeilensumme
über 4096 bf16-Werte sieht nach einem Format-Limit aus, nicht nach einem Bug, und der
fp16-Fall hätte mit 0,22 nie Verdacht erregt. Genau das ist die Klasse stiller
Falschergebnisse, gegen die generierter Kernel-Code abgesichert werden muss.

Ein Toleranz-Fall bleibt zudem echt: Bei tieferen n-ären fp16-Ketten summiert sich ab
:math:`384^4` der Fehler beider GEMM-Schritte über die Toleranz — deshalb zeigt der
Report die Kette bewusst bei :math:`256^4`. Das Tool meldet solche Fälle laut, statt
still eine falsche Zahl zu liefern.

Messqualität: was die Zahlen wert sind
======================================

Ein Bericht, der Durchsätze auf drei Stellen angibt, sollte sagen, wie stabil sie
sind. Die Charge liefert das mit, weil jede Messung eine Verteilung ist:

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

Das Muster ist konsistent und erklärbar: **Die memory-bound-Läufe sind extrem
reproduzierbar** (σ unter 1 %), weil sie durchgehend an der Bandbreite hängen — es
gibt kaum etwas, das variieren kann. **Die Kontraktionen streuen mit ~10 %
deutlich mehr**, und ihr p90 liegt 10–23 % über dem Median: Hier wirken Takt- und
Cache-Effekte, und der L2-Flush zwischen den Iterationen tut genau das, was er
soll — er erzeugt für jede Iteration einen kalten Startzustand, dessen Kosten von
der Verdrängungsreihenfolge abhängen. Deshalb ist ``run_ms`` der **Median**: Ein
Mittelwert würde von den p90-Ausreißern nach oben gezogen.

Zur **Compile-Zeit** gehört eine Einschränkung, die man kennen muss.
``compile_ms`` misst den *ersten Launch im laufenden Prozess*. In dieser Charge
liegt er zwischen 9,8 und 50,7 ms — aber nur, weil viele Läufe einen Kernel-Slug
benutzen, der im selben Prozess schon einmal compiliert wurde (der
:math:`4096^3`-Sweep etwa nutzt dasselbe Artefakt wie der :math:`1024^3`-Sweep).
Für einen **wirklich neuen** Kernel in einem frischen Prozess reichen die Werte im
Store von rund **310 ms bis über 1,7 s** — den Spitzenwert halten die
tf32-Varianten der großen Tensor-Kontraktion ``acspx,bspy->abcyx``, deren
Kernel-Cast den JIT zusätzlich beschäftigt. Die Zahl ist also keine
Kernel-Eigenschaft, sondern eine Prozess-Historie; genau deshalb steht sie
getrennt von ``run_ms`` und geht in keine Kennzahl ein.

Grenzen der Aussagen
====================

Was dieser Bericht **nicht** behauptet:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Grenze
     - Konsequenz
   * - **GB/s sind eine Untergrenze**
     - Die Byte-Zahlen sind der algorithmische Mindest-Traffic ohne
       Tiling-Rereads. Der reale DRAM-Verkehr einer Kontraktion ist höher; „% der
       Bandbreite" ist entsprechend konservativ. Für die memory-bound-Familien
       (jedes Element genau einmal gelesen) ist die Zahl dagegen scharf.
   * - **Kein Autotuning**
     - Alle Kachel-/Swizzle-Zahlen sind *gemessene Einzelkonfigurationen*, keine
       gefundenen Optima. Dass G8 bei :math:`4096^3` das Maximum der geprüften
       Werte ist, heißt nicht, dass es global optimal ist.
   * - **Die Peaks sind Fremdmessungen**
     - Rechen-Peaks stammen aus einem veröffentlichten Microbenchmark, nicht aus
       einem Whitepaper. Alle „%-Peak"-Angaben erben diese Unsicherheit. Die
       Bandbreiten-Decke haben wir dagegen selbst gemessen.
   * - **Nur zwei Operanden pro Kernel**
     - n-äre Ausdrücke werden zerlegt; Diagonalen und Spuren (``ii->i``) sind
       nicht unterstützt und werden laut abgelehnt.
   * - **Epilog nur bei 2-Operanden-Kontraktionen**
     - Auf einer n-ären Kette würde der Epilog still unangewendet bleiben —
       deshalb lehnt das Tool die Kombination ausdrücklich ab, statt sie
       stillschweigend zu ignorieren.
   * - **Eine Maschine, eine GPU**
     - Alle Ergebnisse gelten für **diese** GB10. Es gibt keine Multi-GPU-Läufe
       und keinen Vergleich gegen andere Hardware.
   * - **Reduktions-Fallback nicht aus A02 belegt**
     - Der K-Loop-Pfad des Reduktions-Templates ist gegen ``torch`` verifiziert,
       aber nicht durch eine Assignment-Vorlage abgesichert. Er ist im generierten
       Code als solcher markiert.
