.. _gsc_report_grundlagen:

##########################################
Teil 1 — Grundlagen: Maschine, Modell, API
##########################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Dieser Teil legt die drei Dinge fest: **welche Maschine** gemessen wird, **mit welchem Modell** man ihre Grenzen
sichtbar macht, und **welche Programmierschnittstelle** die Kernel überhaupt
benutzen.

Die Maschine: GB10 (DGX Spark)
==============================

Gemessen wird auf einem **NVIDIA GB10** — dem Grace-Blackwell-SoC der
DGX-Spark-Klasse. Für dieses Projekt sind drei Eigenschaften entscheidend, und
alle drei sind unüblich gegenüber einer klassischen Rechenzentrums-GPU:

.. list-table:: Zielhardware
   :header-rows: 1
   :widths: 30 70

   * - Eigenschaft
     - Wert
   * - Compute Capability
     - ``sm_121`` (Blackwell), 48 SM, 6144 CUDA-Cores,
       192 Tensor-Cores (5. Generation)
   * - Takt
     - ≈ 2,42 GHz typisch (max. 3,0 GHz); SoC-TDP 140 W
   * - Speicher
     - **128 GB LPDDR5x, unified** — CPU und GPU teilen denselben physischen
       Speicher
   * - Speicherbandbreite
     - **273 GB/s** (theoretisch)
   * - Rechen-Peaks (dense)
     - fp16/bf16 ≈ 213 TFLOP/s · fp8 ≈ 214 · tf32 ≈ 53 ·
       fp32 ohne Tensor-Cores ≈ 0,2 · fp64 vernachlässigbar
   * - Software
     - torch 2.11.0+cu130, CUDA-Runtime 13.0, ``cuda.tile`` (cuTile), Treiber
       580.159.03

**Woher die Zahlen kommen.** Die Bandbreite und die Speichergröße stehen in der
DGX-Spark-Hardwaredokumentation. Für die Rechen-Peaks gibt es **kein** offizielles
GB10-Whitepaper. Die Werte stammen aus einem veröffentlichten
``mmapeak``-Microbenchmark und sind damit *gemessene*, nicht theoretische Peaks.
Welche Zahlenformate cuTile auf dieser Maschine tatsächlich rechnen kann, haben wir
nicht angenommen, sondern vorab selbst geprüft (``project-development/analysis/``):
fp16, bf16, tf32, fp8 e4m3/e5m2 compilen, laufen und stimmen gegen eine
fp32-Referenz. Diese Vorab-Analyse ist der Grund, warum
im Tool keine Format-Auswahl steht, die dann doch nicht funktioniert.

**Was „unified memory" praktisch bedeutet.** Es gibt keinen separaten HBM-Stack
und keinen PCIe-Transfer zwischen Host und Device: dieselben LPDDR5x-Chips
bedienen CPU und GPU. Für dieses Projekt hat das zwei Konsequenzen. Erstens
entfällt die klassische „Kopieren-ist-teuer"-Buchführung — ein host-seitiger
``permute``/``reshape`` bewegt Daten im selben Speicher, in dem der Kernel danach
liest. Zweitens — und das prägt jedes Ergebnis dieses Berichts — ist die
Bandbreite mit 273 GB/s **niedrig im Verhältnis zur Rechenleistung**. Zum
Vergleich: Eine H100 mit HBM3 hat rund das Zwölffache an Bandbreite, aber nur
etwa das Vier- bis Fünffache an fp16-Rechenleistung. Ihr Verhältnis von Rechnen
zu Lesen ist damit rund dreimal günstiger — auf der GB10 muss eine Operation
also *deutlich* rechenintensiver sein, bevor sich Rechen-Optimierung überhaupt
lohnt. Die GB10 ist eine Maschine, auf der man Bytes zählt, nicht FLOPs.

.. note::

   **Geteilte Maschine.** Der Lab-Rechner kann von mehreren Personen benutzt werden und
   der Speicher ist unified — ein OOM-Crash trifft nicht nur den eigenen Prozess.
   Deshalb sind alle Größen im Werkzeug bewusst klein gehalten (die Report-Formen
   belegen einige zehn bis wenige hundert MiB), es gibt eine Speicher-Obergrenze in
   der Eingabevalidierung, und **alle** GPU-Läufe serialisieren über einen
   prozessübergreifenden Lock (siehe :ref:`Teil 4 <gsc_report_frontend>`).

Das Roofline-Modell 
===================

Zwei Größen begrenzen jeden Kernel: Er kann nicht mehr rechnen, als die
Rechenwerke schaffen, und er kann nicht schneller rechnen, als die Daten
ankommen. Das Roofline-Modell macht daraus ein Bild, in dem jede Operation ein
**Punkt** ist.

Die Achse: arithmetische Intensität
-----------------------------------

Die entscheidende Kennzahl einer Operation ist, **wie viel gerechnet wird je
gelesenem Byte**:

.. math::

   AI = \frac{\text{FLOP}}{\text{Byte}}

Damit lässt sich der erreichbare Durchsatz nach oben abschätzen:

.. math::

   \text{TFLOP/s}_{\max}(AI) = \min\bigl(\underbrace{P}_{\text{Rechen-Peak}},\;
   \underbrace{B \cdot AI}_{\text{Bandbreiten-Schräge}}\bigr)

mit :math:`B` = Bandbreite in TB/s. Auf der GB10 ist :math:`B = 273\,\text{GB/s}
= 0{,}273\,\text{TB/s}`, die Schräge also :math:`0{,}273 \cdot AI`. Trägt man das
doppelt-logarithmisch auf, ergibt sich eine ansteigende Gerade (Bandbreiten-Limit)
mit einem waagerechten Dach (Rechen-Limit). Der Knick heißt **Ridge-Point** und
liegt dort, wo beide gleich sind:

.. math::

   AI_{\text{ridge}} = \frac{P}{B}

Für die GB10 eingesetzt:

.. list-table:: Ridge-Points der GB10 (theoretische Bandbreite 273 GB/s)
   :header-rows: 1
   :widths: 22 22 28 28

   * - Format
     - Peak :math:`P`
     - :math:`AI_{\text{ridge}} = P/B`
     - Interpretation
   * - fp16 / bf16
     - 213 TFLOP/s
     - **≈ 780 FLOP/Byte**
     - erst jenseits davon rechenlimitiert
   * - fp8 (e4m3/e5m2)
     - 214 TFLOP/s
     - ≈ 784 FLOP/Byte
     - praktisch derselbe Knick
   * - tf32
     - 53 TFLOP/s
     - ≈ 194 FLOP/Byte
     - deutlich früher rechenlimitiert
   * - fp32 (ohne Tensor-Cores)
     - —
     - —
     - kein Tensor-Core-Dach ⇒ kein Ridge

Wo landen unsere Operationen?
-----------------------------

Ein quadratisches GEMM :math:`C = A \cdot B` mit
:math:`M = N = K = n`, fp16-Eingaben (2 Byte) und fp32-Ausgabe (4 Byte):

.. math::

   \text{FLOP} = 2\,n^3, \qquad
   \text{Byte} = \underbrace{2 \cdot 2n^2}_{A,\,B \text{ lesen}}
                 + \underbrace{4 n^2}_{C \text{ schreiben}} = 8n^2

.. math::

   AI = \frac{2n^3}{8n^2} = \frac{n}{4}

Für :math:`n = 1024` ergibt das :math:`AI = 256` — **weit links** vom
Ridge-Point 780. Und das ist kein kleines Beispiel: Selbst :math:`n = 4096` (AI
= 1024, unsere größte Form) liegt gerade eben *rechts* davon. Dies führt zu einer
Konsequenz für dieses Projekt:

.. admonition:: Kernaussage

   Auf der GB10 werden **die meisten Kontraktionen bandbreitenlimitiert sein**. Die operative Decke ist bei :math:`AI = 256`
   nicht der 213-TFLOP/s-Peak, sondern die Schräge:
   :math:`0{,}273 \cdot 256 = 69{,}9` TFLOP/s. Alles, was das Tool an einem GEMM
   tunen kann, spielt sich unterhalb dieser 70 TFLOP/s ab — nicht unterhalb von 213.

Die memory-bound-Familien liegen noch viel weiter links. Eine elementweise
Addition zweier fp16-Matrizen mit fp32-Ausgabe liest 2 × 2 Byte und schreibt
4 Byte je Element, rechnet dafür **ein** FLOP:

.. math::

   AI = \frac{1}{2 \cdot 2 + 4} = 0{,}125\ \text{FLOP/Byte}

Eine reine Kopie hat sogar :math:`AI = 0` — sie rechnet nichts. Genau deshalb
gehören Elementwise, Reduktion und Copy ins Werkzeug: Ohne sie wäre die Roofline
eine Grafik mit vier Punkten in derselben Ecke; mit ihnen spannt sie vier
Größenordnungen auf und zeigt beide Regime.

Bandbreiten-Decke
-----------------

273 GB/s ist ein *theoretischer* Wert. Ohne veröffentlichte STREAM-Zahl haben wir
im Werkzeug zunächst nur die Annahme „real 70–85 %" hinterlegt. Weil das für einen
Bericht, der %-vom-Peak angibt, unbefriedigend ist, misst der Report-Sweep die
Bandbreite inzwischen **selbst** — mit dem billigsten möglichen Kernel, einer
reinen Kopie (0 FLOP):

.. list-table:: Gemessene Bandbreite (Elementwise ``copy``, 4096², verifiziert)
   :header-rows: 1
   :widths: 30 20 22 28

   * - Format
     - Byte/Element
     - erreichte GB/s
     - Anteil an 273 GB/s
   * - fp16 → fp16
     - 4
     - 209,7
     - 76,8 %
   * - fp16 → fp32
     - 6
     - **222,9**
     - **81,7 %**
   * - fp32 → fp32
     - 8
     - 222,2
     - 81,4 %

Damit ist die 70–85 %-Annahme nicht mehr geglaubt, sondern belegt: die praktisch
erreichbare Bandbreite liegt bei **≈ 223 GB/s**. Alle memory-bound-Zahlen in
Teil 5 sind gegen *diese* Decke zu lesen — eine Operation bei 80 % der
theoretischen Bandbreite hat keine 20 % Luft, sondern läuft am Anschlag. Mit der
realen Bandbreite verschiebt sich auch der Ridge-Point nach rechts
(:math:`213 / 0{,}223 \approx 955` FLOP/Byte), die Maschine ist also *noch*
stärker memory-bound als die theoretische Rechnung sagt.

.. note::

   Die Roofline ist eine **obere Schranke**, keine Vorhersage. Dass ein Punkt sie
   nicht erreicht, kann viele Gründe haben (Latenz, Occupancy, Instruktions-Mix,
   Launch-Overhead). Der Wert des Modells liegt darin, dass es die Frage
   *„lohnt es sich hier überhaupt, an der Rechenseite zu drehen?"* beantwortet —
   und auf dieser Maschine lautet die Antwort meistens: nein, dreh an der
   Datenbewegung. Genau das führt später zum Fusions-Kapitel.

cuTile in zwei Seiten
=====================

Alle Kernel dieses Projekts sind in **cuTile** (``cuda.tile``) geschrieben. Wer CUDA kennt, muss
für das Verständnis des generierten Codes vor allem eine Umstellung mitmachen:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - CUDA (klassisch)
     - cuTile
   * - Programmiereinheit
     - **ein Thread** — man schreibt, was ein einzelner Thread tut, und denkt
       Blöcke/Warps mit
     - **eine Kachel** (Tile) — man schreibt, was mit einem ganzen
       Array-Abschnitt passiert
   * - Indexrechnung
     - manuell: ``row = blockIdx.y*blockDim.y + threadIdx.y`` …
     - Kachel-Indizes: ``ct.load(A, index=(i, kk), shape=(TM, TK))``
   * - Randbehandlung
     - explizite ``if (row < M && col < N)``-Masken
     - ``padding_mode=ct.PaddingMode.ZERO`` beim Laden, automatisches Clipping
       beim Speichern
   * - Shared Memory
     - selbst allozieren, selbst synchronisieren
     - implizit — der Compiler entscheidet, wo eine Kachel liegt
   * - Tensor-Cores
     - ``wmma``/``mma``-Intrinsics, Fragment-Layouts
     - ``ct.mma(a, b, acc)`` auf Kacheln

Die Bausteine, die im generierten Code vorkommen
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Baustein
     - Bedeutung
   * - ``@ct.kernel``
     - markiert die Funktion, die auf der GPU läuft
   * - ``ct.bid(n)``
     - Block-Index der :math:`n`-ten Grid-Achse (das cuTile-Pendant zu
       ``blockIdx``) — die einzige Stelle, an der ein Kernel erfährt, *welche*
       Kachel er bearbeitet
   * - ``ct.load(T, index=…, shape=…, padding_mode=…)``
     - lädt eine Kachel. **Wichtig:** ``index`` zählt in **Kachel-Einheiten**, nicht
       in Elementen — ``index=(i, kk), shape=(TM, TK)`` bezeichnet den Block, der
       bei Element :math:`(i \cdot TM,\ kk \cdot TK)` beginnt
   * - ``ct.store(T, index=…, tile=…)``
     - schreibt eine Kachel zurück; Elemente jenseits der Tensorgrenze werden
       **automatisch verworfen**
   * - ``ct.mma(a, b, acc)``
     - Tensor-Core-Multiply-Accumulate auf Kacheln:
       :math:`(TM,TK) \times (TK,TN) + acc \rightarrow (TM,TN)`
   * - ``ct.full(shape, 0, dtype=…)``
     - Akkumulator-Kachel anlegen — hier wird die Akkumulator-Präzision
       festgelegt, unabhängig vom Eingabeformat
   * - ``ct.astype`` · ``ct.reshape`` · ``ct.sum`` · ``ct.maximum`` · ``ct.cdiv``
     - Cast, Umformen, Reduktion entlang einer Achse, elementweises Maximum,
       aufrundende Division
   * - ``ct.Constant[int]``
     - Launch-Argument, das der JIT als **Konstante** behandelt (unsere ``M``,
       ``N``, ``K``)
   * - ``ct.launch(stream, grid, kernel, args)``
     - Start; ``grid`` ist ein 3-Tupel von Blockzahlen
   * - ``ct.TileError``
     - Fehlerklasse des cuTile-JIT — das Tool fängt sie getrennt ab, um
       „compiliert nicht" von „läuft nicht" zu unterscheiden

Zwei Eigenschaften mit Folgen für die Architektur
-------------------------------------------------

Beide sind wir empirisch gestolpert, und beide erklären später Entscheidungen, die
sonst willkürlich aussähen:

1. **Der JIT läuft erst beim ersten ``ct.launch``**, nicht beim Import des Moduls.
   Ein Kernel zu *definieren* kostet nichts; der erste Aufruf kostet Hunderte von
   Millisekunden. Das ist der Grund, warum die Messschicht kalten und warmen Lauf
   getrennt behandelt (:ref:`Teil 3 <gsc_report_pipeline>`) und warum die
   Oberfläche einen echten Hintergrund-Job braucht (:ref:`Teil 4
   <gsc_report_frontend>`).
2. **cuTile liest den Quelltext des Kernels zur Laufzeit** (über
   ``inspect.getsourcelines``) und braucht dafür eine **echte Datei auf der
   Platte**. Ein generierter Kernel, der nur als String im Speicher existiert und
   per ``exec`` ausgeführt wird, scheitert mit ``OSError: could not get source
   code``. Genau deshalb schreibt der Codegen jeden Kernel nach
   ``results/kernels/<slug>.py`` und importiert ihn von dort — was sich als
   glücklicher Zwang erwies: dieselbe Datei ist Compile-Cache, Anzeige in der
   Oberfläche und nachprüfbarer Beleg.

Problemstellung und Risiken
===========================

Für jede Kontraktion existiert ein großer Konfigurationsraum: **Zahlenformat**
(fp16, bf16, tf32, fp8 …), **Kachelung** (``TM``/``TN``/``TK``), **L2-Swizzle**
(Block→Kachel-Umordnung mit Gruppengröße ``GROUP_M``), die **Operations-Familie**
(Kontraktion, elementweise, Reduktion) und die Frage, ob eine nachgelagerte
elementweise Operation als eigener Kernel läuft oder in die Kontraktion
**fusioniert** wird. Jede Achse verschiebt Durchsatz und Genauigkeit, oft
gegenläufig — und der Bezug zum Hardware-Limit bleibt ohne Werkzeug unsichtbar.

Ein Werkzeug, das GPU-Kernel *generiert* und dann *vermisst*, hat dabei sechs
konkrete Fehlerquellen. Sie zu benennen ist keine Formalität: jede hat eine
sichtbare Gegenmaßnahme im Code.

.. list-table:: Risiken und ihre Gegenmaßnahme
   :header-rows: 1
   :widths: 6 34 60

   * - #
     - Risiko
     - Gegenmaßnahme
   * - ①
     - **Stille Falschergebnisse.** Eine vertauschte ``ct.mma``-Orientierung
       liefert plausible, aber falsche Zahlen — und ein falscher Kernel ist oft
       *schneller*.
     - fp32-Verifikation auf **jedem** Kernel vor jeder Messung; ohne bestandene
       Prüfung gibt es keine Durchsatzzahl.
   * - ②
     - **Nicht-teilbare Dimensionen** sind der Normalfall (130 Zeilen bei
       Kachelbreite 128), nicht die Ausnahme.
     - ZERO-Padding beim Laden + automatisches Clipping beim Speichern,
       systematisch über alle drei Familien per GPU-Test belegt.
   * - ③
     - **Nicht nachvollziehbare Artefakte.** „Welcher Code hat diese Zahl
       erzeugt?" muss beantwortbar sein.
     - Jeder Kernel wird als Datei mit lesbarem Config-Namen persistiert; jede
       Messung schreibt eine selbstbeschreibende JSON-Zeile.
   * - ④
     - **View-/Stride-Mathematik.** Der Umbau beliebiger Kontraktionen auf eine
       kanonische Form ist Indexarithmetik — ein Fehler dort ist unsichtbar.
     - Formales zero-copy-Kriterium plus numerische Tests gegen
       ``torch.einsum`` über transponierte, batched und mehrdimensionale Fälle.
   * - ⑤
     - **Fork + CUDA + geteilte GPU.** Eine Weboberfläche mit Worker-Prozessen und
       ein CUDA-Kontext im Elternprozess vertragen sich nicht.
     - Strikte Import-Regel (Hauptprozess bleibt CUDA-frei) und ein
       prozessübergreifender GPU-Lock.
   * - ⑥
     - **Beschädigter Cache.** Ein halb geschriebenes Kernel-Artefakt darf keinen
       Lauf abstürzen lassen.
     - Atomares Schreiben (Temp-Datei + ``os.replace``) und ein Cache, der
       unlesbare Artefakte erkennt und neu erzeugt.
