.. _gsc_presentation:

############
Presentation
############

Am **08.07.2026** wurde das gewählte Projekt — das **cuTile Performance Lab** — in
einem **20-minütigen Vortrag** präsentiert. Gefordert war kein fertiges Projekt,
sondern ein **lauffähiger Prototyp mit ersten Ergebnissen**; gezeigt wurde der Stand
nach den Teil-Zielen TZ 1–6, also die vollständige Kette *Ausdruck → generierter
Kernel → fp32-Verifikation → Messung → Charts* für die Kontraktions-Familie, live auf
der GB10.

Dieses Kapitel hält fest, was vorgetragen wurde. Die technischen Details, alle
Messreihen und die vollständige Argumentation stehen im :ref:`Projektbericht
<gsc_report>`; hier steht nur die Vortragslinie.

Gliederung
==========

Introduction
------------

Einstieg über die Leitfrage, die jede GPU-Tensor-Kontraktion begleitet: *Wie schnell,
wie genau, und wie nah am Hardware-Limit?* Die Assignments 01–06 hatten einzelne
Antworten geliefert — je Kernel, je Format, je Kachelung, jeweils in einem eigenen
Skript und mit von Hand abgetippten Zahlen. Die Projektidee war, diese Antworten
**interaktiv und vergleichbar** zu machen: ein Werkzeug, in das man einen
einsum-Ausdruck eintippt und das daraus einen cuTile-Kernel **erzeugt**, ihn prüft,
misst und einordnet. Abgegrenzt wurde außerdem, was das Tool ausdrücklich *nicht* ist:
kein Autotuner und keine Bibliothek, sondern ein Explorer für den
Konfigurationsraum.

Problem formulation
-------------------

Der Konfigurationsraum wurde als das eigentliche Problem vorgestellt: Zahlenformat
(fp16, bf16, tf32, fp8), Kachelung ``TM``/``TN``/``TK``, L2-Swizzle mit
``GROUP_M`` — jede Achse verschiebt Durchsatz **und** Genauigkeit, oft gegenläufig,
und der Abstand zum Hardware-Peak bleibt ohne Werkzeug unsichtbar.

Dazu die zwei Risiken, die die Umsetzung prägen und die den Aufbau des Tools
erklären: **generierter Kernel-Code ist eine Quelle stiller Falschergebnisse** (eine
vertauschte ``ct.mma``-Orientierung liefert plausible, aber falsche Zahlen — der
Fehler, an dem Assignment 06 gescheitert war), und **nicht-teilbare Dimensionen sind
der Normalfall**, nicht die Ausnahme. Beide führen direkt zur Entwurfsregel des
Projekts: *verify-before-trust* — keine Zahl ohne bestandene fp32-Referenz.

Implemented solution
--------------------

Vorgestellt wurde die Architektur über ihre **eine Naht**:

.. code-block:: text

   run(config: RunConfig) -> RunResult

Die Oberfläche (Plotly Dash) baut nur ``RunConfig`` und liest ``RunResult``; ``run``
wirft nie, sondern kategorisiert jeden Ausgang in einen Status. Dahinter die
Pipeline *parse → Kanonisierung/Reshape → Codegen → compile (+Cache) → verify(fp32) →
Benchmark → Metriken → Store*. Gezeigt wurde außerdem, warum der Codegen bewusst
schlicht ist (f-String-Templates je Familie, ein self-contained Modul je Kernel) und
wie Randfälle einheitlich über ``ct.load(..., padding_mode=ct.PaddingMode.ZERO)``
behandelt werden.

Zum Zeitpunkt des Vortrags trug diese Pipeline die **Kontraktions-Familie**
vollständig (GEMM, batched, transponiert, allgemeine 2-Operanden-Kontraktion) samt
dtype-, Tiling- und Swizzle-Achse, Baselines gegen cuBLAS und der Roofline. Die
memory-bound-Familien, die n-äre Kette und die Epilog-Fusion waren als nächste
Teil-Ziele angekündigt und sind inzwischen umgesetzt (siehe Bericht).

Results
-------

Als erste Ergebnisse wurden gezeigt:

* **Der Codegen ist konkurrenzfähig.** Der generierte fp16-/bf16-Kernel erreicht
  ohne Autotuning knapp drei Viertel des cuBLAS-Durchsatzes bei :math:`1024^3`.
* **Die Kachelwahl ist der stärkste Hebel** — eine ungünstige Kachelung kostet mehr
  als den Faktor fünf, während der L2-Swizzle bei dieser Größe kaum wirkt (und
  numerisch nachweisbar identisch rechnet).
* **Genauigkeit und Durchsatz sind nicht monoton gekoppelt:** fp8 ist am schnellsten
  und am ungenauesten, tf32 war in dieser Messung der schlechteste Kompromiss —
  langsam *und* ungenau.
* **Die GB10 ist memory-bound.** Der Ridge-Point liegt bei ≈ 780 FLOP/Byte, weit
  jenseits der arithmetischen Intensität selbst großer GEMMs — die
  Bandbreiten-Schräge ist die operative Decke. Das war die zentrale Erkenntnis des
  Vortrags und wurde zugleich als Begründung für die geplanten memory-bound-Familien
  genutzt.

Live-Demo
=========

Kern des Vortrags war die Demo statt der Folien: Ausdruck eintippen, Format und
Kachelung wählen, „Run" — der Kernel wird erzeugt, compiliert, gegen fp32 verifiziert
und gemessen, und das Ergebnis erscheint in Durchsatz-, Genauigkeits- und
Roofline-Chart. Gezeigt wurde dabei bewusst auch der **generierte Quelltext** im
Code-Panel (das Tool versteckt nicht, was es tut) und ein Lauf, der mehrere Formate
zum Vergleich nebeneinanderstellt.

Rückmeldung und Konsequenzen für die Restlaufzeit
=================================================

Aus der Diskussion nach dem Vortrag ergaben sich die Schwerpunkte der letzten
Projektwochen, die anschließend als TZ 7–9 umgesetzt wurden:

* Die **memory-bound-Seite** sichtbar machen, statt sie nur aus der Roofline zu
  folgern — Elementwise und Reduktion als eigene Familien mit GB/s als Primärmetrik
  (TZ 7).
* **Mehr als zwei Operanden**: die n-äre Kette über paarweise Zerlegung durch den
  bereits bewiesenen 2-Operanden-Pfad (TZ 7.5).
* Die Frage **„wann lohnt sich Fusion?"** nicht illustrieren, sondern belegen — als
  Trend über die arithmetische Intensität (TZ 9).
