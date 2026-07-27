.. _gsc_projektplan:

############
Projektplan
############

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 2

Gewähltes Projekt
=================

Umgesetzt wird **Idee 2 — der interaktive einsum/GEMM-Performance-Explorer**
(GPU/cuTile, siehe :ref:`Pitch <gsc_pitch_idee2>`). Aus einem einsum-/GEMM-Ausdruck
soll *zur Laufzeit* ein cuTile-Kernel **generiert**, bei verstellbarem Zahlenformat
und verstellbarer Kachelung auf der GPU **gemessen** und in interaktiven Graphen
**visualisiert** werden.

Die Idee setzt auf den Bausteinen der Assignments auf — einsum-Klassifikation,
Kontraktions-Kernel, Kachelung, Swizzling — und bündelt sie zu einem
eigenständigen Werkzeug. Das thematische Ziel ist nicht „ein weiterer Kernel",
sondern den Zusammenhang von **Geschwindigkeit, Genauigkeit und
Hardware-Grenze** sichtbar zu machen.

.. note::

   Die grafische Oberfläche ist die Schauseite. Die eigentliche Substanz sind
   Kernel-Erzeugung, ehrliche Messung und die daraus gewonnenen Erkenntnisse —
   danach richten sich Aufwand und Reihenfolge in diesem Plan.

Ausgangslage: was noch geklärt werden muss
==========================================

Dieser Plan wird bewusst **vor** der Umsetzung geschrieben, und ein Teil der
tragenden Entscheidungen lässt sich zum jetzigen Zeitpunkt noch nicht seriös
treffen. cuTile ist neu, spärlich dokumentiert und wir kennen die Zielmaschine
bisher nur aus den Assignments. Deshalb steht am Anfang eine **Klärungsphase**:
erst lesen, ausprobieren und vermessen, dann festlegen. Die offenen Punkte:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Offene Frage
     - Wie wir sie klären wollen
   * - **Welche Zahlenformate** trägt der vorhandene cuTile-Build überhaupt
       (rechnen, compilieren, korrekt)?
     - Kleines Wegwerf-Analyseskript: je Format ein winziges Batched-GEMM,
       gegen eine Referenz in voller Präzision geprüft. Nur was hier
       nachweislich läuft, kommt in den Umfang. **Das ist der wichtigste
       Punkt** — mehrere Achsen des Werkzeugs hängen daran.
   * - **Welche Akkumulator-Kombinationen** sind zulässig bzw. sinnvoll?
     - Dieselbe Vorstudie: je Format prüfen, in welcher Präzision summiert
       werden muss und was das an Genauigkeit kostet. Ergebnis wird als
       Regeltabelle festgeschrieben, nicht im Kopf behalten.
   * - **Wie sieht die cuTile-API in der Praxis aus** (Kachel laden,
       Multiplizieren, Speichern, Casten)?
     - Dokumentation lesen und die Assignment-Kernel als Vorlage nehmen;
       die belegte Form dann *einmal* als Schablone festhalten.
   * - **Wie lässt sich generierter Quelltext ausführen?**
     - Ausprobieren. Ob ein Kernel als Textbaustein direkt ausführbar ist oder
       vorher als Datei abgelegt werden muss, entscheidet über den Aufbau der
       Codegen-Stufe. Vermutung: eine echte Datei ist nötig.
   * - **Welche Kennwerte der Zielhardware** brauchen wir für eine ehrliche
       Einordnung (Rechen- und Speichergrenzen)?
     - Herstellerangaben und öffentliche Messungen zusammentragen, mit lokalen
       Abfragen abgleichen und die Unsicherheiten **dokumentieren** statt sie
       zu verschweigen.
   * - **Welches GUI-Framework** trägt einen mehrsekündigen GPU-Job, ohne
       einzufrieren?
     - Kandidaten gegeneinander abwägen (siehe unten) und am kleinsten
       möglichen Beispiel prüfen: Knopfdruck → Hintergrundarbeit →
       Fortschritt → Abbruch.
   * - **Wie groß dürfen die Testprobleme sein?**
     - Die Maschine ist geteilt; ein Speicherüberlauf trifft nicht nur uns.
       Wir tasten uns an eine Obergrenze heran und bauen sie als Grenze in die
       Eingabeprüfung ein, statt auf Disziplin zu hoffen.

Erst wenn diese Punkte beantwortet sind, sind die Achsen des Werkzeugs
(Formate, Kachelung, Messgrößen) überhaupt sinnvoll wählbar. Alles, was in der
Vorstudie nicht funktioniert, fällt aus dem Umfang — und wird als Befund
festgehalten, weil ein belegter Ausschluss auch ein Ergebnis ist.

Designentscheidungen und ihre Alternativen
==========================================

Die folgenden Punkte sind die Entscheidungen, an denen der Aufbau hängt. Wo wir
uns schon festlegen können, steht die Begründung dabei; wo die Vorstudie noch
fehlt, steht die Neigung und woran die Entscheidung hängt.

Messen: live oder vorberechnet
------------------------------

*Optionen:* (a) live auf der GPU messen, sobald ein Regler verstellt wird;
(b) einen Messkatalog vorab erzeugen und in der Oberfläche nur noch anzeigen.

Variante (b) wäre robuster und liefe auch ohne GPU — aber sie nimmt dem
Werkzeug genau die Eigenschaft, die es interessant macht: dass man selbst etwas
verstellt und *dieselbe* Maschine antworten sieht. Ein Katalog wäre eine
Diashow. **Wir entscheiden uns für live**, akzeptieren dafür die Wartezeit je
Lauf und müssen die Oberfläche entsprechend bauen (Hintergrundarbeit,
Fortschritt, Abbruch) — dieser Punkt zieht die GUI-Entscheidung nach sich.

Oberfläche: Framework
---------------------

*Optionen:* ein klassisches Web-Frontend von Hand, ein leichtgewichtiges
Skript-Framework, oder ein Framework mit ausgewiesener Unterstützung für
langlaufende Hintergrundaufgaben und interaktive Diagramme.

Ein handgebautes Frontend kostet Zeit, die in die Kernel-Erzeugung gehört. Die
leichtgewichtigen Varianten sind schnell schön, rechnen aber gerne bei jeder
Interaktion das ganze Skript neu — für einen mehrsekündigen GPU-Job das falsche
Modell, und Fortschritt bzw. Abbruch wären Handarbeit. **Neigung: das
Framework mit Hintergrundjobs und eingebauten interaktiven Diagrammen.**
Entschieden wird nach dem kleinen Machbarkeitsversuch aus der Klärungsphase.

Umfang der Operationen
----------------------

*Optionen:* (a) nur die Kontraktions-Familie (Matrixprodukt, Batch, allgemeine
Tensor-Kontraktion); (b) zusätzlich speichergebundene Operationen
(elementweise, Reduktion).

Nur Kontraktionen wären der kürzere Weg, aber alle Messpunkte lägen dann im
selben Bereich — die geplante Einordnung gegenüber der Hardware-Grenze hätte nur
eine Seite und wäre kaum aussagekräftig. **Neigung: (b)**, aber erst nachdem
die Kontraktion vollständig steht; die speichergebundenen Operationen sind der
Kontrast, nicht der Kern. einsum bleibt die gemeinsame Eingabesprache aller
Familien.

Kernel-Erzeugung
----------------

*Optionen:* (a) den Quelltext aus Textschablonen zusammensetzen; (b) einen
richtigen Zwischencode aufbauen und daraus emittieren; (c) auf vorhandene
Bibliotheken zurückgreifen und nur konfigurieren.

(c) fällt aus, weil dann nichts mehr erzeugt wird — das Erzeugen *ist* die
Aufgabe. (b) wäre der saubere Compiler-Weg, ist für den Umfang aber
überdimensioniert. **Entscheidung: (a) Schablonen**, mit einer wichtigen
Zusatzentscheidung: Jede Kontraktion wird **vorher** auf eine einheitliche
Grundform gebracht, sodass nur **eine** Kernel-Struktur erzeugt werden muss.
Der Preis ist eine Umform-Stufe davor; der Gewinn ist, dass es nur eine Stelle
gibt, an der die Orientierung der Multiplikation stimmen muss.

Korrektheit
-----------

*Optionen:* (a) am Ende einmal stichprobenartig prüfen; (b) **jeden** erzeugten
Kernel gegen eine Referenz in voller Präzision prüfen, bevor seine Messwerte
irgendwo auftauchen.

Die gefährlichste Fehlerklasse hier ist ein Kernel, der läuft, plausibel schnell
ist und **das Falsche** rechnet — eine vertauschte Orientierung fällt bei
quadratischen Eingaben nicht auf. Eine solche Zahl in einem Diagramm ist
schlimmer als eine Fehlermeldung. **Entscheidung: (b)**, als harte Regel:
*kein Messwert ohne bestandene Prüfung*. Zusätzlich wollen wir gezielt gegen die
vertauschten Varianten testen, nicht nur gegen „irgendwie ähnlich".

Ergebnis-Ablage
---------------

*Optionen:* (a) Datenbank; (b) eine Zeile je Lauf in einer Textdatei;
(c) nichts speichern, nur anzeigen.

(c) verschenkt die Nachvollziehbarkeit und macht den Bericht später zur
Erinnerungsarbeit. Eine Datenbank ist für die Datenmenge Überbau und schlecht
nachlesbar. **Entscheidung: (b)** — eine Zeile je Lauf, dazu der erzeugte
Kernel als eigene Datei unter einem **lesbaren** Namen aus der Konfiguration.
Der Name wird damit gleichzeitig zum Wiedererkennungs-Schlüssel: gleiche
Konfiguration, gleiche Datei, kein doppeltes Übersetzen. Wichtig dabei: Der
Name muss **alles** enthalten, was den Quelltext verändert — sonst greift man
still auf einen alten, unpassenden Kernel zu. Das ist eine der Stellen, an denen
wir uns leicht selbst betrügen können.

Kopplung Oberfläche ↔ Kern
--------------------------

*Optionen:* (a) die Oberfläche greift überall in den Kern; (b) genau **eine**
Schnittstelle: eine Konfiguration hinein, ein Ergebnis heraus.

(a) ist am Anfang schneller und später unwartbar; außerdem wäre der Kern dann
nicht ohne Oberfläche testbar. **Entscheidung: (b)**, mit zwei Nebenbedingungen,
die wir früh festschreiben wollen: der Kern muss auch ohne Oberfläche
vollständig benutzbar sein (Kommandozeile), und diese eine Schnittstelle darf
**nie** einen Fehler nach außen werfen — Fehler werden zu einem Zustand im
Ergebnis, damit die Oberfläche sie anzeigen kann statt abzustürzen.

Vergleichspunkte
----------------

*Optionen:* absolute Zahlen zeigen; oder gegen Bezugspunkte einordnen.

Eine absolute Zahl sagt niemandem etwas. Wir wollen zwei Bezugspunkte:
eine etablierte Bibliothek als **Obergrenze** („wie nah sind wir dran?") und
einen absichtlich untunten eigenen Kernel als **Untergrenze** („was bringt das
Verstellen überhaupt?"). Beide optional zuschaltbar, weil jede zusätzliche
Messung Zeit kostet.

Bewusst außerhalb des Umfangs
-----------------------------

Aus dem Pitch ausdrücklich **gestrichen**: ein automatisches Durchsuchen des
Konfigurationsraums („Auto-Tune") und die Landschafts-Darstellung dazu. Beides
ist reizvoll, aber es multipliziert die Messzeit auf einer geteilten Maschine
und trägt inhaltlich nichts bei, was der manuelle Vergleich nicht auch zeigt.
Wenn am Ende Luft bleibt, ist der aussichtsreichere Kandidat das
**Verschmelzen** einer Kontraktion mit einer nachgelagerten elementweisen
Operation — das ist eine echte inhaltliche Frage und knüpft an Idee 1 an.

Zielhardware
============

Entwicklung und Messung laufen auf einer **NVIDIA GB10** (Grace-Blackwell,
Compute Capability ``sm_121``) mit 128 GB Unified Memory und rund 273 GB/s
Speicherbandbreite. Wir vermuten, dass vieles davon memory-bound ist, aber das
muss noch untersucht werden.

Die Maschine wird **geteilt**. Das ist keine Randnotiz, sondern eine
Rahmenbedingung: Testprobleme bleiben klein, Läufe werden gegeneinander
serialisiert, und eine Größenprüfung sitzt vor jedem Lauf.

Aufbau: eine Pipeline, eine Naht
================================

Das Werkzeug soll als Kette klar getrennter Stufen entstehen:

   Ausdruck lesen → auf die einheitliche Grundform bringen → Kernel erzeugen →
   übersetzen → gegen die Referenz prüfen → messen → Ergebnis ablegen →
   darstellen.

Die Oberfläche hängt an **einer** Stelle daran (siehe oben). Damit bleibt der
Kern ohne Oberfläche prüfbar, die Oberfläche austauschbar, und beide Wege — mit
und ohne GUI — messen garantiert dasselbe.

Meilensteine
============

Die Umsetzung erfolgt in Meilensteinen. Leitgedanke: **jeder Meilenstein ist
eine vollständige, geprüfte Scheibe durch die ganze Kette** — nicht eine fertige
Stufe. Ein perfekter Codegen ohne Messung ist wertlos, weil wir gar nicht
wüssten, ob er stimmt. Reihenfolge: erst **in die Tiefe** (eine Operation
richtig), dann **in die Breite** (mehr Operationen), dann **Politur**.

.. list-table::
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - Meilenstein
     - Woran wir ihn als erreicht erkennen
   * - **0**
     - Klärung & Vorstudie
     - Die offenen Fragen von oben sind beantwortet und schriftlich belegt:
       welche Formate laufen, welche Kombinationen zulässig sind, wie die
       Kernel-Schablone aussieht, welches GUI-Framework es wird, welche
       Hardware-Kennwerte wir ansetzen. Ergebnis ist Wissen, kein Code.
   * - **1**
     - Tragendes Gerüst, ohne Oberfläche
     - **Eine** Operation läuft komplett durch: erzeugen → übersetzen → prüfen
       → messen → ablegen, bedienbar über die Kommandozeile. Damit steht die
       Kette und ist ab hier nur noch zu erweitern.
   * - **2**
     - Oberfläche um genau diese Operation
     - Eingabe im Browser, Messung auf der GPU, Kennzahlen und der erzeugte
       Quelltext werden angezeigt. Der Lauf blockiert die Oberfläche nicht und
       ist abbrechbar. Ab hier ist das Werkzeug vorführbar.
   * - **3**
     - Die Stellschrauben
     - Zahlenformate, Kachelung und Layout sind wählbar und werden
       gegeneinander gemessen; dazu die ehrlichen Messgrößen (Streuung der
       Laufzeit statt einer Einzelzahl) und die beiden Vergleichspunkte.
   * - **4**
     - Einordnung gegenüber der Hardware
     - Die Messpunkte werden ins Verhältnis zu den Grenzen der Maschine
       gesetzt, sodass ablesbar ist, *woran* eine Operation hängt — und nicht
       nur, wie schnell sie ist.
   * - **5**
     - In die Breite
     - Beliebige Kontraktionen über die einheitliche Grundform, dazu die
       speichergebundenen Operationen als Kontrast. Erst hier hat die
       Einordnung aus Meilenstein 4 beide Seiten.
   * - **6**
     - Nachvollziehbarkeit & Bericht
     - Vergangene Läufe sind wiederauffindbar und vergleichbar; ein
       reproduzierbarer Sammellauf erzeugt genau die Belege, aus denen der
       Bericht seine Zahlen zieht; Randfälle und Fehlerzustände sind
       aufgeräumt.
   * - **+**
     - Puffer / optional
     - Nur wenn 1–6 stehen: das oben genannte **Verschmelzen** von Kontraktion
       und nachgelagerter elementweiser Operation.

Reihenfolge-Begründung
----------------------

Warum nicht zuerst die Oberfläche? Weil eine Oberfläche ohne verlässliche Zahlen
dahinter genau das Ergebnis produziert, das dieses Projekt vermeiden soll:
schöne Diagramme, deren Werte niemand geprüft hat. Und warum nicht zuerst alle
Operationen? Weil jeder Fehler in der Grundform sich sonst über alle Familien
vervielfältigt. Deshalb: eine Operation vollständig und geprüft, dann
verbreitern.

Risiken und wie wir ihnen begegnen
==================================

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Risiko
     - Gegenmaßnahme
   * - Ein erzeugter Kernel rechnet **still das Falsche** (vertauschte
       Orientierung, falsches Format).
     - Prüfung gegen die Referenz vor jeder Anzeige; zusätzlich gezielte Tests
       gegen die vertauschten Varianten. Das ist das **Hauptrisiko** des
       Projekts.
   * - Der Wiedererkennungs-Schlüssel eines Kernels ist unvollständig → ein
       Lauf benutzt still einen fremden, alten Kernel.
     - Alles, was den Quelltext verändert, muss in den Namen eingehen; das wird
       eigens getestet.
   * - cuTile ist neu; eine benötigte Fähigkeit fehlt oder verhält sich anders
       als dokumentiert.
     - Genau dafür ist Meilenstein 0 da: früh ausprobieren, Umfang danach
       zuschneiden, Ausschlüsse als Befund festhalten.
   * - Die Oberfläche friert während der GPU-Arbeit ein oder verträgt sich
       nicht mit dem GPU-Kontext.
     - Hintergrundarbeit von Anfang an einplanen und den GPU-Zugriff strikt aus
       dem Hauptprozess heraushalten; am kleinen Beispiel prüfen, bevor die
       Oberfläche wächst.
   * - Speicherüberlauf auf der **geteilten** Maschine.
     - Kleine Testprobleme, harte Größenprüfung vor dem Lauf, Läufe
       gegeneinander serialisiert.
   * - Messwerte schwanken so stark, dass Vergleiche nichts sagen.
     - Aufwärmen, mehrere getaktete Wiederholungen, Streuung mit ausweisen —
       und den Zustand der GPU zu jedem Lauf mitschreiben.
   * - Der Umfang wächst schneller als die Zeit.
     - Die Meilensteine sind so geschnitten, dass nach jedem ein
       vorzeigbares Werkzeug existiert. Gestrichen wird von hinten.

Aufgabenverteilung
==================

Die Bearbeitung erfolgt **gemeinsam und flexibel** entlang der Meilensteine,
grob in zwei Spuren — „Kern/Pipeline" und „Oberfläche/Darstellung" —, ohne
starre Zuordnung einzelner Dateien. Die eine Naht zwischen Kern und Oberfläche
ist genau deshalb früh festgelegt: sie ist auch die Schnittstelle zwischen den
beiden Spuren und erlaubt paralleles Arbeiten. Die tatsächliche Aufteilung je
Abgabe wird in ``tar/contribution.txt`` festgehalten.

Was am Ende vorliegen soll
==========================

* Ein **lauffähiges Werkzeug**, das aus einem eingegebenen Ausdruck einen
  Kernel erzeugt, ihn prüft, misst und die Ergebnisse verständlich darstellt —
  bedienbar mit und ohne Oberfläche.
* **Nachvollziehbare Belege**: gespeicherte Läufe und die erzeugten Kernel,
  reproduzierbar über einen Sammellauf.
* **Erkenntnisse**, nicht nur Zahlen: der Zusammenhang von Zahlenformat,
  Genauigkeit und Durchsatz; was das Verstellen der Kachelung bringt und was
  nicht; und wo die untersuchten Operationen gegenüber der Hardware-Grenze
  tatsächlich liegen.
* Ein **Projektbericht**, der Aufbau und Begründung jeder Stufe erklärt — nicht
  nur die Ergebnisse auflistet.
