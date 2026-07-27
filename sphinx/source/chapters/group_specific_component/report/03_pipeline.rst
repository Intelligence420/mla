.. _gsc_report_pipeline:

##################################
Teil 3 — Die Pipeline im Detail
##################################

.. contents:: Inhalt dieses Teils
   :local:
   :depth: 2

Dieser Teil geht die acht Stufen aus :ref:`Teil 2 <gsc_report_architektur>` einzeln
durch. Jede Stufe wird nach demselben Muster behandelt: **was geht rein und raus**,
**wie funktioniert es**, und **warum so und nicht anders**.

.. _gsc_report_parse:

Stufe 1 — parse: vom String zur typisierten IR
==============================================

**Rein:** ``RunConfig`` · **Raus:** eine von vier IR-Klassen.

Der Parser routet zuerst auf die **Operations-Familie** und wendet danach die
familienspezifische Analyse an:

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Familie
     - IR
     - Inhalt
   * - ``contraction`` (2 Operanden)
     - ``ContractionIR``
     - Achsen klassifiziert nach M / N / K / Batch + Größen
   * - ``contraction`` (> 2 Operanden)
     - ``NAryContractionIR``
     - zusätzlich der geplante Pfad paarweiser Teil-Kontraktionen
   * - ``elementwise``
     - ``ElementwiseIR``
     - Form, Elementzahl, Arity (1 oder 2)
   * - ``reduction``
     - ``ReductionIR``
     - Achsen zerlegt in ``kept_dims`` / ``reduced_dims`` + gefaltete Größen

Die M/N/K/Batch-Klassifikation
------------------------------

Sie ist das Grundgerüst des Kontraktions-Pfades und besteht aus vier Regeln über die
Indexbuchstaben. Für Operanden :math:`I_0, I_1` und Output :math:`O`:

.. list-table::
   :header-rows: 1
   :widths: 14 42 44

   * - Typ
     - Regel
     - Bedeutung
   * - **Batch**
     - in :math:`I_0` **und** :math:`I_1` **und** :math:`O`
     - unabhängige Kopien derselben Rechnung
   * - **K**
     - in :math:`I_0` **und** :math:`I_1`, **nicht** in :math:`O`
     - wird **summiert** (kontrahiert)
   * - **M**
     - in :math:`I_0` **und** :math:`O`, nicht in :math:`I_1`
     - Zeilen des Ergebnisses
   * - **N**
     - in :math:`I_1` **und** :math:`O`, nicht in :math:`I_0`
     - Spalten des Ergebnisses

Für ``ik,kj->ij`` ergibt das trivial M=[i], N=[j], K=[k], Batch=[]. Interessanter
ist ein Ausdruck, wie er in der Oberfläche tatsächlich vorkommt — dieser Lauf steht
so im Store:

.. code-block:: text

   acspx,bspy->abcyx      alle sieben Indizes der Größe 64

   in beiden Operanden:  s, p     → nicht im Output  ⇒  K = {s, p}     → K = 64² = 4096
   in acspx und Output:  a, c, x                     ⇒  M = {a, c, x}  → M = 64³ = 262144
   in bspy  und Output:  b, y                        ⇒  N = {b, y}     → N = 64² = 4096
   in beiden + Output:   —                           ⇒  Batch = {}     → B = 1

Aus einer 5-dimensionalen und einer 4-dimensionalen Eingabe wird also ein
:math:`262144 \times 4096 \times 4096`-GEMM. Genau das ist der Punkt der Stufe:
**alles, was danach kommt, sieht nur noch M, N, K, B** — der Codegen muss nie
erfahren, dass es sieben Indizes gab.

Grenzen und Absicht
-------------------

Der Parser lehnt **laut** ab, was er nicht kann, statt es still falsch zu rechnen:
mehr als zwei Operanden gehen in den n-är-Zweig, Diagonalen und Spuren (ein Index
zweimal im selben Operanden, ``ii->i``) sind nicht unterstützt. Ein impliziter
Output (``ik,kj`` ohne ``->``) wird nach der einsum-Konvention ergänzt, damit
Eingaben aus der Oberfläche nicht an Formalia scheitern.

Die n-äre Kette
---------------

Mehr als zwei Operanden werden **nicht** zu einem eigenen Kernel-Typ, sondern in
eine Folge paarweiser Kontraktionen zerlegt, die jede durch den bewiesenen
2-Operanden-Pfad läuft. Die Reihenfolge kommt von ``opt_einsum.contract_path``,
falls installiert. Ansonsten greift ein deterministischer Links-nach-rechts-Fold. Für
``ij,jk,kl->il`` ergibt das die Schritte ``ij,jk->ik`` und dann ``kl,ik->il``.

Das ist eine bewusste Entscheidung gegen einen n-är-Kernel: Ein einzelner Kernel
für beliebig viele Operanden wäre ein neues, unbewiesenes Codegen-Muster. 
Paarweise Zerlegung benutzt dagegen ausschließlich die Struktur, die schon gegen
``torch.einsum`` verifiziert ist. Der Preis steht ehrlich in den Ergebnissen: Die
Zwischentensoren kosten Speicherverkehr, weshalb die Kette eine *niedrigere*
arithmetische Intensität hat als ein einzelnes GEMM.

.. _gsc_report_reshape:

Stufe 2 — der B1-Reshape: alles wird ein Batched GEMM
=====================================================

**Rein:** ``ContractionIR`` · **Raus:** ``Canonical`` (M, N, K, B + eine
View-Spezifikation je Operand). Memory-bound-Familien überspringen diese Stufe.

Die Entscheidung: eine kanonische Form
--------------------------------------

Es gäbe zwei Wege, beliebige Kontraktionen zu unterstützen:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Kernel je Ausdruck
     - **Kanonische Form** (gewählt)
   * - Idee
     - der Codegen erzeugt für jede Indexstruktur passenden Code
     - der **Host** bringt jede Kontraktion per ``permute``/``reshape`` auf
       :math:`(B,M,K) \times (B,K,N) \rightarrow (B,M,N)`; der Codegen kennt nur
       diese eine Form
   * - Codegen-Komplexität
     - wächst mit jeder Indexvariante
     - konstant — **eine** Struktur
   * - Korrektheitsrisiko
     - jede Variante ist ein neuer, unbewiesener ``ct.mma``-Fall
     - genau eine Orientierung, dreifach gegen ``torch.einsum`` verifiziert
   * - Kosten
     - keine Host-Vorbereitung
     - Indexarithmetik auf dem Host, im Zweifel eine Kopie

Der zweite Weg verlagert das Risiko von einer Stelle, an der es schwer prüfbar ist
(generierter GPU-Code), an eine, an der es leicht prüfbar ist (View-Mathematik auf
dem Host, testbar gegen ``torch.einsum``). Deshalb ist kanonisch **immer**
:math:`(B,M,K)` — auch bei :math:`B = 1`, wo der Grid einfach eine Achse der Länge
1 bekommt. Ein Sonderfall weniger.

Wie der View gebaut wird
------------------------

Aus den klassifizierten Achsen ergibt sich die Zielreihenfolge direkt:

.. code-block:: text

   Operand A:  natürliche Achsen  →  permute nach [Batch…, M…, K…]  →  reshape (B, M, K)
   Operand B:  natürliche Achsen  →  permute nach [Batch…, K…, N…]  →  reshape (B, K, N)
   Output   :  (B, M, N)  →  reshape in [Batch…, M…, N…]  →  permute in die Output-Reihenfolge

Der Rück-Umbau des Outputs ist nicht Kosmetik: Er wird gebraucht, damit ``verify``
gegen ``torch.einsum(config.expr, …)`` in der **natürlichen** Form vergleichen kann.

Zero-copy oder Kopie? Das formale Kriterium
-------------------------------------------

``reshape`` liefert nur dann einen View, wenn die zu verschmelzenden Achsen im
Speicher zusammenhängen. Das Kriterium dafür ist eine Stride-Adjazenz — dieselbe
Bedingung, die auch der aus Assignment 05/06 portierte ``fuse_dims`` prüft:

.. math::

   \text{stride}_i = \text{stride}_{i+1} \cdot \text{size}_{i+1}
   \quad \text{für alle Nachbarpaare einer Gruppe}

Das Werkzeug wendet das auf die permutierten Strides jeder Gruppe (Batch, M, K bzw.
N) an und **prognostiziert damit vor dem Lauf**, ob der Umbau kopierfrei ist. Am
Beispiel von oben, ``acspx`` mit allen Größen 64 (Row-Major-Strides
:math:`a{:}\,64^4,\ c{:}\,64^3,\ s{:}\,64^2,\ p{:}\,64,\ x{:}\,1`):

.. code-block:: text

   Ziel-Reihenfolge  [a, c, x | s, p]        (M-Gruppe | K-Gruppe)
   Strides danach    [64⁴, 64³, 1 | 64², 64]

   M-Gruppe:  64⁴ == 64³·64 ?  ✓   (a und c hängen zusammen)
              64³ ==   1·64  ?  ✗   (zwischen c und x liegt die s,p-Ebene)
   ⇒ die M-Achsen lassen sich NICHT kopierfrei verschmelzen ⇒ Setup-Kopie

Diese Kopie ist erlaubt und **verfälscht die Messung nicht** — aus zwei Gründen,
die beide explizit sind: Sie passiert (a) **vor** dem Messfenster (in der
Vorbereitung, nicht in der getakteten Schleife) und (b) außerhalb der
Roofline-Metriken, weil diese analytisch aus M/N/K und den dtype-Größen berechnet
werden und nicht aus beobachtetem Speicherverkehr. Der Vorhersage-Wert
(``zero_copy``) bleibt trotzdem interessant, weil er sagt, ob die Vorbereitung in
einem echten Anwendungsfall gratis wäre.

.. note::

   Aus Assignment 05/06 sind ``Config``/``Optimizer`` **vollständig** portiert
   (``split_dim``, ``fuse_dims``, ``permute_dims``, ``make_executable``,
   ``verify``). Der B1-Reshape braucht davon nur ``generate_config`` (Validierung +
   Per-Tensor-Strides), ``fuse_dims``/``permute_dims`` und den Adjazenztest. Die
   Host-Tiling-Heuristik der Assignments (PRIM/SEQ/PAR-Scheduling) wird **nicht**
   aufgerufen — wir kacheln im Template, nicht auf dem Host. Sie ist der
   Vollständigkeit halber mitgeführt und im Modul als ungenutzt markiert. Das
   erschien uns ehrlicher, als einen halben Port zu zeigen.

.. _gsc_report_codegen:

Stufe 3 — Codegen: aus einer Config wird Quelltext
==================================================

**Rein:** ``RunConfig`` · **Raus:** ein vollständiger, ausführbarer
cuTile-Modul-Quelltext als String.

Die Technik: f-String-Templates
-------------------------------

Der Codegen ist absichtlich das einfachste Werkzeug, das die Aufgabe löst: pro
Familie eine Funktion, die einen f-String zusammensetzt. Kein AST-Bau, keine
Template-Engine.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Ansatz
     - Vorteil
     - Warum nicht gewählt
   * - **f-String** (gewählt)
     - der Generator ist *lesbar wie das Ergebnis*; ein Diff zwischen zwei
       generierten Kerneln ist ein Textdiff
     - —
   * - Jinja2 o. ä.
     - Trennung Logik/Text
     - zusätzliche Abhängigkeit, und die Templates sind klein genug
   * - AST / ``ast.unparse``
     - syntaktisch garantiert gültig
     - massiv aufwändiger; Syntaxfehler fängt hier ohnehin der Import sofort
   * - ``torch.compile``/Triton-Codegen
     - viel Maschinerie geschenkt
     - das Projektziel ist ja gerade, cuTile-Kernel selbst zu erzeugen

Der Preis des einfachen Ansatzes ist, dass Quelltext-Fragmente korrekt eingerückt
werden müssen — dafür ist der Gewinn, dass jeder erzeugte Kernel eine gut lesbare,
kommentierte Datei ist, die man einem Menschen zeigen kann. Das ist keine
Nebensache: Die Oberfläche zeigt genau diesen Text an, und die Dateien sind
eingecheckt.

Der generierte GEMM, Zeile für Zeile
------------------------------------

Das ist der Kern des ganzen Werkzeugs — der Kernel, den alle Kontraktions-Läufe
dieses Berichts benutzen (hier fp16 → fp32, Tile 128/128/64, ohne Swizzle):

.. code-block:: python
   :caption: ``results/kernels/ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64.py`` (Auszug)

   TM = 128            # Tile-Literale — vom Codegen eingesetzt
   TN = 128
   TK = 64

   @ct.kernel
   def gemm(A, B, C,
            M: ct.Constant[int],
            N: ct.Constant[int],
            K: ct.Constant[int]):
       i = ct.bid(0)     # welche M-Kachel bearbeitet dieser Block?
       j = ct.bid(1)     # welche N-Kachel?
       bb = ct.bid(2)    # welche Batch-Scheibe?

       acc = ct.full((TM, TN), 0, dtype=ct.float32)

       for kk in range(ct.cdiv(K, TK)):
           a = ct.load(A, index=(bb, i, kk), shape=(1, TM, TK),
                       padding_mode=ct.PaddingMode.ZERO)
           a = ct.reshape(a, (TM, TK))
           b = ct.load(B, index=(bb, kk, j), shape=(1, TK, TN),
                       padding_mode=ct.PaddingMode.ZERO)
           b = ct.reshape(b, (TK, TN))
           acc = ct.mma(a, b, acc)

       ct.store(C, index=(bb, i, j),
                tile=ct.reshape(ct.astype(acc, C.dtype), (1, TM, TN)))

   def launch(A, B, C):
       Bb, M, K = A.shape
       _, _, N = B.shape
       grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), Bb)
       ct.launch(torch.cuda.current_stream().cuda_stream,
                 grid, gemm, (A, B, C, M, N, K))
       return C

Was hier passiert und warum es so aussieht:

* **Ein Block berechnet genau eine Ausgabekachel** :math:`(TM, TN)` von
  :math:`C[bb]`. Das Grid hat deshalb
  :math:`\lceil M/TM \rceil \times \lceil N/TN \rceil \times B` Blöcke. Die
  Batch-Achse ist die dritte Grid-Achse — der Grund, warum :math:`B = 1` kein
  Sonderfall ist, sondern nur ein Grid der Tiefe 1.
* **Der Akkumulator ist unabhängig vom Eingabeformat.** ``ct.full(..., dtype=
  ct.float32)`` legt fest, dass in fp32 summiert wird, auch wenn fp16-Kacheln
  hineinlaufen. Das ist die Stelle, an der die Assignment-Vorgabe „FP16-Eingaben,
  FP32-Akkumulator" im Code lebt — und die Stelle, an der später ein echter Bug saß
  (Teil 5).
* **Die K-Schleife läuft über Kacheln, nicht über Elemente.** ``index=(bb, i, kk)``
  zählt in Kachel-Einheiten: Iteration ``kk`` liest den Block, der bei Spalte
  :math:`kk \cdot TK` beginnt.
* **Ränder brauchen kein Masking.** ``padding_mode=ZERO`` füllt fehlende Elemente
  mit Nullen, und das ist für ein Multiply-Accumulate **neutral**:
  :math:`0 \cdot x + acc = acc`. Beim Schreiben schneidet ``ct.store`` den Überstand
  automatisch ab. Zwei Zeilen Vertrauen statt zwanzig Zeilen Indexlogik — belegt
  durch die ragged-Tests.
* **Der ``ct.reshape``-Tanz** (``(1,TM,TK) → (TM,TK)``) ist notwendig, weil die
  Batch-Scheibe als führende Achse der Länge 1 geladen wird, ``ct.mma`` aber
  2D-Kacheln erwartet.
* **Tile-Größen sind Literale, M/N/K sind Launch-Argumente.** Das ist eine
  Design-Entscheidung mit direkter Folge für den Cache: Weil die Größen nicht im
  Quelltext stehen, ist **ein** Kernel-Artefakt für *alle* Größen gültig — der
  Fusions-Sweep compiliert einen Kernel und misst ihn auf drei Formen. Umgekehrt
  erzeugt eine andere Kachelung einen anderen Text und damit ein anderes Artefakt.
  ``ct.Constant[int]`` sorgt dafür, dass der JIT M/N/K trotzdem als Konstanten
  sehen und einrechnen kann.

Die MMA Orientierung
--------------------

``ct.mma(a, b, acc)`` mit :math:`a = (TM, TK)`, :math:`b = (TK, TN)` ergibt
:math:`(TM, TN)` — **kein** Operanden-Swap, **kein** Permute. Das klingt
selbstverständlich, ist es aber nicht: In Assignment 06 war ein Swap nötig, weil
dort das Output-Layout ``yx`` verlangt war. Eine falsche Orientierung ist der
gefährlichste Fehler in diesem Projekt, weil das Ergebnis *plausibel aussieht*.
Deshalb ist diese eine Struktur dreifach unabhängig gegen ``torch.einsum``
verifiziert, es gibt einen Test, der die Orientierung explizit bewacht, und jedes
Template trägt sie im Docstring — direkt in der generierten Datei.

Zahlenformate
-------------

Der Codegen kennt zwei kleine Abbildungen, aus denen sich das gesamte
Format-Verhalten ergibt:

.. list-table:: Akkumulator-Regeln (``schema.ALLOWED_ACC``)
   :header-rows: 1
   :widths: 26 26 48

   * - Compute-dtype
     - erlaubter Akkumulator
     - Begründung
   * - fp16
     - fp16 **oder** fp32
     - fp16-Akku ist schneller, aber gröber — beides legitim, also wählbar
   * - bf16, tf32
     - **nur** fp32
     - bf16/tf32 sind reine *Compute*-Formate. Ein Akku in bf16 wäre numerisch
       uninteresannt
   * - fp8 e4m3 / e5m2
     - fp16 **oder** fp32
     - wie fp16
   * - fp32
     - fp32
     - Anker/Diagnose

Diese Regeln sind kein Kommentar, sondern Daten — und sie werden **dreimal**
durchgesetzt: in der Oberfläche existieren unzulässige Kombinationen gar nicht
(die Auswahlliste wird aus derselben Tabelle erzeugt), ``run()`` prüft sie früh,
und die Toleranztabelle in ``verify`` enthält keinen Eintrag für sie. Ein still
falsch akkumulierter Lauf müsste an drei Stellen gleichzeitig durchrutschen.

Die zweite Tabelle beantwortet: *braucht dieses Format einen Cast im Kernel?*
Für alle Formate ist die Antwort nein — außer für **tf32**:

.. code-block:: python

   a = ct.astype(a, ct.tfloat32)      # nur bei dtype == "tf32"
   b = ct.astype(b, ct.tfloat32)

Der Hintergrund ist lehrreich: tf32 ist kein Speicherformat, sondern ein
Rechenmodus — die Operanden liegen als fp32 im Speicher (4 Byte). ``ct.mma``
besitzt in diesem cuTile-Build **kein** Präzisions-Flag; ohne den expliziten Cast
liefe die Multiplikation still auf den CUDA-Cores statt auf den Tensor-Cores.
Das Ergebnis wäre *rechnerisch korrekt*, aber **langsamer** (0,2 statt
6 TFLOP/s im Vorab-Test). Genau die Klasse von Fehler, die eine Verifikation
niemals fängt — weil das Ergebnis stimmt. Nur ein Blick auf den Durchsatz verrät
sie, und man muss wissen, wonach man sucht.

L2-Swizzle: dieselben Kacheln, andere Reihenfolge
-------------------------------------------------

Ohne Swizzle bearbeitet Block :math:`(i, j)` die Kachel :math:`(i, j)` — die
Blöcke laufen also zeilenweise über die Ausgabe. Das ist für den L2-Cache
ungünstig: Bis Zeile :math:`i+1` beginnt, sind alle :math:`B`-Spalten längst
verdrängt. Die *grouped-M*-Rasterung ordnet die Zuordnung um, sodass
``GROUP_M`` benachbarte Kachelzeilen zu einer Gruppe zusammengefasst werden und
sich die Blöcke einer Gruppe dieselben :math:`B`-Spalten teilen:

.. code-block:: python

   num_pid_m = ct.cdiv(M, TM)
   num_pid_n = ct.cdiv(N, TN)
   pid = ct.bid(0) * num_pid_n + ct.bid(1)      # lineare Block-Nummer
   num_pid_in_group = GROUP_M * num_pid_n
   group_id = pid // num_pid_in_group
   first_pid_m = group_id * GROUP_M
   group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # letzte Gruppe evtl. kürzer
   local = pid % num_pid_in_group
   i = first_pid_m + (local % group_size_m)
   j = local // group_size_m

Drei Beobachtungen dazu:

* Die Abbildung ist **bijektiv** — dieselbe Kachelmenge, nur andere Reihenfolge.
  Das Ergebnis ist deshalb *numerisch identisch*, was per GPU-Test belegt ist. Der
  Swizzle ist damit die sauberste Tuning-Achse des Werkzeugs: Sie kann nur die
  Laufzeit ändern, nichts anderes.
* ``group_size_m`` wird mit ``min(num_pid_m - first_pid_m, GROUP_M)`` begrenzt,
  damit die letzte Gruppe bei nicht aufgehendem Gitter nicht über den Rand läuft.
  Diese eine Zeile erklärt später, warum die ``GROUP_M``-Achse auf einem kleinen
  Gitter **wirkungslos** ist: Ist ``num_pid_m`` selbst kleiner als ``GROUP_M``,
  entsteht nur eine einzige Gruppe und jede Wahl :math:`\ge` Gittergröße erzeugt
  dieselbe Permutation.
* Bei ``swizzle=False`` erzeugt der Codegen **exakt** die drei ``ct.bid``-Zeilen
  von oben — der Quelltext ist byte-identisch zur Variante ohne diese Funktion.
  Das ist ein wiederkehrendes Prinzip (siehe unten).

Die Epilog-Fusion
-----------------

Eine optionale elementweise Operation wird **auf dem Akkumulator-Tile** angewendet,
bevor ``ct.store`` es wegschreibt. Der Zwischentensor entsteht damit nie:

.. code-block:: python

   for kk in range(ct.cdiv(K, TK)):        # Kontraktions-Loop, unverändert
       acc = ct.mma(a, b, acc)

   d = ct.load(D, index=(bb, i, j), shape=(1, TM, TN),
               padding_mode=ct.PaddingMode.ZERO)
   d = ct.reshape(d, (TM, TN))
   acc = acc + ct.astype(d, ct.float32)    # Epilog auf dem Akku-Tile …
   ct.store(C, index=(bb, i, j),           # … VOR dem Store
            tile=ct.reshape(ct.astype(acc, C.dtype), (1, TM, TN)))

``bias`` braucht einen vierten Operanden :math:`D` in voller Ausgabeform und
verändert damit die Signatur zu ``launch(A, B, D, C)``; ``relu`` ist operandenlos
(``acc = ct.maximum(acc, 0)``). Die Arity wandert durch die ganze Pipeline —
Kernel-Signatur, ``launch``, Mess-Schleife, Metrik-Bytes — was nur deshalb
schmerzfrei ist, weil die Mess-Schicht von Anfang an variadisch gebaut wurde
(„letzter Operand ist der Output").

Anti-Drift-Prinzip
------------------

Dreimal in der Geschichte des Werkzeugs kam eine neue Codegen-Achse dazu (Swizzle,
``GROUP_M``, Epilog). Jedes Mal galt dieselbe Regel:

.. admonition:: Additiv heißt byte-identisch

   Ist die neue Achse nicht gesetzt, soll der erzeugte Quelltext **identisch**
   zu vorher sein.

Ein Teil der Kernel-Artefakte ist als **Referenz
eingecheckt** (die übrigen entstehen bei jedem Lauf lokal neu). Würde eine
Erweiterung den erzeugten Text auch nur um einen Kommentar verändern, schriebe
der nächste Lauf all diese Dateien um: Der Compile-Cache wäre kalt, und jeder
Commit enthielte Pseudo-Änderungen, in denen echte Template-Änderungen untergehen.
Ein Test vergleicht deshalb den unfusionierten Quelltext zeichenweise mit dem
Zustand davor — und genau deshalb steht die Epilog-Zeile im Datei-Header nur,
*wenn* ein Epilog gesetzt ist.

memory-bound-Templates
----------------------

Beide sind bewusst *keine* Varianten des GEMM-Templates, sondern eigene, kleinere
Strukturen — ohne ``ct.mma``, ohne Akkumulator-Loop, ohne B1-Reshape.

**Elementwise** kachelt eine 2D-Sicht ``(rows, cols)`` mit einem echten
``cdiv``-2D-Grid:

.. code-block:: python

   @ct.kernel
   def elementwise(A, C):
       i = ct.bid(0)
       j = ct.bid(1)
       a = ct.load(A, index=(i, j), shape=(TM, TN), padding_mode=ct.PaddingMode.ZERO)
       ct.store(C, index=(i, j), tile=ct.astype(a, C.dtype))   # hier: copy

Dass der Host **die letzte, kontiguierte Achse als ``cols``** faltet, ist kein
Zufall, sondern ein Messergebnis aus Assignment 02: Die inneren Achsen zu kacheln
war dort rund **3,5× schneller**. Die Op selbst (``a + b``, ``a * b``, nur ``a``,
``ct.maximum(a, 0)``) wird als Fragment einsubstituiert und bestimmt zugleich die
Arity — ``add``/``mul`` sind binär, ``copy``/``relu`` unär.

**Reduktion** hat als einziges Template **zwei Pfade**, und der Grund ist
Ehrlichkeit über die Beweislage:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Pfad
     - Wann
     - Status
   * - single-shot
     - reduzierte Achse passt (auf ``next-pow2`` aufgerundet) in **eine** Kachel
       — bis 16384
     - aus Assignment 02 **bewiesen**; ein einziges
       ``ct.sum(..., axis=1)``
   * - K-Loop-Fallback
     - größere Achsen
     - **nicht** in A02 bewiesen und im generierten Code als solcher markiert;
       akkumuliert im GEMM-Muster ``acc += ct.sum(chunk)``

Beide Pfade summieren in ``ct.float32``, und zwar mit einem **Cast vor**
``ct.sum``:

.. code-block:: python

   acc = ct.sum(ct.astype(tile, ct.float32), axis=1)

Dieser Cast ist die Korrektur eines echten Fehlers — die Geschichte dazu steht in
:ref:`Teil 5 <gsc_report_ergebnisse>`. Die Kachelbreite des single-shot-Pfades
(``TILE_K = next-pow2(K)``) wird als **Launch-Konstante** übergeben, nicht als
Literal eingesetzt: Sonst wäre der Quelltext größenabhängig und derselbe Slug
stünde für verschiedene Kernel.

.. _gsc_report_compile:

Stufe 4 — compile: Quelltext ladbar machen und cachen
=====================================================

**Rein:** Quelltext · **Raus:** aufrufbares ``launch(*operanden)``.

Der Weg ist kurz, aber jeder Schritt hat einen Grund:

.. code-block:: text

   emit(config) → Quelltext → results/kernels/<slug>.py → importieren → launch

**Warum eine Datei?** Weil cuTile den Kernel-Quelltext zur JIT-Zeit über
``inspect.getsourcelines`` liest. Ein ``exec(src)`` aus einem String scheitert mit
``OSError: could not get source code``. Diese technische Notwendigkeit hat sich als
Glücksfall erwiesen: Dieselbe Datei ist **Compile-Cache**, **Anzeige in der
Oberfläche** und **nachprüfbarer Beleg** dafür, was gemessen wurde.

Der Slug
--------

Der Dateiname ist die Identität eines Kernels:

.. code-block:: text

   ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64__ep_bias__sw_g16.py
   └─ Ausdruck ─┘  └ Format ┘  └── Kachelung ──┘  └ Epilog ┘└ Swizzle ┘

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Entscheidung
     - Begründung
   * - **lesbar statt Hash**
     - Die vollständige Config steht ohnehin in jeder ``results.jsonl``-Zeile. Ein
       sprechender Name macht das Verzeichnis durchsuchbar und Diffs verständlich —
       ``a3f9c2…py`` hätte keinen dieser Vorteile.
   * - **normalisiert**
     - Whitespace entfernt, Tile-Reihenfolge fest ⇒ logisch gleiche Configs
       ergeben denselben Namen und damit einen Cache-Treffer.
   * - **ohne ``dim_sizes``**
     - M/N/K sind Launch-Argumente; ein Artefakt gilt für alle Größen.
   * - **ohne ``family``**
     - folgt deterministisch aus dem Ausdruck.
   * - **bedingte Suffixe** (``op``, ``epilog``, ``group_m``)
     - nur wenn gesetzt bzw. vom Default abweichend — das ist die Anti-Drift-Regel
       auf Dateinamen-Ebene. Kritisch ist dabei nur die eine Richtung:
       **verschiedener Quelltext darf nie denselben Slug haben**, sonst träfe ein
       Lauf still das falsche gecachte Artefakt.

Zwei Cache-Ebenen und Selbstheilung
-----------------------------------

* **Im Prozess:** ``slug → launch``-dict. Wiederholte Läufe derselben
  Konfiguration importieren nicht neu — wichtig für die Live-Oberfläche.
* **Auf der Platte:** ``kernels/<slug>.py`` wird nur geschrieben, wenn die Datei
  fehlt oder ihr Inhalt abweicht (idempotent), und dann **atomar** (Temp-Datei +
  ``os.replace``). Ein paralleler Leser — insbesondere der JIT selbst — sieht
  nie eine halb geschriebene Datei.
* **Selbstheilung:** Ist ein Artefakt unlesbar oder nicht dekodierbar (``OSError``,
  ``UnicodeDecodeError``), wird es **neu erzeugt** statt den Lauf abzubrechen. Ein
  korrupter Cache ist ein Ärgernis, kein Fehler.

Ein kleines Detail mit Wirkung: Beim Import wird ``sys.dont_write_bytecode``
gesetzt. ``results/kernels/`` ist ein *Daten*-Verzeichnis mit eingecheckten
Dateien — dort hat kein ``__pycache__`` etwas zu suchen, und cuTile braucht ohnehin
den Quelltext, nicht die ``.pyc``.

Der eigentliche cuTile-JIT passiert hier **nicht**: Er läuft lazy beim ersten
``ct.launch`` — und damit in der nächsten Stufe, wo seine Zeit auch gemessen wird.

.. _gsc_report_verify:

Stufe 5+6 — Kalt-Lauf und verify: das Gate
==========================================

Der Kalt-Lauf als Doppelnutzen
------------------------------

Der erste ``launch``-Aufruf kostet den JIT. Statt diese Zeit zu verstecken, wird
sie als ``compile_ms`` per **Wall-Clock** gemessen (``time.perf_counter()`` um
``launch`` + ``torch.cuda.synchronize()``). CUDA-Events wären hier falsch: Sie
messen GPU-Zeit, der JIT ist aber ein **host-seitiger** Kompilierschritt und
würde komplett übersehen. Der Kalt-Lauf füllt zugleich den Output-Tensor — genau
den, den ``verify`` als nächstes beurteilt. Der teure Schritt passiert also genau
einmal und liefert zwei Ergebnisse.

verify ist ein reiner Urteiler
------------------------------

``verify(output, operands, config)`` startet **keinen** Kernel. Es bekommt einen
fertigen Tensor und vergleicht ihn mit der fp32-Referenz derselben Operation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Familie / Op
     - Referenz
   * - Kontraktion
     - ``torch.einsum(expr, A.float(), B.float())``
   * - Kontraktion + Epilog
     - dasselbe einsum, **gefolgt vom Epilog** in fp32 (``+ D`` bzw.
       ``clamp(min=0)``)
   * - Reduktion
     - ``torch.einsum(expr, A.float())`` — die Summe ist als einsum ausdrückbar
   * - Elementwise
     - direkt aus der Op: ``a + b``, ``a * b``, ``a``, ``clamp(min=0)``. ``add``
       und ``copy`` sind **kein** einsum-Ausdruck, deshalb keine einsum-Referenz.
   * - n-äre Kette
     - ``torch.einsum`` über den **vollen** Ausdruck mit allen n Operanden — nicht
       schrittweise, sonst prüfte man die Zerlegung gegen sich selbst

Gemeldet werden ``max_abs_err``, ``mean_abs_err`` und ``rel_err`` (L2-Norm des
Fehlers relativ zur Referenznorm — dimensionslos und damit über Formate
vergleichbar), plus das binäre ``passed`` aus ``torch.allclose``.

Die Toleranztabelle
-------------------

Die Toleranzen sind nach **(dtype, acc_dtype)** gekeyt, nicht global. Eine einzige
Toleranz für alle Formate wäre entweder so lasch, dass sie einen fp16-Bug
durchlässt, oder so streng, dass fp8 immer scheitert:

.. list-table:: ``atol`` / ``rtol`` je Format-Kombination
   :header-rows: 1
   :widths: 24 12 12 52

   * - Kombination
     - ``atol``
     - ``rtol``
     - Begründung
   * - fp16 → fp32
     - 0,2
     - 0,02
     - der Anker (Werte aus Assignment 03/05)
   * - fp16 → fp16
     - 8,0
     - 0,2
     - fp16-Akku rundet grob (gemessen ≈ 0,22 bei 512³)
   * - bf16 → fp32
     - 1,0
     - 0,02
     - gröbere Mantisse als fp16 (8 statt 11 Bit Präzision)
   * - tf32 → fp32
     - 1,0
     - 0,02
     - fp16-artige Mantisse mit fp32-Exponent
   * - fp8 e4m3 → fp32
     - 0,2
     - 0,02
     - straff wie fp16: die fp8-Quantisierung fällt aus der Differenz heraus, weil
       die Referenz *dieselben* quantisierten Werte sieht
   * - fp8 e4m3 → fp16
     - 8,0
     - 0,2
     - grober Akku
   * - fp8 e5m2 → fp16
     - 16,0
     - 0,3
     - noch gröbere Mantisse ⇒ eigenes Gate
   * - fp32 → fp32
     - 0,01
     - 0,001
     - Diagnose-Anker

Wichtig ist das Prinzip dahinter: Die Toleranzen sind aus **gemessenen** Fehlern
der Vorab-Analyse abgeleitet, mit großzügigem Abstand nach oben. Ein korrekter
Kernel soll über wechselnde Größen nie falsch-negativ sein. Ein *grober* Fehler —
und eine vertauschte mma-Orientierung ist grob — wird trotzdem sicher gefangen. Die
Tabelle ist zugleich die dritte Verteidigungslinie der Akkumulator-Regeln: Für eine
unzulässige Kombination existiert kein Eintrag, was zu einem klaren Fehlerstatus
führt statt zu einer stillen Messung.

**Was ein Fehlschlag kostet:** Bei ``passed == False`` bekommt der Lauf
``verify_failed``, die Fehlerzahlen bleiben erhalten — aber es wird **nicht
gemessen**. Es gibt daher im ganzen System keine Durchsatzzahl ohne bestandene
Referenz. Das ist der Satz, auf den sich jede Zahl in Teil 5 stützt.

.. _gsc_report_bench:

Stufe 7 — Messung: Zeit, Verteilung, Kennzahlen
===============================================

Warme Messung mit CUDA-Events
-----------------------------

Nach ``verify`` folgt die eigentliche Messung: ``warmup`` ungetaktete Läufe
(stabilisieren Takt und Caches), dann ``iters`` getaktete Läufe mit **je einem
eigenen Event-Paar**. Gemessen wird also nicht eine Summe über viele Iterationen,
sondern eine **Verteilung**:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Kennzahl
     - Definition und Zweck
   * - ``run_ms``
     - **Median** der Iterationen — die Hauptzahl, robust gegen einzelne Ausreißer
       (auf einer geteilten Maschine kein Luxus)
   * - ``min_ms``
     - schnellste Iteration — die „was geht bestenfalls"-Zahl
   * - ``p90_ms``
     - 90.-Perzentil per **nearest-rank** (interpolationsfrei, damit im Test exakt
       vorhersagbar) — zeigt den Ausreißer-Kopf
   * - ``sigma_ms``
     - **Populations**-Standardabweichung über die Iterationen (nicht Stichprobe:
       wir messen die Grundgesamtheit unserer Läufe)

L2-Flush: kalt messen
---------------------

Zwischen den getakteten Iterationen wird ein 256-MiB-Puffer genullt und damit der
L2-Cache geleert — dasselbe Vorgehen wie ``triton.do_bench``. Ohne diesen Flush
würde die zweite Iteration Daten im Cache finden, die eine erste, echte Ausführung
nie hätte. Der gemessene Durchsatz wäre für kleine Formen systematisch zu gut. Der
Flush wird **vor** dem Start-Event abgesetzt und zählt daher (Stream-Reihenfolge)
nicht in die Messung. Schlägt die Allokation des Puffers fehl, weil die geteilte
GPU voll ist, wird ohne Flush weitergemessen — das ist besser, als einen
verifizierten Lauf an einem Messdetail scheitern zu lassen.

Ein Detail, das man kennen muss: Ist ein Fortschritts-Callback gesetzt (Live-Anzeige
„k/N" in der Oberfläche), wird pro Iteration synchronisiert, damit der Zähler echte
Läufe spiegelt. Die *gemessene* Zeit bleibt davon unberührt, weil sie zwischen den
GPU-Events liegt; der Host wartet lediglich mit.

Kennzahlen: die Formeln
-----------------------

Alle abgeleiteten Größen sind **analytisch** — sie kommen aus M/N/K und den
dtype-Größen, nicht aus Hardware-Zählern:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Größe
     - Formel
   * - GEMM-FLOP
     - :math:`2 \cdot B \cdot M \cdot N \cdot K` (je MAC eine Multiplikation +
       eine Addition)
   * - GEMM-Bytes
     - :math:`B \cdot \bigl(\text{in} \cdot (MK + KN) + \text{out} \cdot MN\bigr)`
   * - Elementwise
     - FLOP :math:`= n` (0 für ``copy``); Bytes
       :math:`= n \cdot (\text{arity} \cdot \text{in} + \text{out})`
   * - Reduktion
     - FLOP :math:`= \text{kept} \cdot \text{reduced}`; Bytes
       :math:`= \text{kept} \cdot \text{reduced} \cdot \text{in}
       + \text{kept} \cdot \text{out}`
   * - n-äre Kette
     - Summe über die Schritte — **inklusive** Zwischentensor-Verkehr
   * - Epilog ``bias``
     - zusätzlich :math:`\text{in} \cdot B \cdot M \cdot N` (der D-Read)
   * - TFLOP/s · GB/s
     - :math:`\text{FLOP}/(t \cdot 10^{-3})/10^{12}` bzw.
       :math:`\text{Byte}/(t \cdot 10^{-3})/10^{9}`
   * - Arithmetische Intensität
     - FLOP / Byte
   * - %-Peak
     - gegen ``hardware.PEAK_TFLOPS[dtype]`` bzw. 273 GB/s; ``None``, wo es keinen
       sinnvollen Nenner gibt (fp32/fp64 haben kein Tensor-Core-Dach)

.. admonition:: Eine Einschränkung dieses Berichts

   ``gemm_bytes`` ist der **algorithmische Mindest-Traffic**: Jeder Operand wird
   genau einmal gelesen. Real liest ein gekachelter Kernel Teile von A und B
   mehrfach (das ist ja der Sinn des L2-Cache und der Grund, warum Kachelung
   überhaupt wirkt). Die ausgewiesenen „erreichten GB/s" sind daher eine
   **Untergrenze** des tatsächlichen DRAM-Verkehrs, und „% Peak-Bandbreite" ist
   entsprechend konservativ. Das ist die übliche Roofline-Konvention (und die
   einzige, die ohne Hardware-Zähler auskommt), aber man muss es wissen:
   Zwei Kernel mit gleicher AI und unterschiedlichem Cache-Verhalten sehen in
   dieser Metrik gleich aus — der Unterschied zeigt sich nur in der Zeit. Genau
   deshalb ist der ``GROUP_M``-Befund in Teil 5 auch keine GB/s-Aussage, sondern
   eine Laufzeit-Aussage.

Eine angenehme Eigenschaft der Formeln: Bei einem batched GEMM skalieren FLOPs
**und** Bytes linear mit :math:`B` — die arithmetische Intensität ist damit
batch-**unabhängig**. Batched Punkte sitzen also an derselben Stelle der Roofline
wie ihre unbatched Variante, was physikalisch richtig ist.

Baselines und der Fusions-Vergleich
-----------------------------------

Beides sind **Zweitmessungen** innerhalb desselben ``run()``, mit derselben
Bench-Schleife (cold-L2) — deshalb direkt vergleichbar:

* **cuBLAS-Obergrenze:** ``torch.matmul`` auf denselben Operanden. Für ``tf32``
  wird ``allow_tf32`` temporär aktiviert (sonst verglichen wir tf32-Kernel gegen
  fp32-cuBLAS). Für fp8 gibt es keinen ``matmul``-Pfad — die Baseline meldet dann
  ehrlich ``available: false``.
* **naive cuTile-Untergrenze:** derselbe Codegen mit Tile 16×16×16 ohne Swizzle,
  also „cuTile ohne Tuning". Sie läuft über den regulären Pfad und bekommt
  ihren eigenen Slug — es ist wirklich derselbe Generator.
* **Sequentieller Fusions-Pfad:** die Plain-Kontraktion (``epilog=None``) plus ein
  separater Elementwise-Kernel, der den Zwischentensor liest. Der Vergleich ist
  damit keine Schätzung, sondern eine **gemessene und ebenfalls gegen fp32
  verifizierte** Alternative. Auch die gesparten Bytes werden nicht geschätzt:
  Der Zwischentensor-Roundtrip ist genau :math:`2 \cdot \text{out} \cdot B \cdot M
  \cdot N`.

Alle drei sind **graceful**: Schlägt eine fehl, verliert der Lauf nur den
Vergleich, nicht sein eigenes verifiziertes Ergebnis.

Provenienz
----------

Direkt **nach** der Messung wird der GPU-Zustand über ``nvidia-smi`` erfasst
(SM-Takt, Speichertakt, Temperatur, Leistung, Auslastung) — keine
Performance-Kennzahl, sondern Reproduzierbarkeits-Metadaten: Unter welchen
Bedingungen entstand diese Zahl? Bewusst ``nvidia-smi`` statt ``pynvml``, weil
Erstes auf dem Host garantiert vorhanden ist. Fehlt es, ist das Feld leer — nie ein
Fehler.

.. _gsc_report_store:

Stufe 8 — Store: JSON Lines und Lauf-Identität
==============================================

Zwei Artefakte, beide unter ``project/results/``:

* ``results.jsonl`` — **eine JSON-Zeile je Lauf**, das vollständige ``RunResult``
  (ohne ``kernel_source``, der liegt schon als Datei vor).
* ``kernels/<slug>.py`` — der generierte Quelltext.

.. list-table:: Warum JSON Lines?
   :header-rows: 1
   :widths: 30 70

   * - Alternative
     - Warum nicht
   * - SQLite
     - Für „eine Zeile je Lauf, gelegentlich alles lesen" ist eine Datenbank
       Overhead; sie wäre nicht git-diff-bar und nicht mit einem Texteditor
       lesbar.
   * - CSV
     - Die Ergebnisse sind **verschachtelt** (``metrics.fusion.speedup``) und das
       Schema wächst. CSV würde entweder flach-verkrüppeln oder brechen.
   * - Ein JSON-Array
     - Anhängen erforderte Lesen + Umschreiben der ganzen Datei; ein Absturz
       mitten im Schreiben zerstörte alles.
   * - **JSON Lines** (gewählt)
     - Anhängen ist ein einziger Schreibvorgang, jede Zeile ist unabhängig
       parsebar, ``pandas.read_json(lines=True)`` lädt es direkt, und ein
       ``git diff`` zeigt genau die neuen Läufe.

**Lauf-Identität.** Alle Läufe *eines* „Vergleichen"-Klicks (bzw. eines
CLI-Sweeps) teilen ``run_id``, ``run_name`` und ``created_at``. Damit ist ein
„Testlauf" eine benannte Einheit, die man in der Oberfläche wieder ansehen,
umbenennen oder löschen kann — und in diesem Bericht die Grundlage dafür, dass
„die Zahlen stammen aus **einer** Charge" überprüfbar ist. Altzeilen ohne diese
Felder bleiben lesbar; der Store synthetisiert dann einen Fallback-Lauf.
