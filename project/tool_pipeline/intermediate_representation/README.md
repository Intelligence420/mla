# `intermediate_representation/` — die Zwischendarstellung (kurz „IR")

Dieses Paket ist das **Frontend + die Mitte** der Tool-Pipeline: Es verwandelt
den rohen einsum-Ausdruck in eine strukturierte, verstandene Form und bringt
diese auf die kanonische GEMM-Gestalt, die der Codegen erwartet.

```
   Frontend                Mitte                     Backend
  parse → IR   →   IR umformen (reshape/B1)   →   codegen → Kernel
     └──────── dieses Paket (ir/) ────────┘         └─ codegen/
```

## Was heißt „Intermediate Representation"?

Zwischen der **Eingabe** (der Text `"ik,kj->ij"`, mit dem man nicht rechnen
kann) und der **Ausgabe** (dem fertigen cuTile-Kernel-Quelltext) liegt eine
**typisierte Datenstruktur**, die *versteht*, worum es geht: „das ist eine
Kontraktion, `i` sind die Zeilen (M), `j` die Spalten (N), über `k` wird
summiert, Größen 512/512/512". Das ist die IR (`ContractionIR`). Jede spätere
Stufe (reshape, codegen, verify) liest aus **dieser** klaren Struktur, statt den
String immer wieder neu zu zerpflücken — die klassische Compiler-Dreiteilung
Frontend → Mitte → Backend.

## Warum reshapen? (einfach)

Die **Tensor-Cores** (`ct.mma`) können nur **eine** Sache: zwei flache
2D-Matrizen multiplizieren (`Zeilen × Spalten`). einsum erlaubt aber viel mehr
(Batch, vertauschte Indizes, viele Dimensionen).

> **Bild:** Eine Maschine faltet nur flache A4-Blätter. Bringt jemand ein
> Notizbuch oder ein Banner, musst du es erst **in A4-Blätter bringen**. Die
> Maschine ändert sich nie — du passt die *Eingabe* an ihre Form an.

**Reshapen = die Daten so umordnen, dass jede Kontraktion aussieht wie eine
schlichte 2D-Matrixmultiplikation** (Zahlen bleiben unverändert, nur
umgruppiert/-sortiert): Batch-Index → Stapel; falsche Reihenfolge → permutieren;
mehrere Zeilen-Indizes → zu einem großen M verschmelzen. Danach hat **jede**
Kontraktion die kanonische Form `(B,M,K)×(B,K,N)→(B,M,N)` — und wir müssen nur
**einen** bewiesenen GEMM-Kernel schreiben und absichern (weniger Code, weniger
Stellen für den stillen mma-Orientierungsfehler).

## Dateien & Implementierungsstand

| Datei | Zweck | Stand |
|---|---|---|
| `parse.py` | einsum-Ausdruck → `ContractionIR` (M/N/K/Batch-Klassifikation, Größen, strenge Validierung) | ✅ **TZ 1** — 2 Operanden, expliziter Output, keine Diagonalen; `is_canonical_gemm()` erkennt den direkt emittierbaren Plain-GEMM |
| `reshape.py` | `ContractionIR` → `Canonical` (kanonische GEMM-Beschreibung, B1) | ✅ **TZ 1** — reiner **Passthrough** (Batch=1) für `ik,kj->ij`; alles, was echte Umformung bräuchte, wird mit TZ-6-Verweis abgelehnt |
| `config.py` | Config-IR (Port aus A05/06 `config.py`) | ⏳ Stub — **TZ 6** |
| `optimizer.py` | IR-Transformationen: split/fuse/permute (`split_dim` = Tile-Injektion), Port aus A05/06 | ⏳ Stub — **TZ 6** |
| `__init__.py` | Paket-Marker | — |

### Wie `parse.py` klassifiziert (A05/06-Schema)

Für einen 2-Operanden-Ausdruck `in0, in1 -> out`:

- **Batch (C)**: Index in *beiden* Operanden **und** im Output.
- **K (kontrahiert)**: Index in *beiden* Operanden, **nicht** im Output (summiert).
- **M**: Index in `in0` **und** Output, nicht in `in1`.
- **N**: Index in `in1` **und** Output, nicht in `in0`.

Beispiel `ik,kj->ij` ⇒ M=[i], N=[j], K=[k], Batch=[] ⇒ `is_canonical_gemm() == True`.

### Was TZ 1 bewusst **nicht** tut

`reshape.py` ist heute nur Passthrough — der echte, view-/stride-basierte
B1-Reshape (beliebige Kontraktion → kanonisches Batched-GEMM, zero-copy) kommt in
**TZ 6** zusammen mit `config.py`/`optimizer.py`. `parse.py` bleibt auf 2 Operanden
mit explizitem Output beschränkt (n-äres einsum, impliziter Output,
Familien-Routing = TZ 6/7). Die `M/N/K/Batch`-Klassifikation ist aber schon die
allgemeine, sodass TZ 6 nur darauf aufsetzt.
