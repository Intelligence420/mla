############################
Installation und Benutzung
############################

Voraussetzungen
============================

Das Projekt wurde für Linux (Ubuntu/WSL) entwickelt und teilt sich in zwei
Hardware-Hälften mit unterschiedlichen Anforderungen:

* **Allgemein:** Python 3 und PyTorch.
* **GPU-Assignments (01–06):** eine NVIDIA-GPU mit CUDA sowie cuTile
  (``cuda.tile``). Diese Assignments laufen auf dem Uni-GPU-Rechner.
* **NPU-Assignments (07–10):** eine AMD-XDNA2-NPU und die MLIR-AIE-Toolchain
  (u. a. ``PEANO_INSTALL_DIR``, ``aiecc.py``, ``pyxrt``). Diese Assignments
  laufen auf dem NPU-Host.
* **Group-Specific Component (``project/``):** dieselbe CUDA-/cuTile-Umgebung wie
  die GPU-Assignments, zusätzlich die GUI-/Plot-Pakete aus
  ``project/requirements.txt``.
* **Dokumentation:** Sphinx mit dem Read-the-Docs-Theme.

Setup
============================

Für die Python-Teile empfiehlt sich eine virtuelle Umgebung:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

Anschließend werden die benötigten Pakete installiert – im Wesentlichen
PyTorch (mit CUDA für den GPU-Teil) sowie Sphinx und das Read-the-Docs-Theme
für die Dokumentation. Die MLIR-AIE-Toolchain für die NPU-Assignments wird
separat auf dem NPU-Host bereitgestellt.

Repo-Aufbau
============================

Der grobe Aufbau des Repositories:

.. code-block:: text

   assignments/NN_assignment/   # Code je Assignment: src/ + Aufgabenstellung
                                #   (NPU-Assignments zusätzlich mit Makefile)
   project/                     # Group-Specific Component: cuTile Performance Lab
   sphinx/                      # Diese Dokumentation (Quellen in source/)
   slides/                      # Vorlesungsfolien (PDF)

Jedes Assignment ``NN`` hat ein Gegenstück unter
``sphinx/source/chapters/NN_*/`` mit Aufgabenstellung (``aufgabe.rst``) und
Lösungsbeschreibung (``loesung.rst``).

GPU-Assignments ausführen (01–06)
============================================

Die GPU-Assignments werden mit aktivierter venv aus dem jeweiligen
Assignment-Ordner heraus gestartet (Beispiel Assignment 01):

.. code-block:: bash

   source .venv/bin/activate
   cd assignments/01_assignment
   python3 src/assignment_01.py

Die Skripte prüfen ihre Ergebnisse gegen eine PyTorch-Referenz und legen –
sofern vorhanden – Plots als PNG neben den Quellcode.

Group-Specific Component ausführen (``project/``)
==================================================

Das **cuTile Performance Lab** (siehe :ref:`gsc_report`) liegt unter ``project/``
und läuft auf dem GPU-Rechner. Es braucht dieselbe Umgebung wie die
GPU-Assignments (``torch`` mit CUDA, ``cuda.tile``, ``triton``) und zusätzlich die
GUI-/Plot-Pakete:

.. code-block:: bash

   source .venv/bin/activate                    # venv mit torch/cuda.tile/triton
   pip install -r project/requirements.txt      # Dash, Plotly, pandas, matplotlib …

Alle weiteren Aufrufe erfolgen **aus dem Ordner** ``project/``:

.. code-block:: bash

   cd project

   python -m tool_pipeline                 # (1) Dash-GUI, dann http://127.0.0.1:8050
   python -m tool_pipeline.cli --sweep     # (2) headless: Report-Sweep (GPU, ~2 min)
   python -m tool_pipeline.report_figures  # (3) Report-Figuren (torch-frei, ohne GUI)
   python -m pytest tests/ -q              # (4) Testsuite

(1) startet die Oberfläche: einsum-Ausdruck, Zahlenformat, Kachelung und Swizzle
wählen, „Run" drücken — das Tool generiert den cuTile-Kernel, verifiziert ihn gegen
fp32, misst ihn und zeigt Charts plus den generierten Quelltext.

(2) fährt ohne GUI den kuratierten Satz Konfigurationen, aus dem der Report seine
Zahlen zieht, und hängt jeden verifizierten Lauf an ``project/results/results.jsonl``
an. Ein Einzellauf geht ebenso, z. B.
``python -m tool_pipeline.cli --family reduction --expr ij->i --size 4096``;
``--show-configs`` listet den Sweep, ohne die GPU anzufassen. (3) erzeugt daraus die
PNGs unter ``sphinx/source/_static/gsc/`` und braucht **keine** GPU.

Die Testsuite (4) teilt sich in headless-Tests (laufen überall) und Codegen-/Messtests,
die eine **CUDA-GPU voraussetzen** — sie compilieren die generierten Kernel wirklich.

NPU-Assignments ausführen (07–10)
============================================

Die NPU-Assignments benötigen die MLIR-AIE-Umgebung und werden auf dem
NPU-Host gebaut und ausgeführt. Der Ablauf ist je Assignment über ein
``Makefile`` gekapselt (Beispiel Assignment 10):

.. code-block:: bash

   cd assignments/10_assignment
   make run_matmul       # baut die Kernel + xclbin und startet den Treiber
   make clean

Das ``Makefile`` übersetzt die Kernel mit der Peano-Toolchain, erzeugt aus der
``.mlir``-Beschreibung eine ``.xclbin`` und lädt diese im Python-Treiber über
``pyxrt``; verifiziert wird gegen eine PyTorch-CPU-Referenz.

Dokumentation generieren
============================

Die Sphinx-Dokumentation (dieser Report) wird gebaut mit:

.. code-block:: bash

   cd sphinx
   make html

Das Ergebnis liegt anschließend unter ``sphinx/build/html/index.html``. Die
Seite wird zusätzlich über GitHub Pages veröffentlicht.
