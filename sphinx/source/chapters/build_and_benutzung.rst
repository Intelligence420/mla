#################
Build & Benutzung
#################

Voraussetzungen
---------------

Die Software wurde für Linux (Ubuntu/WSL) entwickelt. Folgende Pakete werden benötigt:

* Python 3
* PyTorch (auf dem Uni-Rechner mit CUDA, lokal optional CPU-only)
* Sphinx + ``sphinx_rtd_theme`` (für diese Dokumentation)

Eine ausführliche Installationsanleitung befindet sich in der Datei ``SETUP.md`` im Projektroot.

Setup (Virtual Environment)
---------------------------

.. code-block:: bash

   # venv erstellen
   python3 -m venv .venv
   source .venv/bin/activate

   # Abhängigkeiten installieren
   pip install -r requirements.txt

Alternativ können Sphinx und das Theme auch global installiert werden (siehe ``SETUP.md``).

Assignment ausführen
--------------------

.. code-block:: bash

   # venv aktivieren (falls nicht global installiert)
   source .venv/bin/activate

   # Beispiel: Assignment 01
   cd assignments/01_assignment
   python3 src/assignment_01.py

Dokumentation generieren
------------------------

.. code-block:: bash

   # Sphinx (diese Dokumentation)
   cd sphinx && make html

Die generierte Dokumentation befindet sich anschließend unter ``sphinx/build/html/index.html``.
