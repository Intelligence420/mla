"""tool_pipeline.codegen.compile — generierten Quelltext ladbar machen + cachen.

`load_kernel(config)` verwandelt eine `RunConfig` in ein aufrufbares
`launch(*operanden)`-Objekt (Arity je Familie: GEMM/Elementwise-binär
`launch(A, B, C)`, Reduktion/Elementwise-unär `launch(A, C)`; letzter Operand =
Output):

  emit(config) -> Quelltext -> **Datei** results/kernels/<slug>.py -> importieren.

**Wichtig (empirisch verifiziert, TODO 4):** cuTile liest den Kernel-Quelltext
per ``inspect.getsourcelines`` und braucht dafür eine **echte Datei** auf der
Platte — ein reines ``exec(src)`` eines Strings scheitert mit
``OSError: could not get source code``. Deshalb wird der emittierte Text erst
nach ``results/kernels/<slug>.py`` **persistiert** und von dort importiert. Das
persistierte Artefakt ist zugleich Compile-Cache, UI-Code-Anzeige und
reproduzierbarer Beleg (Risiko ③).

**Cache (zweistufig):**
  * *In-Memory* (`_MODULE_CACHE`, Slug → `launch`): vermeidet erneutes Importieren
    innerhalb eines Prozesses (Live-UI, wiederholte Läufe).
  * *Auf Platte* (`results/kernels/<slug>.py`): das Quelltext-Artefakt; wird nur
    neu geschrieben, wenn es fehlt oder der Inhalt abweicht (idempotent).

Der eigentliche cuTile-JIT (mehrere hundert ms) passiert **lazy beim ersten
`ct.launch`** — dessen Zeit misst die Mess-Schicht (`measure/bench.py`) als
kalten-vs-warmen Lauf (Compile-vs-Run-Split), nicht hier.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from ..schema import RunConfig
from ..store.store import config_slug, kernel_file, save_kernel
from .emit import emit

# Slug -> kompiliertes launch(A,B,C)-Callable (prozess-lokal).
_MODULE_CACHE: dict[str, Callable] = {}


@dataclass
class CompileResult:
    """Ergebnis von `load_kernel`."""

    launch: Callable        # launch(*operanden) -> Output (Arity je Familie)
    kernel_path: str        # Pfad des persistierten Quelltexts
    slug: str
    cached: bool            # True = aus dem In-Memory-Cache (kein Import nötig)


def _safe_module_name(slug: str) -> str:
    """Slug → gültiger, kollisionsfreier Python-Modulname (für sys.modules)."""
    return "tp_gen_" + re.sub(r"\W", "_", slug)


def _read_text_or_none(path) -> Optional[str]:
    """Bestehende Kernel-Datei lesen; bei fehlender, korrupter oder nicht
    dekodierbarer Datei ``None`` (⇒ der Aufrufer schreibt sie neu). Härtet den
    Compile-Cache gegen halb geschriebene/beschädigte ``<slug>.py`` (Risiko ⑥):
    ein ``UnicodeDecodeError`` oder ``OSError`` beim Lesen darf keinen Lauf
    crashen, sondern muss zum sauberen Neu-Schreiben führen."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _import_launch(path) -> Callable:
    """Modul aus einer echten Datei importieren und `launch` herausgeben."""
    spec = importlib.util.spec_from_file_location(_safe_module_name(path.stem), str(path))
    module = importlib.util.module_from_spec(spec)
    # results/kernels/ ist ein Daten-Verzeichnis (git-getrackte <slug>.py) → dort
    # kein __pycache__ ablegen. cuTile braucht ohnehin den Quelltext, nicht die .pyc.
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)   # definiert nur Kernel + launch (kein JIT hier)
    finally:
        sys.dont_write_bytecode = prev
    launch = getattr(module, "launch", None)
    if launch is None:
        raise AttributeError(
            f"generierter Modul {path} definiert kein launch(*operanden) "
            f"(Consumer-Konvention verletzt)."
        )
    return launch


def load_kernel(config: RunConfig, source: Optional[str] = None) -> CompileResult:
    """`RunConfig` → aufrufbares `launch(*operanden)` (persistiert + gecacht).

    :param config: der Lauf; bestimmt den Slug (= Cache-Schlüssel + Dateiname).
    :param source: optional bereits emittierter Quelltext; sonst via `emit(config)`.
    """
    slug = config_slug(config)
    path = kernel_file(slug)

    # 1) In-Memory-Cache-Treffer → wiederverwenden, nicht neu importieren.
    if slug in _MODULE_CACHE:
        return CompileResult(_MODULE_CACHE[slug], str(path), slug, cached=True)

    # 2) Quelltext beschaffen + idempotent persistieren. Bei fehlender ODER
    #    korrupter/nicht dekodierbarer Datei (`_read_text_or_none` → None) sowie bei
    #    abweichendem Inhalt wird atomar (save_kernel) neu geschrieben — der Cache
    #    heilt sich so gegen beschädigte Artefakte, statt einen Lauf zu crashen.
    if source is None:
        source = emit(config)
    if _read_text_or_none(path) != source:
        save_kernel(source, slug)

    # 3) Aus der Datei importieren + cachen.
    launch = _import_launch(path)
    _MODULE_CACHE[slug] = launch
    return CompileResult(launch, str(path), slug, cached=False)


def clear_cache() -> None:
    """In-Memory-Cache leeren (v. a. für Tests)."""
    _MODULE_CACHE.clear()
