"""tool_pipeline.measure.provenance — GPU-Zustand für Reproduzierbarkeit.

Erfasst Takt/Temperatur/Leistung/Auslastung der GPU via `nvidia-smi` — reine
**Reproduzierbarkeits-Metadaten** (keine Performance-Kennzahlen), damit jede
`results.jsonl`-Zeile selbst-beschreibend ist: unter welchem GPU-Zustand wurde
gemessen? Bewusst über `nvidia-smi` statt `pynvml`: nvidia-smi ist auf dem Host
vorhanden, `pynvml` ist nicht gepinnt (PLAN §8). Fehlt nvidia-smi oder scheitert
der Aufruf/das Parsen, wird ein **leeres dict** geliefert (graceful) — nie ein
Fehler, der den Lauf kippt. Torch-/cuTile-frei (headless testbar).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Callable

# nvidia-smi-Abfragefeld → (Ausgabe-Key, Typ-Wandler).
_QUERY_FIELDS: tuple[tuple[str, str, Callable], ...] = (
    ("clocks.sm", "sm_clock_mhz", float),
    ("clocks.mem", "mem_clock_mhz", float),
    ("temperature.gpu", "temp_c", float),
    ("power.draw", "power_w", float),
    ("utilization.gpu", "util_pct", float),
)


def gpu_state(index: int = 0, timeout_s: float = 2.0) -> dict[str, Any]:
    """GPU-Zustand (Takt/Temp/Power/Auslastung) via nvidia-smi.

    :param index:     GPU-Index (`--id`).
    :param timeout_s: harter Timeout für den nvidia-smi-Aufruf.
    :returns:         dict der geparsten Felder; **leer**, wenn nvidia-smi fehlt,
                      der Aufruf scheitert oder die Ausgabe unerwartet ist. Einzelne
                      nicht-numerische Felder (z. B. ``[N/A]``) werden ``None``.
    """
    if shutil.which("nvidia-smi") is None:
        return {}
    query = ",".join(f for f, _, _ in _QUERY_FIELDS)
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits", f"--id={index}"],
            capture_output=True, text=True, timeout=timeout_s, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001  (jeder Fehler → graceful leeres dict)
        return {}
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) != len(_QUERY_FIELDS):
        return {}
    state: dict[str, Any] = {}
    for (_raw, key, cast), val in zip(_QUERY_FIELDS, parts):
        try:
            state[key] = cast(val)
        except (ValueError, TypeError):
            state[key] = None      # z. B. "[N/A]" bei manchen Feldern/GPUs
    return state
