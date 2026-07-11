"""tool_pipeline.schema — RunConfig (Eingabe) und RunResult (Ausgabe).

Dies ist der **Vertrag zwischen Core und GUI** (zuerst definieren, T0.2). Die
einzige Naht `run(config) -> result` (siehe `run.py`) spricht ausschließlich
diese beiden Typen; `app/` baut `RunConfig`, der Core liefert `RunResult`.

Bewusst als reine Daten-Container (`@dataclass`) gehalten: **keine** Logik für
das dtype→torch/cuTile-*Mapping*, Parsing oder Messung — das gehört in die
jeweiligen Stufen. Die dtype/acc-*Regeln* (welche Format-Kombis überhaupt
zulässig sind) sind dagegen Vertrags-Daten und liegen hier (`ALLOWED_ACC`),
torch-/cuTile-frei, damit die fork-sichere GUI sie importieren kann.
So bleiben spätere Achsen rein **additiv**: neuer dtype/Tile/Swizzle/Familie/
Baseline = neues Feld oder neuer Status-Zweig, kein Umbau.

TZ 1 nutzt davon nur: `family="contraction"`, `expr="ik,kj->ij"`, `dtype="fp16"`,
`acc_dtype="fp32"`, feste `tile`, `swizzle=False`, `baselines=[]`. Die übrigen
Felder sind für spätere Teil-Ziele schon vorgesehen, werden in TZ 1 aber nur
durchgereicht/geechot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Status-Konstanten für RunResult.status (str-Werte, damit JSON-trivial).
# ---------------------------------------------------------------------------
STATUS_OK = "ok"                       # compiliert, verifiziert, gemessen
STATUS_VERIFY_FAILED = "verify_failed"  # läuft, aber Zahlen != fp32-Referenz
STATUS_COMPILE_ERROR = "compile_error"  # emittierter Quelltext compiliert nicht
STATUS_RUN_ERROR = "run_error"          # compiliert, crasht aber beim Launch/Run


# ---------------------------------------------------------------------------
# dtype/acc-Regeln (Teil des Core↔GUI-Vertrags, TZ 3).
# ---------------------------------------------------------------------------
# Erlaubte Akkumulator-dtypes je Compute-dtype (PLAN §5, empirisch belegt):
# bf16 & tf32 MÜSSEN in fp32 akkumulieren; fp16 & fp8 dürfen fp16 oder fp32;
# fp32 (Anker) nur fp32. **Single Source of Truth der Acc-Regeln** — gelesen von
# `run._build_inputs` (frühe Prüfung, Stufe 2) und den GUI-Controls; `measure.verify`
# hält für exakt dieselben (dtype, acc)-Kombis die Toleranzen (zweite
# Verteidigungslinie). Bewusst torch-/cuTile-frei (fork-sichere GUI kann es laden).
ALLOWED_ACC: dict[str, tuple[str, ...]] = {
    "fp16":    ("fp16", "fp32"),
    "bf16":    ("fp32",),
    "tf32":    ("fp32",),
    "fp8e4m3": ("fp16", "fp32"),
    "fp8e5m2": ("fp16", "fp32"),
    "fp32":    ("fp32",),
}


def check_dtype_combo(dtype: str, acc_dtype: str) -> Optional[str]:
    """Prüft (dtype, acc_dtype) gegen die Acc-Regeln (`ALLOWED_ACC`).

    :returns: ``None`` wenn zulässig, sonst ein deutscher Fehlertext (für einen
              sauberen Fehler-Status statt eines still falschen Laufs).
    """
    allowed = ALLOWED_ACC.get(dtype)
    if allowed is None:
        return (f"Compute-dtype {dtype!r} nicht unterstützt "
                f"(verfügbar: {sorted(ALLOWED_ACC)}).")
    if acc_dtype not in allowed:
        return (f"Akkumulator {acc_dtype!r} für Compute-dtype {dtype!r} "
                f"unzulässig (erlaubt: {list(allowed)}).")
    return None


# ---------------------------------------------------------------------------
# RunConfig — alles, was einen Lauf vollständig beschreibt (= Cache-Eingabe).
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Vollständige Beschreibung eines Laufs (Eingabe von `run()`).

    `expr` ist die Single Source of Truth; `inputs`/`output` werden in
    `__post_init__` daraus abgeleitet, falls nicht explizit gesetzt (reine
    String-Zerlegung — die eigentliche Dim-Klassifikation macht `ir/parse.py`).
    """

    # --- Operation ---
    family: str = "contraction"        # "contraction" | (später) "elementwise"/"reduction"
    expr: str = "ik,kj->ij"            # einsum-Ausdruck; treibt inputs/output
    inputs: Optional[list[str]] = None  # z. B. ["ik", "kj"]; aus expr abgeleitet
    output: Optional[str] = None        # z. B. "ij"; aus expr abgeleitet
    dim_sizes: dict[str, int] = field(
        default_factory=lambda: {"i": 512, "k": 512, "j": 512}
    )

    # --- Zahlenformat (TZ 1: nur fp16 -> fp32) ---
    dtype: str = "fp16"                 # Compute-dtype der Operanden
    acc_dtype: str = "fp32"            # Akkumulator-dtype

    # --- Kachelung / Layout (TZ 1: fest, kein Swizzle) ---
    tile: dict[str, int] = field(
        default_factory=lambda: {"TM": 128, "TN": 128, "TK": 64}
    )
    swizzle: bool = False

    # --- Vergleichsbaselines (TZ 1: leer; cuBLAS/naive kommen in TZ 4) ---
    baselines: list[str] = field(default_factory=list)

    # --- Messung (Benchmark-Iterationen) ---
    # bench_iters = getaktete warme Läufe (→ Verteilung Median/min/p90/σ),
    # bench_warmup = ungetaktete Aufwärm-Läufe (stabilisieren Takt/Caches).
    # Defaults = die bewährten bench.py-Werte. Sie bestimmen NICHT den Kernel-Slug
    # (reiner Messaufwand) → additiv ohne Cache-/Dateinamen-Drift.
    bench_warmup: int = 10
    bench_iters: int = 30

    def __post_init__(self) -> None:
        # inputs/output konsistent aus expr ableiten, wenn nicht vorgegeben.
        if self.inputs is None or self.output is None:
            lhs, _, rhs = self.expr.partition("->")
            if self.inputs is None:
                self.inputs = [s.strip() for s in lhs.split(",") if s.strip()]
            if self.output is None:
                self.output = rhs.strip()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisierbares dict (für Store/Echo/Hash)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        """Aus einem dict rekonstruieren (unbekannte Felder werden ignoriert)."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# RunResult — alles, was ein Lauf zurückgibt (= eine Zeile in results.jsonl).
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Ergebnis eines Laufs (Rückgabe von `run()`).

    `config` echot die Eingabe; die übrigen Felder sind nach Belang gruppiert
    (`accuracy`/`timing`/`metrics`/`provenance`), damit spätere Teil-Ziele dort
    nur Schlüssel ergänzen (z. B. GB/s, %-Peak, Verteilung) statt das Schema
    umzubauen. Bei Fehlern trägt `status` den Grund und `error` den Text.

    `kernel_source` trägt den generierten Quelltext für die GUI-Code-Anzeige (TZ 2).
    Er wird vom Store bewusst **nicht** ins `results.jsonl` geschrieben (Bloat) —
    der Text liegt bereits als `kernels/<slug>.py` vor (`kernel_path`).
    """

    status: str                         # eine der STATUS_*-Konstanten
    config: dict[str, Any]              # Echo der RunConfig (als dict)
    kernel_path: Optional[str] = None   # Pfad des persistierten Kernel-Quelltexts
    kernel_source: Optional[str] = None  # generierter Quelltext (GUI-Code-Panel; NICHT im JSONL)
    accuracy: dict[str, Any] = field(default_factory=dict)   # max_abs_err, passed, ...
    timing: dict[str, Any] = field(default_factory=dict)     # compile_ms, run_ms, ...
    metrics: dict[str, Any] = field(default_factory=dict)    # tflops, ... (TZ 4: GB/s, %-Peak)
    provenance: dict[str, Any] = field(default_factory=dict)  # gpu, dtype, sizes, timestamp
    error: Optional[str] = None         # Fehlertext, falls status != ok

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisierbares dict (eine Zeile in results.jsonl)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
