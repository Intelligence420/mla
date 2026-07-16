"""Headless-Tests der Dash-freien Controls-Logik (TZ 6: allgemeiner Ausdruck).

Die GUI ist schwer headless zu prüfen — deshalb ist die entscheidende Logik
(Ausdruck→Indizes/Klassifikation, Größen→RunConfig, Validierung) bewusst aus dem
Callback herausgezogen und hier ohne Dash-Server + ohne GPU geprüft. Der Rest der
GUI (Layout-Mount, Live-Callback) wird durch das reale Starten der App abgedeckt.

Lauffähig standalone (`python tests/test_app_controls.py`, aus `project/`) **und**
via pytest. Braucht nur `dash`/`schema`/`parse`, KEIN torch/cuTile/GPU.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pipeline.app.components.controls import (  # noqa: E402
    COMBOS,
    FAMILIES,
    FAMILY_PRESETS,
    ID_BASELINE_INFO,
    ID_BASELINES,
    ID_DTYPE_INFO,
    ID_EXPR,
    ID_FAMILY,
    ID_INDEX_SIZES,
    ID_OP,
    ID_PRESET,
    ID_SWIZZLE,
    ID_SWIZZLE_CONFIGS,
    ID_TILE_ADD,
    ID_TILE_INFO,
    ID_TILE_ROWS,
    ID_TILE_TK,
    ID_TILE_TM,
    ID_TILE_TN,
    INDEX_SIZE_TYPE,
    TILE_TM_TYPE,
    PRESETS,
    _DEFAULT_EXPR,
    _DEFAULT_SELECTION,
    _DTYPE_ORDER,
    build_controls,
    combo_key,
    combo_label,
    combos_for_family,
    config_from_controls,
    configs_from_selection,
    default_selection_for_family,
    default_tile_row,
    dim_sizes_from_state,
    expr_indices,
    group_m_from_controls,
    index_categories,
    index_size_inputs,
    mutate_tile_rows,
    parse_combo,
    parse_preset_value,
    preset_options,
    preset_value,
    resolve_expr,
    swizzle_configs_from_state,
    swizzles_from_value,
    tile_from_controls,
    tile_rows,
    tiles_from_state,
    validate_baselines,
    validate_dim_sizes,
    validate_expr,
    validate_group_m,
    validate_selection,
    validate_swizzle,
    validate_swizzle_configs,
    validate_tile,
    validate_tiles,
)
from tool_pipeline.schema import ALLOWED_ACC, RunConfig, check_dtype_combo  # noqa: E402


# --- Ausdruck: Indizes, Auflösung, Klassifikation ----------------------------
def test_expr_indices_in_occurrence_order():
    assert expr_indices("ik,kj->ij") == ["i", "k", "j"]
    assert expr_indices("acspx,bspy->abcyx") == ["a", "c", "s", "p", "x", "b", "y"]
    assert expr_indices("bik,bkj") == ["b", "i", "k", "j"]   # ohne '->' (impliziter Output)


def test_resolve_expr_adds_implicit_output():
    assert resolve_expr("ik,kj->ij") == "ik,kj->ij"          # explizit unverändert
    assert resolve_expr("ik,kj") == "ik,kj->ij"              # impliziter Output ergänzt
    assert resolve_expr("bik,bkj") == "bik,bkj->ij"          # b kontrahiert (einsum-Konvention)


def test_index_categories():
    """Jeder Index bekommt seine Kategorie (M/N/K/Batch) — die Klassifikation sichtbar."""
    cats = index_categories("bik,bkj->bij")
    assert cats == {"b": "Batch", "i": "M", "k": "K", "j": "N"}, cats
    cats2 = index_categories("acspx,bspy->abcyx")
    assert cats2["s"] == "K" and cats2["p"] == "K"
    assert cats2["a"] == cats2["c"] == cats2["x"] == "M"
    assert cats2["b"] == cats2["y"] == "N"


def test_validate_expr():
    """Gültige Ausdrücke → None; nicht zerlegbares n-är/Diagonale/leer → Fehlertext."""
    assert validate_expr("ik,kj->ij") is None
    assert validate_expr("bik,bkj->bij") is None
    assert validate_expr("ik,kj") is None                    # impliziter Output ok
    assert validate_expr("") is not None
    assert validate_expr("ij,jk,kl->il") is None             # n-är-Kette jetzt unterstützt (TZ 7.5-3)
    assert validate_expr("abc,bca,cba->abc") is not None     # n-äres Hadamard (kein GEMM-K) → abgelehnt
    assert validate_expr("ii,ij->ij") is not None            # Diagonale


# --- Größen je Index ---------------------------------------------------------
def test_dim_sizes_from_state():
    ids = [{"type": INDEX_SIZE_TYPE, "index": "i"}, {"type": INDEX_SIZE_TYPE, "index": "k"}]
    vals = [256, 64]
    assert dim_sizes_from_state(ids, vals) == {"i": 256, "k": 64}
    # None/leere Werte bleiben als Roh-Werte erhalten (Validierung fängt sie später).
    assert dim_sizes_from_state([{"index": "i"}], [None]) == {"i": None}


def test_validate_dim_sizes_ok_and_errors():
    assert validate_dim_sizes("ik,kj->ij", {"i": 128, "k": 64, "j": 128}) is None
    assert validate_dim_sizes("ik,kj->ij", {"i": 128, "k": 64}) is not None       # j fehlt
    assert validate_dim_sizes("ik,kj->ij", {"i": 128, "k": 64, "j": 0}) is not None  # < 1
    assert validate_dim_sizes("ik,kj->ij", {"i": 128, "k": 64, "j": 12.5}) is not None  # nicht ganz
    assert validate_dim_sizes("ik,kj->ij", {"i": 128, "k": 64, "j": "x"}) is not None   # nicht Zahl


def test_validate_dim_sizes_memory_guard():
    """Zu große Größen → Fehler (OOM-Schutz), statt die geteilte Maschine zu killen."""
    msg = validate_dim_sizes("ik,kj->ij", {"i": 50000, "k": 50000, "j": 50000})
    assert msg is not None and "GiB" in msg, msg


# --- RunConfig-Bau -----------------------------------------------------------
def test_config_from_controls():
    """Ausdruck + Größen → RunConfig (Ausdruck normalisiert, dim_sizes gesetzt)."""
    cfg = config_from_controls("bik,bkj->bij", {"b": 2, "i": 64, "k": 32, "j": 48})
    assert cfg.expr == "bik,bkj->bij"
    assert cfg.dim_sizes == {"b": 2, "i": 64, "k": 32, "j": 48}
    assert cfg.inputs == ["bik", "bkj"] and cfg.output == "bij"


def test_config_from_controls_resolves_implicit_output():
    """Impliziter Ausdruck → RunConfig mit expliziter Form (sauberer Slug/Echo)."""
    cfg = config_from_controls("ik,kj", {"i": 8, "k": 4, "j": 6})
    assert cfg.expr == "ik,kj->ij"


def test_config_tolerates_float_and_string_sizes():
    cfg = config_from_controls("ik,kj->ij", {"i": 512.0, "k": "128", "j": 256.0})
    assert cfg.dim_sizes == {"i": 512, "k": 128, "j": 256}


# --- Format-Auswahl → RunConfigs ---------------------------------------------
def test_configs_from_selection_canonical_order():
    """Auswahl → eine RunConfig je Kombi in kanonischer COMBOS-Reihenfolge."""
    sel = [combo_key("fp8e4m3", "fp16"), combo_key("fp16", "fp32")]  # 'verkehrt' geklickt
    cfgs = configs_from_selection("ik,kj->ij", {"i": 256, "k": 64, "j": 128}, sel)
    assert [(c.dtype, c.acc_dtype) for c in cfgs] == [("fp16", "fp32"), ("fp8e4m3", "fp16")]
    assert cfgs[0].dim_sizes == {"i": 256, "k": 64, "j": 128}
    assert cfgs[0].expr == "ik,kj->ij"
    assert cfgs[0].dim_sizes is not cfgs[1].dim_sizes


def test_configs_from_selection_batched_expr():
    """Batched-Ausdruck fließt in jede RunConfig (Kontraktions-Familie in der UI)."""
    cfgs = configs_from_selection("bik,bkj->bij", {"b": 4, "i": 128, "k": 128, "j": 128},
                                  [combo_key("fp16", "fp32")])
    assert len(cfgs) == 1 and cfgs[0].expr == "bik,bkj->bij"
    assert cfgs[0].dim_sizes == {"b": 4, "i": 128, "k": 128, "j": 128}


def test_configs_from_selection_fills_tile_swizzle_baselines():
    sel = [combo_key("fp16", "fp32"), combo_key("bf16", "fp32")]
    cfgs = configs_from_selection("ik,kj->ij", {"i": 128, "k": 64, "j": 128}, sel,
                                  tiles=[{"TM": 64, "TN": 64, "TK": 32}],
                                  swizzle_configs=[(True, 8)], baselines=["cublas", "naive"])
    assert len(cfgs) == 2
    for c in cfgs:
        assert c.tile == {"TM": 64, "TN": 64, "TK": 32}
        assert c.swizzle is True and c.baselines == ["cublas", "naive"]
    assert cfgs[0].tile is not cfgs[1].tile and cfgs[0].baselines is not cfgs[1].baselines


def test_configs_from_selection_default_tile_when_none():
    c = configs_from_selection("ik,kj->ij", {"i": 128, "k": 64, "j": 128},
                               [combo_key("fp16", "fp32")])[0]
    assert c.tile == RunConfig().tile
    assert c.swizzle is False and c.baselines == []


def test_configs_from_selection_swizzle_both_expands():
    cfgs = configs_from_selection("ik,kj->ij", {"i": 128, "k": 64, "j": 128},
                                  [combo_key("fp16", "fp32")],
                                  swizzle_configs=[(False, 8), (True, 8)], baselines=["cublas"])
    assert len(cfgs) == 2
    assert [c.swizzle for c in cfgs] == [False, True]
    assert cfgs[0].baselines == ["cublas"] and cfgs[1].baselines == []


# --- Format/Tile/Swizzle/Baseline (unverändert aus TZ 3/4) -------------------
def test_combos_derive_from_schema_rules():
    expected = {(d, a) for d in _DTYPE_ORDER for a in ALLOWED_ACC[d]}
    assert set(COMBOS) == expected, set(COMBOS) ^ expected
    assert ("fp32", "fp32") not in COMBOS


def test_all_combos_are_acc_rule_valid():
    for (d, a) in COMBOS:
        assert check_dtype_combo(d, a) is None, (d, a)


def test_combo_key_roundtrip():
    for (d, a) in COMBOS:
        assert parse_combo(combo_key(d, a)) == (d, a)
    assert combo_label("fp8e4m3", "fp16") == "fp8e4m3 → fp16"


def test_validate_selection():
    assert validate_selection([]) is not None
    assert validate_selection(None) is not None
    assert validate_selection(["fp16:fp32", "bogus:xx"]) is not None
    assert validate_selection([combo_key("bf16", "fp32")]) is None


def test_default_selection_is_valid():
    assert validate_selection(_DEFAULT_SELECTION) is None


def test_validate_tile_accepts_and_rejects():
    assert validate_tile(128, 128, 64) is None
    assert validate_tile("64", "256", "16") is None
    assert validate_tile(48, 128, 64) is not None
    assert validate_tile(128, 128, 256) is not None
    assert validate_tile(128, 128, None) is not None
    assert validate_tile(128, 128, "x") is not None


def test_validate_group_m_accepts_and_rejects():
    """GROUP_M (TZ 7.5): akzeptiert die zulässigen Zweierpotenzen (auch als Strings),
    lehnt Nicht-Werte/Fehlendes ab."""
    for v in (1, 2, 4, 8, 16, 32):
        assert validate_group_m(v) is None and validate_group_m(str(v)) is None
    assert validate_group_m(5) is not None      # keine Option
    assert validate_group_m(64) is not None      # außerhalb
    assert validate_group_m(None) is not None
    assert validate_group_m("") is not None
    assert validate_group_m("x") is not None
    assert group_m_from_controls("16") == 16


def test_configs_from_selection_group_m():
    """group_m wird in jede RunConfig durchgereicht; der Slug trägt es NUR bei
    swizzle & group_m!=8 (bedingt, byte-identisch sonst)."""
    from tool_pipeline.store.store import config_slug
    sel = [combo_key("fp16", "fp32")]
    ds = {"i": 128, "k": 64, "j": 128}
    base = "ik_kj_to_ij__fp16-fp32__TM128_TN128_TK64"
    # swizzle an, GROUP_M=16 → group_m gesetzt + Slug __sw_g16
    c16 = configs_from_selection("ik,kj->ij", ds, sel, swizzle_configs=[(True, 16)])[0]
    assert c16.group_m == 16 and config_slug(c16) == base + "__sw_g16"
    # swizzle an, Default 8 → bares __sw
    c8 = configs_from_selection("ik,kj->ij", ds, sel, swizzle_configs=[(True, 8)])[0]
    assert c8.group_m == 8 and config_slug(c8) == base + "__sw"
    # swizzle aus, GROUP_M=16 → gesetzt aber wirkungslos (kein Suffix)
    coff = configs_from_selection("ik,kj->ij", ds, sel, swizzle_configs=[(False, 16)])[0]
    assert coff.group_m == 16 and config_slug(coff) == base
    # gemischt → False-Config sauber, True-Config mit GROUP_M
    both = configs_from_selection("ik,kj->ij", ds, sel, swizzle_configs=[(False, 8), (True, 16)])
    assert config_slug(both[0]) == base and config_slug(both[1]) == base + "__sw_g16"


def test_validate_baselines():
    assert validate_baselines([]) is None
    assert validate_baselines(None) is None
    assert validate_baselines(["cublas"]) is None
    assert validate_baselines(["cublas", "naive"]) is None
    assert validate_baselines(["bogus"]) is not None


def test_tile_from_controls_coerces():
    assert tile_from_controls("128", "64", "32") == {"TM": 128, "TN": 64, "TK": 32}
    assert tile_from_controls(256.0, 128.0, 16.0) == {"TM": 256, "TN": 128, "TK": 16}


def test_swizzles_from_value():
    assert swizzles_from_value("off") == [False]
    assert swizzles_from_value("on") == [True]
    assert swizzles_from_value("both") == [False, True]
    assert swizzles_from_value(True) == [True]
    assert swizzles_from_value([False, True]) == [False, True]


def test_validate_swizzle():
    for v in ("off", "on", "both", True, False):
        assert validate_swizzle(v) is None
    assert validate_swizzle("bogus") is not None


# --- Presets + Komponentenbaum -----------------------------------------------
def test_presets_are_valid_expressions():
    """Jedes Preset ist ein strukturell gültiger Ausdruck (sonst wäre der Knopf eine Falle)."""
    assert PRESETS, "keine Presets definiert"
    for label, expr in PRESETS:
        assert isinstance(label, str) and label
        assert validate_expr(expr) is None, (label, expr, validate_expr(expr))
    assert _DEFAULT_EXPR in [e for _, e in PRESETS]


# --- Familien (TZ 7): Auswahl, Presets, Op, family-abhängige Validierung -------
def test_family_presets_are_valid():
    """Jedes Familien-Preset ist ein für seine Familie gültiger Ausdruck."""
    assert {k for _, k in FAMILIES} == set(FAMILY_PRESETS)
    for family, presets in FAMILY_PRESETS.items():
        assert presets, family
        for lbl, expr, op in presets:
            assert validate_expr(expr, family) is None, (family, lbl, expr,
                                                         validate_expr(expr, family))


def test_preset_value_roundtrip():
    """preset_value/parse_preset_value sind invers; Op wird mitgeführt."""
    assert parse_preset_value(preset_value("ij,ij->ij", "add")) == ("ij,ij->ij", "add")
    assert parse_preset_value(preset_value("ik,kj->ij", None)) == ("ik,kj->ij", None)
    # preset_options einer Familie tragen die Op im Value.
    opts = preset_options("elementwise")
    exprs_ops = [parse_preset_value(o["value"]) for o in opts]
    assert ("ij,ij->ij", "add") in exprs_ops and ("ij->ij", "copy") in exprs_ops


def test_validate_expr_family_specific():
    """validate_expr ist family-abhängig (Elementwise/Reduktion vs Kontraktion)."""
    assert validate_expr("ij,ij->ij", "elementwise") is None
    assert validate_expr("ij->ij", "elementwise") is None            # copy (unär)
    assert validate_expr("ij,ji->ij", "elementwise") is not None     # transponiert
    assert validate_expr("ij->i", "reduction") is None
    assert validate_expr("ij->", "reduction") is None                # volle Summe
    assert validate_expr("ij->ij", "reduction") is not None          # keine Reduktion
    assert validate_expr("ik,kj->i", "reduction") is not None        # 2 Operanden


def test_index_categories_family_specific():
    """Kategorien je Familie: Elementwise 'elem', Reduktion 'bleibt'/'Σ'."""
    assert index_categories("ij,ij->ij", "elementwise") == {"i": "elem", "j": "elem"}
    assert index_categories("ij->i", "reduction") == {"i": "bleibt", "j": "Σ"}
    assert index_categories("ik,kj->ij")["k"] == "K"                 # Kontraktion default


def test_combos_for_family_memory_bound():
    """memory-bound: nur fp16/bf16/fp32 (inkl. fp32), KEIN fp8/tf32."""
    mb = combos_for_family("elementwise")
    dts = {d for d, _ in mb}
    assert dts == {"fp16", "bf16", "fp32"}, dts
    assert ("fp32", "fp32") in mb and not any(d.startswith("fp8") or d == "tf32" for d, _ in mb)
    assert combos_for_family("contraction") == COMBOS


def test_validate_selection_family():
    """memory-bound lehnt fp8/tf32 mit klarer Meldung ab; fp16/fp32 sind ok."""
    assert validate_selection([combo_key("fp16", "fp32")], "elementwise") is None
    assert validate_selection([combo_key("fp32", "fp32")], "reduction") is None
    assert validate_selection([combo_key("fp8e4m3", "fp16")], "elementwise") is not None
    assert validate_selection([combo_key("tf32", "fp32")], "elementwise") is not None
    # Kontraktion unverändert: fp8/tf32 weiter erlaubt.
    assert validate_selection([combo_key("fp8e4m3", "fp16")], "contraction") is None


def test_default_selection_for_family():
    assert set(default_selection_for_family("elementwise")) == {
        combo_key("fp16", "fp32"), combo_key("bf16", "fp32"), combo_key("fp32", "fp32")}
    assert default_selection_for_family("contraction") == _DEFAULT_SELECTION


def test_configs_from_selection_elementwise():
    """Elementwise: family/op landen in der RunConfig; kein Swizzle/keine Baselines."""
    cfgs = configs_from_selection("ij,ij->ij", {"i": 128, "j": 128},
                                  [combo_key("fp16", "fp32"), combo_key("bf16", "fp32")],
                                  swizzle_configs=[(False, 8), (True, 8)], baselines=["cublas"],
                                  family="elementwise", op="mul")
    # Swizzle-Konfigs werden für memory-bound ignoriert → nur 1 Config je Format.
    assert len(cfgs) == 2
    for c in cfgs:
        assert c.family == "elementwise" and c.op == "mul"
        assert c.swizzle is False and c.baselines == []
        assert c.expr == "ij,ij->ij"


def test_configs_from_selection_reduction_forces_sum():
    """Reduktion: op wird immer 'sum' (unabhängig vom übergebenen op)."""
    cfgs = configs_from_selection("ij->i", {"i": 256, "j": 256},
                                  [combo_key("fp16", "fp32")],
                                  family="reduction", op=None)
    assert len(cfgs) == 1 and cfgs[0].family == "reduction" and cfgs[0].op == "sum"


def test_configs_from_selection_contraction_unchanged():
    """Kontraktion: family='contraction', op=None (Regression: TZ 6 unverändert)."""
    c = configs_from_selection("ik,kj->ij", {"i": 128, "k": 64, "j": 128},
                               [combo_key("fp16", "fp32")])[0]
    assert c.family == "contraction" and c.op is None


# --- TZ 7.5-2: Multi-Config (Tile-Zeilen + Swizzle-Konfigs + Kreuzprodukt) ----
def test_swizzle_configs_from_state():
    """Mehrfachauswahl-Werte → sortierte (swizzle, group_m)-Liste; leer ⇒ nur (False,8)."""
    assert swizzle_configs_from_state(["off"]) == [(False, 8)]
    assert swizzle_configs_from_state([]) == [(False, 8)]
    assert swizzle_configs_from_state(["g8"]) == [(True, 8)]
    # deterministisch sortiert: ohne-Swizzle zuerst, dann GROUP_M aufsteigend
    assert swizzle_configs_from_state(["g16", "off", "g8"]) == [(False, 8), (True, 8), (True, 16)]
    assert swizzle_configs_from_state(["g32", "g1"]) == [(True, 1), (True, 32)]


def test_validate_swizzle_configs():
    assert validate_swizzle_configs(["off", "g8", "g16"]) is None
    assert validate_swizzle_configs([]) is None          # leer ⇒ Default, ok
    assert validate_swizzle_configs(["g7"]) is not None   # keine Option
    assert validate_swizzle_configs(["bogus"]) is not None


def test_tiles_from_state_and_validate_tiles():
    """tiles_from_state zippt die drei ALL-Listen zeilenweise; validate_tiles prüft
    jede Zeile + fängt Duplikate."""
    tiles = tiles_from_state(["128", "64"], ["128", "64"], ["64", "32"])
    assert tiles == [{"TM": "128", "TN": "128", "TK": "64"},
                     {"TM": "64", "TN": "64", "TK": "32"}]
    assert validate_tiles(tiles) is None
    assert validate_tiles(tiles_from_state(["128", "128"], ["128", "128"], ["64", "64"])) is not None  # Duplikat
    assert validate_tiles(tiles_from_state(["48"], ["128"], ["64"])) is not None  # ungültiger Wert
    assert validate_tiles([]) is not None                                          # leer


def test_mutate_tile_rows_add_remove():
    """+/- Zeilen-Mutation: hinzufügen wächst, entfernen schrumpft, mind. 1 bleibt."""
    from tool_pipeline.app.components.controls import ID_TILE_ADD, TILE_RM_TYPE
    rows = [default_tile_row()]
    rows = mutate_tile_rows(rows, ID_TILE_ADD)
    assert len(rows) == 2
    rows = mutate_tile_rows(rows, {"type": TILE_RM_TYPE, "index": 0})
    assert len(rows) == 1
    rows = mutate_tile_rows(rows, {"type": TILE_RM_TYPE, "index": 0})   # letzte Zeile bleibt
    assert len(rows) == 1


def test_tile_rows_renderer_reindexes():
    """tile_rows re-indiziert 0..n-1 (Header + je Zeile TM/TN/TK-Selects + ✕)."""
    from tool_pipeline.app.components.controls import TILE_TM_TYPE
    comps = tile_rows([default_tile_row(), {"TM": 64, "TN": 64, "TK": 32}])
    # 1 Header + 2 Zeilen
    assert len(comps) == 3
    # zweite Zeile: erstes Select hat index 1 vom Typ tile-tm
    row1 = comps[2].to_plotly_json()["props"]["children"]
    tm_sel = row1[0].to_plotly_json()["props"]["children"].to_plotly_json()["props"]
    assert tm_sel["id"] == {"type": TILE_TM_TYPE, "index": 1}
    assert tm_sel["value"] == "64"


def test_configs_cross_product_tiles_x_swizzle():
    """Volles Kreuzprodukt selection × tiles × swizzle_configs; Reihenfolge
    Format→Tile→Swizzle; Baselines nur an der ersten (Tile,Swizzle) je Format;
    jede Config hat einen EIGENEN (kollisionsfreien) Slug."""
    from tool_pipeline.store.store import config_slug
    sel = [combo_key("fp16", "fp32"), combo_key("bf16", "fp32")]
    tiles = [{"TM": 128, "TN": 128, "TK": 64}, {"TM": 64, "TN": 64, "TK": 32}]
    sw = [(False, 8), (True, 16)]
    cfgs = configs_from_selection("ik,kj->ij", {"i": 128, "k": 64, "j": 128}, sel,
                                  tiles=tiles, swizzle_configs=sw, baselines=["cublas"])
    assert len(cfgs) == 8
    assert [c.dtype for c in cfgs] == ["fp16"] * 4 + ["bf16"] * 4    # COMBOS-Reihenfolge
    # je Format: (t0,sw0),(t0,sw1),(t1,sw0),(t1,sw1)
    assert [(c.tile["TM"], c.swizzle) for c in cfgs[:4]] == \
        [(128, False), (128, True), (64, False), (64, True)]
    # Baselines NUR an der ersten (Tile,Swizzle)-Kombi je Format (Index 0 und 4)
    assert cfgs[0].baselines == ["cublas"] and cfgs[4].baselines == ["cublas"]
    assert all(c.baselines == [] for i, c in enumerate(cfgs) if i not in (0, 4))
    # Kollisionsfreiheit: 8 verschiedene Slugs
    assert len({config_slug(c) for c in cfgs}) == 8


def test_validate_dim_sizes_family_memory_guard():
    """OOM-Schutz greift auch family-abhängig (Elementwise: riesige Elementzahl)."""
    assert validate_dim_sizes("ij,ij->ij", {"i": 100, "j": 100}, "elementwise") is None
    msg = validate_dim_sizes("ij,ij->ij", {"i": 100000, "j": 100000}, "elementwise")
    assert msg is not None and "Zu groß" in msg


def test_build_controls_has_family_and_op_ids():
    """Familien- und Op-Auswahl sind im Komponentenbaum."""
    ids = [(c.to_plotly_json().get("props", {}) or {}).get("id")
           for c in _walk(build_controls())]
    assert ID_FAMILY in ids and ID_OP in ids


def test_index_size_inputs_one_per_index():
    """index_size_inputs baut ein Feld je Index mit Pattern-Matching-ID; Werte erhalten."""
    fields = index_size_inputs("bik,bkj->bij", values={"b": 4})
    ids = [(f.to_plotly_json()["props"]["children"][1].to_plotly_json()["props"]["id"])
           for f in fields]
    assert [i["index"] for i in ids] == ["b", "i", "k", "j"]
    assert all(i["type"] == INDEX_SIZE_TYPE for i in ids)
    # der erhaltene Wert für 'b' ist 4, der Rest der Default.
    b_input = fields[0].to_plotly_json()["props"]["children"][1].to_plotly_json()["props"]
    assert b_input["value"] == 4


def _walk(node):
    if hasattr(node, "to_plotly_json"):
        yield node
        ch = (node.to_plotly_json().get("props", {}) or {}).get("children")
        if isinstance(ch, (list, tuple)):
            for c in ch:
                yield from _walk(c)
        elif ch is not None:
            yield from _walk(ch)
    elif isinstance(node, (list, tuple)):
        for c in node:
            yield from _walk(c)


def test_build_controls_has_expr_and_axis_ids():
    """Ausdrucks-/Preset-/Größen-Container + Multi-Config-Tile-/Swizzle-/Baseline-IDs
    sind im Baum (TZ 7.5-2: dynamische Tile-Zeilen statt fester Dropdowns)."""
    # Liste (nicht set): Pattern-Matching-IDs sind dicts (nicht hashbar).
    ids = [(c.to_plotly_json().get("props", {}) or {}).get("id")
           for c in _walk(build_controls())]
    for i in (ID_PRESET, ID_EXPR, ID_INDEX_SIZES, ID_TILE_ROWS, ID_TILE_ADD,
              ID_SWIZZLE_CONFIGS, ID_BASELINES):
        assert i in ids, f"Control-ID {i!r} fehlt im Baum"
    # mindestens eine Tile-Zeile mit Pattern-Matching-TM-Select (Index 0)
    assert any(isinstance(i, dict) and i.get("type") == TILE_TM_TYPE for i in ids), \
        "keine dynamische Tile-Zeile im Baum"


def test_build_controls_default_index_fields():
    """Der Größen-Container ist initial mit je einem Feld pro Default-Index (i,k,j) gefüllt."""
    idx_dict_ids = [(c.to_plotly_json().get("props", {}) or {}).get("id")
                    for c in _walk(build_controls())]
    pm = [i for i in idx_dict_ids if isinstance(i, dict) and i.get("type") == INDEX_SIZE_TYPE]
    assert sorted(i["index"] for i in pm) == ["i", "j", "k"], pm


def test_tooltips_target_markers():
    tips = [c for c in _walk(build_controls()) if c.to_plotly_json().get("type") == "Tooltip"]
    targets = {(t.to_plotly_json().get("props", {}) or {}).get("target") for t in tips}
    for marker in (ID_DTYPE_INFO, ID_TILE_INFO, ID_BASELINE_INFO):
        assert marker in targets, f"Tooltip für {marker!r} fehlt"


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
