"""
Task 2: Matrix Reduction Kernel

a) cuTile kernel: reduce 2D matrix (M, K) along dim K → vector (M,).
   Verify via torch.allclose against torch.sum(mat, dim=1).
b) Report: theoretical impact of M and K on parallelization / per-block load.
"""

import cuda.tile as ct
import cupy as cp
import torch


# ===========================================================================
# Kernel
# ===========================================================================

@ct.kernel
def row_sum_kernel(mat, output, tile_k: ct.Constant[int]):
    """Reduce one row of *mat* by summing along K."""
    # Block-ID holen -> jeder Block verarbeitet eine Zeile
    pid = ct.bid(0)
    # Bedeutung: "Gib mir meine Block-ID entlang Grid-Achse 0."
    # Oder nach Skript Bedeutung: "gets the 1D block ID"

    # Tile laden — shape (1, tile_k), zero-padding für K != 2er-Potenz
    tile = ct.load(mat, index=(pid, 0), shape=(1, tile_k),
                  padding_mode=ct.PaddingMode.ZERO)
    # Doku-Parameters:
    #   array (Array) – The array to load from.
    #   index (tuple[int,...]) – An index in the tile space of shape from array.
    #   shape (tuple[const int,...]) – A tuple of const integers definining the shape of the tile.
    #   order ("C" or "F", or tuple[const int,...]) –
    #   Permutation applied to array axes before the logical tile space is constructed. Can be specified either as a tuple of constants, or as one of the two special string literal values:
    #   ”C” is an alias for (0, 1, 2, ...), i.e. no permutation applied;
    #   ”F” is an alias for (..., 2, 1, 0), i.e. axis order is reversed.
    #   padding_mode (PaddingMode) – The value used to pad the tile when it extends beyond the array boundaries. By default, the padding value is undetermined.
    #   latency (const int) – A hint indicating how heavy DRAM traffic will be. It shall be an integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
    #   allow_tma (const bool) – If False, the load will not use TMA. By default, TMA is allowed.
    # Einträge + Begründungen:
    # index=(pid, 0) bedeutet: "Position im Tile-Space"
    # shape=(1, tile_k) bedeutet: "Größe des Tiles. 1 Zeile, tile_k Spalten"
    # mat[pid * 1  :  pid * 1 + 1,   0 * tile_k  :  0 * tile_k + tile_k]
    #      ↑ index[0] * shape[0]      ↑ index[1] * shape[1]
    # padding_mode=ct.PaddingMode.ZERO bedeutet: "Wenn tile über die Grenzen von mat hinausgeht, mit 0 auffüllen"

        
    # Summe entlang axis=1 (die K-Dimension)
    row_sum = ct.sum(tile, axis=1)          # shape → (1,)
    # tile:      Shape (1, tile_k)
    #            [[3.5, 1.2, 0.8, ..., 0.0, 0.0]] --> und der spaß wird zusammenaddiert

    # Ergebnis speichern
    ct.store(output, index=(pid,), tile=row_sum)
    # Parameters:
    #   array (Array) – The array to store to.
    #   index (tuple[int,...]) – An index in the tile space of array. shape is inferred from the tile argument.
    #   tile (Tile) – The tile to store. The rank of the tile must match rank of the array, unless it is a scalar or 0d tile.
    #   order ("C" or "F", or tuple[const int,...]) – Order of axis mapping. See load().
    #   latency (int, optional) – A hint indicating how heavy DRAM traffic will be. It shall be an integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
    #   allow_tma (bool, optional) – If False, the load will not use TMA. By default, TMA is allowed.
    


# ===========================================================================
# Host / Launch
# ===========================================================================

def row_sum(mat: torch.Tensor) -> torch.Tensor:
    """Launch row_sum_kernel and return per-row sums."""
    M, K = mat.shape
    # mat ist 2D Tensor mit Shape (M, K). Wird hier wieder "entpackt"

    output = torch.empty(M, dtype=mat.dtype, device=mat.device)
    # erstellen leeren Speicherinhaltes. Für Eine Zeilensumme pro Zeile, gleicher Datentyp, gleiches Gerät

    # Tile-Breite = nächste Zweierpotenz >= K
    tile_k = 1
    while tile_k < K:
        tile_k *= 2

    # Grid — ein Block pro Zeile
    grid = (M, 1, 1)

    # Kernel starten
    ct.launch(torch.cuda.current_stream().cuda_stream,
             grid, row_sum_kernel, (mat, output, tile_k))

    return output


# ===========================================================================
# Verifikation
# ===========================================================================

def verify(M: int, K: int):
    """Compare kernel output to torch.sum(mat, dim=1)."""
    mat = torch.randn(M, K, dtype=torch.float16, device="cuda")

    result = row_sum(mat)
    expected = torch.sum(mat, dim=1)

    # FP16 hat begrenzte Präzision → passende Toleranz
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-2)
    # Parameters:
    #   input (Tensor) – first tensor to compare
    #   other (Tensor) – second tensor to compare
    #   atol (float, optional) – absolute tolerance. Default: 1e-08
    #   rtol (float, optional) – relative tolerance. Default: 1e-05
    #   equal_nan (bool, optional) – if True, then two NaN s will be considered equal. Default: False


    print(f"M={M}, K={K} → allclose={close}")
    assert close, f"Mismatch for shape ({M}, {K})"


if __name__ == "__main__":
    # Verschiedene Shapes — auch nicht-Zweierpotenzen für K
    print(f"Task 2: Matrix Reduction Kernel")
    verify(64, 128)
    verify(128, 100)    
    verify(256, 37)     
    
    print(f"Theoretischer Einfluss von M und K auf Parallelisierung / pro-block load:")
    print(f"- M = #Zeilen: größere M  -> mehr Blöcke können parallel arbeiten ->bessere Ausnutzung der GPU ")
    print(f"               kleinere M -> Gpu unterausgelastet, je weniger, desto weniger 'lohnt' sich Parallelisierung")
    print(f"- K = #Spalten: größere K -> desto mehr Arbeit pro Zeile -> höherer pro-block load ")
    print(f"               kleinere K -> GPU unterausgelastet, da die Blöcke schnell fertig sind.")
