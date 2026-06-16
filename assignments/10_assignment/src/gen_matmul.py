#!/usr/bin/env python3
"""
Generator fuer src/matmul.mlir (Assignment 10 - whole NPU).

Tiling (vorgegeben):
  M=256 -> a x p m  (a=2, x=8, p=2, m=8)
  N=128 -> b y q n  (b=2, y=4, q=2, n=8)
  K=1024-> c r k    (c=16, r=8, k=8)

Raeumliche Verteilung: x=8 ueber die 8 Spalten, y=4 ueber die 4 Compute-Tile-Zeilen.
a,b,c sequentiell. in0 wird entlang der Spalten, in1 entlang der Zeilen gebroadcastet.
Pro Spalte werden die 4 Zeilen-Outputs (y) per join zu out_L2L3 zusammengefuehrt.
"""

COLS = 8          # x
ROWS = 4          # y  (Compute-Tile-Zeilen 2..5)
A = 2
B = 2
C = 16
AB = A * B        # fused ab-loop

# in1_L3L2_<ry> wird auf Shim/Mem-Tile der Spalte ry platziert (4 Stueck, frei waehlbar)
def in1_col(ry):
    return ry


def core_fn(col, ry):
    row = ry + 2
    return f"""    %core_{col}_{row} = aie.core(%tile_{col}_{row}) {{
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {{
        %c0_1 = arith.constant 0 : index
        %cAB = arith.constant {AB} : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %cAB step %c1_1 {{
          %buffer_out = aie.objectfifo.acquire @out_L1L2_{col}_{ry}(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {{
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_{col}(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_{ry}(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_{col}(Consume, 1)
            aie.objectfifo.release @in1_L2L1_{ry}(Consume, 1)
          }}
          aie.objectfifo.release @out_L1L2_{col}_{ry}(Produce, 1)
        }}
      }}
      aie.end
    }} {{stack_size = 1024 : i32}}"""


def main():
    L = []
    L.append("module {")
    L.append("  aie.device(npu2) {")
    L.append('    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) attributes {link_with = "matmul.o"}')
    L.append('    func.func private @zero(memref<2x2x8x8xbf16>) attributes {link_with = "zero.o"}')

    # --- Tiles ---
    for col in range(COLS):
        L.append(f"    %shim_{col} = aie.tile({col}, 0)")
    for col in range(COLS):
        L.append(f"    %mem_{col} = aie.tile({col}, 1)")
    for col in range(COLS):
        for row in range(2, 2 + ROWS):
            L.append(f"    %tile_{col}_{row} = aie.tile({col}, {row})")

    # --- in0: per Spalte, Broadcast entlang der Spalte (4 Zeilen) ---
    for col in range(COLS):
        cons = ", ".join(f"%tile_{col}_{row}" for row in range(2, 2 + ROWS))
        L.append(f"    aie.objectfifo @in0_L3L2_{col}(%shim_{col}, {{%mem_{col}}}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>")
        L.append(f"    aie.objectfifo @in0_L2L1_{col}(%mem_{col} dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {{{cons}}}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>")
        L.append(f"    aie.objectfifo.link [@in0_L3L2_{col}] -> [@in0_L2L1_{col}]([] [])")

    # --- in1: per Zeile, Broadcast entlang der Zeile (8 Spalten) ---
    for ry in range(ROWS):
        mc = in1_col(ry)
        row = ry + 2
        cons = ", ".join(f"%tile_{col}_{row}" for col in range(COLS))
        L.append(f"    aie.objectfifo @in1_L3L2_{ry}(%shim_{mc}, {{%mem_{mc}}}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>")
        L.append(f"    aie.objectfifo @in1_L2L1_{ry}(%mem_{mc} dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {{{cons}}}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>")
        L.append(f"    aie.objectfifo.link [@in1_L3L2_{ry}] -> [@in1_L2L1_{ry}]([] [])")

    # --- out: per Spalte, join der 4 Zeilen (y) zu out_L2L3 ---
    for col in range(COLS):
        for ry in range(ROWS):
            row = ry + 2
            L.append(f"    aie.objectfifo @out_L1L2_{col}_{ry}(%tile_{col}_{row}, {{%mem_{col}}}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>")
        # Join produziert ypqmn; toStream reordert pro 256er-Segment pqmn->pmqn
        # (die y-Iteration ueber die 4 Segmente ist durch den join implizit).
        L.append(f"    aie.objectfifo @out_L2L3_{col}(%mem_{col} dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {{%shim_{col}}}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>")
        ins = ", ".join(f"@out_L1L2_{col}_{ry}" for ry in range(ROWS))
        offs = ", ".join(str(ry * 256) for ry in range(ROWS))
        L.append(f"    aie.objectfifo.link [{ins}] -> [@out_L2L3_{col}]([{offs}] [])")

    # --- Cores ---
    for col in range(COLS):
        for ry in range(ROWS):
            L.append(core_fn(col, ry))

    # --- runtime_sequence ---
    L.append("    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {")
    for a in range(A):
        # outputs (S2MM)
        for col in range(COLS):
            for b in range(B):
                off = a * 16384 + col * 2048 + b * 64
                bd = a * 2 + b           # 0,1 fuer a=0 ; 2,3 fuer a=1
                L.append(f"      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, {off}][1, 4, 16, 16][0, 16, 128, 1]) {{id = {bd} : i64, metadata = @out_L2L3_{col}}} : memref<256x128xbf16>")
        # in0 (MM2S, b-repeat via stride 0)
        for col in range(COLS):
            off = a * 131072 + col * 16384
            bd = 4 + a               # 4 fuer a=0, 5 fuer a=1
            L.append(f"      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, {off}][2, 16, 16, 64][0, 64, 1024, 1]) {{id = {bd} : i64, metadata = @in0_L3L2_{col}}} : memref<256x1024xbf16>")
        # in1 (MM2S, b-sweep, unabhaengig von a)
        for ry in range(ROWS):
            off = ry * 16
            bd = 6 + a               # 6 fuer a=0, 7 fuer a=1
            L.append(f"      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, {off}][2, 16, 64, 16][64, 8192, 128, 1]) {{id = {bd} : i64, metadata = @in1_L3L2_{ry}}} : memref<1024x128xbf16>")
        L.append("")
    # drains: pro Spalte AB=4 ausstehende out-Transfers; ein dma_wait drainiert
    # genau den aeltesten -> 4 Waits je Spalte noetig, damit alle vor h.wait()
    # nach L3 geflusht sind.
    for col in range(COLS):
        for _ in range(AB):
            L.append(f"      aiex.npu.dma_wait {{symbol = @out_L2L3_{col}}}")
    L.append("    }")
    L.append("  }")
    L.append("}")
    L.append("")

    out = "\n".join(L)
    with open("src/matmul.mlir", "w") as f:
        f.write(out)
    print(f"wrote src/matmul.mlir ({len(L)} lines)")


if __name__ == "__main__":
    main()
