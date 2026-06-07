#include <aie_api/aie.hpp>

// Accumulating 2x2-tiled GEMM for the A09 L1 layout. Computes ONE (a,b) output
// tile's contribution for ONE c-block; the c-reduction is the surrounding MLIR
// loop, which calls this kernel 16x into the same (zero-initialized) out tile.
//
//   in0: prmk  (p=2,r=8,m=8,k=8)  block(p,r) at p*512 + r*64   (elements)
//   in1: rqkn  (r=8,q=2,k=8,n=8)  block(r,q) at r*128 + q*64
//   out: pqmn  (p=2,q=2,m=8,n=8)  block(p,q) at p*128 + q*64
//   out[p][q] += sum_r in0[p][r] (8x8 m,k) * in1[r][q] (8x8 k,n)
//
// Unlike the A08 kernel (which OVERWROTE out via mm.mul + store), this READS the
// existing out tile and ADDS this c-block's product to it (read-modify-write), so
// repeated calls across the c-loop accumulate -- this is the "explicit OUT load"
// the A08 feedback required. (This aie::mmul exposes no from_vector() to seed the
// accumulator directly, hence the explicit load + aie::add.)
extern "C" void matmul(bfloat16 const *__restrict in0,
                       bfloat16 const *__restrict in1,
                       bfloat16 *__restrict out) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  for (unsigned p = 0; p < 2; ++p)
    for (unsigned q = 0; q < 2; ++q) {
      // Compute this c-block's 8x8 product (mul + mac over r) ...
      MMUL mm;
      mm.mul(aie::load_v<64>(in0 + p * 512 + 0 * 64),
             aie::load_v<64>(in1 + 0 * 128 + q * 64));
      for (unsigned r = 1; r < 8; ++r)
        mm.mac(aie::load_v<64>(in0 + p * 512 + r * 64),
               aie::load_v<64>(in1 + r * 128 + q * 64));
      // ... and add it onto the current out tile (partial sum over c so far).
      auto prev = aie::load_v<64>(out + p * 128 + q * 64);  // out so far (bf16)
      auto prod = mm.template to_vector<bfloat16>();         // A@B of this c-block
      aie::store_v(out + p * 128 + q * 64, aie::add(prod, prev));  // out += A@B
    }
}
