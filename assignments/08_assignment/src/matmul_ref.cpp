#include <aie_api/aie.hpp>

// Reference 2x2-tiled GEMM matching the A08 L1 layout:
//   in0: prmk  (p=2,r=8,m=8,k=8)  block(p,r) at p*512 + r*64   (elements)
//   in1: rqkn  (r=8,q=2,k=8,n=8)  block(r,q) at r*128 + q*64
//   out: pqmn  (p=2,q=2,m=8,n=8)  block(p,q) at p*128 + q*64
// out[p][q] = sum_r in0[p][r] (8x8 m,k) * in1[r][q] (8x8 k,n)
extern "C" void matmul(bfloat16 const *__restrict in0,
                       bfloat16 const *__restrict in1,
                       bfloat16 *__restrict out) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  for (unsigned p = 0; p < 2; ++p)
    for (unsigned q = 0; q < 2; ++q) {
      MMUL mm;
      {
        auto a = aie::load_v<64>(in0 + p * 512 + 0 * 64);
        auto b = aie::load_v<64>(in1 + 0 * 128 + q * 64);
        mm.mul(a, b);
      }
      for (unsigned r = 1; r < 8; ++r) {
        auto a = aie::load_v<64>(in0 + p * 512 + r * 64);
        auto b = aie::load_v<64>(in1 + r * 128 + q * 64);
        mm.mac(a, b);
      }
      aie::store_v(out + p * 128 + q * 64, mm.template to_vector<bfloat16>());
    }
}
