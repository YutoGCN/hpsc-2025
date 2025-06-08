#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 16;
    float x[N], y[N], m[N], fx[N], fy[N];
    for(int i=0; i<N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }

    for(int i=0; i<N; i++) {
        // for(int j=0; j<N; j++) {
        //   if(i != j) {
        //     float rx = x[i] - x[j];
        //     float ry = y[i] - y[j];
        //     float r = std::sqrt(rx * rx + ry * ry);
        //     fx[i] -= rx * m[j] / (r * r * r);
        //     fy[i] -= ry * m[j] / (r * r * r);
        //   }
        // }
        // printf("%d %g %g\n", i, fx[i], fy[i]);
        // fx[i] = fy[i] = 0;

        __m512 xi = _mm512_set1_ps(x[i]);
        __m512 yi = _mm512_set1_ps(y[i]);

        __m512 xvec = _mm512_load_ps(x);
        __m512 yvec = _mm512_load_ps(y);
        __m512 mvec = _mm512_load_ps(m);

        __m512i idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        __mmask16 mask = _mm512_cmp_epi32_mask(idx, _mm512_set1_epi32(i), _MM_CMPINT_NE);

        __m512 rx = _mm512_sub_ps(xi, xvec);
        __m512 ry = _mm512_sub_ps(yi, yvec);
        __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
        __m512 rinv = _mm512_rsqrt14_ps(r2);
        __m512 rinv3 = _mm512_mul_ps(rinv, _mm512_mul_ps(rinv, rinv));

        __m512 fxvec = _mm512_mask_mul_ps(
            _mm512_setzero_ps(), mask,
            _mm512_mul_ps(rx, mvec), rinv3
        );
        __m512 fyvec = _mm512_mask_mul_ps(
            _mm512_setzero_ps(), mask,
            _mm512_mul_ps(ry, mvec), rinv3
        );

        fx[i] -= _mm512_reduce_add_ps(fxvec);
        fy[i] -= _mm512_reduce_add_ps(fyvec);

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
}
