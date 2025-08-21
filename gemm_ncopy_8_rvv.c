/***************************************************************************
Copyright (c) 2022, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"

#if !defined(DOUBLE)
#define VSETVL(n)               __riscv_vsetvl_e32m1(n)
#define VSETVL2(n)              __riscv_vsetvl_e32m2(n)
#define VSETVL4(n)              __riscv_vsetvl_e32m4(n)
#define VSETVL8(n)              __riscv_vsetvl_e32m8(n)
#define FLOAT_V_T               vfloat32m1_t
#define FLOAT2_V_T              vfloat32m2_t
#define FLOAT4_V_T              vfloat32m4_t
#define FLOAT8_V_T              vfloat32m8_t
#define FLOAT_VX2_T             vfloat32m4x2_t
#define FLOAT_VX4_T             vfloat32m2x4_t
#define FLOAT_VX8_T             vfloat32m1x8_t
#define VSET_VX2                __riscv_vset_v_f32m4_f32m4x2
#define VSET_VX4                __riscv_vset_v_f32m2_f32m2x4
#define VSET_VX8                __riscv_vset_v_f32m1_f32m1x8
#define VLEV_FLOAT              __riscv_vle32_v_f32m1
#define VLEV_FLOAT2             __riscv_vle32_v_f32m2
#define VLEV_FLOAT4             __riscv_vle32_v_f32m4
#define VLEV_FLOAT8             __riscv_vle32_v_f32m8
#define VSEV_FLOAT8             __riscv_vse32_v_f32m8
#define VSSEG2_FLOAT            __riscv_vsseg2e32_v_f32m4x2
#define VSSEG4_FLOAT            __riscv_vsseg4e32_v_f32m2x4
#define VSSEG8_FLOAT            __riscv_vsseg8e32_v_f32m1x8
#else
#define VSETVL(n)               __riscv_vsetvl_e64m1(n)
#define VSETVL2(n)              __riscv_vsetvl_e64m2(n)
#define VSETVL4(n)              __riscv_vsetvl_e64m4(n)
#define VSETVL8(n)              __riscv_vsetvl_e64m8(n)
#define FLOAT_V_T               vfloat64m1_t
#define FLOAT2_V_T              vfloat64m2_t
#define FLOAT4_V_T              vfloat64m4_t
#define FLOAT8_V_T              vfloat64m8_t
#define FLOAT_VX2_T             vfloat64m4x2_t
#define FLOAT_VX4_T             vfloat64m2x4_t
#define FLOAT_VX8_T             vfloat64m1x8_t
#define VSET_VX2                __riscv_vset_v_f64m4_f64m4x2
#define VSET_VX4                __riscv_vset_v_f64m2_f64m2x4
#define VSET_VX8                __riscv_vset_v_f64m1_f64m1x8
#define VLEV_FLOAT              __riscv_vle64_v_f64m1
#define VLEV_FLOAT2             __riscv_vle64_v_f64m2
#define VLEV_FLOAT4             __riscv_vle64_v_f64m4
#define VLEV_FLOAT8             __riscv_vle64_v_f64m8
#define VSEV_FLOAT8             __riscv_vse64_v_f64m8
#define VSSEG2_FLOAT            __riscv_vsseg2e64_v_f64m4x2
#define VSSEG4_FLOAT            __riscv_vsseg4e64_v_f64m2x4
#define VSSEG8_FLOAT            __riscv_vsseg8e64_v_f64m1x8
#endif

// Optimizes the implementation in ../generic/gemm_ncopy_8.c

int CNAME(BLASLONG m, BLASLONG n, FLOAT *a, BLASLONG lda, FLOAT *b)
{
    BLASLONG i, j;

    FLOAT *a_offset;
    FLOAT *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    FLOAT *a_offset5, *a_offset6, *a_offset7, *a_offset8;
    FLOAT *b_offset;

    FLOAT_V_T v1, v2, v3, v4, v5, v6, v7, v8;
    FLOAT2_V_T v12, v22, v32, v42;
    FLOAT4_V_T v14, v24;
    FLOAT8_V_T v18;

    FLOAT_VX2_T vx2;
    FLOAT_VX4_T vx4;
    FLOAT_VX8_T vx8;

    size_t vl;

    //fprintf(stderr, "gemm_ncopy_8 m=%ld n=%ld lda=%ld\n", m, n, lda);

    a_offset = a;
    b_offset = b;

    for(j = (n >> 3); j > 0; j--) {
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset3  = a_offset2 + lda;
        a_offset4  = a_offset3 + lda;
        a_offset5  = a_offset4 + lda;
        a_offset6  = a_offset5 + lda;
        a_offset7  = a_offset6 + lda;
        a_offset8  = a_offset7 + lda;
        a_offset += 8 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL(i);

            v1 = VLEV_FLOAT(a_offset1, vl);
            v2 = VLEV_FLOAT(a_offset2, vl);
            v3 = VLEV_FLOAT(a_offset3, vl);
            v4 = VLEV_FLOAT(a_offset4, vl);
            v5 = VLEV_FLOAT(a_offset5, vl);
            v6 = VLEV_FLOAT(a_offset6, vl);
            v7 = VLEV_FLOAT(a_offset7, vl);
            v8 = VLEV_FLOAT(a_offset8, vl);

            vx8 = VSET_VX8(vx8, 0, v1);
            vx8 = VSET_VX8(vx8, 1, v2);
            vx8 = VSET_VX8(vx8, 2, v3);
            vx8 = VSET_VX8(vx8, 3, v4);
            vx8 = VSET_VX8(vx8, 4, v5);
            vx8 = VSET_VX8(vx8, 5, v6);
            vx8 = VSET_VX8(vx8, 6, v7);
            vx8 = VSET_VX8(vx8, 7, v8);

            VSSEG8_FLOAT(b_offset, vx8, vl);

            a_offset1 += vl;
            a_offset2 += vl;
            a_offset3 += vl;
            a_offset4 += vl;
            a_offset5 += vl;
            a_offset6 += vl;
            a_offset7 += vl;
            a_offset8 += vl;
            b_offset += vl*8;
        }
    }

    if (n & 4) {
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset3  = a_offset2 + lda;
        a_offset4  = a_offset3 + lda;
        a_offset += 4 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL2(i);

            v12 = VLEV_FLOAT2(a_offset1, vl);
            v22 = VLEV_FLOAT2(a_offset2, vl);
            v32 = VLEV_FLOAT2(a_offset3, vl);
            v42 = VLEV_FLOAT2(a_offset4, vl);

            vx4 = VSET_VX4(vx4, 0, v12);
            vx4 = VSET_VX4(vx4, 1, v22);
            vx4 = VSET_VX4(vx4, 2, v32);
            vx4 = VSET_VX4(vx4, 3, v42);

            VSSEG4_FLOAT(b_offset, vx4, vl);

            a_offset1 += vl;
            a_offset2 += vl;
            a_offset3 += vl;
            a_offset4 += vl;
            b_offset += vl*4;
        }
    }

    if (n & 2) {
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset += 2 * lda;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL4(i);

            v14 = VLEV_FLOAT4(a_offset1, vl);
            v24 = VLEV_FLOAT4(a_offset2, vl);

            vx2 = VSET_VX2(vx2, 0, v14);
            vx2 = VSET_VX2(vx2, 1, v24);

            VSSEG2_FLOAT(b_offset, vx2, vl);

            a_offset1 += vl;
            a_offset2 += vl;
            b_offset += vl*2;
        }
    }

    if (n & 1) {
        a_offset1  = a_offset;

        for(i = m; i > 0; i -= vl) {
            vl = VSETVL8(i);

            v18 = VLEV_FLOAT8(a_offset1, vl);

            VSEV_FLOAT8(b_offset, v18, vl);

            a_offset1 += vl;
            b_offset += vl;
        }
    }

    return 0;
}
