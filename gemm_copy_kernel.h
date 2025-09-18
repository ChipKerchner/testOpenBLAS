#ifdef TEST_MATRIX
#ifdef TEST_FLOAT
#define CNAME  FP3264GEMM_N_RVV
#ifdef RVV_256
#include "sgemm_kernel_16x8_zvl256b.c"
#else
#include "sgemm_kernel_8x8_zvl128b.c"
#endif
#undef CNAME
#elif defined(TEST_DOUBLE)
#define CNAME  FP3264GEMM_N_RVV
#ifdef RVV_256
#include "dgemm_kernel_8x8_zvl256b.c"
#else
#include "dgemm_kernel_8x4_zvl128b.c"
#endif
#else
#define CNAME  BF16GEMM_N_RVV
#ifdef RVV_256
#ifndef TEST_FLOAT16
#include "sbgemm_kernel_16x8_zvl256b.c"
#else
#include "shgemm_kernel_16x8_zvl256b.c"
#endif
#else
#ifndef TEST_FLOAT16
#include "sbgemm_kernel_8x8_zvl128b.c"
#else
#include "shgemm_kernel_8x8_zvl128b.c"
#endif
#endif
//#include "gemmkernel_2x2.c"
#endif
#undef CNAME

#ifndef TEST_BFLOAT
#define CNAME  FP3264_PACK_MN
#else
#define CNAME  BF16_PACK_MN
#endif
#if GEMM_UNROLL_M == 16
#ifdef VECTORIZE_PACK_N
#include "gemm_ncopy_16_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#undef FLOAT_VX2_T
#undef FLOAT_VX4_T
#undef FLOAT2_V_T
#undef FLOAT4_V_T
#undef VSET_VX2
#undef VSET_VX4
#undef VLEV_FLOAT2
#undef VLEV_FLOAT4
#undef VSSEG2_FLOAT
#undef VSSEG4_FLOAT
#undef VSETVL
#undef VSETVL2
#undef VSETVL4
#undef VSETVL8
#else
#include "gemm_ncopy_16.c"
#endif
#else
#ifdef VECTORIZE_PACK_N
#include "gemm_ncopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#undef FLOAT_VX2_T
#undef FLOAT_VX4_T
#undef FLOAT2_V_T
#undef FLOAT4_V_T
#undef VSET_VX2
#undef VSET_VX4
#undef VLEV_FLOAT2
#undef VLEV_FLOAT4
#undef VSSEG2_FLOAT
#undef VSSEG4_FLOAT
#undef VSETVL
#undef VSETVL2
#undef VSETVL4
#undef VSETVL8
#else
#include "gemm_ncopy_8.c"
#endif
#endif
#undef CNAME
#ifndef TEST_BFLOAT
#define CNAME  FP3264_PACK_NN
#else
#define CNAME  BF16_PACK_NN
#endif
#if GEMM_UNROLL_N == 8
#ifdef VECTORIZE_PACK_N
#include "gemm_ncopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_8.c"
#endif
#else
#ifdef VECTORIZE_PACK_N
#include "gemm_ncopy_4_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_4.c"
#endif
#endif
#undef CNAME

#ifndef TEST_BFLOAT
#define CNAME  FP3264_PACK_MT
#else
#define CNAME  BF16_PACK_MT
#endif
#if GEMM_UNROLL_M == 16
#ifdef VECTORIZE_PACK_T
#include "gemm_tcopy_16_rvv.c"
#undef FLOAT_V_T
#undef FLOAT_V_T_HALF
#undef VLEV_FLOAT
#undef VLEV_FLOAT_HALF
#undef VSEV_FLOAT
#undef VSEV_FLOAT_HALF
#else
#include "gemm_tcopy_16.c"
#endif
#else
#ifdef VECTORIZE_PACK_T
#include "gemm_tcopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_tcopy_8.c"
#endif
#endif
#undef CNAME
#ifndef TEST_BFLOAT
#define CNAME  FP3264_PACK_NT
#else
#define CNAME  BF16_PACK_NT
#endif
#if GEMM_UNROLL_N == 8
#ifdef VECTORIZE_PACK_T
#include "gemm_tcopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_tcopy_8.c"
#endif
#else
#ifdef VECTORIZE_PACK_T
#include "gemm_tcopy_4_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_tcopy_4.c"
#endif
#endif
#undef CNAME

#ifdef TEST_SMALL_MATRIX
//#define B0  // Beta = 0
#define CNAME  FP3264GEMM_NN_RVV_SMALL
#include "gemm_small_kernel_nn_rvv.c"
#undef CNAME
#define CNAME  FP3264GEMM_NT_RVV_SMALL
#include "gemm_small_kernel_nt_rvv.c"
#undef CNAME
#define CNAME  FP3264GEMM_TN_RVV_SMALL
#include "gemm_small_kernel_tn_rvv.c"
#undef CNAME
#define CNAME  FP3264GEMM_TT_RVV_SMALL
#include "gemm_small_kernel_tt_rvv.c"
#undef CNAME

#define CNAME  GEMM_SMALL_M_PERMIT
#include "gemm_small_kernel_permit_rvv.c"
#undef CNAME
#endif
#else

#ifndef TEST_BFLOAT
#define CNAME  FP3264GEMV_N_RVV
#include "gemv_n_vector.c"
#undef VSETVL
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VLSEV_FLOAT
#undef CNAME

#define CNAME  FP3264GEMV_T_RVV
#include "gemv_t_vector.c"
#else
#define CNAME  BF16GEMV_N_RVV
//#include "sbgemv_n.c"
#include "common.h"   // Temp
#undef CNAME

#define CNAME  BF16GEMV_T_RVV
//#include "sbgemv_t.c"
#endif
#endif
#undef CNAME

#ifdef VERIFY_MATRIX
#define CNAME  OLD_GEMM_N
#include "gemmkernel_2x2.c"
#undef CNAME
#endif

