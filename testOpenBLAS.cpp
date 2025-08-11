#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define RVV_256       // Test 256-bit RVV
#ifndef RVV_256
#define RVV_128       // Test 128-bit RVV
#endif

#define TEST_MATRIX   // Test GEMM
#ifndef TEST_MATRIX
#define TEST_VECTOR   // Test GEMV
#endif

#define TEST_FLOAT    // Test FP32
#ifndef TEST_FLOAT
#define TEST_DOUBLE   // Test FP64
#define DOUBLE
#ifndef TEST_DOUBLE
#define TEST_BFLOAT   // Test BF16
#endif
#endif

#ifdef TEST_VECTOR
#define VERIFY_MATRIX        // Verfiy GEMV versus GEMM
#endif

#ifdef VERIFY_MATRIX
//#define VERIFY_OPENBLAS      // Verify versus OpenBLAS code
//#define TEST_SMALL_MATRIX    // Use small matrix (no packing)
#endif

#define VECTORIZE_PACK  // Use vectorize packing (if available)

#define TEST_GENERIC     0   // Generic C code
#define TEST_RVV         1   // Vectorized RVV code
#ifdef VERIFY_OPENBLAS
#define TEST_OPENBLAS    2   // OpenBLAS (with inc) code
#define TEST_MAX         TEST_OPENBLAS
#else
#define TEST_MAX         TEST_RVV
#endif
#define TEST_RVV_SMALL   (TEST_MAX + 1)

#define TEST_NOTRANSPOSE 0
#define TEST_TRANSPOSE   1

#ifdef TEST_MATRIX
#define TEST_SIZE        127
#else
#define TEST_SIZE        4095
#endif

#define TEST_ITER        100

#define TEST_ALPHA       2.0
#define TEST_BETA        4.0

#define TEST_INC         1

#define bfloat16         unsigned short

#ifdef TEST_FLOAT
#define TEST_STR         "FP32"
#define IFLOAT           float
#define GEMM_UNROLL_N    8
#ifdef RVV_256
#define GEMM_UNROLL_M    16
#else
#define GEMM_UNROLL_M    8
#endif
#elif defined(TEST_DOUBLE)
#define TEST_STR         "FP64"
#define IFLOAT           double
#define GEMM_UNROLL_M    8
#ifdef RVV_256
#define GEMM_UNROLL_N    8
#else
#define GEMM_UNROLL_N    4
#endif
#else
#define TEST_STR         "BF16"
#define IFLOAT           bfloat16
#endif
#if defined(TEST_FLOAT) || defined(TEST_BFLOAT)
#define FLOAT            float
#else
#define FLOAT            double
#endif
#ifdef TEST_MATRIX
#define TEST_TYPE        "GEMM"
#else
#define TEST_TYPE        "GEMV"
#endif
#ifdef VERIFY_OPENBLAS
#define BLASLONG         signed long
#else
#define BLASLONG         size_t
#endif

#if __clang__
#define COMP_STR         "LLVM"
#elif __GNUC__
#define COMP_STR         "GCC"
#endif

#define timer_t          uint64_t

#define FORCEINLINE      inline __attribute__((always_inline))

#define BF16_EPSILON     (FLOAT)(1 << ((sizeof(FLOAT) - sizeof(IFLOAT)) * 8))
#ifdef TEST_FLOAT
#define FLOAT_EPSILON    (FLOAT)(FLT_EPSILON)
#else
#define FLOAT_EPSILON    (FLOAT)(DBL_EPSILON)
#endif
#define TRANS_EPSILON    2

#define NBMAX            4096

#ifndef __clang__
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

typedef int funcGEMV(BLASLONG, BLASLONG, FLOAT, IFLOAT *, BLASLONG, IFLOAT *, BLASLONG, FLOAT, FLOAT *, BLASLONG);
typedef int funcGEMM(BLASLONG, BLASLONG, BLASLONG, FLOAT, IFLOAT *, IFLOAT *, FLOAT *, BLASLONG);
typedef int funcPACK(BLASLONG , BLASLONG, IFLOAT *, BLASLONG, IFLOAT *);

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
#include "gemmkernel_2x2.c"
#endif
#undef CNAME

#ifndef TEST_BFLOAT
#define CNAME  FP3264_PACK_MN
#if GEMM_UNROLL_M == 16
#ifdef VECTORIZE_PACK
//#include "gemm_ncopy_16_rvv.c"
#include "gemm_ncopy_16.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_16.c"
#endif
#else
#ifdef VECTORIZE_PACK
#include "gemm_ncopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_8.c"
#endif
#endif
#undef CNAME
#define CNAME  FP3264_PACK_NN
#if GEMM_UNROLL_N == 8
#ifdef VECTORIZE_PACK
#include "gemm_ncopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_8.c"
#endif
#else
#ifdef VECTORIZE_PACK
#include "gemm_ncopy_4_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_ncopy_4.c"
#endif
#endif
#undef CNAME

#define CNAME  FP3264_PACK_MT
#if GEMM_UNROLL_M == 16
#ifdef VECTORIZE_PACK
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
#ifdef VECTORIZE_PACK
#include "gemm_tcopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_tcopy_8.c"
#endif
#endif
#undef CNAME
#define CNAME  FP3264_PACK_NT
#if GEMM_UNROLL_N == 8
#ifdef VECTORIZE_PACK
#include "gemm_tcopy_8_rvv.c"
#undef FLOAT_V_T
#undef VLEV_FLOAT
#undef VSEV_FLOAT
#else
#include "gemm_tcopy_8.c"
#endif
#else
#ifdef VECTORIZE_PACK
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
#define CNAME  FP3264GEMM_N_RVV_SMALL
#undef CNAME
#define CNAME  FP3264GEMM_T_RVV_SMALL
#undef CNAME
#undef
#ifdef TEST_FLOAT
#define CNAME  GEMM_SMALL_M_PERMIT
#elif defined(TEST_DOUBLE)
#define CNAME  FP3264GEMM_N_RVV
#endif
#include "gemm_small_kernel_permit_riscv64.c"
#endif
#else
// Temp
#define BF16_PACK_MN ((funcPACK *)NULL)
#define BF16_PACK_NN ((funcPACK *)NULL)
#define BF16_PACK_MT ((funcPACK *)NULL)
#define BF16_PACK_NT ((funcPACK *)NULL)
#endif
#else
#ifndef TEST_BFLOAT
#define CNAME  FP3264GEMV_N_RVV
#include "gemv_n_vector.c"
#undef CNAME

#define CNAME  FP3264GEMV_T_RVV
#include "gemv_t_vector.c"
#else
#define CNAME  BF16GEMV_N_RVV
#include "sbgemv_n.c"
#undef CNAME

#define CNAME  BF16GEMV_T_RVV
#include "sbgemv_t.c"
#endif
#endif
#undef CNAME

#ifdef VERIFY_MATRIX
#define CNAME  OLD_GEMM_N
#include "gemmkernel_2x2.c"
#undef CNAME
#endif

FORCEINLINE timer_t get_rvv_timer()
{
#if 0
  uint64_t val;
  asm volatile("rdcycle %0" : "=r"(val));
  return val;
#else
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#endif
}

typedef union {
  float f;
  uint32_t i;
  bfloat16 q[2];
} data;

FORCEINLINE float
bfloat16tof32(bfloat16 f16)
{
  data result;
  result.i = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  result.q[0] = f16;
#else
  result.q[1] = f16;
#endif
  return result.f;
}

FORCEINLINE void
not_impl(int which)
{
  if (which == 0) {
    fprintf(stderr, "Not implemented yet.\n\n");
  } else {
    fprintf(stderr, "Not supported.\n\n");
  }
  exit(1);
}

#ifdef TEST_VECTOR
FORCEINLINE void init_T(BLASLONG lda, IFLOAT *input_matrix, IFLOAT *input2, FLOAT *output,
                   BLASLONG j, IFLOAT **ap, IFLOAT **x, FLOAT **y)
{
  BLASLONG line = j * lda;
  *ap = input_matrix + line;
  *y = output + j;
  *x = input2;
}

FORCEINLINE void init_N(BLASLONG lda, IFLOAT *input_matrix, IFLOAT *input2, FLOAT *output,
                   BLASLONG j, IFLOAT *ap[], IFLOAT **xo, FLOAT **y)
{
  BLASLONG line = j * lda;
  ap[0] = input_matrix + line;
  ap[1] = ap[0] + lda;
  ap[2] = ap[1] + lda;
  ap[3] = ap[2] + lda;
  *y = output;
  *xo = input2 + j;
}

static void GEMV_N_beta(BLASLONG n, FLOAT *output_vector, FLOAT *input_vector, FLOAT beta)
{
  if (beta == (FLOAT)0) {
    memset(output_vector, 0, sizeof(FLOAT) * n);
  } else if (beta == (FLOAT)1) {
    if (output_vector != input_vector) {
      memcpy(output_vector, input_vector, sizeof(FLOAT) * n);
    }
  } else {
    for (BLASLONG j = 0; j < n; j++) {
      output_vector[j] *= beta;
    }
  }
}

int BF16GEMV_T_generic(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  BF16GEMV_N_beta(N, output, input, beta);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT t = 0;
    for (BLASLONG i = 0; i < M; i++) {
      t += bfloat16tof32(input_matrix[line + i]) * bfloat16tof32(input_vector[i]);
    }
    output[j] += (t * alpha);
  }
  return 0;
}

int FP3264GEMV_T_generic(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  GEMV_N_beta(M, N, output, input, beta);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT t = 0;
    for (BLASLONG i = 0; i < M; i++) {
      t += input_matrix[line + i] * input_vector[i];
    }
    output[j] += (t * alpha);
  }
  return 0;
}

int BF16GEMV_T_RVV(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  IFLOAT *ap, *x;
  FLOAT *y;
  BF16GEMV_N_beta(N, output, input, beta);
  BLASLONG lda = M;
  BLASLONG j = 0;
#ifdef USE_BFGEMV_8_T_RVV
  for (; j + 8 <= N; j += 8) {
    init_T(lda, input_matrix, input_vector, output, j, &ap, &x, &y);
    BF16GEMV_T_RVV_8(M, lda, ap, x, y, alpha);
  }
  if (N & 4) {
#else
  while (j + 4 <= N) {
#endif
    init_T(lda, input_matrix, input_vector, output, j, &ap, &x, &y);
    j += 4;
    BF16GEMV_T_RVV_4(M, lda, ap, x, y, alpha);
  }
  if (N & 2) {
    init_T(lda, input_matrix, input_vector, output, j, &ap, &x, &y);
    j += 2;
    BF16GEMV_T_RVV_2(M, lda, ap, x, y, alpha);
  }
  if (N & 1) {
    init_T(lda, input_matrix, input_vector, output, j, &ap, &x, &y);
    BF16GEMV_T_RVV_1(M, lda, ap, x, y, alpha);
  }
  return 0;
}

int BF16GEMV_N_generic(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  GEMV_N_beta(M, output, input, beta);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT inp = bfloat16tof32(input_vector[j]) * alpha;
    for (BLASLONG i = 0; i < M; i++) {
      output[i] += bfloat16tof32(input_matrix[line + i]) * inp;
    }
  }
  return 0;
}

int FP3264GEMV_N_generic(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  GEMV_N_beta(M, output, input, beta);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT inp = bfloat16tof32(input_vector[j]) * alpha;
    for (BLASLONG i = 0; i < M; i++) {
      output[i] += input_matrix[line + i] * inp;
    }
  }
  return 0;
}

int BF16GEMV_N_RVV(BLASLONG M, BLASLONG N, IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output, FLOAT alpha, FLOAT beta, FLOAT *input)
{
  IFLOAT *ap[4], *xo;
  FLOAT *y;
  BF16GEMV_N_beta(M, output, input, beta);
  BLASLONG lda = M;
  BLASLONG j = 0;
#ifdef USE_BFGEMV_8_N_RVV
  BLASLONG lda4 = M * 4;
  for (; j + 8 <= N; j += 8) {
    init_N(lda, input_matrix, input_vector, output, j, ap, &xo, &y);
    BF16GEMV_N_RVV_8(M, ap, xo, y, lda4, alpha);
  }
  if (N & 4) {
#else
  while (j + 4 <= N) {
#endif
    init_N(lda, input_matrix, input_vector, output, j, ap, &xo, &y);
    j += 4;
    BF16GEMV_N_RVV_4(M, ap, xo, y, alpha);
  }
  if (N & 2) {
    init_N(lda, input_matrix, input_vector, output, j, ap, &xo, &y);
    j += 2;
    BF16GEMV_N_RVV_2(M, ap, xo, y, alpha);
  }
  if (N & 1) {
    init_N(lda, input_matrix, input_vector, output, j, ap, &xo, &y);
    BF16GEMV_N_RVV_1(M, ap, xo, y, alpha);
  }
  return 0;
}
#endif

int FP3264GEMM_N_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line2 = j * K;
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      BLASLONG line = i * K;
      for (BLASLONG k = 0; k < K; k++) {
        t += A[line + k] * B[line2 + k];
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int FP3264GEMM_T_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      for (BLASLONG k = 0; k < K; k++) {
        t += A[k * M + i] * B[k * N + j];
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int BF16GEMM_N_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line2 = j * K;
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      BLASLONG line = i * K;
      for (BLASLONG k = 0; k < K; k++) {
        t += bfloat16tof32(A[line + k]) * bfloat16tof32(B[line2 + k]);
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int BF16GEMM_T_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      for (BLASLONG k = 0; k < K; k++) {
        t += bfloat16tof32(A[k * M + i]) * bfloat16tof32(B[k * N + j]);
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

#ifdef TEST_MATRIX
typedef funcGEMM func;
#else
typedef funcGEMV func;
#endif

func *func_ptr(int test, int orient)
{
  switch (orient) {
    case TEST_NOTRANSPOSE:
      switch (test) {
        case TEST_GENERIC:
#ifdef TEST_MATRIX
#ifndef TEST_BFLOAT
          return FP3264GEMM_N_generic;
#else
          return BF16GEMM_N_generic;
#endif
#else
#ifndef TEST_BFLOAT
          return FP3264GEMV_N_generic;
#else
          return BF16GEMV_N_generic;
#endif
#endif
          break;
        case TEST_RVV:
#ifdef TEST_MATRIX
#ifndef TEST_BFLOAT
          return FP3264GEMM_N_RVV;
#else
          return BF16GEMM_N_RVV;
#endif
#else
#ifndef TEST_BFLOAT
          return FP3264GEMV_N_RVV;
#else
          return BF16GEMV_N_RVV;
#endif
#endif
          break;
#ifdef TEST_SMALL_MATRIX
        case TEST_RVV_SMALL:
          return FP3264GEMM_N_RVV_SMALL;
#endif
      }
      break;
    case TEST_TRANSPOSE:
      switch (test) {
        case TEST_GENERIC:
#ifdef TEST_MATRIX
#ifndef TEST_BFLOAT
          return FP3264GEMM_T_generic;
#else
          return BF16GEMM_T_generic;
#endif
#else
#ifndef TEST_BFLOAT
          return FP3264GEMV_T_generic;
#else
          return BF16GEMV_T_generic;
#endif
#endif
          break;
        case TEST_RVV:
#ifdef TEST_MATRIX
#ifndef TEST_BFLOAT
          return FP3264GEMM_N_RVV;
#else
          return BF16GEMM_N_RVV;
#endif
#else
#ifndef TEST_BFLOAT
          return FP3264GEMV_N_RVV;
#else
          return BF16GEMV_N_RVV;
#endif
#endif
          break;
#ifdef TEST_SMALL_MATRIX
        case TEST_RVV_SMALL:
          return FP3264GEMM_T_RVV_SMALL;
#endif
      }
      break;
  }
  return NULL;
}

FORCEINLINE bfloat16 
float32tobf16(float inp)
{
  data inp1;
  inp1.f = inp;
  inp1.i += ((inp1.i >> 16) & 0x1) + 0x7fff;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return inp1.q[0];
#else
  return inp1.q[1];
#endif
}

#define USE_NEW_RAND

#ifdef USE_NEW_RAND
#define RAND_MUL   1103515245
#define RAND_ADD   12345

static int rand_state = 1;
#endif

FORCEINLINE int rand_value()
{
#ifdef USE_NEW_RAND
  int val = ((rand_state * RAND_MUL) + RAND_ADD) & RAND_MAX;
  rand_state = val;
  return val;
#else
  return rand();
#endif
}

FORCEINLINE void rand_seed(unsigned int seed)
{
#ifdef USE_NEW_RAND
  if (seed == 0) {
    seed = 1;
  }
  rand_state = seed;
#else
  srand(seed);
#endif
}

FORCEINLINE FLOAT get_rand(int x)
{
  return ((FLOAT)x / (FLOAT) RAND_MAX) + 0.5;
}

#ifdef TEST_MATRIX
FORCEINLINE void init_array(BLASLONG i, int y, IFLOAT *input_array0)
{
  FLOAT x = get_rand(y);
#ifndef TEST_BFLOAT
  input_array0[i] = x;
#else
  input_array0[i] = float32tobf16(x);
#endif
}

FORCEINLINE void init_array2(BLASLONG i, int y, FLOAT *input_array0)
{
  FLOAT x = get_rand(y);
#ifndef TEST_BFLOAT
  input_array0[i] = x;
#else
  input_array0[i] = bfloat16tof32(float32tobf16(x));
#endif
}

void init(IFLOAT *input_matrix, IFLOAT *input_matrix2, FLOAT *output_matrix,
          BLASLONG M, BLASLONG N, BLASLONG K)
{
  for (BLASLONG j = 0; j < M; j++) {
    BLASLONG line = j * K;
    for (BLASLONG i = 0; i < K; i++) {
      init_array(line + i, rand_value(), input_matrix);
    }
  }
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * K;
    for (BLASLONG i = 0; i < K; i++) {
      init_array(line + i, rand_value(), input_matrix2);
    }
  }
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    for (BLASLONG i = 0; i < M; i++) {
      init_array2(line + i, rand_value(), output_matrix);
    }
  }
}

int verifyOut(FLOAT *output0, FLOAT *output1, FLOAT tol, BLASLONG M, BLASLONG N, BLASLONG K, const char *str)
{
  FLOAT maxOut = (FLOAT)0;
  BLASLONG m = 0, n = 0;
  for (BLASLONG j = 0; j < N; j++) {
    FLOAT *out0 = output0 + (j * M);
    FLOAT *out1 = output1 + (j * M);
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT diff = fabs(out0[i] - out1[i]);
      if (diff > maxOut) {
        maxOut = diff;
        m = i;
        n = j;
      }
    }
  }
  if (maxOut > tol) {
    fprintf(stderr, "Bad %s %s result %13.4f %13.4f %4ld %4ld (%8.6f %8.6f %4ld %4ld %4ld)\n\n", TEST_STR, str, output0[(n * M) + m], output1[(n * M) + m], m, n, maxOut, tol, M, N, K);
    return 1;
  }
  return 0;
}
#else
FORCEINLINE void init_array(BLASLONG i, int y, IFLOAT *input_array0, FLOAT *input_array1)
{
  FLOAT x = get_rand(y);
#ifndef TEST_BFLOAT
  input_array0[i] = input_array1[i] = x;
#else
  input_array0[i] = float32tobf16(x);
  input_array1[i] = bfloat16tof32(input_array0[i]);
#endif
}

void init(IFLOAT *input_matrix, FLOAT *input_matrix2, IFLOAT *input,
          BLASLONG M, BLASLONG N, BLASLONG in, FLOAT *input, BLASLONG out)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    for (BLASLONG i = 0; i < M; i++) {
      init_array(line + i, rand_value(), input_matrix, input_matrix2);
    }
  }
  for (BLASLONG j = 0; j < in; j++) {
    init_array(j, rand_value(), input_vector, input_vector1);
  }
  for (BLASLONG j = 0; j < out; j++) {
    input[j] = get_rand(rand_value());
  }
}

int verifyOut(FLOAT *output0, FLOAT *output1, BLASLONG out, FLOAT tol, BLASLONG size, const char *str)
{
  FLOAT maxOut = (FLOAT)0;
  BLASLONG i = 0;
  for (BLASLONG j = 0; j < out; j++) {
    FLOAT diff = fabs(output0[j] - output1[j]);
    if (diff > maxOut) {
      maxOut = diff;
      i = j;
    }
  }
  if (maxOut > tol) {
    fprintf(stderr, "Bad %s result %13.4f %13.4f %4ld %4ld (%8.6f %8.6f)\n\n", str, output0[i], output1[i], i, size, maxOut, tol);
    return 1;
  }
  return 0;
}
#endif

int test_F32(int orient, FLOAT *input_matrix1, FLOAT *input_vector, FLOAT *output0, FLOAT *output1,
              BLASLONG M, BLASLONG N, BLASLONG out, FLOAT alpha, FLOAT beta, FLOAT *input, FLOAT tol)
{
#if 0
  if (orient == TEST_NOTRANSPOSE) {
    GEMV_N_beta(M, output1, input, beta);
    for (BLASLONG j = 0; j < N; j++) {
      BLASLONG line = j * M;
      for (BLASLONG i = 0; i < M; i++) {
        output1[i] += input_matrix1[line + i] * input_vector[j] * alpha;
      }
    }
  } else {
    GEMV_N_beta(N, output1, input, beta);
    for (BLASLONG j = 0; j < N; j++) {
      BLASLONG line = j * M;
      FLOAT t = 0;
      for (BLASLONG i = 0; i < M; i++) {
        t += input_matrix1[line + i] * input_vector[i];
      }
      output1[j] += (t * alpha);
    }
  }

  return verifyOut(output0, output1, out, tol, M, "FP32");
#else
  return 0;
#endif
}

#ifdef TEST_MATRIX
int verify(int test, int orient, BLASLONG M, BLASLONG N, BLASLONG K, IFLOAT *input_matrix0,
           IFLOAT *input_matrix1, FLOAT *output0, FLOAT *output1, FLOAT alpha)
#else
int verify(int test, int orient, BLASLONG M, BLASLONG N, BLASLONG out, FLOAT *input_matrix1,
           FLOAT *input_vector1, FLOAT *output0, FLOAT *output1, FLOAT *output2, FLOAT alpha, FLOAT beta,
           FLOAT *input)
#endif
{
#ifdef TEST_MATRIX
  FLOAT tol = (FLOAT)(((orient == TEST_NOTRANSPOSE) ? ((test <= TEST_RVV) ? N : 0) : M) * K * TRANS_EPSILON) * FLOAT_EPSILON * alpha;

  if (verifyOut(output0, output1, tol, M, N, K, TEST_TYPE)) {
#else
  int test2 = TEST_RVV;
  FLOAT tol = (FLOAT)((orient == TEST_NOTRANSPOSE) ? ((test <= test2) ? 0 : N) : M * TRANS_EPSILON) / BF16_EPSILON;

  if ((orient == TEST_TRANSPOSE) && (M > NBMAX)) {
    tol = (FLOAT)M / (FLOAT)NBMAX;
  }
  if (verifyOut(output0, output1, out, tol, M, TEST_TYPE)) {
#endif
    return 1;
  }

#if 0
  if (test_F32(orient, input_matrix1, input_vector1, output1, output2, M, N, out, alpha, beta, input, tol)) {
    return 1;
  }
#endif
  return 0;
}

#ifdef VERIFY_OPENBLAS
void BF16GEMV_N_beta(BLASLONG n, FLOAT *output_vector, FLOAT *input_vector, FLOAT beta)
{
  if (beta == (FLOAT)0) {
    memset(output_vector, 0, sizeof(FLOAT) * n);
  } else if (beta == (FLOAT)1) {
    if (output_vector != input_vector) {
      memcpy(output_vector, input_vector, sizeof(FLOAT) * n);
    }
  } else {
    for (BLASLONG i = 0; i < n; i++) {
       output_vector[i] = input_vector[i] * beta;
    }
  }
}

void verifyGEMV(funcGEMV *test_ptr, int orient, BLASLONG M, BLASLONG N, BLASLONG out,
           IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output2, FLOAT alpha, FLOAT beta,
           FLOAT *input, BLASLONG inc)
{
  if (inc != 1) {
    BLASLONG in  = (orient == TEST_TRANSPOSE) ? M : N;
    FLOAT output3[out * inc];
    IFLOAT input_vector1[in * inc];

    for (BLASLONG j = 0; j < in; j++) {
      input_vector1[j * inc] = input_vector[j];
    }
    for (BLASLONG j = 0; j < out; j++) {
      output3[j * inc] = input[j];
    }
    test_ptr(M, N, alpha, input_matrix, M, input_vector1, inc, beta, output3, inc);
    for (BLASLONG j = 0; j < out; j++) {
     output2[j] = output3[j * inc];
    }
  } else {
    memcpy(output2, input, sizeof(FLOAT) * out);
    test_ptr(M, N, alpha, input_matrix, M, input_vector, inc, beta, output2, inc);
  }
}

void verifyGEMM(funcGEMM *test_ptr, funcPACK *pack_ptr, int orient, BLASLONG M, BLASLONG N, BLASLONG out,
        IFLOAT *input_matrix, IFLOAT *input_vector, FLOAT *output2, IFLOAT *input_matrix1, FLOAT alpha,
        FLOAT *input, FLOAT beta)
{
  BF16GEMV_N_beta(out, output2, input, beta);

  if (orient != TEST_NOTRANSPOSE) {
    pack_ptr(M, N, input_matrix, M, input_matrix1);
    test_ptr(N, 1, M, alpha, input_matrix1, input_vector, output2, N);
  } else {
    pack_ptr(N, M, input_matrix, M, input_matrix1);
    test_ptr(M, 1, N, alpha, input_matrix1, input_vector, output2, M);
  }
}

int verifyOpenBLAS(int orient, BLASLONG M, BLASLONG N, BLASLONG out, IFLOAT *input_matrix,
           IFLOAT *input_vector, IFLOAT *input_matrix1, FLOAT *output1, FLOAT *output2, FLOAT alpha,
           FLOAT beta, FLOAT *input, BLASLONG inc)
{
#ifdef VERIFY_OLD_MATRIX_BF16
  if (verifyOldGEMM(orient, M, N, out, input_matrix, input_vector, input_matrix1, output1, output2, alpha, beta, input)) {
    return 1;
  }
#endif

#ifdef VERIFY_OLD_VECTOR_BF16
  if (verifyOldGEMV(orient, M, N, out, input_matrix, input_vector, output1, output2, alpha, beta, input, inc)) {
    return 1;
  }
#endif

#ifdef VERIFY_NEW_VECTOR_BF16
  if (verifyNewGEMV(orient, M, N, out, input_matrix, input_vector, output1, output2, alpha, beta, input, inc)) {
    return 1;
  }
#endif

  return 0;
}
#endif

int main(int argc, char **argv)
{
  int test = TEST_GENERIC;
  int orient = TEST_TRANSPOSE;
  BLASLONG M = TEST_SIZE;
  BLASLONG N = TEST_SIZE;
  BLASLONG K = TEST_SIZE;
  int iter = TEST_ITER;
  FLOAT alpha = TEST_ALPHA;
  FLOAT beta = TEST_BETA;
  BLASLONG inc = TEST_INC;
  int all = 0;

  if (argc > 1) {
    test = atoi(argv[1]);
    if (argc > 2) {
      orient = atoi(argv[2]);
      if (argc > 3) {
        M = atol(argv[3]);
        if (argc > 4) {
          N = atol(argv[4]);
          if (argc > 5) {
            K = atol(argv[5]);
            if (argc > 6) {
              iter = atoi(argv[6]);
               if (argc > 7) {
                 alpha = atof(argv[7]);
                 if (argc > 8) {
                   beta = atof(argv[8]);
                  if (argc > 9) {
                    inc = atol(argv[9]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if ((signed long)(M) < 0) {
    all = 1;
    M = -M;
  }
  if ((signed long)(N) < 0) {
    all = 2;
    N = -N;
  } else if (all) {
    N = M;
  }

  if ((argc == 1) || (test < TEST_GENERIC) || (test > TEST_MAX) || (orient < TEST_NOTRANSPOSE) ||
      (orient > TEST_TRANSPOSE) || (iter <= 0) || (inc < 1)) {
    fprintf(stderr, "testOpenBLAS (%s %s) ", TEST_STR, TEST_TYPE);
    fprintf(stderr, "type [orient M N K iter alpha beta inc]\n\n");
    fprintf(stderr, "type = 0 (generic C = default), 1 (RVV)\n");
#ifdef VERIFY_OPENBLAS
    fprintf(stderr, "type = 2 (generic - OpenBLAS)\n");
#endif
    fprintf(stderr, "orient = 0 (no transpose), 1 (transpose = default)\n");
    fprintf(stderr, "iter = test iterations.\n");
    fprintf(stderr, "inc = incx & incy.\n");
    fprintf(stderr, "if M < 0 all Ms up to negative M are tested.\n");
    return 1;
  }

  printf("Testing %s %s %s ", COMP_STR, TEST_STR, TEST_TYPE);
  printf("%d %d %4ld %4ld %4ld %3d %4.1f %4.1f %2ld\n\n", test, orient, M, N, K, iter, alpha, beta, inc);

  rand_seed((unsigned int)(get_rvv_timer()));

  func *gen_ptr = func_ptr(TEST_GENERIC, orient);
  func *test_ptr = func_ptr(test, orient);
#ifdef TEST_SMALL_MATRIX
  func *small_ptr = func_ptr(TEST_RVV_SMALL, orient);
#endif
#ifdef VERIFY_OPENBLAS
#ifdef TEST_MATRIX
  funcGEMM *test2_ptr = ((test == TEST_OPENBLAS) ? ((orient == TEST_NOTRANSPOSE) ? NEW_BF16_GEMV_N : NEW_BF16_GEMV_T) : NULL);
#else
  funcGEMV *test2_ptr = ((test == TEST_OPENBLAS) ? ((orient == TEST_NOTRANSPOSE) ? NEW_BF16_GEMV_N : NEW_BF16_GEMV_T) : NULL);
#endif
  funcGEMM *test3_ptr = ((test == TEST_OPENBLAS) ? OLD_BF16_GEMM : NULL);
#endif
#ifndef TEST_BFLOAT
  funcPACK *packm_ptr = ((orient != TEST_NOTRANSPOSE) ? FP3264_PACK_MT : FP3264_PACK_MN);
  funcPACK *packn_ptr = ((orient != TEST_NOTRANSPOSE) ? FP3264_PACK_NT : FP3264_PACK_NN);
#else
  funcPACK *pack_ptr = ((orient != TEST_NOTRANSPOSE) ? BF16_PACK_T : BF16_PACK_N);
#endif

  if ((gen_ptr == NULL) ||
#ifdef VERIFY_OPENBLAS
     ((test_ptr == NULL) &&
     ((test == TEST_OPENBLAS) && (test2_ptr == NULL)) ||
     ((test == TEST_OPENBLAS) && (test3_ptr == NULL)))
#else
     (test_ptr == NULL)
#endif
     ) {
    fprintf(stderr, "Unsupported test.\n");
    return 1;
  }

  for (BLASLONG size = (all) ? 1 : M; size <= M; size += 1) {
    if ((all == 1) || (argc < 4)) {
      N = size;
#ifdef TEST_MATRIX
      K = size;
#endif
    }
    
    BLASLONG M0 = size;
    BLASLONG N0 = N;

    BLASLONG in  = (orient == TEST_TRANSPOSE) ? M0 : N0;
    BLASLONG out = (orient == TEST_TRANSPOSE) ? N0 : M0;

#ifdef TEST_MATRIX
    FLOAT *output_matrix0 = NULL, *output_matrix1 = NULL, *output_matrix2 = NULL;
#else
    IFLOAT input_vector0[in];
    FLOAT input_vector1[in], output0[out], output1[out], output2[out], input[out];
#endif
    IFLOAT *input_matrix0 = NULL, *input_matrix1 = NULL;
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
    IFLOAT *input_matrix01 = NULL, *input_matrix11 = NULL;
#endif

    input_matrix0 = (IFLOAT *)malloc(in * K * sizeof(IFLOAT));
    if (input_matrix0 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
    input_matrix1 = (IFLOAT *)malloc(K * out * sizeof(IFLOAT));
    if (input_matrix1 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
    if (test >= TEST_RVV) {
      input_matrix01 = (IFLOAT *)malloc(in * K * sizeof(IFLOAT));
      if (input_matrix01 == NULL) {
        fprintf(stderr, "Bad malloc\n");
        return 1;
      }
      input_matrix11 = (IFLOAT *)malloc(K * out * sizeof(IFLOAT));
      if (input_matrix11 == NULL) {
        fprintf(stderr, "Bad malloc\n");
        return 1;
      }
    }
#endif
#ifdef TEST_MATRIX
    output_matrix0 = (FLOAT *)malloc(M0 * N0 * sizeof(FLOAT));
    if (output_matrix0 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
    output_matrix1 = (FLOAT *)malloc(M0 * N0 * sizeof(FLOAT));
    if (output_matrix1 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
    output_matrix2 = (FLOAT *)malloc(M0 * N0 * sizeof(FLOAT));
    if (output_matrix2 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
#endif

#ifdef TEST_MATRIX
    init(input_matrix0, input_matrix1, output_matrix0, in, out, K);
    memcpy(output_matrix2, output_matrix0, M0 * N0 * sizeof(FLOAT));
#else
    init(input_matrix0, input_matrix1, input_vector0, input_vector1, M0, N0, in, input, out);
#endif
#ifdef TEST_MATRIX
    gen_ptr(in, out, K, alpha, input_matrix0, input_matrix1, output_matrix0, in);
#else
    gen_ptr(M0, N0, input_matrix0, input_vector0, output0, alpha, beta, input);
#endif

    int stop = (all) ? 1 : (int)iter;
    timer_t start = get_rvv_timer();
    for (int i = 0; i < stop; i++) {

      if (test <= TEST_RVV) {
#ifdef TEST_MATRIX
        memcpy(output_matrix1, output_matrix2, M0 * N0 * sizeof(FLOAT));
        if (test == TEST_RVV) {
#ifdef TEST_SMALL_MATRIX
          if (GEMM_SMALL_M_PERMIT(orient, orient, in, out, K, alpha, 0.0) && 0) {
            small_ptr(in, out, K, alpha, input_matrix0, input_matrix1, output_matrix1, in);
          } else
#endif
          {
            if (orient == TEST_TRANSPOSE) {
              packm_ptr(K, in, input_matrix0, in, input_matrix01);
              packn_ptr(K, out, input_matrix1, out, input_matrix11);
            } else {
              packm_ptr(K, in, input_matrix0, K, input_matrix01);
              packn_ptr(K, out, input_matrix1, K, input_matrix11);
            }
            test_ptr(in, out, K, alpha, input_matrix01, input_matrix11, output_matrix1, in);
          }
        } else {
          test_ptr(in, out, K, alpha, input_matrix0, input_matrix1, output_matrix1, in);
        }
#else
        test_ptr(M0, N0, input_matrix0, input_vector0, output1, alpha, beta, input);
#endif
      }
#ifdef VERIFY_OPENBLAS
      else {
        if (test2_ptr && (test == TEST_OPENBLAS) {
          verifyGEMV(test2_ptr, orient, M0, N0, out, input_matrix0, input_vector0, output1, alpha, beta, input, inc);
        }
#ifdef VERIFY_MATRIX
        else if (test3_ptr && (test == TEST_OPENBLAS)) {
          verifyGEMM(test3_ptr, packm_ptr, orient, M0, N0, out, input_matrix0, input_vector0, output1, input_matrix1, alpha, input, beta);
        }
#endif
      }
#endif
    }
    timer_t end = get_rvv_timer();
    if (!all) {
      printf("Total time = %16ld\n\n", end - start);
    }

#ifdef TEST_MATRIX
    if (verify(test, orient, M0, N0, K, input_matrix0, input_matrix1, output_matrix0, output_matrix1, alpha)) {
#else
    if (verify(test, orient, M0, N0, out, input_matrix1, input_vector1, output0, output1, output2, alpha, beta, input)) {
#endif
      return 1;
    }

#ifdef VERIFY_OPENBLAS
    if (test == TEST_RVV) {
      if (verifyOpenBLAS(orient, M0, N0, out, input_matrix1, input_vector0, input_matrix1, output1, output2, alpha, beta, input, inc)) {
        return 1;
      }
    }
#endif

    free(input_matrix0);
    free(input_matrix1);
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
    if (test >= TEST_RVV) {
      free(input_matrix01);
      free(input_matrix11);
    }
#endif
#ifdef TEST_MATRIX
    free(output_matrix0);
    free(output_matrix1);
    free(output_matrix2);
#endif
  }

  if (all) {
#ifdef TEST_MATRIX
    FLOAT tol = (FLOAT)(((orient == TEST_NOTRANSPOSE) ? ((test <= TEST_RVV) ? N : 0) : M) * K * TRANS_EPSILON) * FLOAT_EPSILON * alpha;
#else
    FLOAT tol = (FLOAT)((orient == TEST_NOTRANSPOSE) ? ((test <= TEST_RVV) ? N : 0) : M * TRANS_EPSILON) / BF16_EPSILON;
#endif
    printf("All %s tests successful from %4d to %4ld (%4ld %8.6f)\n\n", (all == 2) ? "rectangular" : "square", 1, M, N, tol);
  }

  return 0;
}

