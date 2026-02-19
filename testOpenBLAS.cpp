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

#if (defined(RVV_256) && !defined(USE_256)) || (!defined(RVV_256) && defined(USE_256))
#warning "Compiler size mismatch"
#endif

#ifdef RVV_256
#define TEST_VLEN        "VLEN256"
#else
#define TEST_VLEN        "VLEN128"
#endif

#define TEST_VERIFY     // Verify

#ifdef TEST_VERIFY
#define TEST_PACKING    // Include packing
#define TEST_INITIALIZE // Include initializing
#endif

#define TEST_MATRIX   // Test GEMM
#ifndef TEST_MATRIX
#define TEST_VECTOR   // Test GEMV
#endif

#define TEST_FLOAT    // Test FP32
#ifndef TEST_FLOAT
#define TEST_DOUBLE   // Test FP64
#ifndef TEST_DOUBLE
#define TEST_BFLOAT   // Test BF16
#ifdef TEST_BFLOAT
#define BFLOAT16
#define BF16_WIDEN_ONE // Widen arrays first and use FP32
#ifndef TEST_VERIFY
#define BF16_DONT_CONV // Don't convert BF16 to FP32
#endif
#else
#define TEST_FLOAT16  // Test FP16
#define HFLOAT16
#define FP16_NARROW   // Accumulate in FP16 and widen at end
#endif
#else
#define DOUBLE
#endif
#endif

#define GEMM_SWITCH_INPUT // Switch M & N inputs
#define GEMM_RIGHT_EDGE   // One pass on right edge
#define GEMM_BOTTOM_EDGE  // One pass on bottom edge

#ifdef TEST_VECTOR
//#define VERIFY_MATRIX        // Verfiy GEMV versus GEMM
#endif

#if defined(TEST_MATRIX) && defined(TEST_PACKING) && defined(TEST_INITIALIZE)
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)    // Temp
//#define TEST_SMALL_MATRIX    // Use small matrix (no packing)
#endif
#endif

#ifdef VERIFY_MATRIX
//#define VERIFY_OPENBLAS      // Verify versus OpenBLAS code
#endif

#define VECTORIZE_PACK  // Use vectorize packing (if available)
#define VECTORIZE_MEMSET // Since there is no vectorized memset yet

#ifdef VECTORIZE_PACK
#ifndef TEST_DOUBLE
#define VECTORIZE_PACK_N     // Vectorize n_copy
#endif
#define VECTORIZE_PACK_T     // Vectorize t_copy
#endif

//#define FASTER_GENERIC_C     // Pack data and use cache friendly algorithm

#ifdef RVV_256
#define RVV_VLENB        32
#define RISCV64_ZVL256B

#if __riscv_v_min_vlen != 256
#warning "Better to compile with ZVL256B"
#endif
#else
#define RVV_VLENB        16
#define RISCV64_ZVL128B

#if __riscv_v_min_vlen != 128
#warning "Better to compile with ZVL128B"
#endif
#endif

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
#define TEST_SIZE        511
#endif

#define TEST_ITER        100

#define TEST_ALPHA       2.0

#define TEST_INC         1

#define TEST_SET_SEED    // Set random number generator seed to time

#ifndef TEST_FLOAT16
#define bfloat16         __bf16
#else
#define bfloat16         _Float16
#endif

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
#ifndef TEST_FLOAT16
#ifdef BF16_WIDEN_ONE
#define TEST_STR         "BF16_W"
#else
#define TEST_STR         "BF16"
#endif
#else
#ifdef FP16_NARROW
#define TEST_STR         "FP16_N"
#else
#define TEST_STR         "FP16"
#endif
#endif
#define IFLOAT           bfloat16
#define GEMM_UNROLL_N    8
#ifdef RVV_256
#define GEMM_UNROLL_M    16
#else
#define GEMM_UNROLL_M    8
#endif
#endif
#if defined(TEST_FLOAT) || defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
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

#ifdef TEST_BFLOAT
#define FLOAT_EPSILON    ((FLOAT)(1) / (FLOAT)(1 << 7))
#elif defined(TEST_FLOAT16)
#ifdef FP16_NARROW
#define FLOAT_EPSILON    ((FLOAT)(4) / (FLOAT)(1 << 10))
#else
#define FLOAT_EPSILON    ((FLOAT)(1) / (FLOAT)(1 << 10))
#endif
#elif defined(TEST_FLOAT)
#define FLOAT_EPSILON    (FLOAT)(FLT_EPSILON)
#else
#define FLOAT_EPSILON    (FLOAT)(DBL_EPSILON)
#endif
#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
#define TRANS_EPSILON    1
#else
#define TRANS_EPSILON    8
#endif

#ifdef TEST_DOUBLE
#ifdef RVV_256
#define SET_N            8
#else
#define SET_N            4
#endif
#else
#ifdef RVV_256
#define SET_N            16
#else
#define SET_N            8
#endif
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define NBMAX            4096

#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
typedef int funcGEMV(BLASLONG, BLASLONG, FLOAT, IFLOAT *, BLASLONG, IFLOAT *, BLASLONG, FLOAT, FLOAT *, BLASLONG);
#else
typedef int funcGEMV(BLASLONG, BLASLONG, BLASLONG, FLOAT, IFLOAT *, BLASLONG, IFLOAT *, BLASLONG, FLOAT *, BLASLONG, FLOAT *);
#endif
typedef int funcGEMM(BLASLONG, BLASLONG, BLASLONG, FLOAT, IFLOAT *, IFLOAT *, FLOAT *, BLASLONG);
typedef int funcPACK(BLASLONG , BLASLONG, IFLOAT *, BLASLONG, IFLOAT *);

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

#include "gemm_copy_kernel.h"

#ifdef B0
typedef int funcGEMMsmall(BLASLONG, BLASLONG, BLASLONG, IFLOAT *, BLASLONG, FLOAT, IFLOAT *, BLASLONG, FLOAT *, BLASLONG);
#else
typedef int funcGEMMsmall(BLASLONG, BLASLONG, BLASLONG, IFLOAT *, BLASLONG, FLOAT, IFLOAT *, BLASLONG, FLOAT, FLOAT *, BLASLONG);
#endif

#ifdef TEST_MATRIX
#ifdef B0
#define TEST_BETA        0.0
#else
#define TEST_BETA        1.0
#endif
#else
#define TEST_BETA        4.0
#endif

typedef union {
  float f;
  uint32_t i;
  bfloat16 q[2];
} data;

typedef union {
  bfloat16 f;
  uint16_t s;
} data2;

FORCEINLINE float
bfloat16tof32(bfloat16 f16)
{
  data result;
#ifndef TEST_FLOAT16
  result.i = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  result.q[0] = f16;
#else
  result.q[1] = f16;
#endif
#else
  data2 result2;
  result2.f = f16;
  uint32_t sign = result2.s >> 15;
  uint32_t exponent = (result2.s >> 10) & 0x1F;
  uint32_t fraction = (result2.s & 0x3FF);

  if (exponent == 0) {
    if (fraction == 0) {
      // zero
      result.i = (sign << 31);
    } else {
      // can be represented as ordinary value in float32
      // 2 ** -14 * 0.0101
      // => 2 ** -16 * 1.0100
      exponent = 127 - 14;
      while ((fraction & (1 << 10)) == 0) {
        exponent--;
        fraction <<= 1;
      }
      fraction &= 0x3FF;
      result.i = (sign << 31) | (exponent << 23) | (fraction << 13);
    }
  } else if (exponent == 0x1F) {
    /* Inf or NaN */
    result.i = (sign << 31) | (0xFF << 23) | (fraction << 13);
  } else {
    /* ordinary number */
    result.i = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
  }
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

#include "gemm_generic.c"

#ifdef TEST_MATRIX
typedef funcGEMM func;
#else
typedef funcGEMV func;
#endif

func *func_ptr(int test, int orient, int orient2)
{
  switch (orient) {
    case TEST_NOTRANSPOSE:
      switch (test) {
        case TEST_GENERIC:
#ifdef TEST_MATRIX
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          if (orient2 == TEST_NOTRANSPOSE) {
            return FP3264GEMM_NN_generic;
          } else {
            return FP3264GEMM_NT_generic;
          }
#else
          if (orient2 == TEST_NOTRANSPOSE) {
            return BF16GEMM_NN_generic;
          } else {
            return BF16GEMM_NT_generic;
          }
#endif
#else
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMV_N_generic;
#else
          return BF16GEMV_N_generic;
#endif
#endif
          break;
        case TEST_RVV:
#ifdef TEST_MATRIX
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMM_N_RVV;
#else
          return BF16GEMM_N_RVV;
#endif
#else
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMV_N_RVV;
#else
          return BF16GEMV_N_RVV;
#endif
#endif
          break;
      }
      break;
    case TEST_TRANSPOSE:
      switch (test) {
        case TEST_GENERIC:
#ifdef TEST_MATRIX
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          if (orient2 == TEST_NOTRANSPOSE) {
            return FP3264GEMM_TN_generic;
          } else {
            return FP3264GEMM_TT_generic;
          }
#else
          if (orient2 == TEST_NOTRANSPOSE) {
            return BF16GEMM_TN_generic;
          } else {
            return BF16GEMM_TT_generic;
          }
#endif
#else
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMV_T_generic;
#else
          return BF16GEMV_T_generic;
#endif
#endif
          break;
        case TEST_RVV:
#ifdef TEST_MATRIX
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMM_N_RVV;
#else
          return BF16GEMM_N_RVV;
#endif
#else
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
          return FP3264GEMV_T_RVV;
#else
          return BF16GEMV_T_RVV;
#endif
#endif
          break;
      }
      break;
  }
  return NULL;
}

funcGEMMsmall *func_small_ptr(int test, int orient, int orient2)
{
  switch (orient) {
    case TEST_NOTRANSPOSE:
      switch (test) {
#ifdef TEST_SMALL_MATRIX
        case TEST_RVV_SMALL:
          if (orient2 == TEST_NOTRANSPOSE) {
            return FP3264GEMM_TN_RVV_SMALL;
          } else {
            return FP3264GEMM_TT_RVV_SMALL;
          }
#endif
      }
      break;
    case TEST_TRANSPOSE:
      switch (test) {
#ifdef TEST_SMALL_MATRIX
        case TEST_RVV_SMALL:
          if (orient2 == TEST_NOTRANSPOSE) {
            return FP3264GEMM_NN_RVV_SMALL;
          } else {
            return FP3264GEMM_NT_RVV_SMALL;
          }
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
#ifndef TEST_FLOAT16
  inp1.i += ((inp1.i >> 16) & 0x1) + 0x7fff;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return inp1.q[0];
#else
  return inp1.q[1];
#endif
#else
  data2 out;
  out.s = ((inp1.i >> 16) & 0x8000) |
    ((((inp1.i & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
    ((inp1.i >> 13) & 0x03ff);
  return out.f;
#endif
}

unsigned int openblas_seed = 1;

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

FORCEINLINE void init_array2(BLASLONG i, int y, FLOAT *input_array0)
{
  FLOAT x = get_rand(y);
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
  input_array0[i] = x;
#else
  input_array0[i] = bfloat16tof32(float32tobf16(x));
#endif
}

#ifdef TEST_MATRIX
FORCEINLINE void init_array(BLASLONG i, int y, IFLOAT *input_array0)
{
  FLOAT x = get_rand(y);
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
  input_array0[i] = x;
#else
  input_array0[i] = float32tobf16(x);
#endif
}

#ifdef VECTORIZE_MEMSET
#if defined(TEST_FLOAT) || defined(TEST_DOUBLE)
#define SET_DATA  uint32_t
#define SET_VEC   __riscv_vse32_v_u32m8
#define SET_MOVE  __riscv_vmv_v_x_u32m8
#define SET_Z     vuint32m8_t
#else
#define SET_DATA  uint16_t
#define SET_VEC   __riscv_vse16_v_u16m8
#define SET_MOVE  __riscv_vmv_v_x_u16m8
#define SET_Z     vuint16m8_t
#endif
#define SET_SIZE  sizeof(SET_DATA)

#define SET_DATA8 uint8_t
#define SET_VEC8  __riscv_vse8_v_u8m8
#if defined(TEST_FLOAT) || defined(TEST_DOUBLE)
#define SET_MOVE8 __riscv_vreinterpret_v_u32m8_u8m8
#else
#define SET_MOVE8 __riscv_vreinterpret_v_u16m8_u8m8
#endif
#define SET_Z8    vuint8m8_t

#ifdef RVV_256
#define VECTOR_BYTES (32 / SET_SIZE)
#else
#define VECTOR_BYTES (16 / SET_SIZE)
#endif

void memset_zero(void *input, BLASLONG size, bool dir)
{
  if (size) {
    const BLASLONG gvl = VECTOR_BYTES;

    SET_Z z = SET_MOVE(0, gvl);

    if (dir) {
       SET_DATA *input2 = (SET_DATA *)(input);
       while (size >= (gvl * 16 * SET_SIZE)) {
         SET_VEC(input2 + (0 * gvl), z, gvl);
         SET_VEC(input2 + (1 * gvl), z, gvl);
         SET_VEC(input2 + (2 * gvl), z, gvl);
         SET_VEC(input2 + (3 * gvl), z, gvl);
         SET_VEC(input2 + (4 * gvl), z, gvl);
         SET_VEC(input2 + (5 * gvl), z, gvl);
         SET_VEC(input2 + (6 * gvl), z, gvl);
         SET_VEC(input2 + (7 * gvl), z, gvl);
         SET_VEC(input2 + (8 * gvl), z, gvl);
         SET_VEC(input2 + (9 * gvl), z, gvl);
         SET_VEC(input2 + (10 * gvl), z, gvl);
         SET_VEC(input2 + (11 * gvl), z, gvl);
         SET_VEC(input2 + (12 * gvl), z, gvl);
         SET_VEC(input2 + (13 * gvl), z, gvl);
         SET_VEC(input2 + (14 * gvl), z, gvl);
         SET_VEC(input2 + (15 * gvl), z, gvl);
         input2 += 16 * gvl;
         size -= 16 * (gvl * SET_SIZE);
      }
      while (size >= (gvl * 4 * SET_SIZE)) {
         SET_VEC(input2 + (0 * gvl), z, gvl);
         SET_VEC(input2 + (1 * gvl), z, gvl);
         SET_VEC(input2 + (2 * gvl), z, gvl);
         SET_VEC(input2 + (3 * gvl), z, gvl);
         input2 += 4 * gvl;
         size -= 4 * (gvl * SET_SIZE);
      }
      if (size >= (gvl * SET_SIZE)) {
         SET_VEC(input2, z, gvl);
         input2 += gvl;
         size -= (gvl * SET_SIZE);
      }
      SET_DATA8 *input8 = (SET_DATA8 *)(input2);
      SET_Z8 z8 = SET_MOVE8(z);
      SET_VEC8(input8, z8, size);
    } else {
      SET_DATA *input2 = (SET_DATA *)((unsigned char *)(input) + size);
      while (size >= (gvl * 16 * SET_SIZE)) {
         SET_VEC(input2 - (1 * gvl), z, gvl);
         SET_VEC(input2 - (2 * gvl), z, gvl);
         SET_VEC(input2 - (3 * gvl), z, gvl);
         SET_VEC(input2 - (4 * gvl), z, gvl);
         SET_VEC(input2 - (5 * gvl), z, gvl);
         SET_VEC(input2 - (6 * gvl), z, gvl);
         SET_VEC(input2 - (7 * gvl), z, gvl);
         SET_VEC(input2 - (8 * gvl), z, gvl);
         SET_VEC(input2 - (9 * gvl), z, gvl);
         SET_VEC(input2 - (10 * gvl), z, gvl);
         SET_VEC(input2 - (11 * gvl), z, gvl);
         SET_VEC(input2 - (12 * gvl), z, gvl);
         SET_VEC(input2 - (13 * gvl), z, gvl);
         SET_VEC(input2 - (14 * gvl), z, gvl);
         SET_VEC(input2 - (15 * gvl), z, gvl);
         SET_VEC(input2 - (16 * gvl), z, gvl);
         input2 -= 16 * gvl;
         size -= 16 * (gvl * SET_SIZE);
      }
      while (size >= (gvl * 4 * SET_SIZE)) {
         SET_VEC(input2 - (1 * gvl), z, gvl);
         SET_VEC(input2 - (2 * gvl), z, gvl);
         SET_VEC(input2 - (3 * gvl), z, gvl);
         SET_VEC(input2 - (4 * gvl), z, gvl);
         input2 -= 4 * gvl;
         size -= 4 * (gvl * SET_SIZE);
      }
      if (size >= (gvl * SET_SIZE)) {
         SET_VEC(input2 - (1 * gvl), z, gvl);
         input2 -= gvl;
         size -= (gvl * SET_SIZE);
      }
      SET_DATA8 *input8 = (SET_DATA8 *)((unsigned char *)(input2) - size);
      SET_Z8 z8 = SET_MOVE8(z);
      SET_VEC8(input8, z8, size);
    }
  }
}
#else
#define memset_zero(ptr, size, dir)  memset(ptr, 0, size)
#endif

void init(IFLOAT *input_matrix, IFLOAT *input_matrix2, FLOAT *output_matrix,
          BLASLONG M, BLASLONG N, BLASLONG K)
{
#ifdef TEST_INITIALIZE
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
#else
  memset(input_matrix, 0, M * K * sizeof(IFLOAT));
  memset(input_matrix2, 0, N * K * sizeof(IFLOAT));
#endif
}

int verifyOut(FLOAT *output0, FLOAT *output1, FLOAT tol, BLASLONG M, BLASLONG N, BLASLONG K, const char *str, int orient, int orient2, FLOAT *err)
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
  *err = maxOut;
  if (maxOut > tol) {
    fprintf(stderr, "Bad %s %s result %13.4f %13.4f %4ld %4ld (%8.6f %8.6f %4ld %4ld %4ld %d %d - %10u)\n\n", TEST_STR, str, output0[(n * M) + m], output1[(n * M) + m], m, n, maxOut, tol, M, N, K, orient, orient2, openblas_seed);
    return 1;
  }
  return 0;
}
#else
FORCEINLINE void init_array(BLASLONG i, int y, IFLOAT *input_array0, IFLOAT *input_array1)
{
  FLOAT x = get_rand(y);
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
  input_array0[i] = input_array1[i] = x;
#else
  input_array0[i] = input_array1[i] = float32tobf16(x);
#endif
}

void init(IFLOAT *input_matrix, IFLOAT *input_matrix2, IFLOAT *input_vector, IFLOAT *input_vector1,
          BLASLONG M, BLASLONG N, BLASLONG in, FLOAT *input, BLASLONG out, BLASLONG inc)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    for (BLASLONG i = 0; i < M; i++) {
      init_array(line + i, rand_value(), input_matrix, input_matrix2);
    }
  }
  for (BLASLONG j = 0; j < in * inc; j += inc) {
    init_array(j, rand_value(), input_vector, input_vector1);
  }
  for (BLASLONG j = 0; j < out * inc; j += inc) {
    init_array2(j, rand_value(), input);
  }
}

int verifyOut(FLOAT *output0, FLOAT *output1, BLASLONG out, FLOAT tol, BLASLONG size, BLASLONG size2, const char *str, int orient, BLASLONG inc, FLOAT *err)
{
  FLOAT maxOut = (FLOAT)0;
  BLASLONG i = 0;
  for (BLASLONG j = 0; j < out * inc; j += inc) {
    FLOAT diff = fabs(output0[j] - output1[j]);
    if (diff > maxOut) {
      maxOut = diff;
      i = j;
    }
  }
  *err = maxOut;
  if (maxOut > tol) {
    fprintf(stderr, "Bad %s %s result %13.4f %13.4f %4ld (%8.6f %8.6f %4ld %4ld %d)\n\n", TEST_STR, str, output0[i], output1[i], i, maxOut, tol, size, size2, orient);
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
int verify(int test, int orient, int orient2, BLASLONG M, BLASLONG N, BLASLONG K, IFLOAT *input_matrix0,
           IFLOAT *input_matrix1, FLOAT *output0, FLOAT *output1, FLOAT alpha, FLOAT *err)
#else
int verify(int test, int orient, int orient2, BLASLONG M, BLASLONG N, BLASLONG out, IFLOAT *input_matrix1,
           IFLOAT *input_vector1, FLOAT *output0, FLOAT *output1, FLOAT *output2, FLOAT alpha, FLOAT beta,
           FLOAT *input, BLASLONG inc, FLOAT *err)
#endif
{
  FLOAT tol = (FLOAT)(K * TRANS_EPSILON) * FLOAT_EPSILON * alpha;

#ifdef TEST_MATRIX
  if (verifyOut(output0, output1, tol, M, N, K, TEST_TYPE, orient, orient2, err)) {
#else
  if (verifyOut(output0, output1, out, tol, M, N, TEST_TYPE, orient, inc, err)) {
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
  GEMV_N_beta(out, output2, input, beta);

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
  int orient2 = TEST_TRANSPOSE;
  BLASLONG M = TEST_SIZE;
  BLASLONG N = TEST_SIZE;
  BLASLONG K = TEST_SIZE;
  int iter = TEST_ITER;
  FLOAT alpha = TEST_ALPHA;
  FLOAT beta = TEST_BETA;
  BLASLONG inc = TEST_INC;
#if defined(TEST_PACKING) && defined(TEST_INITIALIZE)
  FLOAT err = 0;
#endif
  int all = 0;

#ifdef RVV_256
  if (RVV_VLENB > __riscv_vlenb()) {
    fprintf(stderr, "Compiled for RVV_256\n");
    return 1;
  }
#endif

  if (argc > 1) {
    test = atoi(argv[1]);
    if (argc > 2) {
      orient = atoi(argv[2]);
      if (argc > 3) {
#ifdef TEST_SMALL_MATRIX
        orient2 = atoi(argv[3]);
#else
        orient2 = orient;
#endif
        if (argc > 4) {
          M = atol(argv[4]);
          if (argc > 5) {
            N = atol(argv[5]);
            if (argc > 6) {
              K = atol(argv[6]);
              if (argc > 7) {
                iter = atoi(argv[7]);
                if (argc > 8) {
                  alpha = atof(argv[8]);
                  if (argc > 9) {
#ifndef TEST_MATRIX
                    beta = atof(argv[9]);
#endif
                    if (argc > 10) {
                      inc = atol(argv[10]);
                    }
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
      (orient > TEST_TRANSPOSE) || (orient2 < TEST_NOTRANSPOSE) || (orient2 < TEST_NOTRANSPOSE) ||
      (iter <= 0) || (inc < 1)) {
    fprintf(stderr, "testOpenBLAS (%s %s) ", TEST_STR, TEST_TYPE);
    fprintf(stderr, "type [orient orient2 M N K iter alpha beta inc]\n\n");
    fprintf(stderr, "type = 0 (generic C = default), 1 (RVV)\n");
#ifdef VERIFY_OPENBLAS
    fprintf(stderr, "type = 2 (generic - OpenBLAS)\n");
#endif
    fprintf(stderr, "orient = 0 (no transpose), 1 (transpose = default)\n");
    fprintf(stderr, "orient2 = only used in small matrices\n");
    fprintf(stderr, "iter = test iterations.\n");
    fprintf(stderr, "inc = incx & incy.\n");
    fprintf(stderr, "if M < 0 all Ms up to negative M are tested.\n");
    return 1;
  }

  printf("Testing %s %s %s %s ", COMP_STR, TEST_STR, TEST_VLEN, TEST_TYPE);
#ifdef TEST_MATRIX
#if !defined(TEST_PACKING) || !defined(TEST_INITIALIZE)
  printf("NO_VERIFY ");
#endif
#endif
  const char *release = __DATE__ " " __TIME__;
  printf("%d %d %d %4ld %4ld %4ld %3d %4.1f %4.1f %2ld %s\n", test, orient, orient2, M, N, K, iter, alpha, beta, inc, release);
#ifdef TEST_VERIFY
  printf("\n");
#endif

#ifdef GEMM_SWITCH_INPUT
  BLASLONG swap = M;
  M = N;
  N = swap;
#endif

#ifdef TEST_SET_SEED
  const char *openblas_str = getenv("OPENBLAS_SEED");
  if (openblas_str) {
    openblas_seed = atoi(openblas_str);
  } else {
    openblas_seed = (unsigned int)(get_rvv_timer());
  }
#endif
  rand_seed(openblas_seed);

  func *gen_ptr = func_ptr(TEST_GENERIC, orient, orient2);
  func *test_ptr = func_ptr(test, orient, orient2);
#ifdef TEST_SMALL_MATRIX
  funcGEMMsmall *small_ptr = func_small_ptr(TEST_RVV_SMALL, orient, orient2);
#endif
#ifdef VERIFY_OPENBLAS
#ifdef TEST_MATRIX
  funcGEMM *test2_ptr = ((test == TEST_OPENBLAS) ? ((orient == TEST_NOTRANSPOSE) ? NEW_BF16_GEMV_N : NEW_BF16_GEMV_T) : NULL);
#else
  funcGEMV *test2_ptr = ((test == TEST_OPENBLAS) ? ((orient == TEST_NOTRANSPOSE) ? NEW_BF16_GEMV_N : NEW_BF16_GEMV_T) : NULL);
#endif
  funcGEMM *test3_ptr = ((test == TEST_OPENBLAS) ? OLD_BF16_GEMM : NULL);
#endif
#ifdef TEST_MATRIX
#ifdef TEST_PACKING
  funcPACK *packm_ptr, *packn_ptr;
#if !defined(TEST_BFLOAT) && !defined(TEST_FLOAT16)
  if (orient == orient2) {
    packm_ptr = ((orient != TEST_NOTRANSPOSE) ? FP3264_PACK_MT : FP3264_PACK_MN);
    packn_ptr = ((orient != TEST_NOTRANSPOSE) ? FP3264_PACK_NT : FP3264_PACK_NN);
  } else {
    packm_ptr = ((orient2 != TEST_NOTRANSPOSE) ? FP3264_PACK_MN : FP3264_PACK_MT);
    packn_ptr = ((orient2 != TEST_NOTRANSPOSE) ? FP3264_PACK_NT : FP3264_PACK_NN);
  }
#else
  if (orient == orient2) {
    packm_ptr = ((orient != TEST_NOTRANSPOSE) ? BF16_PACK_MT : BF16_PACK_MN);
    packn_ptr = ((orient != TEST_NOTRANSPOSE) ? BF16_PACK_NT : BF16_PACK_NN);
  } else {
    packm_ptr = ((orient2 != TEST_NOTRANSPOSE) ? BF16_PACK_MN : BF16_PACK_MT);
    packn_ptr = ((orient2 != TEST_NOTRANSPOSE) ? BF16_PACK_NT : BF16_PACK_NN);
  }
#endif
#endif
#endif

  if ((gen_ptr == NULL) ||
#ifdef VERIFY_OPENBLAS
     ((test_ptr == NULL) &&
     ((test == TEST_OPENBLAS) && (test2_ptr == NULL)) ||
     ((test == TEST_OPENBLAS) && (test3_ptr == NULL)))
#else
#ifdef TEST_SMALL_MATRIX
     (small_ptr == NULL) ||
#endif
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
    IFLOAT input_vector0[in * inc], input_vector1[in * inc];
    FLOAT output0[N0 * inc], output1[N0 * inc], output2[N0 * inc], input[N0 * inc];
#endif
    IFLOAT *input_matrix0 = NULL, *input_matrix1 = NULL;
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
    IFLOAT *input_matrix01 = NULL, *input_matrix11 = NULL;
#endif

#ifdef TEST_MATRIX
    input_matrix0 = (IFLOAT *)malloc(in * K * sizeof(IFLOAT));
#else
    input_matrix0 = (IFLOAT *)malloc(in * out * sizeof(IFLOAT));
#endif
    if (input_matrix0 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
#ifdef TEST_MATRIX
    input_matrix1 = (IFLOAT *)malloc(K * out * sizeof(IFLOAT));
#else
    input_matrix1 = (IFLOAT *)malloc(in * out * sizeof(IFLOAT));
#endif
    if (input_matrix1 == NULL) {
      fprintf(stderr, "Bad malloc\n");
      return 1;
    }
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
#ifdef FASTER_GENERIC_C
    if (test >= TEST_GENERIC) {
#else
    if (test == TEST_RVV) {
#endif
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
    init(input_matrix0, input_matrix1, input_vector0, input_vector1, M0, N0, in, input, N0, inc);
#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
    memcpy(output0, input, N0 * inc * sizeof(FLOAT));
#endif
#endif
#ifdef TEST_INITIALIZE
#ifdef TEST_MATRIX
    gen_ptr(in, out, K, alpha, input_matrix0, input_matrix1, output_matrix0, in);
#else
#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
    gen_ptr(in, out, alpha, input_matrix0, in, input_vector0, inc, beta, output0, inc);
#else
    gen_ptr(in, out, K, alpha, input_matrix0, in, input_vector0, inc, output0, inc, input);
#endif
#endif
#endif

    timer_t start;
    {
#ifdef TEST_INITIALIZE
    bool warmup = true;
#else
    bool warmup = false;
#ifdef TEST_MATRIX
    memset(output_matrix1, 0, M0 * N0 * sizeof(FLOAT));
#ifdef FASTER_GENERIC_C
    if (test >= TEST_GENERIC) {
#else
    if (test == TEST_RVV) {
#endif
      memset(input_matrix01, 0, in * K * sizeof(IFLOAT));
      memset(input_matrix11, 0, K * out * sizeof(IFLOAT));
    }
#endif
#endif
again:
    int stop = (all || warmup) ? 1 : (int)iter;
    start = get_rvv_timer();
    for (int i = 0; i < stop; i++) {

      if (test <= TEST_RVV) {
#ifdef TEST_MATRIX
#ifdef TEST_INITIALIZE
        memcpy(output_matrix1, output_matrix2, M0 * N0 * sizeof(FLOAT));
#else
        memset_zero(output_matrix1, MIN(N0, SET_N) * sizeof(FLOAT), false);
#endif
#ifdef FASTER_GENERIC_C
        if (test >= TEST_GENERIC) {
#else
        if (test == TEST_RVV) {
#endif
#ifdef TEST_SMALL_MATRIX
          if ((GEMM_SMALL_M_PERMIT(orient, orient2, in, out, K, alpha, beta) != 0)) {
#ifdef B0
            if (orient == TEST_TRANSPOSE) {
              small_ptr(in, out, K, input_matrix0, in, alpha, input_matrix1, out, output_matrix1, in);
            } else {
              small_ptr(in, out, K, input_matrix0, K, alpha, input_matrix1, K, output_matrix1, in);
            }
#else
            if (orient == TEST_TRANSPOSE) {
              small_ptr(in, out, K, input_matrix0, in, alpha, input_matrix1, out, beta, output_matrix1, in);
            } else {
              small_ptr(in, out, K, input_matrix0, K, alpha, input_matrix1, K, beta, output_matrix1, in);
            }
#endif
          } else
#endif
          {
#ifdef TEST_PACKING
            if (orient == TEST_TRANSPOSE) {
              packm_ptr(K, in, input_matrix0, in, input_matrix01);
              packn_ptr(K, out, input_matrix1, out, input_matrix11);
            } else {
              packm_ptr(K, in, input_matrix0, K, input_matrix01);
              packn_ptr(K, out, input_matrix1, K, input_matrix11);
            }
#else
#ifdef TEST_INITIALIZE
            memcpy(input_matrix01, input_matrix0, in * K * sizeof(IFLOAT));
            memcpy(input_matrix11, input_matrix1, K * out * sizeof(IFLOAT));
#endif
#endif
            test_ptr(in, out, K, alpha, input_matrix01, input_matrix11, output_matrix1, in);
          }
        } else {
          test_ptr(in, out, K, alpha, input_matrix0, input_matrix1, output_matrix1, in);
        }
#else
#ifdef TEST_INITIALIZE
        memcpy(output1, input, N0 * inc * sizeof(FLOAT));
#endif
#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
        test_ptr(in, out, alpha, input_matrix0, in, input_vector0, inc, beta, output1, inc);
#else
        test_ptr(in, out, K, alpha, input_matrix0, in, input_vector0, inc, output1, inc, input);
#endif
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
    if (warmup) {
      warmup = false;
      goto again;
    }
    }
    timer_t end = get_rvv_timer();
    if (!all) {
      printf("Total time = %16ld\n", end - start);
#ifdef TEST_VERIFY
      printf("\n");
#endif
    }

#if defined(TEST_PACKING) && defined(TEST_INITIALIZE)
#ifdef TEST_MATRIX
    if (verify(test, orient, orient2, M0, N0, K, input_matrix0, input_matrix1, output_matrix0, output_matrix1, alpha, &err)) {
#else
    if (verify(test, orient, orient2, M0, N0, N0, input_matrix1, input_vector1, output0, output1, output2, alpha, beta, input, inc, &err)) {
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
#endif

    free(input_matrix0);
    free(input_matrix1);
#if defined(VERIFY_OPENBLAS) || defined(TEST_MATRIX)
#ifdef FASTER_GENERIC_C
    if (test >= TEST_GENERIC) {
#else
    if (test == TEST_RVV) {
#endif
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

#if defined(TEST_PACKING) && defined(TEST_INITIALIZE)
  if (all) {
    FLOAT tol = (FLOAT)(K * TRANS_EPSILON) * FLOAT_EPSILON * alpha;
    printf("All %s tests successful from %4d to %4ld (%4ld - %8.6f %8.6f - %10u)\n\n", (all == 2) ? "rectangular" : "square", 1, M, N, err, tol, openblas_seed);
  }
#endif

  return 0;
}

