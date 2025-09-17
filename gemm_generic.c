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
      output_vector[j] = input_vector[j] * beta;
    }
  }
}

#ifdef TEST_BFLOAT
int BF16GEMV_T_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(N, output, buffer, 1.0);
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
#endif

int FP3264GEMV_T_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(N, output, buffer, 1.0);
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

#ifdef TEST_BFLOAT
int BF16GEMV_T_RVV(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
#if 0
  IFLOAT *ap, *x;
  FLOAT *y;
#endif
  GEMV_N_beta(N, output, buffer, 1.0);
#if 0
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
#endif
  return 0;
}

int BF16GEMV_N_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(M, output, buffer, 1.0);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT inp = bfloat16tof32(input_vector[j]) * alpha;
    for (BLASLONG i = 0; i < M; i++) {
      output[i] += bfloat16tof32(input_matrix[line + i]) * inp;
    }
  }
  return 0;
}
#endif

int FP3264GEMV_N_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(M, output, buffer, 1.0);
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line = j * M;
    FLOAT inp = input_vector[j] * alpha;
    for (BLASLONG i = 0; i < M; i++) {
      output[i] += input_matrix[line + i] * inp;
    }
  }
  return 0;
}

#ifdef TEST_BFLOAT
int BF16GEMV_N_RVV(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
#if 0
  IFLOAT *ap[4], *xo;
  FLOAT *y;
#endif
  GEMV_N_beta(M, output, buffer, 1.0);
#if 0
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
#endif
  return 0;
}
#endif

void GEMM_beta(BLASLONG M, BLASLONG N, FLOAT *output_vector, FLOAT *input_vector, FLOAT beta)
{
  if (beta == (FLOAT)0) {
    memset(output_vector, 0, sizeof(FLOAT) * M * N);
  } else if (beta == (FLOAT)1) {
    if (output_vector != input_vector) {
      memcpy(output_vector, input_vector, sizeof(FLOAT) * M * N);
    }
  } else {
    for (BLASLONG i = 0; i < M * N; i++) {
       output_vector[i] = input_vector[i] * beta;
    }
  }
}
#endif

int FP3264GEMM_NN_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
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

int FP3264GEMM_NT_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      BLASLONG line = i * K;
      for (BLASLONG k = 0; k < K; k++) {
        t += A[line + k] * B[k * N + j];
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int FP3264GEMM_TN_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line2 = j * K;
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      for (BLASLONG k = 0; k < K; k++) {
        t += A[k * M + i] * B[line2 + k];
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int FP3264GEMM_TT_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc)
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

#ifdef TEST_BFLOAT
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
#endif

