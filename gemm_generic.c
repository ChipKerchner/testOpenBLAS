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

static void GEMV_N_beta(BLASLONG n, FLOAT *output_vector, FLOAT *input_vector, FLOAT beta, BLASLONG inc)
{
  if (inc == 1) {
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
  } else {
    for (BLASLONG j = 0; j < n * inc; j += inc) {
      output_vector[j] = input_vector[j] * beta;
    }
  }
}

#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
int BF16GEMV_T_generic(BLASLONG M, BLASLONG N, FLOAT alpha, IFLOAT *a, BLASLONG lda, IFLOAT *x, BLASLONG inc_x, FLOAT beta, FLOAT *y, BLASLONG inc_y)
{
  BLASLONG ij = 0;
  for (BLASLONG j = 0; j < N; j++, ij += inc_y) {
    BLASLONG line = j * M;
    FLOAT t = 0;
    BLASLONG ii = 0;
    for (BLASLONG i = 0; i < M; i++, ii += inc_x) {
      t += bfloat16tof32(a[line + i]) * bfloat16tof32(x[ii]);
    }
    y[ij] = y[ij] * beta + (t * alpha);
  }
  return 0;
}
#endif

int FP3264GEMV_T_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(N, output, buffer, 1.0, inc_y);
  BLASLONG ij = 0;
  for (BLASLONG j = 0; j < N; j++, ij += inc_y) {
    BLASLONG line = j * M;
    FLOAT t = 0;
    BLASLONG ii = 0;
    for (BLASLONG i = 0; i < M; i++, ii += inc_x) {
      t += input_matrix[line + i] * input_vector[ii];
    }
    output[ij] += (t * alpha);
  }
  return 0;
}

#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
int BF16GEMV_N_generic(BLASLONG M, BLASLONG N, FLOAT alpha, IFLOAT *a, BLASLONG lda, IFLOAT *x, BLASLONG inc_x, FLOAT beta, FLOAT *y, BLASLONG inc_y)
{
  GEMV_N_beta(M, y, y, beta, inc_y);
  BLASLONG ij = 0;
  for (BLASLONG j = 0; j < N; j++, ij += inc_x) {
    BLASLONG line = j * M;
    FLOAT inp = bfloat16tof32(x[ij]) * alpha;
    BLASLONG ii = 0;
    for (BLASLONG i = 0; i < M; i++, ii += inc_y) {
      y[ii] += bfloat16tof32(a[line + i]) * inp;
    }
  }
  return 0;
}
#endif

int FP3264GEMV_N_generic(BLASLONG M, BLASLONG N, BLASLONG dummy1, FLOAT alpha, IFLOAT *input_matrix, BLASLONG lda, IFLOAT *input_vector, BLASLONG inc_x, FLOAT *output, BLASLONG inc_y, FLOAT *buffer)
{
  GEMV_N_beta(M, output, buffer, 1.0, inc_y);
  BLASLONG ij = 0;
  for (BLASLONG j = 0; j < N; j++, ij += inc_x) {
    BLASLONG line = j * M;
    FLOAT inp = input_vector[ij] * alpha;
    BLASLONG ii = 0;
    for (BLASLONG i = 0; i < M; i++, ii += inc_y) {
      output[ii] += input_matrix[line + i] * inp;
    }
  }
  return 0;
}

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

#if defined(TEST_BFLOAT) || defined(TEST_FLOAT16)
int BF16GEMM_NN_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, IFLOAT* A, IFLOAT* B, FLOAT* C, BLASLONG ldc)
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

int BF16GEMM_NT_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, IFLOAT* A, IFLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      BLASLONG line = i * K;
      for (BLASLONG k = 0; k < K; k++) {
        t += bfloat16tof32(A[line + k]) * bfloat16tof32(B[k * N + j]);
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int BF16GEMM_TN_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, IFLOAT* A, IFLOAT* B, FLOAT* C, BLASLONG ldc)
{
  for (BLASLONG j = 0; j < N; j++) {
    BLASLONG line2 = j * K;
    for (BLASLONG i = 0; i < M; i++) {
      FLOAT t = 0;
      for (BLASLONG k = 0; k < K; k++) {
        t += bfloat16tof32(A[k * M + i]) * bfloat16tof32(B[line2 + k]);
      }
      C[i] += (t * alpha);
    }
    C += ldc;
  }
  return 0;
}

int BF16GEMM_TT_generic(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, IFLOAT* A, IFLOAT* B, FLOAT* C, BLASLONG ldc)
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

