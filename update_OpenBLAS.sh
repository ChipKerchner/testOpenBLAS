export HOME2="/proj_sw/user_dev/ckerchner"
export OPENBLAS_HOME="${HOME2}/OpenBLAS/kernel/riscv64"
#exit 1
# Vectorized packing
cp ${OPENBLAS_HOME}/gemm_ncopy_4_rvv.c    .
cp ${OPENBLAS_HOME}/gemm_ncopy_8_rvv.c    .
cp ${OPENBLAS_HOME}/gemm_ncopy_16_rvv.c   .
cp ${OPENBLAS_HOME}/gemm_tcopy_4_rvv.c    .
cp ${OPENBLAS_HOME}/gemm_tcopy_8_rvv.c    .
cp ${OPENBLAS_HOME}/gemm_tcopy_16_rvv.c   .
cp ${OPENBLAS_HOME}/gemm_ncopy_8fp_rvv.c  .
cp ${OPENBLAS_HOME}/gemm_ncopy_16fp_rvv.c .
cp ${OPENBLAS_HOME}/gemm_tcopy_8fp_rvv.c  .
cp ${OPENBLAS_HOME}/gemm_tcopy_16fp_rvv.c .
