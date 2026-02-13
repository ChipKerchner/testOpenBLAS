export HOME2="/proj_sw/user_dev/ckerchner"
export OPENBLAS_HOME="${HOME2}/OpenBLAS/kernel/riscv64"
cp shgemm_kernel_8x8_zvl128b.c  ${OPENBLAS_HOME}
cp shgemm_kernel_16x8_zvl256b.c ${OPENBLAS_HOME}
cp sbgemm_kernel_8x8_zvl128b.c  ${OPENBLAS_HOME}
cp sbgemm_kernel_16x8_zvl256b.c ${OPENBLAS_HOME}
exit 1
# Vectorized packing
cp gemm_ncopy_4_rvv.c      ${OPENBLAS_HOME}
cp gemm_ncopy_8_rvv.c      ${OPENBLAS_HOME}
cp gemm_ncopy_16_rvv.c     ${OPENBLAS_HOME}
cp gemm_tcopy_4_rvv.c      ${OPENBLAS_HOME}
cp gemm_tcopy_8_rvv.c      ${OPENBLAS_HOME}
cp gemm_tcopy_16_rvv.c     ${OPENBLAS_HOME}
cp gemm_ncopy_8fp_rvv.c    ${OPENBLAS_HOME}
cp gemm_ncopy_16fp_rvv.c   ${OPENBLAS_HOME}
cp gemm_tcopy_8fp_rvv.c    ${OPENBLAS_HOME}
cp gemm_tcopy_16fp_rvv.c   ${OPENBLAS_HOME}
cp KERNEL.RISCV64_ZVL128B  ${OPENBLAS_HOME}
cp KERNEL.RISCV64_ZVL256B  ${OPENBLAS_HOME}
# Small kernels
#cp gemm_small_kernel_permit_riscv64.c ${OPENBLAS_HOME}
#cp sgemm_small_kernel_nn_riscv64.c    ${OPENBLAS_HOME}
#cp dgemm_small_kernel_nn_riscv64.c    ${OPENBLAS_HOME}
#cp sgemm_small_kernel_nt_riscv64.c    ${OPENBLAS_HOME}
#cp dgemm_small_kernel_nt_riscv64.c    ${OPENBLAS_HOME}
#cp sgemm_small_kernel_tn_riscv64.c    ${OPENBLAS_HOME}
#cp dgemm_small_kernel_tn_riscv64.c    ${OPENBLAS_HOME}
#cp sgemm_small_kernel_tt_riscv64.c    ${OPENBLAS_HOME}
#cp dgemm_small_kernel_tt_riscv64.c    ${OPENBLAS_HOME}
