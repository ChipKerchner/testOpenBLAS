export OPENBLAS_HOME="${HOME}/OpenBLAS/kernel/riscv64"
cp sbgemv_common.c         ${OPENBLAS_HOME}
cp sbgemv_common_power10.c ${OPENBLAS_HOME}
cp sbgemv_n.c              ${OPENBLAS_HOME}
cp sbgemv_n_vsx.c          ${OPENBLAS_HOME}
cp sbgemv_n_power10.c      ${OPENBLAS_HOME}
cp sbgemv_t.c              ${OPENBLAS_HOME}
cp sbgemv_t_vsx.c          ${OPENBLAS_HOME}
cp sbgemv_t_power10.c      ${OPENBLAS_HOME}
cp gemm_common.c           ${OPENBLAS_HOME}

