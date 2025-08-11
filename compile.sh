export OPT_FLAGS="-O3"
#export OPT_FLAGS="-O0 -g"
export COMMON_FLAGS="${OPT_FLAGS} -march=rv64imafdcv_zvl256b_zvfh_zfh -Wall -DRISCV64_ZVL256B"
#export COMMON_FLAGS="${OPT_FLAGS} -march=rv64imafdcv_zvl128b_zvfh_zfh -Wall -DRISCV64_ZVL128B"
#export COMMON_GFLAGS="${COMMON_FLAGS}"
# Linux
#riscv64-unknown-linux-gnu-g++ ${COMMON_GFLAGS} testOpenBLAS.cpp -o testOpenBLAS
riscv64-unknown-linux-gnu-clang++ ${COMMON_FLAGS} -Wno-vla-cxx-extension testOpenBLAS.cpp -o testOpenBLAS
