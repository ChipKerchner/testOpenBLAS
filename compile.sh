export USE_GCC=0
export USE_ASM=0
export USE_DEBUG=0
export USE_256=1
export USE_BF16=0
export USE_ASCALON=1
export TEST_NAME="testOpenBLAS"

if [[ $USE_GCC -eq 1 ]]; then
export COMPILER="riscv64-unknown-linux-gnu-g++"
else
export COMPILER="riscv64-unknown-linux-gnu-clang++"
fi

if [[ $USE_DEBUG -eq 1 ]]; then
export OPT_FLAGS="-O0 -g"
else
export OPT_FLAGS="-O3"
fi

if [[ $USE_ASM -eq 1 ]]; then
export OPT_FLAGS="${OPT_FLAGS} -S"
export OUTPUT=".s"
else
export OUTPUT=""
fi

if [[ $USE_256 -eq 1 ]]; then
export ARCH_FLAG="rv64gcv_zvl256b_zvfh_zfh"
else
export ARCH_FLAG="rv64gcv_zvl128b_zvfh_zfh"
fi

if [[ $USE_BF16 -eq 1 && $USE_GCC -eq 1 ]]; then
export ARCH_FLAG="${ARCH_FLAG}_zvfbfwma"
fi

if [[ $USE_ASCALON -eq 1 ]]; then
export ARCH_FLAG="${ARCH_FLAG}_zvbb"
fi

export COMMON_FLAGS="${OPT_FLAGS} -march=${ARCH_FLAG} -mrvv-vector-bits=zvl -mabi=lp64d -DEIGEN_RISCV64_USE_RVV10 -Wall -DEIGEN_RISCV64_DEFAULT_LMUL=1"

# Linux
${COMPILER} ${COMMON_FLAGS} ${TEST_NAME}.cpp -o ${TEST_NAME}${OUTPUT}
