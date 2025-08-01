export ARCH=1  # RVV
export SIZE=127

# Test big matrix
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 2.0 4.0
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 2.0 4.0

# Test sizes (square matrices)
./testOpenBLAS $ARCH 0 -384
./testOpenBLAS $ARCH 1 -384

# Test alphas
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 0.0 4.0
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 0.0 4.0
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 1.0 4.0
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 1.0 4.0

# Test betas
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 2.0 0.0
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 2.0 0.0
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 2.0 1.0
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 2.0 1.0

# Test retangular matrices
./testOpenBLAS $ARCH 0 -384 -512
./testOpenBLAS $ARCH 1 -384 -512

# Test inc != 1
./testOpenBLAS $ARCH 0 $SIZE $SIZE $SIZE 1 2.0 4.0 2
./testOpenBLAS $ARCH 1 $SIZE $SIZE $SIZE 1 2.0 4.0 2
