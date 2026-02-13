export ARCH=1  # RVV
export SIZE=127

# Test big matrix
./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 2.0 4.0
./testOpenBLAS $ARCH 1 1 $SIZE $SIZE $SIZE 1 2.0 4.0

# Test sizes (square matrices)
./testOpenBLAS $ARCH 0 0 -192
./testOpenBLAS $ARCH 1 1 -192

# Test alphas
./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 0.0 4.0
./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 1.0 4.0

# Test betas
#./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 2.0 0.0
#./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 2.0 1.0

# Test retangular matrices
./testOpenBLAS $ARCH 0 0 -128 -192
./testOpenBLAS $ARCH 1 1 -128 -192

# Test inc != 1
./testOpenBLAS $ARCH 0 0 $SIZE $SIZE $SIZE 1 2.0 4.0 2

# Test small
./testOpenBLAS $ARCH 0 0 64 64 64 1 2.0 4.0
./testOpenBLAS $ARCH 1 1 64 64 64 1 2.0 4.0
