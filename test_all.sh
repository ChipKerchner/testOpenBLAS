export SIZE=127

./testOpenBLAS 0 0 $SIZE $SIZE $SIZE
./testOpenBLAS 1 0 $SIZE $SIZE $SIZE
#./testOpenBLAS 2 0 $SIZE $SIZE $SIZE
./testOpenBLAS 0 1 $SIZE $SIZE #SIZE
./testOpenBLAS 1 1 $SIZE $SIZE $SIZE
#./testOpenBLAS 2 1 $SIZE $SIZE $SIZE

