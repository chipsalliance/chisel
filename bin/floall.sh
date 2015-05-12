#!/bin/bash
shift 1
for file in "$@"; do
    echo -n $file "-> "
    (cd generated; $CHISEL_BIN/fir2flo.sh $file > $file.out; grep "res = " $file.out)
done
