#!/bin/bash
shift 1

for file in "$@"; do
    echo $file
    (cd generated; $CHISEL_BIN/flo2app.sh $file >& $file.appout)
done
