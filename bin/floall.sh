#!/bin/bash
shift 1
        
for file in "$@"; do
    echo -n $file "-> "
    (cd generated; ../chisel3/bin/fir2flo.sh $file > $file.out; grep "res = " $file.out)
done
