#!/bin/bash
shift 1

for file in "$@"; do
    echo $file
    (cd generated; ../bin/flo-app.sh $file >& $file.appout)
done
