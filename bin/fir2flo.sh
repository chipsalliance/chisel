#!/bin/bash

$CHISEL_BIN/firrtl -i $1.fir -o $1.flo -X flo # -x X # -p c # tkwTgc
if [ $? ] ; then
   $CHISEL_BIN/filter < $1.flo > tmp && mv tmp $1.flo
fi
