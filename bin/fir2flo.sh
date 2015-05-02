#!/bin/bash

$HOME/bar/firrtl/utils/bin/firrtl -i $1.fir -o $1.flo -x X # -p tkwTgc
$HOME/bar/chisel3-tests/bin/filter < $1.flo > tmp; mv tmp $1.flo
