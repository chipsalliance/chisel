; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: FIRRTL Parsing Error: Expected a module declaration here.

circuit Statement-in-Circuit : 
   node x : UInt(1)
