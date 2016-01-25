; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    input a : {f:UInt<3>[3]}[2]
    input b : {f:UInt<3>[3]}[2]
    input p : UInt<1>
    input i : UInt<1>
    node x = mux(p,mux(p,a[i],a[1]).f,b[i].f)[2]



;CHECK: Pull Muxes
;CHECK: node x = mux(p, mux(p, a[i].f[2], a[1].f[2]), b[i].f[2])   
;CHECK: Finished Pull Muxes
;CHECK: Done!


