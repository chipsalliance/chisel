; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Expand Connects
circuit Top : 
  module Top : 
    input a : {f:UInt<3>[3], flip g:UInt<3>[3]}[2]
    input b : {f:UInt<3>[3], flip g:UInt<3>[3]}[2]
    input p : UInt<1>
    input i : UInt<1>
    b[0].g[0] <= UInt(0)
    b[0].g[1] <= UInt(0)
    b[0].g[2] <= UInt(0)
    b[1].g[0] <= UInt(0)
    b[1].g[1] <= UInt(0)
    b[1].g[2] <= UInt(0)
    a[0].g[0] <= UInt(0)
    a[0].g[1] <= UInt(0)
    a[0].g[2] <= UInt(0)
    a[1].g[0] <= UInt(0)
    a[1].g[1] <= UInt(0)
    a[1].g[2] <= UInt(0)
    node x = mux(p,a[i].f,b[i].f)



;CHECK: Finished Expand Connects
;CHECK: Done!


