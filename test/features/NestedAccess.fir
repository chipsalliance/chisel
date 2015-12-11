; RUN: firrtl -i %s -o %s.v -X verilog -p cw 2>&1 | tee %s.out | FileCheck %s
;CHECK: Expand Connects
circuit Top : 
  module Top : 
    input i : UInt<1>
    input j : UInt<1>
    wire a : { x : UInt<42> flip y : UInt<42>[2]}[2][3]
    wire b : { x : UInt<42> flip y : UInt<42>[2]}
    a[0][0].x <= UInt(0)
    a[0][0].y[0] <= UInt(0)
    a[0][0].y[1] <= UInt(0)
    a[0][1].x <= UInt(0)
    a[0][1].y[0] <= UInt(0)
    a[0][1].y[1] <= UInt(0)
    a[1][0].x <= UInt(0)
    a[1][0].y[0] <= UInt(0)
    a[1][0].y[1] <= UInt(0)
    a[1][1].x <= UInt(0)
    a[1][1].y[0] <= UInt(0)
    a[1][1].y[1] <= UInt(0)
    a[2][0].x <= UInt(0)
    a[2][0].y[0] <= UInt(0)
    a[2][0].y[1] <= UInt(0)
    a[2][1].x <= UInt(0)
    a[2][1].y[0] <= UInt(0)
    a[2][1].y[1] <= UInt(0)
    b.x <= UInt(0)
    a[i][j] <= b
;CHECK: Finished Expand Connects
;CHECK: Done!

