; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Lower To Ground
circuit Top : 
  module Top : 
    wire a : { w : UInt<42>, x : UInt<10>, flip y : UInt<42>, z : SInt<42>}
    wire b : { w : UInt<42>, x : UInt<20>, y : UInt<42>, z : UInt<42>}
    a <> b 
    ; CHECK:     a_w := b_w
    ; CHECK:     a_x := b_x
    ; CHECK-NOT: a_y := b_y
    ; CHECK-NOT: b_y := a_y
    ; CHECK-NOT: a_z := b_z
    

    wire c : { x : { y : UInt<1>, z : UInt<1>}}[4]
    wire d : { x : { y : UInt<1>}}[2]
    c <> d
    ; CHECK:     c_0_x_y := d_0_x_y
    ; CHECK:     c_1_x_y := d_1_x_y
    ; CHECK-NOT: c_2_x_y := d_2_x_y
    ; CHECK-NOT: c_3_x_y := d_3_x_y
    ; CHECK-NOT: c_0_x_z := d_0_x_z
    ; CHECK-NOT: c_1_x_z := d_1_x_z
    ; CHECK-NOT: c_2_x_z := d_2_x_z
    ; CHECK-NOT: c_3_x_z := d_3_x_z

;CHECK: Finished Lower To Ground
;CHECK: Done!
