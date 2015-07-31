; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Lower To Ground
circuit Top : 
  module Top : 
    wire a : { w : UInt<42>}
    a.w := UInt(1)
    wire b : { w : UInt<42>, x : UInt<20>}
    b.w := UInt(1)
    b.x := UInt(1)
    a <> b 
    ; CHECK:     a$w := b$w
    ; CHECK-NOT: a$x := b$x
    ; CHECK-NOT: a$y := b$y
    ; CHECK-NOT: b$y := a$y
    ; CHECK-NOT: a$z := b$z
    

    wire c : { x : { y : UInt<1>, z : UInt<1>}}[4]
    c[0].x.z := UInt(1)
    c[1].x.z := UInt(1)
    c[2].x.y := UInt(1)
    c[2].x.z := UInt(1)
    c[3].x.y := UInt(1)
    c[3].x.z := UInt(1)
    wire d : { x : { y : UInt<1>}}[2]
    d[0].x.y := UInt(1)
    d[1].x.y := UInt(1)
    c <> d
    ; CHECK:     c$0$x$y := d$0$x$y
    ; CHECK:     c$1$x$y := d$1$x$y
    ; CHECK-NOT: c$2$x$y := d$2$x$y
    ; CHECK-NOT: c$3$x$y := d$3$x$y
    ; CHECK-NOT: c$0$x$z := d$0$x$z
    ; CHECK-NOT: c$1$x$z := d$1$x$z
    ; CHECK-NOT: c$2$x$z := d$2$x$z
    ; CHECK-NOT: c$3$x$z := d$3$x$z

;CHECK: Finished Lower To Ground
;CHECK: Done!
