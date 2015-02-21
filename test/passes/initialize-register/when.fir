; RUN: firrtl %s abcd | tee %s.out | FileCheck %s 
; CHECK: circuit top :
         circuit top :
            module top :
               input a : UInt(16)
               input b : UInt(16)
               output z : UInt
               when greater(1, 2) :
                  reg r1: UInt
                  r1.init := UInt(12)
; CHECK:          wire [[R1:gen[0-9]*]] : UInt
; CHECK-NOT:      reg:r1 := n:[[R1]]
; CHECK:          n:[[R1]] := Null
; CHECK:          n:[[R1]] := UInt(12)
; CHECK-NOT:      r1.init := UInt(12)
                  reg r2: UInt
; CHECK:          wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:      reg:r2 := n:[[R2]]
; CHECK:          n:[[R2]] := Null

; CHECK:          when port:reset :
; CHECK-DAG:         reg:r2 := n:[[R2]]
; CHECK-DAG:         reg:r1 := n:[[R1]]
               else :
                  reg r1: UInt
                  r1.init := UInt(12)
; CHECK:          wire [[R1:gen[0-9]*]] : UInt
; CHECK-NOT:      reg:r1 := n:[[R1]]
; CHECK:          n:[[R1]] := Null
; CHECK:          n:[[R1]] := UInt(12)
; CHECK-NOT:      r1.init := UInt(12)

                  reg r2: UInt
; CHECK:          wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:      reg:r2 := n:[[R2]]
; CHECK:          n:[[R2]] := Null

; CHECK:          when port:reset :
; CHECK-DAG:         reg:r2 := n:[[R2]]
; CHECK-DAG:         reg:r1 := n:[[R1]]
