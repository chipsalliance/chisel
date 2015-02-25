; RUN: firrtl %s abcd c | tee %s.out | FileCheck %s 

; CHECK: Initialize Registers
         circuit top :
            module top :
               input a : UInt(16)
               input b : UInt(16)
               output z : UInt
               when greater(1, 2) :
                  reg r1: UInt
                  r1.init := UInt(12)
; CHECK:          wire [[R1:gen[0-9]*]] : UInt
; CHECK-NOT:      r1 := [[R1]]
; CHECK:          [[R1]] := Null
; CHECK:          [[R1]] := UInt(12)
; CHECK-NOT:      r1.init := UInt(12)
                  reg r2: UInt
; CHECK:          wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:      r2 := [[R2]]
; CHECK:          [[R2]] := Null

; CHECK:          when reset :
; CHECK-DAG:         r2 := [[R2]]
; CHECK-DAG:         r1 := [[R1]]
               else :
                  reg r1: UInt
                  r1.init := UInt(12)
; CHECK:          wire [[R1:gen[0-9]*]] : UInt
; CHECK-NOT:      r1 := [[R1]]
; CHECK:          [[R1]] := Null
; CHECK:          [[R1]] := UInt(12)
; CHECK-NOT:      r1.init := UInt(12)

                  reg r2: UInt
; CHECK:          wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:      r2 := [[R2]]
; CHECK:          [[R2]] := Null

; CHECK:          when reset :
; CHECK-DAG:         r2 := [[R2]]
; CHECK-DAG:         r1 := [[R1]]

; CHECK: Finished Initialize Registers
