; RUN: firrtl %s abcd c | tee %s.out | FileCheck %s 

; CHECK: Initialize Registers
         circuit top :
            module top :
               input a : UInt(16)
               input b : UInt(16)
               output z : UInt

               reg r1 : UInt
; CHECK:       wire [[R1:gen[0-9]*]] : UInt
; CHECK:       [[R1]] := Null

               reg r2 : UInt
               r2.init := UInt(0)
; CHECK-NOT:   r2.init := UInt(0)
; CHECK:       wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:   r2 := [[R2]]
; CHECK:       [[R2]] := Null
; CHECK:       [[R2]] := UInt(0)
      
; CHECK:       when reset :
; CHECK-DAG:      r1 := [[R1]]
; CHECK-DAG:      r2 := [[R2]]

; CHECK: Finished Initialize Registers
