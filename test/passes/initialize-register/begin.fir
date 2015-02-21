; RUN: firrtl %s abcd | tee %s.out | FileCheck %s 

         circuit top :
            module top :
               input a : UInt(16)
               input b : UInt(16)
               output z : UInt

               reg r1 : UInt
; CHECK:       wire [[R1:gen[0-9]*]] : UInt
; CHECK:       n:[[R1]] := Null

               reg r2 : UInt
               r2.init := UInt(0)
; CHECK:       wire [[R2:gen[0-9]*]] : UInt
; CHECK-NOT:   reg:r2 := n:[[R2]]
; CHECK:       n:[[R2]] := Null
; CHECK:       n:[[R2]] := UInt(0)
      
; CHECK:       when port:reset :
; CHECK-DAG:      reg:r1 := n:[[R1]]
; CHECK-DAG:      reg:r2 := n:[[R2]]
