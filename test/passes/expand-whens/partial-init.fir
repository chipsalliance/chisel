; RUN: firrtl %s abcefghipj c | tee %s.out | FileCheck %s 

; CHECK: Expand Whens
circuit top :
   module top :
      reg r : UInt(1)[10]
      r.init.3 := UInt(0)

; CHECK: Finished Expand Whens
