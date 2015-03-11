; RUN: firrtl %s abcdef cg | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module source :
      output bundle : { male data : UInt(16), female ready : UInt(1) }
   module sink :
      input bundle : { male data : UInt(16), female ready : UInt(1) }
   module top :
      inst src of source
      inst snk of sink
      snk.bundle := src.bundle

; CHECK: Finished Resolve Genders
