; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module source :
      output bundle : { data : UInt<16>, flip ready : UInt<1> }
   module sink :
      input bundle : { data : UInt<16>, flip ready : UInt<1> }
   module top :
      inst src of source
      inst snk of sink
      snk.bundle := src.bundle

; CHECK: Finished Resolve Genders
