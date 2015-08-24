; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Top : 
   module Top :
      skip
      when UInt(1) :
         skip
      else :
         skip
      skip

