; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit Top :
   module Top :
      input with : Clock
      output unsigned : Clock
      wire trireg : Clock
      unsigned <= with
      trireg <= with

;CHECK: Verilog Rename

;CHECK: input with$ : Clock
;CHECK: output unsigned$ : Clock
;CHECK: wire trireg$ : Clock
;CHECK: trireg$ <= with$
;CHECK: unsigned$ <= with$   

;CHECK: Done!
