; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower To Ground
circuit Top : 
  module Top : 
    input x : {y : UInt<1>}
    input p : UInt<1>
    input clk : Clock
    printf(clk,"Hello World!\n")
    printf(clk,"Hello World! %x\n", x.y)
    when p :
       printf(clk,"In consequence\n")
    else :
       printf(clk,"In alternate\n")

;CHECK: printf(clk, "Hello World!\n")
;CHECK: printf(clk, "Hello World! %x\n", x$y)
;CHECK: when p : printf(clk, "In consequence\n")
;CHECK: when not(p) : printf(clk, "In alternate\n")   
;CHECK: Done!
