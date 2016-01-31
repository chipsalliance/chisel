; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Expand Whens
circuit Top : 
  module Top : 
    input x : {y : UInt<1>}
    input p : UInt<1>
    input clk : Clock
    input en : UInt<1>
    printf(clk,en,"Hello World!\n")
    printf(clk,en,"Hello World! %x\n", x.y)
    when p :
       printf(clk,en,"In consequence\n")
    else :
       printf(clk,en,"In alternate\n")

;CHECK: printf(clk, en, "Hello World!\n")
;CHECK: printf(clk, en, "Hello World! %x\n", x.y)
;CHECK: printf(clk, and(p, en), "In consequence\n")
;CHECK: printf(clk, and(eq(p, UInt<1>("h0")), en), "In alternate\n")   
;CHECK: Done!
