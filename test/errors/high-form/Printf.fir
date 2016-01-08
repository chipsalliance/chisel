; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit Top : 
  module Top : 
    input x : {y : UInt<1>}
    input p : UInt<1>
    input clk : Clock
    printf(clk,p,"Hello World%!\n",x)
    printf(clk,p,"Hello World%")
    printf(clk,p,"Hello World%d %s %h %x",x,x,x)

;CHECK: Bad printf format: "%!"
;CHECK: Bad printf format: trailing "%"
;CHECK: Bad printf format: incorrect number of arguments
;CHECK: Bad printf format: "%h"
;CHECK: Bad printf format: incorrect number of arguments
