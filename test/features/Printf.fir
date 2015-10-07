; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower To Ground
circuit Top : 
  module Top : 
    input x : {y : UInt<1>}
    input p : UInt<1>
    printf("Hello World!\n")
    printf("Hello World! %x\n", x.y)
    when p :
       printf("In consequence\n")
    else :
       printf("In alternate\n")

;CHECK: printf("Hello World!\n")
;CHECK: printf("Hello World! %x\n", x$y)
;CHECK: when p : printf("In consequence\n")
;CHECK: when not(p) : printf("In alternate\n")   
;CHECK: Done!
