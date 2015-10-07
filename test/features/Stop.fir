; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Lower To Ground

circuit Top : 
  module Top : 
    input p : UInt<1>
    input q : UInt<1>
    when p :
       stop(0)
    when q :
       stop(1)
    stop(3)

;CHECK: when p : stop(0)
;CHECK: when q : stop(1)
;CHECK: stop(3)   

;CHECK: Done!

