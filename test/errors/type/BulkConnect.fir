; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: BulkConnect.fir@10.4: [module Top]  Type mismatch.
;CHECK: BulkConnect.fir@14.4: [module Top]  Type mismatch.
;CHECK: BulkConnect.fir@26.4: [module Top]  Type mismatch.
;CHECK: BulkConnect.fir@30.4: [module Top]  Type mismatch.
circuit Top : 
  module Top : 
    wire a : { w : UInt<42>}
    wire b : { w : SInt<42>}
    a <> b 

    wire c : { w : UInt<10>}
    wire d : { flip w : UInt<12> }
    c <> d

    wire e : { w : UInt<10>}
    wire f : { x : UInt<12> }
    e <> f

    wire g : { w : { y : UInt<10> }}
    wire h : { w : { x : UInt<12> }}
    g <> h

    wire i : { w : { flip y : UInt<10> }}
    wire j : { w : { y : UInt<12> }}
    i <> j

    wire k : { w : { y : SInt<10> }}
    wire l : { w : { y : UInt<12> }}
    k <> l


