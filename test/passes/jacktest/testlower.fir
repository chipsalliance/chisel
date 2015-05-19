; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens

circuit BundleWire : 
  module BundleWire : 
    input in : { y : UInt<32>, x : UInt<32> }
    output outs : { y : UInt<32>, x : UInt<32> }[4]
    
    wire coords : { y : UInt<32>, x : UInt<32> }[4]
    coords[0] := in
    outs[0] := coords[0]
    coords[1] := in
    outs[1] := coords[1]
    coords[2] := in
    outs[2] := coords[2]
    coords[3] := in
    outs[3] := coords[3]

; CHECK: Finished Expand Whens
