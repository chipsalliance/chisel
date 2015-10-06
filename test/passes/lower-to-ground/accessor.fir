; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      input clk : Clock
      wire i : UInt<2>
      wire j : UInt<32>

      wire a : UInt<32>[4]
      ; CHECK: wire a{{[_$]+}}0 : UInt<32>
      ; CHECK: wire a{{[_$]+}}1 : UInt<32>
      ; CHECK: wire a{{[_$]+}}2 : UInt<32>
      ; CHECK: wire a{{[_$]+}}3 : UInt<32>

      infer accessor b = a[i]
      ; CHECK: indexer b = (a{{[_$]+}}0 a{{[_$]+}}1 a{{[_$]+}}2 a{{[_$]+}}3)[i] : UInt<32>
      j := b

      infer accessor c = a[i]
      ; CHECK: indexer (a{{[_$]+}}0 a{{[_$]+}}1 a{{[_$]+}}2 a{{[_$]+}}3)[i] = c : UInt<32>
      c := j
      
      cmem p : UInt<32>[4],clk
      infer accessor t = p[i]
      ; CHECK: read accessor t = p[i]
      j := t
      infer accessor r = p[i]
      ; CHECK: write accessor r = p[i]
      r := j

; CHECK: Finished Lower To Ground
