; RUN: firrtl -i %s -o %s.v -X verilog -p ct 2>&1 | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire z : { x : UInt, flip y: SInt}
      z.x := UInt(1)
      z.y := SInt(1)
      node x = z.x            ;CHECK: node x = z@<t:{ x : UInt, flip y : SInt}>.x@<t:UInt>
      node y = z.y            ;CHECK: node y = z@<t:{ x : UInt, flip y : SInt}>.y@<t:SInt>
      wire a : UInt<3>[10]    ;CHECK: wire a : UInt<3>[10]@<t:UInt>@<t:UInt<3>[10]@<t:UInt>>
      a[0] := UInt(1)
      a[1] := UInt(1)
      a[2] := UInt(1)
      a[3] := UInt(1)
      a[4] := UInt(1)
      a[5] := UInt(1)
      a[6] := UInt(1)
      a[7] := UInt(1)
      a[8] := UInt(1)
      a[9] := UInt(1)
      node b = a[2]           ;CHECK: node b = a@<t:UInt<3>[10]@<t:UInt>>[2]@<t:UInt>
      read accessor c = a[UInt(3)] ;CHECK: read accessor c = a@<t:UInt<3>[10]@<t:UInt>>[UInt("h3")@<t:UInt>]
; CHECK: Finished Infer Types

; CHECK: Resolve Genders 
; CHECK: read accessor c = a@<t:UInt<3>[10]@<t:UInt>>[UInt("h3")@<t:UInt>]
; CHECK: Finished Resolve Genders 

