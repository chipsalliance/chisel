; RUN: firrtl -i %s -o %s.flo -X flo -p ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire z : { x : UInt, flip y: SInt}
      node x = z.x            ;CHECK: node x = z@<t:{ x : UInt@<t:UInt>, flip y : SInt@<t:SInt>}>.x@<t:UInt>
      node y = z.y            ;CHECK: node y = z@<t:{ x : UInt@<t:UInt>, flip y : SInt@<t:SInt>}>.y@<t:SInt>
      wire a : UInt<3>[10]    ;CHECK: wire a : UInt<3>[10]@<t:UInt>@<t:UInt<3>[10]@<t:UInt>>
      node b = a[2]           ;CHECK: node b = a@<t:UInt<3>[10]@<t:UInt>>[2]@<t:UInt>
      read accessor c = a[UInt(3)] ;CHECK: read accessor c = a@<t:UInt<3>[10]@<t:UInt>>[UInt(3)]
; CHECK: Finished Infer Types
; CHECK: Resolve Genders 
; CHECK: read accessor c = a@<t:UInt<3>[10]@<t:UInt>>[UInt(3)]   
; CHECK: Finished Resolve Genders 


