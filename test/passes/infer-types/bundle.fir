; RUN: firrtl %s abcde ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module subtracter :
      wire z : {input x : UInt, output y: SInt}
      node x = z.x ;CHECK: node x = z@<t:{input x : UInt@<t:UInt>, output y : SInt@<t:SInt>}>.x@<t:UInt>
      node y = z.y ;CHECK: node y = z@<t:{input x : UInt@<t:UInt>, output y : SInt@<t:SInt>}>.y@<t:SInt>

      wire a : UInt(3)[10] ;CHECK: wire a : UInt(3)[10]@<t:UInt>@<t:UInt(3)[10]@<t:UInt>>
      node b = a.2 ;CHECK: node b = a@<t:UInt(3)[10]@<t:UInt>>.2@<t:UInt>
      accessor c = a[UInt(3)] ;CHECK: unknown accessor c = a@<t:UInt(3)[10]@<t:UInt>>[UInt(3)]
; CHECK: Finished Infer Types
