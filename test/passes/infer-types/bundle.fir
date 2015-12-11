; RUN: firrtl -i %s -o %s.v -X verilog -p ct 2>&1 | tee %s.out | FileCheck %s

circuit top :
   module top :
      wire z : { x : UInt, flip y: SInt}
      z.x <= UInt(1)
      z.y <= SInt(1)
      node x = z.x            
      node y = z.y            
      wire a : UInt<3>[10]    
      a[0] <= UInt(1)
      a[1] <= UInt(1)
      a[2] <= UInt(1)
      a[3] <= UInt(1)
      a[4] <= UInt(1)
      a[5] <= UInt(1)
      a[6] <= UInt(1)
      a[7] <= UInt(1)
      a[8] <= UInt(1)
      a[9] <= UInt(1)
      node b = a[2]           
      node c = a[UInt(3)]     

;CHECK: Infer Types
;CHECK: node x = z@<t:{ x : UInt, flip y : SInt}>.x@<t:UInt>
;CHECK: node y = z@<t:{ x : UInt, flip y : SInt}>.y@<t:SInt>
;CHECK: wire a : UInt<3>[10]@<t:UInt>@<t:UInt[10]@<t:UInt>>
;CHECK: node b = a@<t:UInt[10]@<t:UInt>>[2]@<t:UInt>
;CHECK: node c = a@<t:UInt[10]@<t:UInt>>[UInt("h3")@<t:UInt>]
;CHECK: Finished Infer Types


