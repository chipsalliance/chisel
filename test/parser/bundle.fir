; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
circuit top :
   module top :
      wire z : { x : UInt, flip y: SInt}
      z.x := UInt(1)
      z.y := SInt(1)
      node x = z.x            
      node y = z.y            
      wire a : UInt<3>[10]    
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
      node b = a[2]           
      read accessor c = a[UInt(3)] 

; CHECK: circuit top :
; CHECK:    module top :
; CHECK:       wire z : { x : UInt, flip y : SInt}
; CHECK:       z.x := UInt("h1")
; CHECK:       z.y := SInt("h1")
; CHECK:       node x = z.x            
; CHECK:       node y = z.y            
; CHECK:       wire a : UInt<3>[10]    
; CHECK:       a[0] := UInt("h1")
; CHECK:       a[1] := UInt("h1")
; CHECK:       a[2] := UInt("h1")
; CHECK:       a[3] := UInt("h1")
; CHECK:       a[4] := UInt("h1")
; CHECK:       a[5] := UInt("h1")
; CHECK:       a[6] := UInt("h1")
; CHECK:       a[7] := UInt("h1")
; CHECK:       a[8] := UInt("h1")
; CHECK:       a[9] := UInt("h1")
; CHECK:       node b = a[2]           
; CHECK:       read accessor c = a[UInt("h3")] 

