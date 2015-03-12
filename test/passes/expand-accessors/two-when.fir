; RUN: firrtl %s abcdefg c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      mem m : UInt(1)[2]
      wire i : UInt(1)
      wire p : UInt(1)
      when p :
        wire p2 : UInt(1)
        when p2 :
          accessor a = m[i] 
          i := a
          accessor b = m[i]
          b := i
            ;CHECK : wire a : UInt(1)
            ;CHECK : a := ReadPort(m, i, bit-and(p2, bit-and(p, UInt(1))))
            ;CHECK : i := a
            ;CHECK : wire b : UInt(1)
            ;CHECK : WritePort(m, i, bit-and(p2, bit-and(p, UInt(1)))) := b
            ;CHECK : b := i
        else :
          accessor c = m[i] 
          i := c
          accessor d = m[i]
          d := i
            ;CHECK : wire c : UInt(1)
            ;CHECK : c := ReadPort(m, i, bit-and(equal-uu(UInt(0), p2), bit-and(p, UInt(1))))
            ;CHECK : i := c
            ;CHECK : wire d : UInt(1)
            ;CHECK : WritePort(m, i, bit-and(equal-uu(UInt(0), p2), bit-and(p, UInt(1)))) := d
            ;CHECK : d := i
      else :
        when p2 :
          accessor w = m[i] 
          i := w
          accessor x = m[i]
          x := i
            ;CHECK : wire w : UInt(1)
            ;CHECK : w := ReadPort(m, i, bit-and(p2, bit-and(equal-uu(UInt(0), p), UInt(1))))
            ;CHECK : i := w
            ;CHECK : wire x : UInt(1)
            ;CHECK : WritePort(m, i, bit-and(p2, bit-and(equal-uu(UInt(0), p), UInt(1)))) := x
            ;CHECK : x := i
        else :
          accessor y = m[i] 
          i := y
          accessor z = m[i]
          z := i
            ;CHECK : wire y : UInt(1)
            ;CHECK : y := ReadPort(m, i, bit-and(equal-uu(UInt(0), p2), bit-and(equal-uu(UInt(0), p), UInt(1))))
            ;CHECK : i := y
            ;CHECK : wire z : UInt(1)
            ;CHECK : WritePort(m, i, bit-and(equal-uu(UInt(0), p2), bit-and(equal-uu(UInt(0), p), UInt(1)))) := z
            ;CHECK : z := i

; CHECK: Finished Expand Accessors
