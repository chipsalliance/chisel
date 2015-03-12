; RUN: firrtl %s abcdefg c | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      mem m : UInt(1)[2]
      wire i : UInt(1)
      wire p : UInt(1)
      when p :
        accessor a = m[i] ;CHECK: a := ReadPort(m, i, bit-and(p, UInt(1)))
        i := a
        accessor b = m[i] ;CHECK: WritePort(m, i, bit-and(p, UInt(1))) := b
        b := i
      else :
        accessor c = m[i] ;CHECK: c := ReadPort(m, i, bit-and(equal-uu(UInt(0), p), UInt(1)))
        i := c
        accessor d = m[i] ;CHECK: WritePort(m, i, bit-and(equal-uu(UInt(0), p), UInt(1))) := d
        d := i

; CHECK: Finished Expand Accessors
