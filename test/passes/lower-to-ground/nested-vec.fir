; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower Types
circuit top :
   module top :
      input clk : Clock
      input i : UInt<1>
      wire j : { x : UInt<32>, flip y : UInt<32> }
      wire k : { x : UInt<32>, y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}y : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}y : UInt<32>
      a[0].x <= UInt(0)
      a[0].y <= UInt(0)
      a[1].x <= UInt(0)
      a[1].y <= UInt(0)
      j.x <= UInt(0)
      j.y <= UInt(0)
      k.x <= UInt(0)
      k.y <= UInt(0)

      wire b : { x : UInt<32>, flip y : UInt<32> }
      b <= a[i]
      j <= b

      mem m : 
         data-type => { x : UInt<32>, y : UInt<32> }
         depth => 2
         read-latency => 0
         write-latency => 1
         writer => c

      m.c.addr <= i
      m.c.clk <= clk
      m.c.en <= UInt(1)
      m.c.mask.x <= UInt(1)
      m.c.mask.y <= UInt(1)
      m.c.data <= k

;CHECK: mem m_x :          
;CHECK:    data-type => UInt<32>
;CHECK:    depth => 2
;CHECK:    write-latency => 1
;CHECK:    read-latency => 0
;CHECK:    writer => c
;CHECK: mem m_y :          
;CHECK:    data-type => UInt<32>
;CHECK:    depth => 2
;CHECK:    write-latency => 1
;CHECK:    read-latency => 0
;CHECK:    writer => c
;CHECK: m_x.c.data <= k_x
;CHECK: m_y.c.data <= k_y
;CHECK: m_x.c.mask <= UInt<1>("h1")
;CHECK: m_y.c.mask <= UInt<1>("h1")
;CHECK: m_x.c.addr <= i
;CHECK: m_y.c.addr <= i
;CHECK: m_x.c.en <= UInt<1>("h1")
;CHECK: m_y.c.en <= UInt<1>("h1")
;CHECK: m_x.c.clk <= clk
;CHECK: m_y.c.clk <= clk   

; CHECK: Finished Lower Types
; CHECK: Done!

