; See LICENSE for license details.
circuit ChirrtlMems :
  module ChirrtlMems :
    input clock : Clock
    input reset : UInt<1>

    cmem ram : UInt<32>[16]
    node newClock = clock

    wire wen : UInt<1>
    wen <= not(reset) ; Don't const prop me!

    reg raddr : UInt<4>, clock with : (reset => (reset, UInt(0)))
    raddr <= add(raddr, UInt(1))
    infer mport r = ram[raddr], newClock

    when wen :
      node newerClock = clock
      reg waddr : UInt<4>, clock with : (reset => (reset, UInt(0)))
      waddr <= add(waddr, UInt(1))
      infer mport w = ram[waddr], newerClock
      w <= waddr

      when eq(waddr, UInt(0)) :
        raddr <= UInt(0)

    when not(reset) :
      when gt(waddr, UInt(1)) :
        when neq(r, raddr) :
          printf(clock, UInt(1), "Assertion failed! r =/= raddr\n")
          stop(clock, UInt(1), 1) ; Failure!
      when eq(raddr, UInt(15)) :
        stop(clock, UInt(1), 0) ; Success!
    
