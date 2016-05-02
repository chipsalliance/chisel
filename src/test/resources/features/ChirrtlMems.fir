
circuit ChirrtlMems :
  module ChirrtlMems :
    input clk : Clock
    input reset : UInt<1>

    cmem ram : UInt<32>[16]
    node newClock = clk

    wire wen : UInt<1>
    wen <= not(reset) ; Don't const prop me!

    reg raddr : UInt<4>, clk with : (reset => (reset, UInt(0)))
    raddr <= add(raddr, UInt(1))
    infer mport r = ram[raddr], newClock

    when wen :
      node newerClock = clk
      reg waddr : UInt<4>, clk with : (reset => (reset, UInt(0)))
      waddr <= add(waddr, UInt(1))
      infer mport w = ram[waddr], newerClock
      w <= waddr

      when eq(waddr, UInt(0)) :
        raddr <= UInt(0)

    when not(reset) :
      when gt(waddr, UInt(1)) :
        when neq(r, raddr) :
          printf(clk, UInt(1), "Assertion failed! r =/= raddr\n")
          stop(clk, UInt(1), 1) ; Failure!
      when eq(raddr, UInt(15)) :
        stop(clk, UInt(1), 0) ; Success!
    
