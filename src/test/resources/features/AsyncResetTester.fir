
circuit AsyncResetTester :
  module AsyncResetTester :
    input clock : Clock
    input reset : UInt<1>

    reg div : UInt<2>, clock with : (reset => (reset, UInt(0)))
    div <= tail(add(div, UInt(1)), 1)

    reg slowClkReg : UInt<1>, clock with : (reset => (reset, UInt(0)))
    slowClkReg <= eq(div, UInt(0))
    node slowClk = asClock(slowClkReg)

    reg counter : UInt<4>, clock with : (reset => (reset, UInt(0)))
    counter <= tail(add(counter, UInt(1)), 1)

    reg asyncResetReg : UInt<1>, clock with : (reset => (reset, UInt(0)))
    asyncResetReg <= eq(counter, UInt(2))
    node asyncReset = asAsyncReset(asyncResetReg)

    reg r : UInt<8>, slowClk with : (reset => (asyncReset, UInt("h55")))
    ; We always set the register on slowClk
    when UInt(1) :
      r <= UInt("hff")

    when and(leq(counter, UInt(2)), neq(counter, UInt(0))) :
      when neq(r, UInt("hff")) :
        printf(clock, UInt(1), "Assertion 1 failed!\n")
        stop(clock, UInt(1), 1)
    ; Do the async reset
    when eq(counter, UInt(3)) :
      when neq(r, UInt("h55")) :
        printf(clock, UInt(1), "Assertion 2 failed!\n")
        stop(clock, UInt(1), 1)
    ; Back to normal value
    when eq(counter, UInt(5)) :
      when neq(r, UInt("hff")) :
        printf(clock, UInt(1), "Assertion 3 failed!\n")
        stop(clock, UInt(1), 1)
    ; Success!
    when eq(counter, UInt(6)) :
      stop(clock, UInt(1), 0)

