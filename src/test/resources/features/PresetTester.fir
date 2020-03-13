
circuit PresetTester :

  module Test :
    input clock : Clock
    input reset : AsyncReset
    input x : UInt<4>
    output z : UInt<4>
    reg r : UInt<4>, clock with : (reset => (reset, UInt(12)))
    r <= x
    z <= r
    
  module PresetTester :
    input clock : Clock
    input reset : UInt<1>

    reg div : UInt<2>, clock with : (reset => (reset, UInt(0)))
    div <= tail(add(div, UInt(1)), 1)
    
    reg slowClkReg : UInt<1>, clock with : (reset => (reset, UInt(0)))
    slowClkReg <= eq(div, UInt(0))
    node slowClk = asClock(slowClkReg)

    reg counter : UInt<4>, clock with : (reset => (reset, UInt(0)))
    counter <= tail(add(counter, UInt(1)), 1)

    reg x : UInt<5>, slowClk with : (reset => (reset, UInt(9)))
    wire z : UInt<5>
    
    wire preset : AsyncReset 
    preset <= asAsyncReset(UInt(0)) ; should be annotated as Preset
    
    inst i of Test
    i.clock <= slowClk
    i.reset <= preset
    i.x <= x
    z <= i.z

    when eq(counter, UInt(0)) :
      when neq(z, UInt(12)) :
        printf(clock, UInt(1), "Assertion 1 failed! z=%d \n",z)
        stop(clock, UInt(1), 1)
    ; Do the async reset
    when eq(counter, UInt(1)) :
      when neq(z, UInt(9)) :
        printf(clock, UInt(1), "Assertion 2 failed! z=%d \n",z)
        stop(clock, UInt(1), 1)
    ; Success!
    when eq(counter, UInt(3)) :
      stop(clock, UInt(1), 0)

