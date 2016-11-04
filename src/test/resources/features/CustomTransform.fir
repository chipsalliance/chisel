; See LICENSE for license details.
circuit CustomTransform :
  ; Replaced in custom transform by an implementation
  extmodule Delay :
    input clk : Clock
    input reset : UInt<1>
    input a : UInt<32>
    input en : UInt<1>
    output b : UInt<32>

  module CustomTransform :
    input clk : Clock
    input reset : UInt<1>

    reg cycle : UInt<32>, clk with : (reset => (reset, UInt<32>(0)))
    cycle <= tail(add(cycle, UInt<32>(1)), 1)

    inst delay of Delay
    delay.clk <= clk
    delay.reset <= reset
    delay.a <= UInt(0)
    delay.en <= UInt(0)

    when eq(cycle, UInt(0)) :
      delay.en <= UInt(1)
      delay.a <= UInt("hdeadbeef")
    when eq(cycle, UInt(1)) :
      when neq(delay.b, UInt("hdeadbeef")) :
        printf(clk, UInt(1), "Assertion failed!\n")
        stop(clk, UInt(1), 1)
    when eq(cycle, UInt(2)) :
      printf(clk, UInt(1), "Success!\n")
      stop(clk, UInt(1), 0)

