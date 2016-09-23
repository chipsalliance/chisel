circuit SimpleExtModuleTester :
  extmodule SimpleExtModule :
    input foo : UInt<16>
    output bar : UInt<16>

  module SimpleExtModuleTester :
    input clk : Clock
    input reset : UInt<1>

    inst dut of SimpleExtModule

    dut.foo <= UInt(1234)

    when not(reset) :
      when neq(dut.bar, UInt(1234)) :
        printf(clk, not(reset), "Assertion failed\nTest Failed!\n")
        stop(clk, not(reset), 1)
      else :
        stop(clk, not(reset), 0)

