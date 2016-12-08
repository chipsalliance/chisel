; See LICENSE for license details.
circuit RenamedExtModuleTester :
  extmodule ThisIsJustTheFirrtlName :
    input foo : UInt<16>
    output bar : UInt<16>

    defname = SimpleExtModule

  module RenamedExtModuleTester :
    input clock : Clock
    input reset : UInt<1>

    inst dut of ThisIsJustTheFirrtlName

    dut.foo <= UInt(1234)

    when not(reset) :
      when neq(dut.bar, UInt(1234)) :
        printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
        stop(clock, not(reset), 1)
      else :
        stop(clock, not(reset), 0)

