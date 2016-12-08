; See LICENSE for license details.
circuit MultiExtModuleTester :
  extmodule SimpleExtModule :
    input foo : UInt<16>
    output bar : UInt<16>

  extmodule AdderExtModule :
    input foo : UInt<16>
    output bar : UInt<16>

  module MultiExtModuleTester :
    input clock : Clock
    input reset : UInt<1>

    inst dut1 of SimpleExtModule
    inst dut2 of SimpleExtModule
    inst dut3 of AdderExtModule

    dut1.foo <= UInt(1234)
    dut2.foo <= UInt(5678)
    dut3.foo <= UInt(100)

    when not(reset) :
      when neq(dut1.bar, UInt(1234)) :
        printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
        stop(clock, not(reset), 1)
      else :
        when neq(dut2.bar, UInt(5678)) :
          printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
          stop(clock, not(reset), 1)
        else :
          when neq(dut3.bar, UInt(101)) :
            printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
            stop(clock, not(reset), 1)
          else :
            stop(clock, not(reset), 0)

