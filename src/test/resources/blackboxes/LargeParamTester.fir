; See LICENSE for license details.
circuit LargeParamTester :
  extmodule LargeParam :
    output out : UInt<1>

    defname = LargeParam
    parameter WIDTH = 1
    parameter DATA = 0

  extmodule LargeParam_1 :
    output out : UInt<128>

    defname = LargeParam
    parameter WIDTH = 128
    parameter DATA = 9223372036854775807000

  extmodule LargeParam_2 :
    output out : SInt<128>

    defname = LargeParam
    parameter WIDTH = 128
    parameter DATA = -9223372036854775807000

  module LargeParamTester  :
    input clock : Clock
    input reset : UInt<1>

    inst mod1 of LargeParam
    inst mod2 of LargeParam_1
    inst mod3 of LargeParam_2

    when not(reset) :
      when neq(mod1.out, UInt(0)) :
        printf(clock, UInt(1), "Assertion failed\nTest Failed!\n")
        stop(clock, UInt(1), 1)
      when neq(mod2.out, UInt<128>(9223372036854775807000)) :
        printf(clock, UInt(1), "Assertion failed\nTest Failed!\n")
        stop(clock, UInt(1), 1)
      when neq(mod3.out, SInt<128>(-9223372036854775807000)) :
        printf(clock, UInt(1), "Assertion failed\nTest Failed!\n")
        stop(clock, UInt(1), 1)
      stop(clock, UInt(1), 0)

