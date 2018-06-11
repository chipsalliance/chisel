; See LICENSE for license details.
circuit ParameterizedExtModuleTester :
  extmodule ParameterizedExtModule_1 :
    input foo : UInt<16>
    output bar : UInt<16>

    defname = ParameterizedExtModule
    parameter VALUE = 1
    parameter STRING = "one"
    parameter REAL = -1.7
    parameter TYP = 'bit'

  extmodule ParameterizedExtModule_2 :
    input foo : UInt<16>
    output bar : UInt<16>

    defname = ParameterizedExtModule
    parameter VALUE = '2\'d2'
    parameter STRING = "two"
    parameter REAL = 2.6E50
    parameter TYP = 'bit [1:0]'

  module ParameterizedExtModuleTester  :
    input clock : Clock
    input reset : UInt<1>

    inst dut1 of ParameterizedExtModule_1
    inst dut2 of ParameterizedExtModule_2

    dut1.foo <= UInt(1000)
    dut2.foo <= UInt(1000)

    when not(reset) :
      when neq(dut1.bar, UInt(1003)) :
        printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
        stop(clock, not(reset), 1)
      else :
        when neq(dut2.bar, UInt(1008)) :
          printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
          stop(clock, not(reset), 1)
        else :
          stop(clock, not(reset), 0)

