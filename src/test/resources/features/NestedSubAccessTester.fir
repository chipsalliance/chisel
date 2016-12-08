; See LICENSE for license details.
circuit NestedSubAccessTester :
  module NestedSubAccess :
    input foo : UInt<1>[4]
    input index : UInt<2>
    output out : UInt<4>

    wire vec : UInt<4>[2]
    vec[0] <= UInt(3)
    vec[1] <= UInt(4)

    out <= vec[foo[index]]

  module NestedSubAccessTester :
    input clock : Clock
    input reset : UInt<1>

    inst dut of NestedSubAccess

    dut.foo is invalid
    dut.index <= UInt(2)
    dut.foo[2] <= UInt(1)

    when neq(dut.out, UInt(4)) :
      printf(clock, not(reset), "Assertion failed\nTest Failed!\n")
      stop(clock, not(reset), 1)
    else :
      stop(clock, not(reset), 0)


