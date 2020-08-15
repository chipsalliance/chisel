// See LICENSE for license details.

package firrtl.backends.experimental.smt

private class Btor2Spec extends SMTBackendBaseSpec {

  it should "convert a hello world module" in {
    val src =
      """circuit m:
        |  module m:
        |    input clock: Clock
        |    input a: UInt<8>
        |    output b: UInt<16>
        |    b <= a
        |    assert(clock, eq(a, b), UInt(1), "")
        |""".stripMargin

    val expected =
      """1 sort bitvec 8
        |2 input 1 a
        |3 sort bitvec 16
        |4 uext 3 2 8
        |5 output 4 ; b
        |6 sort bitvec 1
        |7 uext 3 2 8
        |8 eq 6 7 4
        |9 not 6 8
        |10 bad 9 ; assert_
        |""".stripMargin

    assert(toBotr2Str(src) == expected)
  }

  it should "include FileInfo in the output" in {
    val src =
      """circuit m: @[circuit 0:0]
        |  module m: @[module 0:0]
        |    input clock: Clock @[clock 0:0]
        |    input a: UInt<8> @[a 0:0]
        |    output b: UInt<16> @[b 0:0]
        |    b <= a @[b_a 0:0]
        |    assert(clock, eq(a, b), UInt(1), "") @[assert 0:0]
        |""".stripMargin

    val expected =
      """; @ module 0:0
        |1 sort bitvec 8
        |2 input 1 a ; @ a 0:0
        |3 sort bitvec 16
        |4 uext 3 2 8
        |5 output 4 ; b @ b 0:0, b_a 0:0
        |6 sort bitvec 1
        |7 uext 3 2 8
        |8 eq 6 7 4
        |9 not 6 8
        |10 bad 9 ; assert_ @ assert 0:0
        |""".stripMargin

    assert(toBotr2Str(src) == expected)
  }
}
