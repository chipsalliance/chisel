// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt

import org.scalatest.flatspec.AnyFlatSpec

class Btor2Spec extends AnyFlatSpec {
  behavior.of("btor2 backend")

  it should "convert a hello world module" in {
    val src =
      """circuit m:
        |  module m:
        |    input clock: Clock
        |    input a: UInt<8>
        |    output b: UInt<16>
        |    b <= a
        |    assert(clock, eq(a, b), UInt(1), "") : a_eq_b
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
        |10 bad 9 ; a_eq_b
        |""".stripMargin

    assert(SMTBackendHelpers.toBotr2Str(src) == expected)
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
        |10 bad 9 ; assert_0 @ assert 0:0
        |""".stripMargin

    assert(SMTBackendHelpers.toBotr2Str(src) == expected)
  }
}
