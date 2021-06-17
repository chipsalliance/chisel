// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt

import org.scalatest.flatspec.AnyFlatSpec

class FirrtlExpressionSemanticsSpec extends AnyFlatSpec {

  private def primopSys(op: String, resTpe: String, inTpes: Seq[String], consts: Seq[Int]): TransitionSystem = {
    val inputs = inTpes.zipWithIndex.map { case (tpe, ii) => s"    input i$ii : $tpe" }.mkString("\n")
    val args = (inTpes.zipWithIndex.map { case (_, ii) => s"i$ii" } ++ consts.map(_.toString)).mkString(", ")
    val src =
      s"""circuit m:
         |  module m:
         |$inputs
         |    output res: $resTpe
         |    res <= $op($args)
         |
         |""".stripMargin
    SMTBackendHelpers.toSys(src)
  }

  def primop(op: String, resTpe: String, inTpes: Seq[String], consts: Seq[Int]): String = {
    val sys = primopSys(op, resTpe, inTpes, consts)
    assert(sys.signals.length >= 1)
    sys.signals.last.e.toString
  }

  private def primopSys(
    signed:            Boolean,
    op:                String,
    resWidth:          Int,
    inWidth:           Seq[Int],
    consts:            Seq[Int],
    resAlwaysUnsigned: Boolean
  ): TransitionSystem = {
    val tpe = if (signed) "SInt" else "UInt"
    val resTpe = if (resAlwaysUnsigned) "UInt" else tpe
    val inTpes = inWidth.map(w => s"$tpe<$w>")
    primopSys(op, s"$resTpe<$resWidth>", inTpes, consts)
  }

  def primop(
    signed:            Boolean,
    op:                String,
    resWidth:          Int,
    inWidth:           Seq[Int],
    consts:            Seq[Int] = List(),
    resAlwaysUnsigned: Boolean = false
  ): String = {
    val sys = primopSys(signed, op, resWidth, inWidth, consts, resAlwaysUnsigned)
    assert(sys.signals.length >= 1)
    sys.signals.last.e.toString
  }

  it should "correctly translate the add primitive operation with different operand sizes" in {
    assert(primop(false, "add", 5, List(3, 5)) == "add(zext(i0, 3), zext(i1, 1))[4:0]")
    assert(primop(false, "add", 5, List(3, 4)) == "add(zext(i0, 2), zext(i1, 1))")
    assert(primop(true, "add", 5, List(3, 5)) == "add(sext(i0, 3), sext(i1, 1))[4:0]")
    assert(primop(true, "add", 5, List(3, 4)) == "add(sext(i0, 2), sext(i1, 1))")

    // could be simplified to just `add(i0, i1)`
    assert(primop(false, "add", 8, List(8, 8)) == "add(zext(i0, 1), zext(i1, 1))[7:0]")
  }

  it should "correctly translate the `add` primitive operation" in {
    assert(primop(false, "add", 8, List(7, 7)) == "add(zext(i0, 1), zext(i1, 1))")
  }

  it should "correctly translate the `sub` primitive operation" in {
    assert(primop(false, "sub", 8, List(7, 7)) == "sub(zext(i0, 1), zext(i1, 1))")
  }

  it should "correctly translate the `mul` primitive operation" in {
    assert(primop(false, "mul", 8, List(4, 4)) == "mul(zext(i0, 4), zext(i1, 4))")
  }

  it should "correctly translate the `div` primitive operation" in {
    // division is a little bit more complicated because the result of division by zero is undefined
    //val sys = primopSys(false, "div", 8, List(8, 8), List(), false)
    //println(sys.serialize)

    assert(
      primop(false, "div", 8, List(8, 8)) ==
        "ite(res_invalid_cond, res_invalid, udiv(i0, i1))"
    )
    assert(
      primop(false, "div", 8, List(8, 4)) ==
        "ite(res_invalid_cond, res_invalid, udiv(i0, zext(i1, 4)))"
    )

    // signed division increases result width by 1
    assert(
      primop(true, "div", 8, List(7, 7)) ==
        "ite(res_invalid_cond, res_invalid, sdiv(sext(i0, 1), sext(i1, 1)))"
    )
    assert(
      primop(true, "div", 8, List(7, 4))
        == "ite(res_invalid_cond, res_invalid, sdiv(sext(i0, 1), sext(i1, 4)))"
    )
  }

  it should "correctly translate the `rem` primitive operation" in {
    // rem can decrease the size of operands, but we should only do that decrease on the result
    assert(primop(false, "rem", 4, List(4, 8)) == "urem(zext(i0, 4), i1)[3:0]")
    assert(primop(false, "rem", 4, List(8, 4)) == "urem(i0, zext(i1, 4))[3:0]")
    assert(primop(true, "rem", 4, List(4, 8)) == "srem(sext(i0, 4), i1)[3:0]")
    assert(primop(true, "rem", 4, List(8, 4)) == "srem(i0, sext(i1, 4))[3:0]")
    // TODO: add test to make sure we are using the correct mod/rem operation for signed and unsigned
    //       https://groups.google.com/g/stp-users/c/od43h8q5RSI has some tests that we could copy and
    //       use with a SMT solver
  }

  it should "correctly translate the comparison primitive operations" in {
    // some comparisons are represented as the negation of others
    assert(primop(false, "lt", 1, List(8, 8)) == "not(ugeq(i0, i1))")
    assert(primop(false, "leq", 1, List(8, 8)) == "not(ugt(i0, i1))")
    assert(primop(false, "gt", 1, List(8, 8)) == "ugt(i0, i1)")
    assert(primop(false, "geq", 1, List(8, 8)) == "ugeq(i0, i1)")
    assert(primop(false, "eq", 1, List(8, 8)) == "eq(i0, i1)")
    assert(primop(false, "neq", 1, List(8, 8)) == "not(eq(i0, i1))")

    assert(primop(true, "lt", 1, List(8, 8), resAlwaysUnsigned = true) == "not(sgeq(i0, i1))")
    assert(primop(true, "leq", 1, List(8, 8), resAlwaysUnsigned = true) == "not(sgt(i0, i1))")
    assert(primop(true, "gt", 1, List(8, 8), resAlwaysUnsigned = true) == "sgt(i0, i1)")
    assert(primop(true, "geq", 1, List(8, 8), resAlwaysUnsigned = true) == "sgeq(i0, i1)")
    assert(primop(true, "eq", 1, List(8, 8), resAlwaysUnsigned = true) == "eq(i0, i1)")
    assert(primop(true, "neq", 1, List(8, 8), resAlwaysUnsigned = true) == "not(eq(i0, i1))")

    // it should always extend the width to the max of both
    assert(primop(false, "gt", 1, List(7, 8)) == "ugt(zext(i0, 1), i1)")
  }

  it should "correctly translate the `pad` primitive operation" in {
    // firrtl pad takes new width as argument, whereas the smt zext takes the number of bits to extend by
    assert(primop(false, "pad", 8, List(3), List(8)) == "zext(i0, 5)")
    assert(primop(false, "pad", 8, List(3), List(5)) == "zext(zext(i0, 2), 3)")

    // there is no negative padding, instead the result is just e
    assert(primop(false, "pad", 3, List(3), List(2)) == "i0")

    assert(primop(true, "pad", 8, List(3), List(8)) == "sext(i0, 5)")
    assert(primop(true, "pad", 8, List(3), List(5)) == "sext(sext(i0, 2), 3)")
  }

  it should "correctly translate the asX primitive operations" in {
    // these are all essentially no-ops
    assert(primop(false, "asUInt", 3, List(3)) == "i0")
    assert(primop(true, "asSInt", 3, List(3)) == "i0")
  }

  it should "correctly translate the `shl` primitive operation" in {
    assert(primop(false, "shl", 6, List(3), List(3)) == "concat(i0, 3'b0)")
    assert(primop(true, "shl", 6, List(3), List(3)) == "concat(i0, 3'b0)")
    assert(primop(false, "shl", 3, List(3), List(0)) == "i0")
  }

  it should "correctly translate the `shr` primitive operation" in {
    assert(primop(false, "shr", 6, List(9), List(3)) == "i0[8:3]")
    assert(primop(true, "shr", 6, List(9), List(3)) == "i0[8:3]")

    // "If n is greater than or equal to the bit-width of e,
    // the resulting value will be zero for unsigned types and the sign bit for signed types."
    assert(primop(false, "shr", 1, List(3), List(3)) == "1'b0")
    assert(primop(false, "shr", 1, List(3), List(4)) == "1'b0")
    assert(primop(true, "shr", 1, List(3), List(3)) == "i0[2]")
    assert(primop(true, "shr", 1, List(3), List(4)) == "i0[2]")
  }

  it should "correctly translate the `dshl` primitive operation" in {
    assert(primop(false, "dshl", 31, List(16, 4)) == "logical_shift_left(zext(i0, 15), zext(i1, 27))")
    assert(primop(false, "dshl", 19, List(16, 2)) == "logical_shift_left(zext(i0, 3), zext(i1, 17))")
    assert(
      primop("dshl", "SInt<19>", List("SInt<16>", "UInt<2>"), List()) ==
        "logical_shift_left(sext(i0, 3), zext(i1, 17))"
    )
  }

  it should "correctly translate the `dshr` primitive operation" in {
    assert(primop(false, "dshr", 16, List(16, 4)) == "logical_shift_right(i0, zext(i1, 12))")
    assert(primop(false, "dshr", 16, List(16, 2)) == "logical_shift_right(i0, zext(i1, 14))")
    assert(
      primop("dshr", "SInt<16>", List("SInt<16>", "UInt<2>"), List()) ==
        "arithmetic_shift_right(i0, zext(i1, 14))"
    )
  }

  it should "correctly translate the `cvt` primitive operation" in {
    // for signed operands, this is a no-op
    assert(primop(true, "cvt", 3, List(3)) == "i0")

    // for unsigned, a zero is prepended
    assert(primop("cvt", "SInt<16>", List("UInt<15>"), List()) == "concat(1'b0, i0)")
    assert(primop("cvt", "SInt<16>", List("UInt<14>"), List()) == "sext(concat(1'b0, i0), 1)")
  }

  it should "correctly translate the `neg` primitive operation" in {
    assert(primop(true, "neg", 4, List(3)) == "sub(sext(3'b0, 1), sext(i0, 1))")
    assert(primop("neg", "SInt<4>", List("UInt<3>"), List()) == "sub(zext(3'b0, 1), zext(i0, 1))")
  }

  it should "correctly translate the `not` primitive operation" in {
    assert(primop(false, "not", 4, List(4)) == "not(i0)")
    assert(primop("not", "UInt<4>", List("SInt<4>"), List()) == "not(i0)")
  }

  it should "correctly translate the binary bitwise primitive operations" in {
    assert(primop(false, "and", 4, List(4, 3)) == "and(i0, zext(i1, 1))")
    assert(primop("and", "UInt<4>", List("SInt<4>", "SInt<3>"), List()) == "and(i0, sext(i1, 1))")

    assert(primop(false, "or", 4, List(4, 3)) == "or(i0, zext(i1, 1))")
    assert(primop("or", "UInt<4>", List("SInt<4>", "SInt<3>"), List()) == "or(i0, sext(i1, 1))")

    assert(primop(false, "xor", 4, List(4, 3)) == "xor(i0, zext(i1, 1))")
    assert(primop("xor", "UInt<4>", List("SInt<4>", "SInt<3>"), List()) == "xor(i0, sext(i1, 1))")
  }

  it should "correctly translate the bitwise reduction primitive operation" in {
    // zero width special cases are removed by the firrtl compiler
    assert(primop(false, "andr", 1, List(0)) == "1'b1")
    assert(primop(false, "orr", 1, List(0)) == "redor(1'b0)")
    assert(primop(false, "xorr", 1, List(0)) == "redxor(1'b0)")

    assert(primop(false, "andr", 1, List(3)) == "redand(i0)")
    assert(primop(true, "andr", 1, List(3), resAlwaysUnsigned = true) == "redand(i0)")

    assert(primop(false, "orr", 1, List(3)) == "redor(i0)")
    assert(primop(true, "orr", 1, List(3), resAlwaysUnsigned = true) == "redor(i0)")

    assert(primop(false, "xorr", 1, List(3)) == "redxor(i0)")
    assert(primop(true, "xorr", 1, List(3), resAlwaysUnsigned = true) == "redxor(i0)")
  }

  it should "correctly translate the `cat` primitive operation" in {
    assert(primop(false, "cat", 7, List(4, 3)) == "concat(i0, i1)")
    assert(primop(true, "cat", 7, List(4, 3), resAlwaysUnsigned = true) == "concat(i0, i1)")
  }

  it should "correctly translate the `bits` primitive operation" in {
    assert(primop(false, "bits", 1, List(4), List(2, 2)) == "i0[2]")
    assert(primop(false, "bits", 2, List(4), List(2, 1)) == "i0[2:1]")
    assert(primop(false, "bits", 1, List(4), List(2, 1)) == "i0[2:1][0]")
    assert(primop(false, "bits", 3, List(4), List(2, 1)) == "zext(i0[2:1], 1)")

    assert(primop(true, "bits", 1, List(4), List(2, 2), resAlwaysUnsigned = true) == "i0[2]")
    assert(primop(true, "bits", 2, List(4), List(2, 1), resAlwaysUnsigned = true) == "i0[2:1]")
    assert(primop(true, "bits", 1, List(4), List(2, 1), resAlwaysUnsigned = true) == "i0[2:1][0]")
    assert(primop(true, "bits", 3, List(4), List(2, 1), resAlwaysUnsigned = true) == "zext(i0[2:1], 1)")
  }

  it should "correctly translate the `head` primitive operation" in {
    // "The result of the head operation are the n most significant bits of e"
    assert(primop(false, "head", 1, List(4), List(1)) == "i0[3]")
    assert(primop(false, "head", 1, List(5), List(1)) == "i0[4]")
    assert(primop(false, "head", 3, List(5), List(3)) == "i0[4:2]")
  }

  it should "correctly translate the `tail` primitive operation" in {
    // "The tail operation truncates the n most significant bits from e"
    assert(primop(false, "tail", 3, List(4), List(1)) == "i0[2:0]")
    assert(primop(false, "tail", 4, List(5), List(1)) == "i0[3:0]")
    assert(primop(false, "tail", 2, List(5), List(3)) == "i0[1:0]")
  }
}
