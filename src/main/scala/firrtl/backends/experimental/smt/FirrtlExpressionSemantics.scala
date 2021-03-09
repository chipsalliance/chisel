// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.ir
import firrtl.PrimOps
import firrtl.passes.CheckWidths.WidthTooBig

private trait TranslationContext {
  def getReference(name: String, tpe: ir.Type): BVExpr = BVSymbol(name, FirrtlExpressionSemantics.getWidth(tpe))
}

private object FirrtlExpressionSemantics {
  def getWidth(tpe: ir.Type): Int = tpe match {
    case ir.UIntType(ir.IntWidth(w))   => w.toInt
    case ir.SIntType(ir.IntWidth(w))   => w.toInt
    case ir.ClockType                  => 1
    case ir.ResetType                  => 1
    case ir.AnalogType(ir.IntWidth(w)) => w.toInt
    case other                         => throw new RuntimeException(s"Cannot handle type $other")
  }

  def toSMT(e: ir.Expression)(implicit ctx: TranslationContext): BVExpr = {
    val eSMT = e match {
      case ir.DoPrim(op, args, consts, _) => onPrim(op, args, consts)
      case r: ir.Reference => ctx.getReference(r.serialize, r.tpe)
      case r: ir.SubField  => ctx.getReference(r.serialize, r.tpe)
      case r: ir.SubIndex  => ctx.getReference(r.serialize, r.tpe)
      case ir.UIntLiteral(value, ir.IntWidth(width)) => BVLiteral(value, width.toInt)
      case ir.SIntLiteral(value, ir.IntWidth(width)) => BVLiteral(value, width.toInt)
      case ir.Mux(cond, tval, fval, _) =>
        val width = List(tval, fval).map(getWidth).max
        BVIte(toSMT(cond), toSMT(tval, width), toSMT(fval, width))
      case v: ir.ValidIf =>
        throw new RuntimeException(s"Unsupported expression: ValidIf ${v.serialize}")
    }
    assert(
      eSMT.width == getWidth(e),
      "We aim to always produce a SMT expression of the same width as the firrtl expression."
    )
    eSMT
  }

  /** Ensures that the result has the desired width by appropriately extending it. */
  def toSMT(e: ir.Expression, width: Int, allowNarrow: Boolean = false)(implicit ctx: TranslationContext): BVExpr =
    forceWidth(toSMT(e), isSigned(e), width, allowNarrow)

  private def forceWidth(eSMT: BVExpr, eSigned: Boolean, width: Int, allowNarrow: Boolean = false): BVExpr = {
    if (eSMT.width == width) { eSMT }
    else if (width < eSMT.width) {
      assert(allowNarrow, s"Narrowing from ${eSMT.width} bits to $width bits is not allowed!")
      BVSlice(eSMT, width - 1, 0)
    } else {
      BVExtend(eSMT, width - eSMT.width, eSigned)
    }
  }

  // see "Primitive Operations" section in the Firrtl Specification
  private def onPrim(
    op:     ir.PrimOp,
    args:   Seq[ir.Expression],
    consts: Seq[BigInt]
  )(
    implicit ctx: TranslationContext
  ): BVExpr = {
    (op, args, consts) match {
      case (PrimOps.Add, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max + 1
        BVOp(Op.Add, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Sub, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max + 1
        BVOp(Op.Sub, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Mul, Seq(e1, e2), _) =>
        val width = args.map(getWidth).sum
        BVOp(Op.Mul, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Div, Seq(num, den), _) =>
        val (width, op) = if (isSigned(num)) {
          (getWidth(num) + 1, Op.SignedDiv)
        } else { (getWidth(num), Op.UnsignedDiv) }
        BVOp(op, toSMT(num, width), forceWidth(toSMT(den), isSigned(den), width))
      case (PrimOps.Rem, Seq(num, den), _) =>
        val op = if (isSigned(num)) Op.SignedRem else Op.UnsignedRem
        val width = args.map(getWidth).max
        val resWidth = args.map(getWidth).min
        val res = BVOp(op, toSMT(num, width), toSMT(den, width))
        if (res.width > resWidth) { BVSlice(res, resWidth - 1, 0) }
        else { res }
      case (PrimOps.Lt, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVNot(BVComparison(Compare.GreaterEqual, toSMT(e1, width), toSMT(e2, width), isSigned(e1)))
      case (PrimOps.Leq, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVNot(BVComparison(Compare.Greater, toSMT(e1, width), toSMT(e2, width), isSigned(e1)))
      case (PrimOps.Gt, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVComparison(Compare.Greater, toSMT(e1, width), toSMT(e2, width), isSigned(e1))
      case (PrimOps.Geq, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVComparison(Compare.GreaterEqual, toSMT(e1, width), toSMT(e2, width), isSigned(e1))
      case (PrimOps.Eq, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVEqual(toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Neq, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVNot(BVEqual(toSMT(e1, width), toSMT(e2, width)))
      case (PrimOps.Pad, Seq(e), Seq(n)) =>
        val width = getWidth(e)
        if (n <= width) { toSMT(e) }
        else { BVExtend(toSMT(e), n.toInt - width, isSigned(e)) }
      case (PrimOps.AsUInt, Seq(e), _)       => checkForClockInCast(PrimOps.AsUInt, e); toSMT(e)
      case (PrimOps.AsSInt, Seq(e), _)       => checkForClockInCast(PrimOps.AsSInt, e); toSMT(e)
      case (PrimOps.AsFixedPoint, Seq(e), _) => throw new AssertionError("Fixed-Point numbers need to be lowered!")
      case (PrimOps.AsClock, Seq(e), _)      => toSMT(e)
      case (PrimOps.AsAsyncReset, Seq(e), _) =>
        checkForClockInCast(PrimOps.AsAsyncReset, e)
        throw new AssertionError(s"Asynchronous resets are not supported! Cannot cast ${e.serialize}.")
      case (PrimOps.Shl, Seq(e), Seq(n)) =>
        if (n == 0) { toSMT(e) }
        else {
          val zeros = BVLiteral(0, n.toInt)
          BVConcat(toSMT(e), zeros)
        }
      case (PrimOps.Shr, Seq(e), Seq(n)) =>
        val width = getWidth(e)
        // "If n is greater than or equal to the bit-width of e,
        // the resulting value will be zero for unsigned types
        // and the sign bit for signed types"
        if (n >= width) {
          if (isSigned(e)) { BV1BitZero }
          else { BVSlice(toSMT(e), width - 1, width - 1) }
        } else {
          BVSlice(toSMT(e), width - 1, n.toInt)
        }
      case (PrimOps.Dshl, Seq(e1, e2), _) =>
        val width = getWidth(e1) + (1 << getWidth(e2)) - 1
        BVOp(Op.ShiftLeft, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Dshr, Seq(e1, e2), _) =>
        val width = getWidth(e1)
        val o = if (isSigned(e1)) Op.ArithmeticShiftRight else Op.ShiftRight
        BVOp(o, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Cvt, Seq(e), _) =>
        if (isSigned(e)) { toSMT(e) }
        else { BVConcat(BV1BitZero, toSMT(e)) }
      case (PrimOps.Neg, Seq(e), _) => BVNegate(BVExtend(toSMT(e), 1, isSigned(e)))
      case (PrimOps.Not, Seq(e), _) => BVNot(toSMT(e))
      case (PrimOps.And, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVOp(Op.And, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Or, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVOp(Op.Or, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Xor, Seq(e1, e2), _) =>
        val width = args.map(getWidth).max
        BVOp(Op.Xor, toSMT(e1, width), toSMT(e2, width))
      case (PrimOps.Andr, Seq(e), _)           => BVReduceAnd(toSMT(e))
      case (PrimOps.Orr, Seq(e), _)            => BVReduceOr(toSMT(e))
      case (PrimOps.Xorr, Seq(e), _)           => BVReduceXor(toSMT(e))
      case (PrimOps.Cat, Seq(e1, e2), _)       => BVConcat(toSMT(e1), toSMT(e2))
      case (PrimOps.Bits, Seq(e), Seq(hi, lo)) => BVSlice(toSMT(e), hi.toInt, lo.toInt)
      case (PrimOps.Head, Seq(e), Seq(n)) =>
        val width = getWidth(e)
        assert(n >= 0 && n <= width)
        BVSlice(toSMT(e), width - 1, width - n.toInt)
      case (PrimOps.Tail, Seq(e), Seq(n)) =>
        val width = getWidth(e)
        assert(n >= 0 && n <= width)
        assert(n < width, "While allowed by the firrtl standard, we do not support 0-bit values in this backend!")
        BVSlice(toSMT(e), width - n.toInt - 1, 0)
    }
  }

  /** For now we strictly forbid casting clocks to anything else.
    * Eventually this should be replaced by a more sophisticated clock analysis pass.
    */
  private def checkForClockInCast(cast: ir.PrimOp, signal: ir.Expression): Unit = {
    assert(signal.tpe != ir.ClockType, s"Cannot cast (${cast.serialize}) clock expression ${signal.serialize}!")
  }

  private val BV1BitZero = BVLiteral(0, 1)

  def isSigned(e: ir.Expression): Boolean = e.tpe match {
    case _: ir.SIntType => true
    case _ => false
  }
  private def getWidth(e: ir.Expression): Int = getWidth(e.tpe)
}
