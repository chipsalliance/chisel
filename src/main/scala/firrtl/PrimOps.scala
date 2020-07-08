// See LICENSE for license details.

package firrtl

import logger.LazyLogging
import firrtl.ir._
import Implicits.{constraint2bound, constraint2width, width2constraint}
import firrtl.constraint._

/** Definitions and Utility functions for [[ir.PrimOp]]s */
object PrimOps extends LazyLogging {
  def t1(e: DoPrim): Type = e.args.head.tpe
  def t2(e: DoPrim): Type = e.args(1).tpe
  def t3(e: DoPrim): Type = e.args(2).tpe
  def w1(e: DoPrim): Width = getWidth(t1(e))
  def w2(e: DoPrim): Width = getWidth(t2(e))
  def p1(e: DoPrim): Width = t1(e) match {
    case FixedType(w, p) => p
    case IntervalType(min, max, p) => p
    case _ => sys.error(s"Cannot get binary point from ${t1(e)}")
  }
  def p2(e: DoPrim): Width = t2(e) match {
    case FixedType(w, p) => p
    case IntervalType(min, max, p) => p
    case _ => sys.error(s"Cannot get binary point from ${t1(e)}")
  }
  def c1(e: DoPrim) = IntWidth(e.consts.head)
  def c2(e: DoPrim) = IntWidth(e.consts(1))
  def o1(e: DoPrim) = e.consts(0)
  def o2(e: DoPrim) = e.consts(1)
  def o3(e: DoPrim) = e.consts(2)

  /** Addition */
  case object Add extends PrimOp {
    override def toString = "add"
    override def propagateType(e: DoPrim): Type = {
      (t1(e), t2(e)) match {
        case (_: UIntType, _: UIntType) => UIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
        case (_: SIntType, _: SIntType) => SIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
        case (_: FixedType, _: FixedType) => FixedType(IsAdd(IsAdd(IsMax(p1(e), p2(e)), IsMax(IsAdd(w1(e), IsNeg(p1(e))), IsAdd(w2(e), IsNeg(p2(e))))), IntWidth(1)), IsMax(p1(e), p2(e)))
        case (IntervalType(l1, u1, p1), IntervalType(l2, u2, p2)) => IntervalType(IsAdd(l1, l2), IsAdd(u1, u2), IsMax(p1, p2))
        case _ => UnknownType
      }
    }
  }

  /** Subtraction */
  case object Sub extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
      case (_: FixedType, _: FixedType) => FixedType(IsAdd(IsAdd(IsMax(p1(e), p2(e)),IsMax(IsAdd(w1(e), IsNeg(p1(e))), IsAdd(w2(e), IsNeg(p2(e))))),IntWidth(1)), IsMax(p1(e), p2(e)))
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, p2)) => IntervalType(IsAdd(l1, IsNeg(u2)), IsAdd(u1, IsNeg(l2)), IsMax(p1, p2))
      case _ => UnknownType
    }
    override def toString = "sub"
  }

  /** Multiplication */
  case object Mul extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(IsAdd(w1(e), w2(e)))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(w1(e), w2(e)))
      case (_: FixedType, _: FixedType) => FixedType(IsAdd(w1(e), w2(e)), IsAdd(p1(e), p2(e)))
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, p2)) =>
        IntervalType(
          IsMin(Seq(IsMul(l1, l2), IsMul(l1, u2), IsMul(u1, l2), IsMul(u1, u2))),
          IsMax(Seq(IsMul(l1, l2), IsMul(l1, u2), IsMul(u1, l2), IsMul(u1, u2))),
          IsAdd(p1, p2)
        )
      case _ => UnknownType
    }
    override def toString = "mul" }

  /** Division */
  case object Div extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(w1(e))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(w1(e), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "div" }

  /** Remainder */
  case object Rem extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(MIN(w1(e), w2(e)))
      case (_: SIntType, _: SIntType) => SIntType(MIN(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "rem" }
  /** Less Than */
  case object Lt extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "lt" }
  /** Less Than Or Equal To */
  case object Leq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "leq" }
  /** Greater Than */
  case object Gt extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "gt" }
  /** Greater Than Or Equal To */
  case object Geq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "geq" }
  /** Equal To */
  case object Eq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "eq" }
  /** Not Equal To */
  case object Neq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case (_: FixedType, _: FixedType) => Utils.BoolType
      case (_: IntervalType, _: IntervalType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "neq" }
  /** Padding */
  case object Pad extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsMax(w1(e), c1(e)))
      case _: SIntType => SIntType(IsMax(w1(e), c1(e)))
      case _: FixedType => FixedType(IsMax(w1(e), c1(e)), p1(e))
      case _ => UnknownType
    }
    override def toString = "pad" }
  /** Static Shift Left */
  case object Shl extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsAdd(w1(e), c1(e)))
      case _: SIntType => SIntType(IsAdd(w1(e), c1(e)))
      case _: FixedType => FixedType(IsAdd(w1(e),c1(e)), p1(e))
      case IntervalType(l, u, p) => IntervalType(IsMul(l, Closed(BigDecimal(BigInt(1) << o1(e).toInt))), IsMul(u, Closed(BigDecimal(BigInt(1) << o1(e).toInt))), p)
      case _ => UnknownType
    }
    override def toString = "shl" }
  /** Static Shift Right */
  case object Shr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsMax(IsAdd(w1(e), IsNeg(c1(e))), IntWidth(1)))
      case _: SIntType => SIntType(IsMax(IsAdd(w1(e), IsNeg(c1(e))), IntWidth(1)))
      case _: FixedType => FixedType(IsMax(IsMax(IsAdd(w1(e), IsNeg(c1(e))), IntWidth(1)), p1(e)), p1(e))
      case IntervalType(l, u, IntWidth(p)) =>
        val shiftMul = Closed(BigDecimal(1) / BigDecimal(BigInt(1) << o1(e).toInt))
        // BP is inferred at this point
        val bpRes = Closed(BigDecimal(1) / BigDecimal(BigInt(1) << p.toInt))
        val bpResInv = Closed(BigDecimal(BigInt(1) << p.toInt))
        val newL = IsMul(IsFloor(IsMul(IsMul(l, shiftMul), bpResInv)), bpRes)
        val newU = IsMul(IsFloor(IsMul(IsMul(u, shiftMul), bpResInv)), bpRes)
        // BP doesn't grow
        IntervalType(newL, newU, IntWidth(p))
      case _ => UnknownType
    }
    override def toString = "shr"
  }
  /** Dynamic Shift Left */
  case object Dshl extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsAdd(w1(e), IsAdd(IsPow(w2(e)), Closed(-1))))
      case _: SIntType => SIntType(IsAdd(w1(e), IsAdd(IsPow(w2(e)), Closed(-1))))
      case _: FixedType => FixedType(IsAdd(w1(e), IsAdd(IsPow(w2(e)), Closed(-1))), p1(e))
      case IntervalType(l, u, p) =>
        val maxShiftAmt = IsAdd(IsPow(w2(e)), Closed(-1))
        val shiftMul = IsPow(maxShiftAmt)
        // Magnitude matters! i.e. if l is negative, shifting by the largest amount makes the outcome more negative
        // whereas if l is positive, shifting by the largest amount makes the outcome more positive (in this case, the lower bound is the previous l)
        val newL = IsMin(l, IsMul(l, shiftMul))
        val newU = IsMax(u, IsMul(u, shiftMul))
        // BP doesn't grow
        IntervalType(newL, newU, p)
      case _ => UnknownType
    }
    override def toString = "dshl"
  }
  /** Dynamic Shift Right */
  case object Dshr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(w1(e))
      case _: SIntType => SIntType(w1(e))
      case _: FixedType => FixedType(w1(e), p1(e))
      // Decreasing magnitude -- don't need more bits
      case IntervalType(l, u, p) => IntervalType(l, u, p)
      case _ => UnknownType
    }
    override def toString = "dshr"
  }
  /** Arithmetic Convert to Signed */
  case object Cvt extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => SIntType(IsAdd(w1(e), IntWidth(1)))
      case _: SIntType => SIntType(w1(e))
      case _ => UnknownType
    }
    override def toString = "cvt"
  }
  /** Negate */
  case object Neg extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => SIntType(IsAdd(w1(e), IntWidth(1)))
      case _: SIntType => SIntType(IsAdd(w1(e), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "neg"
  }
  /** Bitwise Complement */
  case object Not extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(w1(e))
      case _: SIntType => UIntType(w1(e))
      case _ => UnknownType
    }
    override def toString = "not"
  }
  /** Bitwise And */
  case object And extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: SIntType | _: UIntType, _: SIntType | _: UIntType) => UIntType(IsMax(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "and"
  }
  /** Bitwise Or */
  case object Or extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: SIntType | _: UIntType, _: SIntType | _: UIntType) => UIntType(IsMax(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "or"
  }
  /** Bitwise Exclusive Or */
  case object Xor extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: SIntType | _: UIntType, _: SIntType | _: UIntType) => UIntType(IsMax(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "xor"
  }
  /** Bitwise And Reduce */
  case object Andr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "andr"
  }
  /** Bitwise Or Reduce */
  case object Orr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "orr"
  }
  /** Bitwise Exclusive Or Reduce */
  case object Xorr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "xorr"
  }
  /** Concatenate */
  case object Cat extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType | _: SIntType | _: FixedType | _: IntervalType, _: UIntType | _: SIntType | _: FixedType | _: IntervalType) => UIntType(IsAdd(w1(e), w2(e)))
      case (t1, t2) => UnknownType
    }
    override def toString = "cat"
  }
  /** Bit Extraction */
  case object Bits extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType | _: FixedType | _: IntervalType) => UIntType(IsAdd(IsAdd(c1(e), IsNeg(c2(e))), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "bits"
  }
  /** Head */
  case object Head extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType | _: FixedType | _: IntervalType) => UIntType(c1(e))
      case _ => UnknownType
    }
    override def toString = "head"
  }
  /** Tail */
  case object Tail extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType | _: FixedType | _: IntervalType) => UIntType(IsAdd(w1(e), IsNeg(c1(e))))
      case _ => UnknownType
    }
    override def toString = "tail"
  }
  /** Increase Precision **/
  case object IncP extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: FixedType => FixedType(IsAdd(w1(e),c1(e)), IsAdd(p1(e), c1(e)))
      // Keeps the same exact value, but adds more precision for the future i.e. aaa.bbb -> aaa.bbb00
      case IntervalType(l, u, p) => IntervalType(l, u, IsAdd(p, c1(e)))
      case _ => UnknownType
    }
    override def toString = "incp"
  }
  /** Decrease Precision **/
  case object DecP extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: FixedType => FixedType(IsAdd(w1(e),IsNeg(c1(e))), IsAdd(p1(e), IsNeg(c1(e))))
      case IntervalType(l, u, IntWidth(p)) =>
        val shiftMul = Closed(BigDecimal(1) / BigDecimal(BigInt(1) << o1(e).toInt))
        // BP is inferred at this point
        // newBPRes is the only difference in calculating bpshr from shr
        // y = floor(x * 2^(-amt + bp)) gets rid of precision --> y * 2^(-bp + amt)
        // without amt, same op as shr
        val newBPRes = Closed(BigDecimal(BigInt(1) << o1(e).toInt) / BigDecimal(BigInt(1) << p.toInt))
        val bpResInv = Closed(BigDecimal(BigInt(1) << p.toInt))
        val newL = IsMul(IsFloor(IsMul(IsMul(l, shiftMul), bpResInv)), newBPRes)
        val newU = IsMul(IsFloor(IsMul(IsMul(u, shiftMul), bpResInv)), newBPRes)
        // BP doesn't grow
        IntervalType(newL, newU, IsAdd(IntWidth(p), IsNeg(c1(e))))
      case _ => UnknownType
    }
    override def toString = "decp"
  }
  /** Set Precision **/
  case object SetP extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: FixedType => FixedType(IsAdd(c1(e), IsAdd(w1(e), IsNeg(p1(e)))), c1(e))
      case IntervalType(l, u, p) =>
        val newBPResInv = Closed(BigDecimal(BigInt(1) << o1(e).toInt))
        val newBPRes = Closed(BigDecimal(1) / BigDecimal(BigInt(1) << o1(e).toInt))
        val newL = IsMul(IsFloor(IsMul(l, newBPResInv)), newBPRes)
        val newU = IsMul(IsFloor(IsMul(u, newBPResInv)), newBPRes)
        IntervalType(newL, newU, c1(e))
      case _ => UnknownType
    }
    override def toString = "setp"
  }
  /** Interpret As UInt */
  case object AsUInt extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(w1(e))
      case _: SIntType => UIntType(w1(e))
      case _: FixedType => UIntType(w1(e))
      case ClockType => UIntType(IntWidth(1))
      case AsyncResetType => UIntType(IntWidth(1))
      case ResetType => UIntType(IntWidth(1))
      case AnalogType(w) => UIntType(w1(e))
      case _: IntervalType => UIntType(w1(e))
      case _ => UnknownType
    }
    override def toString = "asUInt"
  }
  /** Interpret As SInt */
  case object AsSInt extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => SIntType(w1(e))
      case _: SIntType => SIntType(w1(e))
      case _: FixedType => SIntType(w1(e))
      case ClockType => SIntType(IntWidth(1))
      case AsyncResetType => SIntType(IntWidth(1))
      case ResetType => SIntType(IntWidth(1))
      case _: AnalogType => SIntType(w1(e))
      case _: IntervalType => SIntType(w1(e))
      case _ => UnknownType
    }
    override def toString = "asSInt"
  }
  /** Interpret As Clock */
  case object AsClock extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => ClockType
      case _: SIntType => ClockType
      case ClockType => ClockType
      case AsyncResetType => ClockType
      case ResetType => ClockType
      case _: AnalogType => ClockType
      case _: IntervalType => ClockType
      case _ => UnknownType
    }
    override def toString = "asClock"
  }
  /** Interpret As AsyncReset */
  case object AsAsyncReset extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType | _: SIntType | _: AnalogType | ClockType | AsyncResetType | ResetType | _: IntervalType | _: FixedType => AsyncResetType
      case _ => UnknownType
    }
    override def toString = "asAsyncReset"
  }
  /** Interpret as Fixed Point **/
  case object AsFixedPoint extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => FixedType(w1(e), c1(e))
      case _: SIntType => FixedType(w1(e), c1(e))
      case _: FixedType => FixedType(w1(e), c1(e))
      case ClockType => FixedType(IntWidth(1), c1(e))
      case _: AnalogType => FixedType(w1(e), c1(e))
      case AsyncResetType => FixedType(IntWidth(1), c1(e))
      case ResetType => FixedType(IntWidth(1), c1(e))
      case _: IntervalType => FixedType(w1(e), c1(e))
      case _ => UnknownType
    }
    override def toString = "asFixedPoint"
  }
  /** Interpret as Interval (closed lower bound, closed upper bound, binary point) **/
  case object AsInterval extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      // Chisel shifts up and rounds first.
      case _: UIntType | _: SIntType | _: FixedType | ClockType | AsyncResetType | ResetType | _: AnalogType | _: IntervalType    =>
        IntervalType(Closed(BigDecimal(o1(e))/BigDecimal(BigInt(1) << o3(e).toInt)), Closed(BigDecimal(o2(e))/BigDecimal(BigInt(1) << o3(e).toInt)), IntWidth(o3(e)))
      case _ => UnknownType
    }
    override def toString = "asInterval"
  }
  /** Try to fit the first argument into the type of the smaller argument **/
  case object Squeeze extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, _)) =>
        val low = IsMax(l1, l2)
        val high = IsMin(u1, u2)
        IntervalType(IsMin(low, u2), IsMax(l2, high), p1)
      case _ => UnknownType
    }
    override def toString = "squz"
  }
  /** Wrap First Operand Around Range/Width of Second Operand **/
  case object Wrap extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, _)) => IntervalType(l2, u2, p1)
      case _ => UnknownType
    }
    override def toString = "wrap"
  }
  /** Clip First Operand At Range/Width of Second Operand **/
  case object Clip extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (IntervalType(l1, u1, p1), IntervalType(l2, u2, _)) =>
        val low = IsMax(l1, l2)
        val high = IsMin(u1, u2)
        IntervalType(IsMin(low, u2), IsMax(l2, high), p1)
      case _ => UnknownType
    }
    override def toString = "clip"
  }

  private[firrtl] lazy val builtinPrimOps: Seq[PrimOp] =
    Seq(Add, Sub, Mul, Div, Rem, Lt, Leq, Gt, Geq, Eq, Neq, Pad, AsUInt, AsSInt, AsInterval, AsClock, AsAsyncReset, Shl, Shr,
        Dshl, Dshr, Neg, Cvt, Not, And, Or, Xor, Andr, Orr, Xorr, Cat, Bits, Head, Tail, AsFixedPoint, IncP, DecP,
        SetP, Wrap, Clip, Squeeze)
  private lazy val strToPrimOp: Map[String, PrimOp] = {
    builtinPrimOps.map { case op : PrimOp=> op.toString -> op }.toMap
  }

  /** Seq of String representations of [[ir.PrimOp]]s */
  lazy val listing: Seq[String] = builtinPrimOps map (_.toString)
  /** Gets the corresponding [[ir.PrimOp]] from its String representation */
  def fromString(op: String): PrimOp = strToPrimOp(op)

  // Width Constraint Functions
  def PLUS(w1: Width, w2: Width): Constraint = IsAdd(w1, w2)
  def MAX(w1: Width, w2: Width): Constraint = IsMax(w1, w2)
  def MINUS(w1: Width, w2: Width): Constraint = IsAdd(w1, IsNeg(w2))
  def MIN(w1: Width, w2: Width): Constraint = IsMin(w1, w2)

  def set_primop_type(e: DoPrim): DoPrim = DoPrim(e.op, e.args, e.consts, e.op.propagateType(e))
}
