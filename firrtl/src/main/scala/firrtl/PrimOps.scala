// SPDX-License-Identifier: Apache-2.0

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
        case _ => UnknownType
      }
    }
  }

  /** Subtraction */
  case object Sub extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(IsMax(w1(e), w2(e)), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "sub"
  }

  /** Multiplication */
  case object Mul extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(IsAdd(w1(e), w2(e)))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "mul"
  }

  /** Division */
  case object Div extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(w1(e))
      case (_: SIntType, _: SIntType) => SIntType(IsAdd(w1(e), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "div"
  }

  /** Remainder */
  case object Rem extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => UIntType(MIN(w1(e), w2(e)))
      case (_: SIntType, _: SIntType) => SIntType(MIN(w1(e), w2(e)))
      case _ => UnknownType
    }
    override def toString = "rem"
  }

  /** Less Than */
  case object Lt extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "lt"
  }

  /** Less Than Or Equal To */
  case object Leq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "leq"
  }

  /** Greater Than */
  case object Gt extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "gt"
  }

  /** Greater Than Or Equal To */
  case object Geq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "geq"
  }

  /** Equal To */
  case object Eq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "eq"
  }

  /** Not Equal To */
  case object Neq extends PrimOp {
    override def propagateType(e: DoPrim): Type = (t1(e), t2(e)) match {
      case (_: UIntType, _: UIntType) => Utils.BoolType
      case (_: SIntType, _: SIntType) => Utils.BoolType
      case _ => UnknownType
    }
    override def toString = "neq"
  }

  /** Padding */
  case object Pad extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsMax(w1(e), c1(e)))
      case _: SIntType => SIntType(IsMax(w1(e), c1(e)))
      case _ => UnknownType
    }
    override def toString = "pad"
  }

  /** Static Shift Left */
  case object Shl extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsAdd(w1(e), c1(e)))
      case _: SIntType => SIntType(IsAdd(w1(e), c1(e)))
      case _ => UnknownType
    }
    override def toString = "shl"
  }

  /** Static Shift Right */
  case object Shr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsMax(IsAdd(w1(e), IsNeg(c1(e))), IntWidth(1)))
      case _: SIntType => SIntType(IsMax(IsAdd(w1(e), IsNeg(c1(e))), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "shr"
  }

  /** Dynamic Shift Left */
  case object Dshl extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(IsAdd(w1(e), IsAdd(IsPow(w2(e)), Closed(-1))))
      case _: SIntType => SIntType(IsAdd(w1(e), IsAdd(IsPow(w2(e)), Closed(-1))))
      case _ => UnknownType
    }
    override def toString = "dshl"
  }

  /** Dynamic Shift Right */
  case object Dshr extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(w1(e))
      case _: SIntType => SIntType(w1(e))
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
      case (
            _: UIntType | _: SIntType,
            _: UIntType | _: SIntType
          ) =>
        UIntType(IsAdd(w1(e), w2(e)))
      case (t1, t2) => UnknownType
    }
    override def toString = "cat"
  }

  /** Bit Extraction */
  case object Bits extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) =>
        UIntType(IsAdd(IsAdd(c1(e), IsNeg(c2(e))), IntWidth(1)))
      case _ => UnknownType
    }
    override def toString = "bits"
  }

  /** Head */
  case object Head extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) => UIntType(c1(e))
      case _ => UnknownType
    }
    override def toString = "head"
  }

  /** Tail */
  case object Tail extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case (_: UIntType | _: SIntType) => UIntType(IsAdd(w1(e), IsNeg(c1(e))))
      case _ => UnknownType
    }
    override def toString = "tail"
  }

  /** Interpret As UInt */
  case object AsUInt extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => UIntType(w1(e))
      case _: SIntType => UIntType(w1(e))
      case ClockType      => UIntType(IntWidth(1))
      case AsyncResetType => UIntType(IntWidth(1))
      case ResetType      => UIntType(IntWidth(1))
      case AnalogType(w)  => UIntType(w1(e))
      case _              => UnknownType
    }
    override def toString = "asUInt"
  }

  /** Interpret As SInt */
  case object AsSInt extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => SIntType(w1(e))
      case _: SIntType => SIntType(w1(e))
      case ClockType      => SIntType(IntWidth(1))
      case AsyncResetType => SIntType(IntWidth(1))
      case ResetType      => SIntType(IntWidth(1))
      case _: AnalogType => SIntType(w1(e))
      case _ => UnknownType
    }
    override def toString = "asSInt"
  }

  /** Interpret As Clock */
  case object AsClock extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType => ClockType
      case _: SIntType => ClockType
      case ClockType      => ClockType
      case AsyncResetType => ClockType
      case ResetType      => ClockType
      case _: AnalogType => ClockType
      case _ => UnknownType
    }
    override def toString = "asClock"
  }

  /** Interpret As AsyncReset */
  case object AsAsyncReset extends PrimOp {
    override def propagateType(e: DoPrim): Type = t1(e) match {
      case _: UIntType | _: SIntType | _: AnalogType | ClockType | AsyncResetType | ResetType =>
        AsyncResetType
      case _ => UnknownType
    }
    override def toString = "asAsyncReset"
  }

  // format: off
  private[firrtl] lazy val builtinPrimOps: Seq[PrimOp] = Seq(
    Add, Sub, Mul, Div, Rem, Lt, Leq, Gt, Geq, Eq, Neq, Pad, AsUInt, AsSInt, AsClock,
    AsAsyncReset, Shl, Shr, Dshl, Dshr, Neg, Cvt, Not, And, Or, Xor, Andr, Orr, Xorr, Cat, Bits,
    Head, Tail
  )
  // format: on
  private lazy val strToPrimOp: Map[String, PrimOp] = {
    builtinPrimOps.map { case op: PrimOp => op.toString -> op }.toMap
  }

  /** Seq of String representations of [[ir.PrimOp]]s */
  lazy val listing: Seq[String] = builtinPrimOps.map(_.toString)

  /** Gets the corresponding [[ir.PrimOp]] from its String representation */
  def fromString(op: String): PrimOp = strToPrimOp(op)

  // Width Constraint Functions
  def PLUS(w1:  Width, w2: Width): Constraint = IsAdd(w1, w2)
  def MAX(w1:   Width, w2: Width): Constraint = IsMax(w1, w2)
  def MINUS(w1: Width, w2: Width): Constraint = IsAdd(w1, IsNeg(w2))
  def MIN(w1:   Width, w2: Width): Constraint = IsMin(w1, w2)

  def set_primop_type(e: DoPrim): DoPrim = DoPrim(e.op, e.args, e.consts, e.op.propagateType(e))
}
