// SPDX-License-Identifier: Apache-2.0

package firrtl

import logger.LazyLogging
import firrtl.ir._

/** Definitions and Utility functions for [[ir.PrimOp]]s */
object PrimOps extends LazyLogging {

  /** Addition */
  case object Add extends PrimOp {
    override def toString = "add"
  }

  /** Subtraction */
  case object Sub extends PrimOp {
    override def toString = "sub"
  }

  /** Multiplication */
  case object Mul extends PrimOp {
    override def toString = "mul"
  }

  /** Division */
  case object Div extends PrimOp {
    override def toString = "div"
  }

  /** Remainder */
  case object Rem extends PrimOp {
    override def toString = "rem"
  }

  /** Less Than */
  case object Lt extends PrimOp {
    override def toString = "lt"
  }

  /** Less Than Or Equal To */
  case object Leq extends PrimOp {
    override def toString = "leq"
  }

  /** Greater Than */
  case object Gt extends PrimOp {
    override def toString = "gt"
  }

  /** Greater Than Or Equal To */
  case object Geq extends PrimOp {
    override def toString = "geq"
  }

  /** Equal To */
  case object Eq extends PrimOp {
    override def toString = "eq"
  }

  /** Not Equal To */
  case object Neq extends PrimOp {
    override def toString = "neq"
  }

  /** Padding */
  case object Pad extends PrimOp {
    override def toString = "pad"
  }

  /** Static Shift Left */
  case object Shl extends PrimOp {
    override def toString = "shl"
  }

  /** Static Shift Right */
  case object Shr extends PrimOp {
    override def toString = "shr"
  }

  /** Dynamic Shift Left */
  case object Dshl extends PrimOp {
    override def toString = "dshl"
  }

  /** Dynamic Shift Right */
  case object Dshr extends PrimOp {
    override def toString = "dshr"
  }

  /** Arithmetic Convert to Signed */
  case object Cvt extends PrimOp {
    override def toString = "cvt"
  }

  /** Negate */
  case object Neg extends PrimOp {
    override def toString = "neg"
  }

  /** Bitwise Complement */
  case object Not extends PrimOp {
    override def toString = "not"
  }

  /** Bitwise And */
  case object And extends PrimOp {
    override def toString = "and"
  }

  /** Bitwise Or */
  case object Or extends PrimOp {
    override def toString = "or"
  }

  /** Bitwise Exclusive Or */
  case object Xor extends PrimOp {
    override def toString = "xor"
  }

  /** Bitwise And Reduce */
  case object Andr extends PrimOp {
    override def toString = "andr"
  }

  /** Bitwise Or Reduce */
  case object Orr extends PrimOp {
    override def toString = "orr"
  }

  /** Bitwise Exclusive Or Reduce */
  case object Xorr extends PrimOp {
    override def toString = "xorr"
  }

  /** Concatenate */
  case object Cat extends PrimOp {
    override def toString = "cat"
  }

  /** Bit Extraction */
  case object Bits extends PrimOp {
    override def toString = "bits"
  }

  /** Head */
  case object Head extends PrimOp {
    override def toString = "head"
  }

  /** Tail */
  case object Tail extends PrimOp {
    override def toString = "tail"
  }

  /** Interpret As UInt */
  case object AsUInt extends PrimOp {
    override def toString = "asUInt"
  }

  /** Interpret As SInt */
  case object AsSInt extends PrimOp {
    override def toString = "asSInt"
  }

  /** Interpret As Clock */
  case object AsClock extends PrimOp {
    override def toString = "asClock"
  }

  /** Interpret As AsyncReset */
  case object AsAsyncReset extends PrimOp {
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

  /** Gets the corresponding [[ir.PrimOp]] from its String representation */
  def fromString(op: String): PrimOp = strToPrimOp(op)
}
