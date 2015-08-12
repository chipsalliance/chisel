/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:

    * Redistributions of source code must retain the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the Regents nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
*/

package Chisel
import Chisel._
import Builder.pushOp
import ChiselError._

/// FLO

case class FloLit(num: Float) extends Arg {
  def name = s"Flo(${num.toString})"
}

case class DblLit(num: Double) extends Arg {
  def name = s"Dbl(${num.toString})"
}

object Flo {
  def apply(x: Float): Flo = new Flo(NO_DIR, Some(FloLit(x)))
  def apply(x: Double): Flo = Flo(x.toFloat)
  def apply(dir: Direction = null): Flo = new Flo(dir)
}

object FloPrimOp {
  val FloNeg = PrimOp("flo-neg")
  val FloAdd = PrimOp("flo-add")
  val FloSub = PrimOp("flo-sub")
  val FloMul = PrimOp("flo-mul")
  val FloDiv = PrimOp("flo-div")
  val FloMod = PrimOp("flo-mod")
  val FloEqual = PrimOp("flo-equal")
  val FloNotEqual = PrimOp("flo-not-equal")
  val FloGreater = PrimOp("flo-greater")
  val FloLess = PrimOp("flo-less")
  val FloLessEqual = PrimOp("flo-less-equal")
  val FloGreaterEqual = PrimOp("flo-greater-equal")
  val FloPow = PrimOp("flo-pow")
  val FloSin = PrimOp("flo-sin")
  val FloCos = PrimOp("flo-cos")
  val FloTan = PrimOp("flo-tan")
  val FloAsin = PrimOp("flo-asin")
  val FloAcos = PrimOp("flo-acos")
  val FloAtan = PrimOp("flo-atan")
  val FloSqrt = PrimOp("flo-sqrt")
  val FloFloor = PrimOp("flo-floor")
  val FloCeil = PrimOp("flo-ceil")
  val FloRound = PrimOp("flo-round")
  val FloLog = PrimOp("flo-log")
  val FloToBits = PrimOp("flo-to-bits")
  val BitsToFlo = PrimOp("bits-to-flo")
}
import FloPrimOp._

sealed abstract class FloBase[T <: Data](dir: Direction, width: Width) extends Element(dir, width) {
  protected def unop(op: PrimOp): T =
    pushOp(DefPrim(cloneType, op, this.ref)).asInstanceOf[T]
  protected def binop(op: PrimOp, other: T): T =
    pushOp(DefPrim(cloneType, op, this.ref, other.ref)).asInstanceOf[T]
  protected def compop(op: PrimOp, other: T): Bool =
    pushOp(DefPrim(Bool(), op, this.ref, other.ref))

  def toUInt = toBits
}

class Flo(dir: Direction = NO_DIR, val value:Option[FloLit] = None) extends FloBase[Flo](dir, Width(32)) with Num[Flo] {
  type T = Flo;
  override def floLitValue: Float = value.get.num
  def cloneTypeWidth(width: Width): this.type = cloneType
  override def fromBits(n: Bits): this.type =
    pushOp(DefPrim(cloneType, BitsToFlo, this.ref)).asInstanceOf[this.type]
  override def toBits: UInt =
    pushOp(DefPrim(UInt(width=32), FloToBits, this.ref))
  def toType: Kind = FloType(isFlip)
  def cloneType: this.type = new Flo(dir).asInstanceOf[this.type]

  def fromInt(x: Int): Flo = 
    Flo(x.toFloat).asInstanceOf[this.type]

  def unary_-() = unop(FloNeg)
  def +  (b: Flo) = binop(FloAdd, b)
  def -  (b: Flo) = binop(FloSub, b)
  def *  (b: Flo) = binop(FloMul, b)
  def /  (b: Flo) = binop(FloDiv, b)
  def %  (b: Flo) = binop(FloMod, b)
  def ===(b: Flo) = compop(FloEqual, b)
  def != (b: Flo) = compop(FloNotEqual, b)
  def >  (b: Flo) = compop(FloGreater, b)
  def <  (b: Flo) = compop(FloLess, b)
  def <= (b: Flo) = compop(FloLessEqual, b)
  def >= (b: Flo) = compop(FloGreaterEqual, b)
  def pow (b: Flo) = binop(FloPow, b)
  def sin = unop(FloSin)
  def cos = unop(FloCos)
  def tan = unop(FloTan)
  def asin = unop(FloAsin)
  def acos = unop(FloAcos)
  def atan = unop(FloAtan)
  def sqrt = unop(FloSqrt)
  def floor = unop(FloFloor)
  def ceil = unop(FloCeil)
  def round = unop(FloRound)
  def log = unop(FloLog)
}

/// DBL

import java.lang.Double.doubleToLongBits

object Dbl {
  def apply(x: Float): Dbl = Dbl(x.toDouble);
  def apply(x: Double): Dbl = new Dbl(NO_DIR, Some(DblLit(x)))
  def apply(dir: Direction = NO_DIR): Dbl = new Dbl(dir)
}

object DblPrimOp {
  val DblNeg = PrimOp("dbl-neg")
  val DblAdd = PrimOp("dbl-add")
  val DblSub = PrimOp("dbl-sub")
  val DblMul = PrimOp("dbl-mul")
  val DblDiv = PrimOp("dbl-div")
  val DblMod = PrimOp("dbl-mod")
  val DblEqual = PrimOp("dbl-equal")
  val DblNotEqual = PrimOp("dbl-not-equal")
  val DblGreater = PrimOp("dbl-greater")
  val DblLess = PrimOp("dbl-less")
  val DblLessEqual = PrimOp("dbl-less-equal")
  val DblGreaterEqual = PrimOp("dbl-greater-equal")
  val DblPow = PrimOp("dbl-pow")
  val DblSin = PrimOp("dbl-sin")
  val DblCos = PrimOp("dbl-cos")
  val DblTan = PrimOp("dbl-tan")
  val DblAsin = PrimOp("dbl-asin")
  val DblAcos = PrimOp("dbl-acos")
  val DblAtan = PrimOp("dbl-atan")
  val DblSqrt = PrimOp("dbl-sqrt")
  val DblFloor = PrimOp("dbl-floor")
  val DblCeil = PrimOp("dbl-ceil")
  val DblRound = PrimOp("dbl-round")
  val DblLog = PrimOp("dbl-log")
  val DblToBits = PrimOp("dbl-to-bits")
  val BitsToDbl = PrimOp("bits-to-dbl")
}
import DblPrimOp._

class Dbl(dir: Direction, val value: Option[DblLit] = None) extends FloBase[Dbl](dir, Width(64)) with Num[Dbl] {
  type T = Dbl;
  override def dblLitValue: Double = value.get.num
  def cloneTypeWidth(width: Width): this.type = cloneType
  override def fromBits(n: Bits): this.type =
    pushOp(DefPrim(cloneType, BitsToDbl, this.ref)).asInstanceOf[this.type]
  override def toBits: UInt =
    pushOp(DefPrim(UInt(width=64), DblToBits, this.ref))
  def toType: Kind = DblType(isFlip)
  def cloneType: this.type = new Dbl(dir).asInstanceOf[this.type]

  def fromInt(x: Int): this.type = 
    Dbl(x.toDouble).asInstanceOf[this.type]

  def unary_-() = unop(DblNeg)
  def +  (b: Dbl) = binop(DblAdd, b)
  def -  (b: Dbl) = binop(DblSub, b)
  def *  (b: Dbl) = binop(DblMul, b)
  def /  (b: Dbl) = binop(DblDiv, b)
  def %  (b: Dbl) = binop(DblMod, b)
  def ===(b: Dbl) = compop(DblEqual, b)
  def != (b: Dbl) = compop(DblNotEqual, b)
  def >  (b: Dbl) = compop(DblGreater, b)
  def <  (b: Dbl) = compop(DblLess, b)
  def <= (b: Dbl) = compop(DblLessEqual, b)
  def >= (b: Dbl) = compop(DblGreaterEqual, b)
  def pow (b: Dbl) = binop(DblPow, b)
  def sin = unop(DblSin)
  def cos = unop(DblCos)
  def tan = unop(DblTan)
  def asin = unop(DblAsin)
  def acos = unop(DblAcos)
  def atan = unop(DblAtan)
  def sqrt = unop(DblSqrt)
  def floor = unop(DblFloor)
  def ceil = unop(DblCeil)
  def round = unop(DblRound)
  def log = unop(DblLog)
}

object Sin {
  def apply (x: Flo): Flo = x.sin
  def apply (x: Dbl): Dbl = x.sin
}

object Cos {
  def apply (x: Flo): Flo = x.cos
  def apply (x: Dbl): Dbl = x.cos
}

object Tan {
  def apply (x: Flo): Flo = x.tan
  def apply (x: Dbl): Dbl = x.tan
}

object ASin {
  def apply (x: Flo): Flo = x.asin
  def apply (x: Dbl): Dbl = x.asin
}

object ACos {
  def apply (x: Flo): Flo = x.acos
  def apply (x: Dbl): Dbl = x.acos
}

object ATan {
  def apply (x: Flo): Flo = x.atan
  def apply (x: Dbl): Dbl = x.atan
}

object Sqrt {
  def apply (x: Flo): Flo = x.sqrt
  def apply (x: Dbl): Dbl = x.sqrt
}

object Floor {
  def apply (x: Flo): Flo = x.floor
  def apply (x: Dbl): Dbl = x.floor
}

object Ceil {
  def apply (x: Flo): Flo = x.ceil
  def apply (x: Dbl): Dbl = x.ceil
}

object Round {
  def apply (x: Flo): Flo = x.round
  def apply (x: Dbl): Dbl = x.round
}

object Log {
  def apply (x: Flo): Flo = x.log
  def apply (x: Dbl): Dbl  = x.log
  def apply (x: Flo, p: Flo): Flo = Log(x)/Log(p)
  def apply (x: Dbl, p: Dbl): Dbl = Log(x)/Log(p)
}

object Pow {
  def apply (x: Flo, y: Flo): Flo = x.pow(y)
  def apply (x: Dbl, y: Dbl): Dbl = x.pow(y)
}
