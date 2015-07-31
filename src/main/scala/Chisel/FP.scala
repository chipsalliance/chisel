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
import Builder._
import ChiselError._
import Commands.NoLits

/// FLO

object Flo {
  def apply(x: Float): Flo = floLit(x)
  def apply(x: Double): Flo = Flo(x.toFloat)
  def floLit(value: Float): Flo = {
    val b = new Flo(NO_DIR, Some(value))
    pushCommand(DefFlo(b, value))
    b
  }
  def apply(dir: Direction = null): Flo = 
    new Flo(dir)
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

class Flo(dir: Direction = NO_DIR, val value:Option[Float] = None) extends Element(dir, 32) with Num[Flo] {
  type T = Flo;
  override def floLitValue: Float = value.get
  def cloneTypeWidth(width: Int): this.type = cloneType
  override def fromBits(n: Bits): this.type = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, BitsToFlo, Array(this.ref), NoLits))
    d
  }
  override def toBits: UInt = {
    val d = UInt(dir, 32)
    pushCommand(DefPrim(d, d.toType, FloToBits, Array(this.ref), NoLits))
    d
  }
  def toType: Kind = FloType(isFlip)
  def cloneType: this.type = new Flo(dir).asInstanceOf[this.type]
  def flatten: IndexedSeq[Bits] = IndexedSeq(toBits)

  def fromInt(x: Int): Flo = 
    Flo(x.toFloat).asInstanceOf[this.type]

  private def flo_unop(op: PrimOp): Flo = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref), NoLits))
    d
  }
  private def flo_binop(op: PrimOp, other: Flo): Flo = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def flo_compop(op: PrimOp, other: Flo): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }

  def unary_-() = flo_unop(FloNeg)
  def +  (b: Flo) = flo_binop(FloAdd, b)
  def -  (b: Flo) = flo_binop(FloSub, b)
  def *  (b: Flo) = flo_binop(FloMul, b)
  def /  (b: Flo) = flo_binop(FloDiv, b)
  def %  (b: Flo) = flo_binop(FloMod, b)
  def ===(b: Flo) = flo_compop(FloEqual, b)
  def != (b: Flo) = flo_compop(FloNotEqual, b)
  def >  (b: Flo) = flo_compop(FloGreater, b)
  def <  (b: Flo) = flo_compop(FloLess, b)
  def <= (b: Flo) = flo_compop(FloLessEqual, b)
  def >= (b: Flo) = flo_compop(FloGreaterEqual, b)
  def pow (b: Flo) = flo_binop(FloPow, b)
  def sin = flo_unop(FloSin)
  def cos = flo_unop(FloCos)
  def tan = flo_unop(FloTan)
  def asin = flo_unop(FloAsin)
  def acos = flo_unop(FloAcos)
  def atan = flo_unop(FloAtan)
  def sqrt = flo_unop(FloSqrt)
  def floor = flo_unop(FloFloor)
  def ceil = flo_unop(FloCeil)
  def round = flo_unop(FloRound)
  def log = flo_unop(FloLog)
  def toSInt () = SInt(OUTPUT).fromBits(toBits)
  def toUInt () = UInt(OUTPUT).fromBits(toBits)
}

/// DBL

import java.lang.Double.doubleToLongBits

object Dbl {
  def apply(x: Float): Dbl = Dbl(x.toDouble);
  def apply(x: Double): Dbl = dblLit(x)
  def dblLit(value: Double): Dbl = {
    val b = new Dbl(NO_DIR, Some(value))
    pushCommand(DefDbl(b, value))
    b
  }
  def apply(dir: Direction = NO_DIR): Dbl = 
    new Dbl(dir)
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

class Dbl(dir: Direction, val value: Option[Double] = None) extends Element(dir, 64) with Num[Dbl] { 
  // setIsSigned

  // override def setIsTypeNode = {inputs(0).setIsSigned; super.setIsTypeNode}

  type T = Dbl;
  override def dblLitValue: Double = value.get
  def cloneTypeWidth(width: Int): this.type = cloneType
  override def fromBits(n: Bits): this.type = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, BitsToDbl, Array(this.ref), NoLits))
    d
  }
  override def toBits: UInt = {
    val d = UInt(dir, 64)
    pushCommand(DefPrim(d, d.toType, DblToBits, Array(this.ref), NoLits))
    d
  }
  def toType: Kind = DblType(isFlip)
  def cloneType: this.type = new Dbl(dir).asInstanceOf[this.type]
  def flatten: IndexedSeq[Bits] = IndexedSeq(toBits)

  def fromInt(x: Int): this.type = 
    Dbl(x.toDouble).asInstanceOf[this.type]

  private def dbl_unop(op: PrimOp): Dbl = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref), NoLits))
    d
  }
  private def dbl_binop(op: PrimOp, other: Dbl): Dbl = {
    val d = cloneType
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def dbl_compop(op: PrimOp, other: Dbl): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }

  def unary_-() = dbl_unop(DblNeg)
  def +  (b: Dbl) = dbl_binop(DblAdd, b)
  def -  (b: Dbl) = dbl_binop(DblSub, b)
  def *  (b: Dbl) = dbl_binop(DblMul, b)
  def /  (b: Dbl) = dbl_binop(DblDiv, b)
  def %  (b: Dbl) = dbl_binop(DblMod, b)
  def ===(b: Dbl) = dbl_compop(DblEqual, b)
  def != (b: Dbl) = dbl_compop(DblNotEqual, b)
  def >  (b: Dbl) = dbl_compop(DblGreater, b)
  def <  (b: Dbl) = dbl_compop(DblLess, b)
  def <= (b: Dbl) = dbl_compop(DblLessEqual, b)
  def >= (b: Dbl) = dbl_compop(DblGreaterEqual, b)
  def pow (b: Dbl) = dbl_binop(DblPow, b)
  def sin = dbl_unop(DblSin)
  def cos = dbl_unop(DblCos)
  def tan = dbl_unop(DblTan)
  def asin = dbl_unop(DblAsin)
  def acos = dbl_unop(DblAcos)
  def atan = dbl_unop(DblAtan)
  def sqrt = dbl_unop(DblSqrt)
  def floor = dbl_unop(DblFloor)
  def ceil = dbl_unop(DblCeil)
  def round = dbl_unop(DblRound)
  def log = dbl_unop(DblLog)
  def toSInt () = SInt(OUTPUT).fromBits(toBits)
  def toUInt () = UInt(OUTPUT).fromBits(toBits)
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
