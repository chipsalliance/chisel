/*
Copyright (c) 2014 - 2016 The Regents of the University of
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

package firrtl

import firrtl.ir._
import firrtl.Utils.{max, min, pow_minus_one}

import com.typesafe.scalalogging.LazyLogging

/** Definitions and Utility functions for [[ir.PrimOp]]s */
object PrimOps extends LazyLogging {
  /** Addition */
  case object Add extends PrimOp { override def toString = "add" }
  /** Subtraction */
  case object Sub extends PrimOp { override def toString = "sub" }
  /** Multiplication */
  case object Mul extends PrimOp { override def toString = "mul" }
  /** Division */
  case object Div extends PrimOp { override def toString = "div" }
  /** Remainder */
  case object Rem extends PrimOp { override def toString = "rem" }
  /** Less Than */
  case object Lt extends PrimOp { override def toString = "lt" }
  /** Less Than Or Equal To */
  case object Leq extends PrimOp { override def toString = "leq" }
  /** Greater Than */
  case object Gt extends PrimOp { override def toString = "gt" }
  /** Greater Than Or Equal To */
  case object Geq extends PrimOp { override def toString = "geq" }
  /** Equal To */
  case object Eq extends PrimOp { override def toString = "eq" }
  /** Not Equal To */
  case object Neq extends PrimOp { override def toString = "neq" }
  /** Padding */
  case object Pad extends PrimOp { override def toString = "pad" }
  /** Interpret As UInt */
  case object AsUInt extends PrimOp { override def toString = "asUInt" }
  /** Interpret As SInt */
  case object AsSInt extends PrimOp { override def toString = "asSInt" }
  /** Interpret As Clock */
  case object AsClock extends PrimOp { override def toString = "asClock" }
  /** Static Shift Left */
  case object Shl extends PrimOp { override def toString = "shl" }
  /** Static Shift Right */
  case object Shr extends PrimOp { override def toString = "shr" }
  /** Dynamic Shift Left */
  case object Dshl extends PrimOp { override def toString = "dshl" }
  /** Dynamic Shift Right */
  case object Dshr extends PrimOp { override def toString = "dshr" }
  /** Arithmetic Convert to Signed */
  case object Cvt extends PrimOp { override def toString = "cvt" }
  /** Negate */
  case object Neg extends PrimOp { override def toString = "neg" }
  /** Bitwise Complement */
  case object Not extends PrimOp { override def toString = "not" }
  /** Bitwise And */
  case object And extends PrimOp { override def toString = "and" }
  /** Bitwise Or */
  case object Or extends PrimOp { override def toString = "or" }
  /** Bitwise Exclusive Or */
  case object Xor extends PrimOp { override def toString = "xor" }
  /** Bitwise And Reduce */
  case object Andr extends PrimOp { override def toString = "andr" }
  /** Bitwise Or Reduce */
  case object Orr extends PrimOp { override def toString = "orr" }
  /** Bitwise Exclusive Or Reduce */
  case object Xorr extends PrimOp { override def toString = "xorr" }
  /** Concatenate */
  case object Cat extends PrimOp { override def toString = "cat" }
  /** Bit Extraction */
  case object Bits extends PrimOp { override def toString = "bits" }
  /** Head */
  case object Head extends PrimOp { override def toString = "head" }
  /** Tail */
  case object Tail extends PrimOp { override def toString = "tail" }

  private lazy val builtinPrimOps: Seq[PrimOp] =
    Seq(Add, Sub, Mul, Div, Rem, Lt, Leq, Gt, Geq, Eq, Neq, Pad, AsUInt, AsSInt, AsClock, Shl, Shr,
        Dshl, Dshr, Neg, Cvt, Not, And, Or, Xor, Andr, Orr, Xorr, Cat, Bits, Head, Tail)
  private lazy val strToPrimOp: Map[String, PrimOp] = builtinPrimOps map (op => op.toString -> op) toMap

  /** Seq of String representations of [[ir.PrimOp]]s */
  lazy val listing: Seq[String] = builtinPrimOps map (_.toString)
  /** Gets the corresponding [[ir.PrimOp]] from its String representation */
  def fromString(op: String): PrimOp = strToPrimOp(op)

  // Borrowed from Stanza implementation
   def set_primop_type (e:DoPrim) : DoPrim = {
      //println-all(["Inferencing primop type: " e])
      def PLUS (w1:Width,w2:Width) : Width = (w1, w2) match {
        case (IntWidth(i), IntWidth(j)) => IntWidth(i + j)
        case _ => PlusWidth(w1,w2)
      }
      def MAX (w1:Width,w2:Width) : Width = (w1, w2) match {
        case (IntWidth(i), IntWidth(j)) => IntWidth(max(i,j))
        case _ => MaxWidth(Seq(w1,w2))
      }
      def MINUS (w1:Width,w2:Width) : Width = (w1, w2) match {
        case (IntWidth(i), IntWidth(j)) => IntWidth(i - j)
        case _ => MinusWidth(w1,w2)
      }
      def POW (w1:Width) : Width = w1 match {
        case IntWidth(i) => IntWidth(pow_minus_one(BigInt(2), i))
        case _ => ExpWidth(w1)
      }
      def MIN (w1:Width,w2:Width) : Width = (w1, w2) match {
        case (IntWidth(i), IntWidth(j)) => IntWidth(min(i,j))
        case _ => MinWidth(Seq(w1,w2))
      }
      val o = e.op
      val a = e.args
      val c = e.consts
      def t1 () = a(0).tpe
      def t2 () = a(1).tpe
      def t3 () = a(2).tpe
      def w1 () = Utils.widthBANG(a(0).tpe)
      def w2 () = Utils.widthBANG(a(1).tpe)
      def w3 () = Utils.widthBANG(a(2).tpe)
      def c1 () = IntWidth(c(0))
      def c2 () = IntWidth(c(1))
      o match {
         case Add => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Sub => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),Utils.ONE))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Mul => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(PLUS(w1(),w2()))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(w1(),w2()))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(w1(),w2()))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(w1(),w2()))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Div => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(w1())
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(w1(),Utils.ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(w1())
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(w1(),Utils.ONE))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Rem => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(MIN(w1(),w2()))
               case (t1:UIntType, t2:SIntType) => UIntType(MIN(w1(),w2()))
               case (t1:SIntType, t2:UIntType) => SIntType(MIN(w1(),PLUS(w2(),Utils.ONE)))
               case (t1:SIntType, t2:SIntType) => SIntType(MIN(w1(),w2()))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Lt => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Leq => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Gt => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Geq => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Eq => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Neq => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => Utils.BoolType
               case (t1:SIntType, t2:UIntType) => Utils.BoolType
               case (t1:UIntType, t2:SIntType) => Utils.BoolType
               case (t1:SIntType, t2:SIntType) => Utils.BoolType
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Pad => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(MAX(w1(),c1()))
               case (t1:SIntType) => SIntType(MAX(w1(),c1()))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case AsUInt => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => UIntType(w1())
               case ClockType => UIntType(Utils.ONE)
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case AsSInt => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(w1())
               case (t1:SIntType) => SIntType(w1())
               case ClockType => SIntType(Utils.ONE)
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case AsClock => {
            val t = (t1()) match {
               case (t1:UIntType) => ClockType
               case (t1:SIntType) => ClockType
               case ClockType => ClockType
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Shl => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(PLUS(w1(),c1()))
               case (t1:SIntType) => SIntType(PLUS(w1(),c1()))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Shr => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(MAX(MINUS(w1(),c1()),Utils.ONE))
               case (t1:SIntType) => SIntType(MAX(MINUS(w1(),c1()),Utils.ONE))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Dshl => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(PLUS(w1(),POW(w2())))
               case (t1:SIntType) => SIntType(PLUS(w1(),POW(w2())))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Dshr => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => SIntType(w1())
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Cvt => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(PLUS(w1(),Utils.ONE))
               case (t1:SIntType) => SIntType(w1())
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Neg => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(PLUS(w1(),Utils.ONE))
               case (t1:SIntType) => SIntType(PLUS(w1(),Utils.ONE))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Not => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => UIntType(w1())
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case And => {
            val t = (t1(),t2()) match {
               case (_:SIntType|_:UIntType, _:SIntType|_:UIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Or => {
            val t = (t1(),t2()) match {
               case (_:SIntType|_:UIntType, _:SIntType|_:UIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Xor => {
            val t = (t1(),t2()) match {
               case (_:SIntType|_:UIntType, _:SIntType|_:UIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Andr => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => Utils.BoolType
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Orr => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => Utils.BoolType
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Xorr => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => Utils.BoolType
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Cat => {
            val t = (t1(),t2()) match {
               case (_:UIntType|_:SIntType,_:UIntType|_:SIntType) => UIntType(PLUS(w1(),w2()))
               case (t1, t2) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Bits => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => UIntType(PLUS(MINUS(c1(),c2()),Utils.ONE))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Head => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => UIntType(c1())
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
         case Tail => {
            val t = (t1()) match {
               case (_:UIntType|_:SIntType) => UIntType(MINUS(w1(),c1()))
               case (t1) => UnknownType
            }
            DoPrim(o,a,c,t)
         }
      
     }
   }

}
