
package firrtl

import com.typesafe.scalalogging.LazyLogging

import Utils._
import DebugUtils._

object PrimOps extends LazyLogging {

  private val mapPrimOp2String = Map[PrimOp, String](
    ADD_OP -> "add",
    SUB_OP -> "sub",
    MUL_OP -> "mul",
    DIV_OP -> "div",
    REM_OP -> "rem",
    LESS_OP -> "lt",
    LESS_EQ_OP -> "leq",
    GREATER_OP -> "gt",
    GREATER_EQ_OP -> "geq",
    EQUAL_OP -> "eq",
    NEQUAL_OP -> "neq",
    PAD_OP -> "pad",
    AS_UINT_OP -> "asUInt",
    AS_SINT_OP -> "asSInt",
    AS_CLOCK_OP -> "asClock",
    SHIFT_LEFT_OP -> "shl",
    SHIFT_RIGHT_OP -> "shr",
    DYN_SHIFT_LEFT_OP -> "dshl",
    DYN_SHIFT_RIGHT_OP -> "dshr",
    NEG_OP -> "neg",
    CONVERT_OP -> "cvt",
    NOT_OP -> "not",
    AND_OP -> "and",
    OR_OP -> "or",
    XOR_OP -> "xor",
    AND_REDUCE_OP -> "andr",
    OR_REDUCE_OP -> "orr",
    XOR_REDUCE_OP -> "xorr",
    CONCAT_OP -> "cat",
    BITS_SELECT_OP -> "bits",
    HEAD_OP -> "head",
    TAIL_OP -> "tail"
  )
  lazy val listing: Seq[String] = PrimOps.mapPrimOp2String.map { case (k,v) => v } toSeq
  private val mapString2PrimOp = mapPrimOp2String.map(_.swap)
  def fromString(op: String): PrimOp = mapString2PrimOp(op)

  implicit class PrimOpImplicits(op: PrimOp){
    def getString(): String = mapPrimOp2String(op)
  }

  // Borrowed from Stanza implementation
   def set_primop_type (e:DoPrim) : DoPrim = {
      //println-all(["Inferencing primop type: " e])
      def PLUS (w1:Width,w2:Width) : Width = PlusWidth(w1,w2)
      def MAX (w1:Width,w2:Width) : Width = MaxWidth(Seq(w1,w2))
      def MINUS (w1:Width,w2:Width) : Width = MinusWidth(w1,w2)
      def POW (w1:Width) : Width = ExpWidth(w1)
      def MIN (w1:Width,w2:Width) : Width = MinWidth(Seq(w1,w2))
      val o = e.op
      val a = e.args
      val c = e.consts
      def t1 () = tpe(a(0))
      def t2 () = tpe(a(1))
      def t3 () = tpe(a(2))
      def w1 () = widthBANG(tpe(a(0)))
      def w2 () = widthBANG(tpe(a(1)))
      def w3 () = widthBANG(tpe(a(2)))
      def c1 () = IntWidth(c(0))
      def c2 () = IntWidth(c(1))
      o match {
         case ADD_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case SUB_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(MAX(w1(),w2()),ONE))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case MUL_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(PLUS(w1(),w2()))
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(w1(),w2()))
               case (t1:SIntType, t2:UIntType) => SIntType(PLUS(w1(),w2()))
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(w1(),w2()))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case DIV_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(w1())
               case (t1:UIntType, t2:SIntType) => SIntType(PLUS(w1(),ONE))
               case (t1:SIntType, t2:UIntType) => SIntType(w1())
               case (t1:SIntType, t2:SIntType) => SIntType(PLUS(w1(),ONE))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case REM_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(MIN(w1(),w2()))
               case (t1:UIntType, t2:SIntType) => UIntType(MIN(w1(),w2()))
               case (t1:SIntType, t2:UIntType) => SIntType(MIN(w1(),PLUS(w2(),ONE)))
               case (t1:SIntType, t2:SIntType) => SIntType(MIN(w1(),w2()))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case LESS_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case LESS_EQ_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case GREATER_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case GREATER_EQ_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case EQUAL_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case NEQUAL_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => BoolType()
               case (t1:SIntType, t2:UIntType) => BoolType()
               case (t1:UIntType, t2:SIntType) => BoolType()
               case (t1:SIntType, t2:SIntType) => BoolType()
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case PAD_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(MAX(w1(),c1()))
               case (t1:SIntType) => SIntType(MAX(w1(),c1()))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case AS_UINT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => UIntType(w1())
               case (t1:ClockType) => UIntType(ONE)
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case AS_SINT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(w1())
               case (t1:SIntType) => SIntType(w1())
               case (t1:ClockType) => SIntType(ONE)
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case AS_CLOCK_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => ClockType()
               case (t1:SIntType) => ClockType()
               case (t1:ClockType) => ClockType()
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case SHIFT_LEFT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(PLUS(w1(),c1()))
               case (t1:SIntType) => SIntType(PLUS(w1(),c1()))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case SHIFT_RIGHT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(MINUS(w1(),c1()))
               case (t1:SIntType) => SIntType(MINUS(w1(),c1()))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case DYN_SHIFT_LEFT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(PLUS(w1(),POW(w2())))
               case (t1:SIntType) => SIntType(PLUS(w1(),POW(w2())))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case DYN_SHIFT_RIGHT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => SIntType(w1())
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case CONVERT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(PLUS(w1(),ONE))
               case (t1:SIntType) => SIntType(w1())
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case NEG_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => SIntType(PLUS(w1(),ONE))
               case (t1:SIntType) => SIntType(PLUS(w1(),ONE))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case NOT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(w1())
               case (t1:SIntType) => UIntType(w1())
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case AND_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:SIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case OR_OP => {
            val t = (t1(),t2()) match {
               case (t1:SIntType, t2:SIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case XOR_OP => {
            val t = (t1(),t2()) match {
               case (t1:UIntType, t2:UIntType) => UIntType(MAX(w1(),w2()))
               case (t1:SIntType, t2:UIntType) => UIntType(MAX(w1(),w2()))
               case (t1:UIntType, t2:SIntType) => UIntType(MAX(w1(),w2()))
               case (t1:SIntType, t2:SIntType) => UIntType(MAX(w1(),w2()))
               case (t1,t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case AND_REDUCE_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => BoolType()
               case (t1:SIntType) => BoolType()
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case OR_REDUCE_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => BoolType()
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case XOR_REDUCE_OP => {
            val t = (t1()) match {
               case (t1:SIntType) => BoolType()
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case CONCAT_OP => {
            val t = (t1(),t2()) match {
               case (t1:SIntType, t2:SIntType) => UIntType(PLUS(w1(),w2()))
               case (t1, t2) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case BITS_SELECT_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(PLUS(MINUS(c1(),c2()),ONE))
               case (t1:SIntType) => UIntType(PLUS(MINUS(c1(),c2()),ONE))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case HEAD_OP => {
            val t = (t1()) match {
               case (t1:UIntType) => UIntType(c1())
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }
         case TAIL_OP => {
            val t = (t1()) match {
               case (t1:SIntType) => UIntType(MINUS(w1(),c1()))
               case (t1) => UnknownType()
            }
            DoPrim(o,a,c,t)
         }

     }
   }

}
