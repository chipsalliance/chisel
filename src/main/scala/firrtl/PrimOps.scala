
package firrtl

import com.typesafe.scalalogging.LazyLogging

import Utils._
import DebugUtils._

object PrimOps extends LazyLogging {

  private val mapPrimOp2String = Map[PrimOp, String](
    AddOp -> "add",
    SubOp -> "sub",
    MulOp -> "mul",
    DivOp -> "div",
    RemOp -> "rem",
    LessOp -> "lt",
    LessEqOp -> "leq",
    GreaterOp -> "gt",
    GreaterEqOp -> "geq",
    EqualOp -> "eq",
    NEqualOp -> "neq",
    PadOp -> "pad",
    AsUIntOp -> "asUInt",
    AsSIntOp -> "asSInt",
    AsClockOp -> "asClock",
    ShiftLeftOp -> "shl",
    ShiftRightOp -> "shr",
    DynShiftLeftOp -> "dshl",
    DynShiftRightOp -> "dshr",
    ConvertOp -> "cvt",
    NegOp -> "neg",
    BitNotOp -> "not",
    BitAndOp -> "and",
    BitOrOp -> "or",
    BitXorOp -> "xor",
    BitAndReduceOp -> "andr",
    BitOrReduceOp -> "orr",
    BitXorReduceOp -> "xorr",
    ConcatOp -> "cat",
    BitsSelectOp -> "bits",
    HeadOp -> "head",
    TailOp -> "tail"
  )
  private val mapString2PrimOp = mapPrimOp2String.map(_.swap)
  def fromString(op: String): PrimOp = mapString2PrimOp(op)

  implicit class PrimOpImplicits(op: PrimOp){
    def getString(): String = mapPrimOp2String(op)
  }

  // Borrowed from Stanza implementation
  def lowerAndTypePrimOp(e: DoPrim): DoPrim = {
    def uAnd(op1: Expression, op2: Expression): Type = {
      (op1.getType, op2.getType) match {
        case (t1: UIntType, t2: UIntType) => UIntType(UnknownWidth)
        case (t1: SIntType, t2) => SIntType(UnknownWidth)
        case (t1, t2: SIntType) => SIntType(UnknownWidth)
        case _ => UnknownType
      }
    }
    def ofType(op: Expression): Type = {
      op.getType match {
        case t: UIntType => UIntType(UnknownWidth)
        case t: SIntType => SIntType(UnknownWidth)
        case _ => UnknownType
      }
    }

    logger.debug(s"lowerAndTypePrimOp on ${e.op.getClass.getSimpleName}")
    // TODO fix this
    val tpe = UIntType(UnknownWidth)
    //val tpe = e.op match {
    //  case Add => uAnd(e.args(0), e.args(1))
    //  case Sub => SIntType(UnknownWidth)
    //  case Addw => uAnd(e.args(0), e.args(1))
    //  case Subw => uAnd(e.args(0), e.args(1))
    //  case Mul => uAnd(e.args(0), e.args(1))
    //  case Div => uAnd(e.args(0), e.args(1))
    //  case Mod => ofType(e.args(0))
    //  case Quo => uAnd(e.args(0), e.args(1))
    //  case Rem => ofType(e.args(1))
    //  case Lt => UIntType(UnknownWidth)
    //  case Leq => UIntType(UnknownWidth)
    //  case Gt => UIntType(UnknownWidth)
    //  case Geq => UIntType(UnknownWidth)
    //  case Eq => UIntType(UnknownWidth)
    //  case Neq => UIntType(UnknownWidth)
    //  case Eqv => UIntType(UnknownWidth)
    //  case Neqv => UIntType(UnknownWidth)
    //  case Mux => ofType(e.args(1))
    //  case Pad => ofType(e.args(0))
    //  case AsUInt => UIntType(UnknownWidth)
    //  case AsSInt => SIntType(UnknownWidth)
    //  case Shl => ofType(e.args(0))
    //  case Shr => ofType(e.args(0))
    //  case Dshl => ofType(e.args(0))
    //  case Dshr => ofType(e.args(0))
    //  case Cvt => SIntType(UnknownWidth)
    //  case Neg => SIntType(UnknownWidth)
    //  case Not => ofType(e.args(0))
    //  case And => ofType(e.args(0))
    //  case Or => ofType(e.args(0))
    //  case Xor => ofType(e.args(0))
    //  case Andr => UIntType(UnknownWidth)
    //  case Orr => UIntType(UnknownWidth)
    //  case Xorr => UIntType(UnknownWidth)
    //  case Cat => UIntType(UnknownWidth)
    //  case Bit => UIntType(UnknownWidth)
    //  case Bits => UIntType(UnknownWidth)
    //  case _ => ???
    //}
    DoPrim(e.op, e.args, e.consts, tpe)
  }

}
