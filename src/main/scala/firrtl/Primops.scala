
package firrtl

import Utils._
import DebugUtils._

object Primops {

  // Borrowed from Stanza implementation
  def lowerAndTypePrimop(e: DoPrimOp)(implicit logger: Logger): DoPrimOp = {
    def uAnd(op1: Exp, op2: Exp): Type = {
      (op1.getType, op2.getType) match {
        case (t1: UIntType, t2: UIntType) => UIntType(UnknownWidth)
        case (t1: SIntType, t2) => SIntType(UnknownWidth)
        case (t1, t2: SIntType) => SIntType(UnknownWidth)
        case _ => UnknownType
      }
    }
    def ofType(op: Exp): Type = {
      op.getType match {
        case t: UIntType => UIntType(UnknownWidth)
        case t: SIntType => SIntType(UnknownWidth)
        case _ => UnknownType
      }
    }
   
    logger.debug(s"lowerAndTypePrimop on ${e.op.getClass.getSimpleName}")
    val tpe = e.op match {
      case Add => uAnd(e.args(0), e.args(1))
      case Sub => SIntType(UnknownWidth)
      case Addw => uAnd(e.args(0), e.args(1))
      case Subw => uAnd(e.args(0), e.args(1))
      case Mul => uAnd(e.args(0), e.args(1))
      case Div => uAnd(e.args(0), e.args(1))
      case Mod => ofType(e.args(0))
      case Quo => uAnd(e.args(0), e.args(1))
      case Rem => ofType(e.args(1))
      case Lt => UIntType(UnknownWidth)
      case Leq => UIntType(UnknownWidth)
      case Gt => UIntType(UnknownWidth)
      case Geq => UIntType(UnknownWidth)
      case Eq => UIntType(UnknownWidth)
      case Neq => UIntType(UnknownWidth)
      case Mux => ofType(e.args(1))
      case Pad => ofType(e.args(0))
      case AsUInt => UIntType(UnknownWidth)
      case AsSInt => SIntType(UnknownWidth)
      case Shl => ofType(e.args(0))
      case Shr => ofType(e.args(0))
      case Dshl => ofType(e.args(0))
      case Dshr => ofType(e.args(0))
      case Cvt => SIntType(UnknownWidth)
      case Neg => SIntType(UnknownWidth)
      case Not => ofType(e.args(0))
      case And => ofType(e.args(0))
      case Or => ofType(e.args(0))
      case Xor => ofType(e.args(0))
      case Andr => UIntType(UnknownWidth)
      case Orr => UIntType(UnknownWidth)
      case Xorr => UIntType(UnknownWidth)
      case Cat => UIntType(UnknownWidth)
      case Bit => UIntType(UnknownWidth)
      case Bits => UIntType(UnknownWidth)
      case _ => ??? 
    }
    DoPrimOp(e.op, e.args, e.consts, tpe)
  }

}
