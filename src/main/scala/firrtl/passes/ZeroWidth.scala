// See LICENSE for license details.

package firrtl.passes

import scala.collection.mutable
import firrtl.PrimOps._
import firrtl.ir._
import firrtl._
import firrtl.Mappers._
import firrtl.Utils.throwInternalError


object ZeroWidth extends Pass {
  private val ZERO = BigInt(0)
  private def removeZero(t: Type): Option[Type] = t match {
    case GroundType(IntWidth(ZERO)) => None
    case BundleType(fields) =>
      fields map (f => (f, removeZero(f.tpe))) collect {
        case (Field(name, flip, _), Some(t)) => Field(name, flip, t)
      } match {
        case Nil => None
        case seq => Some(BundleType(seq))
      }
    case VectorType(t, size) => removeZero(t) map (VectorType(_, size))
    case x => Some(x)
  }
  private def onExp(e: Expression): Expression = removeZero(e.tpe) match {
    case None => e.tpe match {
      case UIntType(x) => UIntLiteral(ZERO, IntWidth(BigInt(1)))
      case SIntType(x) => SIntLiteral(ZERO, IntWidth(BigInt(1)))
      case _ => throwInternalError
    }
    case Some(t) => 
      def replaceType(x: Type): Type = t
      (e map replaceType) map onExp
  }
  private def onStmt(s: Statement): Statement = s match {
    case (_: DefWire| _: DefRegister| _: DefMemory) =>
      var removed = false
      def applyRemoveZero(t: Type): Type = removeZero(t) match {
        case None => removed = true; t
        case Some(tx) => tx
      }
      val sxx = (s map onExp) map applyRemoveZero
      if(removed) EmptyStmt else sxx
    case Connect(info, loc, exp) => removeZero(loc.tpe) match {
      case None => EmptyStmt
      case Some(t) => Connect(info, loc, onExp(exp))
    }
    case DefNode(info, name, value) => removeZero(value.tpe) match {
      case None => EmptyStmt
      case Some(t) => DefNode(info, name, onExp(value))
    }
    case sx => sx map onStmt
  }
  private def onModule(m: DefModule): DefModule = {
    val ports = m.ports map (p => (p, removeZero(p.tpe))) collect {
      case (Port(info, name, dir, _), Some(t)) => Port(info, name, dir, t)
    }
    m match {
      case ext: ExtModule => ext.copy(ports = ports)
      case in: Module => in.copy(ports = ports, body = onStmt(in.body))
    }
  }
  def run(c: Circuit): Circuit = {
    InferTypes.run(c.copy(modules = c.modules map onModule))
  }
}
