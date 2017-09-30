// See LICENSE for license details.

package firrtl.passes

import scala.collection.mutable
import firrtl.PrimOps._
import firrtl.ir._
import firrtl._
import firrtl.Mappers._
import firrtl.Utils.throwInternalError


object ZeroWidth extends Transform {
  def inputForm = UnknownForm
  def outputForm = UnknownForm
  private val ZERO = BigInt(0)
  private def getRemoved(x: IsDeclaration): Seq[String] = {
    var removedNames: Seq[String] = Seq.empty
    def onType(name: String)(t: Type): Type = {
      removedNames = Utils.create_exps(name, t) map {e => (e, e.tpe)} collect {
        case (e, GroundType(IntWidth(ZERO))) => e.serialize
      }
      t
    }
    x match {
      case s: Statement => s map onType(s.name)
      case Port(_, name, _, t) => onType(name)(t)
    }
    removedNames
  }
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
  private def onExp(e: Expression): Expression = e match {
    case DoPrim(Cat, args, consts, tpe) =>
      val nonZeros = args.flatMap { x =>
        x.tpe match {
          case UIntType(IntWidth(ZERO)) => Seq.empty[Expression]
          case SIntType(IntWidth(ZERO)) => Seq.empty[Expression]
          case other => Seq(x)
        }
      }
      nonZeros match {
        case Nil => UIntLiteral(ZERO, IntWidth(BigInt(1)))
        case Seq(x) => x
        case seq => DoPrim(Cat, seq, consts, tpe) map onExp
      }
    case other => other.tpe match {
      case UIntType(IntWidth(ZERO)) => UIntLiteral(ZERO, IntWidth(BigInt(1)))
      case SIntType(IntWidth(ZERO)) => SIntLiteral(ZERO, IntWidth(BigInt(1)))
      case _ => e map onExp
    }
  }
  private def onStmt(renames: RenameMap)(s: Statement): Statement = s match {
    case (_: DefWire| _: DefRegister| _: DefMemory) =>
      // List all removed expression names, and delete them from renames
      renames.delete(getRemoved(s.asInstanceOf[IsDeclaration]))
      // Create new types without zero-width wires
      var removed = false
      def applyRemoveZero(t: Type): Type = removeZero(t) match {
        case None => removed = true; t
        case Some(tx) => tx
      }
      val sxx = (s map onExp) map applyRemoveZero
      // Return new declaration
      if(removed) EmptyStmt else sxx
    case Connect(info, loc, exp) => removeZero(loc.tpe) match {
      case None => EmptyStmt
      case Some(t) => Connect(info, loc, onExp(exp))
    }
    case IsInvalid(info, exp) => removeZero(exp.tpe) match {
      case None => EmptyStmt
      case Some(t) => IsInvalid(info, onExp(exp))
    }
    case DefNode(info, name, value) => removeZero(value.tpe) match {
      case None => EmptyStmt
      case Some(t) => DefNode(info, name, onExp(value))
    }
    case sx => sx map onStmt(renames)
  }
  private def onModule(renames: RenameMap)(m: DefModule): DefModule = {
    renames.setModule(m.name)
    // For each port, record deleted subcomponents
    m.ports.foreach{p => renames.delete(getRemoved(p))}
    val ports = m.ports map (p => (p, removeZero(p.tpe))) flatMap {
      case (Port(info, name, dir, _), Some(t)) => Seq(Port(info, name, dir, t))
      case (Port(_, name, _, _), None) =>
        renames.delete(name)
        Nil
    }
    m match {
      case ext: ExtModule => ext.copy(ports = ports)
      case in: Module => in.copy(ports = ports, body = onStmt(renames)(in.body))
    }
  }
  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val renames = RenameMap()
    renames.setCircuit(c.main)
    val result = InferTypes.run(c.copy(modules = c.modules map onModule(renames)))
    CircuitState(result, outputForm, state.annotations, Some(renames))
  }
}
