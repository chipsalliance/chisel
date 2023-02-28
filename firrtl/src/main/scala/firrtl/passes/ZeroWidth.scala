// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.PrimOps._
import firrtl.ir._
import firrtl._
import firrtl.renamemap.MutableRenameMap
import firrtl.Mappers._
import firrtl.options.Dependency

object ZeroWidth extends Transform with DependencyAPIMigration {

  override def prerequisites =
    Seq(
      Dependency(PullMuxes),
      Dependency(ReplaceAccesses),
      Dependency(ExpandConnects),
      Dependency(RemoveAccesses),
      Dependency[ExpandWhensAndCheck]
    ) ++ firrtl.stage.Forms.Deduped

  override def invalidates(a: Transform): Boolean = a match {
    case InferTypes => true
    case _          => false
  }

  private def makeZero(tpe: ir.Type): ir.Type = tpe match {
    case ClockType => UIntType(IntWidth(0))
    case a: UIntType      => a.copy(IntWidth(0))
    case a: SIntType      => a.copy(IntWidth(0))
    case a: AggregateType => a.map(makeZero)
  }

  private def onEmptyMemStmt(s: Statement): Statement = s match {
    case d @ DefMemory(info, name, tpe, _, _, _, rs, ws, rws, _) =>
      removeZero(tpe) match {
        case None =>
          DefWire(
            info,
            name,
            MemPortUtils
              .memType(d)
              .map(makeZero)
          )
        case Some(_) => d
      }
    case sx => sx.map(onEmptyMemStmt)
  }

  private def onModuleEmptyMemStmt(m: DefModule): DefModule = {
    m match {
      case ext: ExtModule => ext
      case in:  Module    => in.copy(body = onEmptyMemStmt(in.body))
    }
  }

  /**
    * Replace zero width mems before running the rest of the ZeroWidth transform.
    * Dealing with mems is a bit tricky because the address, en, clk ports
    * of the memory are not width zero even if data is.
    *
    * This replaces memories with a DefWire() bundle that contains the address, en,
    * clk, and data fields implemented as zero width wires. Running the rest of the ZeroWidth
    * transform will remove these dangling references properly.
    */
  def executeEmptyMemStmt(state: CircuitState): CircuitState = {
    val c = state.circuit
    val result = c.copy(modules = c.modules.map(onModuleEmptyMemStmt))
    state.copy(circuit = result)
  }

  // This is slightly different and specialized version of create_exps, TODO unify?
  private def findRemovable(expr: => Expression, tpe: Type): Seq[Expression] = tpe match {
    case GroundType(width) =>
      width match {
        case IntWidth(ZERO) => List(expr)
        case _              => List.empty
      }
    case BundleType(fields) =>
      if (fields.isEmpty) List(expr)
      else fields.flatMap(f => findRemovable(WSubField(expr, f.name, f.tpe, SourceFlow), f.tpe))
    case VectorType(vtpe, size) =>
      if (size == 0) List(expr)
      else { // Only invoke findRemovable multiple times if a zero-width element is found
        val es0 = findRemovable(WSubIndex(expr, 0, vtpe, SourceFlow), vtpe)
        if (es0.isEmpty) es0
        else {
          es0 ++ (1 until size).flatMap(i => findRemovable(WSubIndex(expr, i, vtpe, SourceFlow), vtpe))
        }
      }
  }

  private val ZERO = BigInt(0)
  private def getRemoved(x: IsDeclaration): Seq[String] = {
    var removedNames: Seq[String] = Seq.empty
    def onType(name: String)(t: Type): Type = {
      removedNames = findRemovable(WRef(name), t).map(_.serialize)
      t
    }
    x match {
      case s: Statement => s.map(onType(s.name))
      case Port(_, name, _, t) => onType(name)(t)
    }
    removedNames
  }
  private[passes] def removeZero(t: Type): Option[Type] = t match {
    case GroundType(IntWidth(ZERO)) => None
    case BundleType(fields) =>
      fields.map(f => (f, removeZero(f.tpe))).collect {
        case (Field(name, flip, _), Some(t)) => Field(name, flip, t)
      } match {
        case Nil => None
        case seq => Some(BundleType(seq))
      }
    case VectorType(t, size) => removeZero(t).map(VectorType(_, size))
    case x                   => Some(x)
  }
  private def onExp(e: Expression): Expression = e match {
    case DoPrim(Cat, args, consts, tpe) =>
      val nonZeros = args.flatMap { x =>
        x.tpe match {
          case UIntType(IntWidth(ZERO)) => Seq.empty[Expression]
          case SIntType(IntWidth(ZERO)) => Seq.empty[Expression]
          case other                    => Seq(x)
        }
      }
      nonZeros match {
        case Nil => UIntLiteral(ZERO, IntWidth(BigInt(1)))
        // We may have an SInt, Cat has type UInt so cast
        case Seq(x) => castRhs(tpe, x)
        case seq    => DoPrim(Cat, seq, consts, tpe).map(onExp)
      }
    case DoPrim(Andr, Seq(x), _, _) if (bitWidth(x.tpe) == 0) => UIntLiteral(1) // nothing false
    // The width of the result type of dshl is a function of the width of the shift.  This has to be special cased for
    // the zero-width shift case to prevent increasing the result width.  Canonicalize a dshl by a zero-width element as
    // just returning the unshifted expression.
    case DoPrim(Dshl, Seq(x, a), _, _) if (bitWidth(a.tpe) == 0) => x
    case other =>
      other.tpe match {
        case UIntType(IntWidth(ZERO)) => UIntLiteral(ZERO, IntWidth(BigInt(1)))
        case SIntType(IntWidth(ZERO)) => SIntLiteral(ZERO, IntWidth(BigInt(1)))
        case _                        => e.map(onExp)
      }
  }
  private def onStmt(renames: MutableRenameMap)(s: Statement): Statement = s match {
    case d @ DefWire(info, name, tpe) =>
      renames.delete(getRemoved(d))
      removeZero(tpe) match {
        case None    => EmptyStmt
        case Some(t) => DefWire(info, name, t)
      }
    case d @ DefRegister(info, name, tpe, clock, reset, init) =>
      renames.delete(getRemoved(d))
      removeZero(tpe) match {
        case None => EmptyStmt
        case Some(t) =>
          DefRegister(info, name, t, onExp(clock), onExp(reset), onExp(init))
      }
    case d: DefMemory =>
      renames.delete(getRemoved(d))
      removeZero(d.dataType) match {
        case None =>
          Utils.throwInternalError(s"private pass ZeroWidthMemRemove should have removed this memory: $d")
        case Some(t) => d.copy(dataType = t)
      }
    case Connect(info, loc, exp) =>
      removeZero(loc.tpe) match {
        case None    => EmptyStmt
        case Some(t) => Connect(info, loc, onExp(exp))
      }
    case IsInvalid(info, exp) =>
      removeZero(exp.tpe) match {
        case None    => EmptyStmt
        case Some(t) => IsInvalid(info, onExp(exp))
      }
    case DefNode(info, name, value) =>
      removeZero(value.tpe) match {
        case None    => EmptyStmt
        case Some(t) => DefNode(info, name, onExp(value))
      }
    case sx => sx.map(onStmt(renames)).map(onExp)
  }
  private def onModule(renames: MutableRenameMap)(m: DefModule): DefModule = {
    renames.setModule(m.name)
    // For each port, record deleted subcomponents
    m.ports.foreach { p => renames.delete(getRemoved(p)) }
    val ports = m.ports.map(p => (p, removeZero(p.tpe))).flatMap {
      case (Port(info, name, dir, _), Some(t)) => Seq(Port(info, name, dir, t))
      case (Port(_, name, _, _), None) =>
        renames.delete(name)
        Nil
    }
    m match {
      case ext: ExtModule => ext.copy(ports = ports)
      case in:  Module    => in.copy(ports = ports, body = onStmt(renames)(in.body))
    }
  }
  def execute(state: CircuitState): CircuitState = {
    // run executeEmptyMemStmt first to remove zero-width memories
    // then run InferTypes to update widths for addr, en, clk, etc
    val c = InferTypes.run(executeEmptyMemStmt(state).circuit)
    val renames = MutableRenameMap()
    renames.setCircuit(c.main)
    val result = c.copy(modules = c.modules.map(onModule(renames)))
    state.copy(circuit = result, renames = Some(renames))
  }
}
