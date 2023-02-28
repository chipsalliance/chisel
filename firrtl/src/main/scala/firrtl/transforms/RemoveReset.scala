// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.WrappedExpression.we
import firrtl.annotations.PresetRegAnnotation
import firrtl.options.Dependency

import scala.collection.{immutable, mutable}

/** Remove Synchronous Reset
  *
  * @note This pass must run after LowerTypes
  */
object RemoveReset extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.MidForm ++
    Seq(Dependency(passes.LowerTypes))

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform): Boolean = false

  private case class Reset(cond: Expression, value: Expression, info: Info)

  /** Return an immutable set of all invalid expressions in a module
    * @param m a module
    */
  private def computeInvalids(m: DefModule): immutable.Set[WrappedExpression] = {
    val invalids = mutable.HashSet.empty[WrappedExpression]

    def onStmt(s: Statement): Unit = s match {
      case IsInvalid(_, expr)                                 => invalids += we(expr)
      case Connect(_, lhs, rhs) if invalids.contains(we(rhs)) => invalids += we(lhs)
      case other                                              => other.foreach(onStmt)
    }

    m.foreach(onStmt)
    invalids.toSet
  }

  private def onModule(m: DefModule, isPreset: String => Boolean): DefModule = {
    val resets = mutable.HashMap.empty[String, Reset]
    val asyncResets = mutable.HashMap.empty[String, Reset]
    val invalids = computeInvalids(m)
    lazy val namespace = Namespace(m)
    def onStmt(stmt: Statement): Statement = {
      stmt match {
        case reg @ DefRegister(_, name, _, _, reset, init) if isPreset(name) =>
          // registers that are preset annotated should already be in canonical form
          if (reset != Utils.False()) {
            throw new RuntimeException(
              s"[${m.name}] register `$name` has a PresetRegAnnotation, but the reset is not UInt(0)!"
            )
          }
          if (!Utils.isLiteral(init)) {
            throw new RuntimeException(
              s"[${m.name}] register `$name` has a PresetRegAnnotation, " +
                s"but the init value is not a literal! ${init.serialize}"
            )
          }
          // no change necessary
          reg
        /* A register is initialized to an invalid expression */
        case reg @ DefRegister(_, _, _, _, _, init) if invalids.contains(we(init)) =>
          reg.copy(reset = Utils.zero, init = WRef(reg))
        case reg @ DefRegister(_, rname, _, _, Utils.zero, _) =>
          reg.copy(init = WRef(reg)) // canonicalize
        case reg @ DefRegister(info, rname, _, _, reset, init) if reset.tpe != AsyncResetType =>
          // Add register reset to map
          resets(rname) = Reset(reset, init, info)
          reg.copy(reset = Utils.zero, init = WRef(reg))
        case reg @ DefRegister(info, rname, _, _, reset, init) if reset.tpe == AsyncResetType =>
          asyncResets(rname) = Reset(reset, init, info)
          reg
        case Connect(info, ref @ WRef(rname, _, RegKind, _), expr) if resets.contains(rname) =>
          val reset = resets(rname)
          val muxType = Utils.mux_type_and_widths(reset.value, expr)
          // Use reg source locator for mux enable and true value since that's where they're defined
          val infox = MultiInfo(reset.info, reset.info, info)
          Connect(infox, ref, Mux(reset.cond, reset.value, expr, muxType))
        case Connect(info, ref @ WRef(rname, _, RegKind, _), expr) if asyncResets.contains(rname) =>
          val reset = asyncResets(rname)
          // The `muxType` for async always blocks is located in [[VerilogEmitter.VerilogRender.regUpdate]]:
          // addUpdate(info, Mux(reset, tv, fv, mux_type_and_widths(tv, fv)), Seq.empty)
          val infox = MultiInfo(reset.info, reset.info, info)
          Connect(infox, ref, expr)
        /* Synchronously reset register that has reset value but only an invalid connection */
        case IsInvalid(iinfo, ref @ WRef(rname, tpe, RegKind, _)) if resets.contains(rname) =>
          // We need to mux with the invalid value to be consistent with async reset registers
          val dummyWire = DefWire(iinfo, namespace.newName(rname), tpe)
          val wireRef = Reference(dummyWire).copy(flow = SourceFlow)
          val invalid = IsInvalid(iinfo, wireRef)
          // Now mux between the invalid wire and the reset value
          val Reset(cond, init, info) = resets(rname)
          val muxType = Utils.mux_type_and_widths(init, wireRef)
          val connect = Connect(info, ref, Mux(cond, init, wireRef, muxType))
          Block(Seq(dummyWire, invalid, connect))
        case other => other.map(onStmt)
      }
    }
    m.map(onStmt)
  }

  def execute(state: CircuitState): CircuitState = {
    // If registers are annotated with the [[PresetRegAnnotation]], they will take on their
    // reset value when the circuit "starts" (i.e. at the beginning of simulation or when the FPGA
    // bit-stream is initialized) and thus we need to special-case them.
    val presetRegs = PresetRegAnnotation.collect(state.annotations, state.circuit.main)
    val c = state.circuit.mapModule(m => onModule(m, presetRegs.getOrElse(m.name, _ => false)))
    state.copy(circuit = c)
  }
}
