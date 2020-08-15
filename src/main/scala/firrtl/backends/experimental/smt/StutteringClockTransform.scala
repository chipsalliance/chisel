// See LICENSE for license details.
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.{CircuitState, DependencyAPIMigration, Namespace, PrimOps, RenameMap, Transform, Utils, ir}
import firrtl.annotations.{Annotation, CircuitTarget, PresetAnnotation, ReferenceTarget, SingleTargetAnnotation}
import firrtl.ir.EmptyStmt
import firrtl.options.Dependency
import firrtl.passes.PassException
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency

import scala.collection.mutable

case class GlobalClockAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
  override def duplicate(n: ReferenceTarget): Annotation = this.copy(n)
}

/** Converts every input clock into a clock enable input and adds a single global clock.
  * - all registers and memory ports will be connected to the new global clock
  * - all registers and memory ports will be guarded by the enable signal of their original clock
  * - the clock enabled signal can be understood as a clock tick or posedge
  * - this transform can be used in order to (formally) verify designs with multiple clocks or asynchronous resets
  */
class StutteringClockTransform extends Transform with DependencyAPIMigration {
  override def prerequisites: Seq[TransformDependency] = Forms.LowForm
  override def invalidates(a: Transform): Boolean = false

  // this pass needs to run *before* converting to a transition system
  override def optionalPrerequisiteOf: Seq[TransformDependency] = Seq(Dependency(FirrtlToTransitionSystem))
  // since this pass only runs on the main module, inlining needs to happen before
  override def optionalPrerequisites: Seq[TransformDependency] = Seq(Dependency[firrtl.passes.InlineInstances])


  override protected def execute(state: CircuitState): CircuitState = {
    if(state.circuit.modules.size > 1) {
      logger.warn("WARN: StutteringClockTransform currently only supports running on a single module.\n" +
        s"All submodules of ${state.circuit.main} will be ignored! Please inline all submodules if this is not what you want.")
    }

    // get main module
    val main = state.circuit.modules.find(_.name == state.circuit.main).get match {
      case m: ir.Module => m
      case e: ir.ExtModule => unsupportedError(s"Cannot run on extmodule $e")
    }
    mainName = main.name

    val namespace = Namespace(main)

    // create a global clock
    val globalClocks = state.annotations.collect { case GlobalClockAnnotation(c) => c }
    assert(globalClocks.size < 2, "There can only be a single global clock: " + globalClocks.mkString(", "))
    val (globalClock, portsWithGlobalClock) = globalClocks.headOption match {
      case Some(clock) =>
        assert(clock.module == main.name, "GlobalClock needs to be an input of the main module!")
        assert(main.ports.exists(_.name == clock.ref), "GlobalClock needs to be an input port!")
        assert(main.ports.find(_.name == clock.ref).get.direction == ir.Input, "GlobalClock needs to be an input port!")
        (clock.ref, main.ports)
      case None =>
        val name = namespace.newName("global_clock")
        (name, ir.Port(ir.NoInfo, name, ir.Input, ir.ClockType) +: main.ports)
    }

    // replace all other clocks with enable signals, unless they are the global clock
    val clocks = portsWithGlobalClock.filter(p => p.tpe == ir.ClockType && p.name != globalClock).map(_.name)
    val clockToEnable = clocks.map{c =>
      c -> ir.Reference(namespace.newName(c + "_en"), Bool, firrtl.PortKind, firrtl.SourceFlow)
    }.toMap
    val portsWithEnableSignals = portsWithGlobalClock.map { p =>
      if(clockToEnable.contains(p.name)) { p.copy(name = clockToEnable(p.name).name, tpe = Bool) } else { p }
    }
    // replace async reset with synchronous reset (since everything will we synchronous with the global clock)
    // unless it is a preset reset
    val asyncResets = portsWithEnableSignals.filter(_.tpe == ir.AsyncResetType).map(_.name)
    val isPresetReset = state.annotations.collect{ case PresetAnnotation(r) if r.module == main.name => r.ref }.toSet
    val resetsToChange = asyncResets.filterNot(isPresetReset).toSet
    val portsWithSyncReset = portsWithEnableSignals.map { p =>
      if(resetsToChange.contains(p.name)) { p.copy(tpe = Bool) } else { p }
    }

    // discover clock and reset connections
    val scan = scanClocks(main, clockToEnable, resetsToChange)

    // rename clocks to clock enable signals
    val mRef = CircuitTarget(state.circuit.main).module(main.name)
    val renameMap = RenameMap()
    scan.clockToEnable.foreach { case (clk, en) =>
      renameMap.record(mRef.ref(clk), mRef.ref(en.name))
    }

    // make changes
    implicit val ctx: Context = new Context(globalClock, scan)
    val newMain = main.copy(ports = portsWithSyncReset).mapStmt(onStatement)

    val nonMainModules = state.circuit.modules.filterNot(_.name == state.circuit.main)
    val newCircuit = state.circuit.copy(modules = nonMainModules :+ newMain)
    state.copy(circuit = newCircuit, renames = Some(renameMap))
  }

  private def onStatement(s: ir.Statement)(implicit ctx: Context): ir.Statement = {
    s.foreachExpr(checkExpr)
    s match {
      // memory field connects
      case c @ ir.Connect(_, ir.SubField(ir.SubField(ir.Reference(mem, _, _, _), port, _, _), field, _, _), _)
        if ctx.isMem(mem) && ctx.memPortToClockEnable.contains(mem + "." + port) =>
        // replace clock with the global clock
        if(field == "clk") {
          c.copy(expr = ctx.globalClock)
        } else if(field == "en") {
          val m = ctx.memInfo(mem)
          val isWritePort = m.writers.contains(port)
          assert(isWritePort || m.readers.contains(port))

          // for write ports we guard the write enable with the clock enable signal, similar to registers
          if(isWritePort) {
            val clockEn = ctx.memPortToClockEnable(mem + "." + port)
            val guardedEnable = and(clockEn, c.expr)
            c.copy(expr = guardedEnable)
          } else { c }
        } else { c}
      // register field connects
      case c @ ir.Connect(_, r : ir.Reference, next) if ctx.registerToEnable.contains(r.name) =>
        val clockEnable = ctx.registerToEnable(r.name)
        val guardedNext = mux(clockEnable, next, r)
        c.copy(expr = guardedNext)
      // remove other clock wires and nodes
      case ir.Connect(_, loc, expr) if expr.tpe == ir.ClockType && ctx.isRemovedClock(loc.serialize) => EmptyStmt
      case ir.DefNode(_, name, value) if value.tpe == ir.ClockType && ctx.isRemovedClock(name) => EmptyStmt
      case ir.DefWire(_, name, tpe) if tpe == ir.ClockType && ctx.isRemovedClock(name) => EmptyStmt
      // change async reset to synchronous reset
      case ir.Connect(info, loc: ir.Reference, expr: ir.Reference) if expr.tpe == ir.AsyncResetType && ctx.isResetToChange(loc.serialize) =>
        ir.Connect(info, loc.copy(tpe=Bool), expr.copy(tpe=Bool))
      case d @ ir.DefNode(_, name, value: ir.Reference) if value.tpe == ir.AsyncResetType && ctx.isResetToChange(name) =>
        d.copy(value = value.copy(tpe=Bool))
      case d @ ir.DefWire(_, name, tpe) if tpe == ir.AsyncResetType && ctx.isResetToChange(name) => d.copy(tpe=Bool)
      // change memory clock and synchronize reset
      case ir.DefRegister(info, name, tpe, clock, reset, init) if ctx.registerToEnable.contains(name) =>
        val clockEnable = ctx.registerToEnable(name)
        val newReset = reset match {
          case r @ ir.Reference(name, _, _, _) if ctx.isResetToChange(name) => r.copy(tpe=Bool)
          case other => other
        }
        val synchronizedReset = if(reset.tpe == ir.AsyncResetType) { newReset } else { and(newReset, clockEnable) }
        ir.DefRegister(info, name, tpe, ctx.globalClock, synchronizedReset, init)
      case other => other.mapStmt(onStatement)
    }
  }

  private def scanClocks(m: ir.Module, initialClockToEnable: Map[String, ir.Reference], resetsToChange: Set[String]): ScanCtx = {
    implicit val ctx: ScanCtx = new ScanCtx(initialClockToEnable, resetsToChange)
    m.foreachStmt(scanClocksAndResets)
    ctx
  }

  private def scanClocksAndResets(s: ir.Statement)(implicit ctx: ScanCtx): Unit = {
    s.foreachExpr(checkExpr)
    s match {
      // track clock aliases
      case ir.Connect(_, loc, expr) if expr.tpe == ir.ClockType =>
        val locName = loc.serialize
        ctx.clockToEnable.get(expr.serialize).foreach { clockEn =>
          ctx.clockToEnable(locName) = clockEn
          // keep track of memory clocks
          if(loc.isInstanceOf[ir.SubField]) {
            val parts = locName.split('.')
            if(ctx.mems.contains(parts.head)) {
              assert(parts.length == 3 && parts.last == "clk")
              ctx.memPortToClockEnable.append(parts.dropRight(1).mkString(".") -> clockEn)
            }
          }
        }
      case ir.DefNode(_, name, value) if value.tpe == ir.ClockType =>
        ctx.clockToEnable.get(value.serialize).foreach(c => ctx.clockToEnable(name) = c)
      // track reset aliases
      case ir.Connect(_, loc, expr) if expr.tpe == ir.AsyncResetType && ctx.resetsToChange(expr.serialize) =>
        ctx.resetsToChange.add(loc.serialize)
      case ir.DefNode(_, name, value) if value.tpe == ir.AsyncResetType && ctx.resetsToChange(value.serialize) =>
        ctx.resetsToChange.add(name)
      // modify clocked elements
      case ir.DefRegister(_, name, _, clock, _, _) =>
        ctx.clockToEnable.get(clock.serialize).foreach { clockEnable =>
          ctx.registerToEnable.append(name -> clockEnable)
        }
      case m : ir.DefMemory =>
        assert(m.readwriters.isEmpty, "Combined read/write ports are not supported!")
        assert(m.readLatency == 0 || m.readLatency == 1, "Only read-latency 1 and read latency 0 are supported!")
        assert(m.writeLatency == 1, "Only write-latency 1 is supported!")
        if(m.readers.nonEmpty && m.readLatency == 1) {
          unsupportedError("Registers memory read ports are not properly implemented yet :(")
        }
        ctx.mems(m.name) = m
      case other => other.foreachStmt(scanClocksAndResets)
    }
  }

  // we rely on people not casting clocks or async resets
  private def checkExpr(expr: ir.Expression): Unit = expr match {
    case ir.DoPrim(PrimOps.AsUInt, Seq(e), _, _) if e.tpe == ir.ClockType =>
      unsupportedError(s"Clock casts are not supported: ${expr.serialize}")
    case ir.DoPrim(PrimOps.AsSInt, Seq(e), _, _) if e.tpe == ir.ClockType =>
      unsupportedError(s"Clock casts are not supported: ${expr.serialize}")
    case ir.DoPrim(PrimOps.AsUInt, Seq(e), _, _) if e.tpe == ir.AsyncResetType =>
      unsupportedError(s"AsyncReset casts are not supported: ${expr.serialize}")
    case ir.DoPrim(PrimOps.AsSInt, Seq(e), _, _) if e.tpe == ir.AsyncResetType =>
      unsupportedError(s"AsyncReset casts are not supported: ${expr.serialize}")
    case ir.DoPrim(PrimOps.AsAsyncReset, _, _, _) =>
      unsupportedError(s"AsyncReset casts are not supported: ${expr.serialize}")
    case ir.DoPrim(PrimOps.AsClock, _, _, _) =>
      unsupportedError(s"Clock casts are not supported: ${expr.serialize}")
    case other => other.foreachExpr(checkExpr)
  }

  private class ScanCtx(initialClockToEnable: Map[String, ir.Reference], initialResetsToChange: Set[String]) {
    // keeps track of which clock signals will be replaced by which clock enable signal
    val clockToEnable = mutable.HashMap[String, ir.Reference]() ++ initialClockToEnable
    // kepp track of asynchronous resets that need to be changed to bool
    val resetsToChange = mutable.HashSet[String]() ++ initialResetsToChange
    // registers whose next function needs to be guarded with a clock enable
    val registerToEnable = mutable.ArrayBuffer[(String, ir.Reference)]()
    // memory enables which need to be guarded with clock enables
    val memPortToClockEnable = mutable.ArrayBuffer[(String, ir.Reference)]()
    // keep track of memory names
    val mems = mutable.HashMap[String, ir.DefMemory]()
  }

  private class Context(globalClockName: String, scanResults: ScanCtx) {
    val globalClock: ir.Reference = ir.Reference(globalClockName, ir.ClockType, firrtl.PortKind, firrtl.SourceFlow)
    // keeps track of which clock signals will be replaced by which clock enable signal
    val isRemovedClock: String => Boolean = scanResults.clockToEnable.contains
    // registers whose next function needs to be guarded with a clock enable
    val registerToEnable: Map[String, ir.Reference] = scanResults.registerToEnable.toMap
    // memory enables which need to be guarded with clock enables
    val memPortToClockEnable: Map[String, ir.Reference] = scanResults.memPortToClockEnable.toMap
    // keep track of memory names
    val isMem: String => Boolean = scanResults.mems.contains
    val memInfo: String => ir.DefMemory = scanResults.mems
    val isResetToChange: String => Boolean = scanResults.resetsToChange.contains
  }

  private var mainName: String = "" // for debugging
  private def unsupportedError(msg: String): Nothing =
    throw new UnsupportedFeatureException(s"StutteringClockTransform: [$mainName] $msg")

  private def mux(cond: ir.Expression, a: ir.Expression, b: ir.Expression): ir.Expression = {
    ir.Mux(cond, a, b, Utils.mux_type_and_widths(a, b))
  }
  private def and(a: ir.Expression, b: ir.Expression): ir.Expression =
    ir.DoPrim(PrimOps.And, List(a, b), List(), Bool)
  private val Bool = ir.UIntType(ir.IntWidth(1))
}

private class UnsupportedFeatureException(s: String) extends PassException(s)