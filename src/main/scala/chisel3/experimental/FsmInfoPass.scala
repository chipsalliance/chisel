// Copyright 2021 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package chisel3.experimental

import chisel3.experimental.EnumAnnotations.{EnumComponentAnnotation, EnumDefAnnotation}
import firrtl.annotations._
import firrtl._
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency

import scala.collection.mutable

case class FsmInfoAnnotation(target: ReferenceTarget, states: Seq[(BigInt, String)], transitions: Seq[(BigInt, BigInt)], start: Option[BigInt])
  extends SingleTargetAnnotation[ReferenceTarget] {
  override def duplicate(n: ReferenceTarget) = copy(target=n)
}

/** Annotates FSMs in the design with information about all available states and transitions. */
object FsmInfoPass extends Transform with DependencyAPIMigration {
  val Prefix = "f"

  override def prerequisites: Seq[TransformDependency] = Forms.LowForm
  override def invalidates(a: Transform): Boolean = false

  override protected def execute(state: CircuitState): CircuitState = {
    val enums = state.annotations.collect { case a: EnumDefAnnotation => a.typeName -> a }.toMap
    val components = state.annotations.collect { case a : EnumComponentAnnotation => a }

    // if there are no enums, we won't be able to find any FSMs
    if(enums.isEmpty) return state

    val c = CircuitTarget(state.circuit.main)
    val infos = state.circuit.modules.flatMap(onModule(_, c, enums, components))

    state.copy(annotations = infos ++: state.annotations)
  }

  private def onModule(m: ir.DefModule, c: CircuitTarget, enums: Map[String, EnumDefAnnotation], components: Seq[EnumComponentAnnotation]): List[Annotation] = m match {
    case mod: ir.Module =>
      val localComponents = components
        .filter(c => toReferenceTarget(c.target).module == mod.name)
        .map(c => toReferenceTarget(c.target).ref -> c).toMap
      if (localComponents.isEmpty) {
        List()
      } else {
        val enumRegs = new ModuleScanner(localComponents).run(mod)
        enumRegs.map { case EnumReg(enumTypeName, regDef, next) =>
          analyzeFSM(c.module(mod.name), regDef, next, enums(enumTypeName).definition)
        }.toList
      }
    case other => List()
  }

  private def analyzeFSM(module: ModuleTarget, regDef: ir.DefRegister, nextExpr: ir.Expression, states: Map[String, BigInt]): FsmInfoAnnotation = {
    val (resetState, next) = destructReset(nextExpr)
    // println(s"Next: ${next.serialize}")
    val allStates = states.values.toSet
    val transitions = destructMux(next).flatMap { case (guard, nx) =>
      val from = guardStates(guard, regDef.name, allStates).getOrElse(allStates)
      val to = nextStates(nx, regDef.name, allStates, from)
      from.flatMap(f => to.map(t => f -> t))
    }.sortBy(_._1)

    FsmInfoAnnotation(module.ref(regDef.name),
      states = states.toSeq.sorted.map{ case (n, i) => i -> n },
      transitions = transitions,
      start = resetState
    )
  }

  // tries to extract the reset value
  private def destructReset(e: ir.Expression): (Option[BigInt], ir.Expression) = e match {
    case ir.Mux(ir.Reference("reset", _, _, _), rval: ir.UIntLiteral, oval, _) => (Some(rval.value), oval)
    case ir.Mux(ir.DoPrim(PrimOps.Not, Seq(ir.Reference("reset", _, _, _)), _, _), oval, rval: ir.UIntLiteral, _) => (Some(rval.value), oval)
    case _ => (None, e)
  }

  private def destructMux(e: ir.Expression): List[(ir.Expression, ir.Expression)] = e match {
    case ir.Mux(cond, tval, fval, _) =>
      val tru = destructMux(tval)
      val fals = destructMux(fval)
      tru.map{ case (guard, value) => (Utils.and(cond, guard), value) } ++
        fals.map{ case (guard, value) => (Utils.and(Utils.not(cond), guard), value) }
    case other => List((Utils.True(), other))
  }
  private def nextStates(e: ir.Expression, name: String, allStates: Set[BigInt], guards: Set[BigInt]): Set[BigInt] = e match {
    case c: ir.UIntLiteral => Set(c.value)
    case r: ir.Reference if r.name == name => guards
    case _ => allStates
  }
  private def guardStates(e: ir.Expression, name: String, allStates: Set[BigInt]): Option[Set[BigInt]] = e match {
    case ir.DoPrim(PrimOps.Eq, Seq(r: ir.Reference, c: ir.UIntLiteral), _, _) if r.name == name =>
      Some(Set(c.value))
    case ir.DoPrim(PrimOps.Eq, Seq(c: ir.UIntLiteral, r: ir.Reference), _, _) if r.name == name =>
      Some(Set(c.value))
    case ir.DoPrim(PrimOps.Neq, Seq(r: ir.Reference, c: ir.UIntLiteral), _, _) if r.name == name =>
      Some(allStates -- Set(c.value))
    case ir.DoPrim(PrimOps.Neq, Seq(c: ir.UIntLiteral, r: ir.Reference), _, _) if r.name == name =>
      Some(allStates -- Set(c.value))
    case ir.DoPrim(PrimOps.Or, Seq(a, b), _, _) =>
      val aStates = guardStates(a, name, allStates)
      val bStates = guardStates(b, name, allStates)
      combineOr(aStates, bStates)
    case ir.DoPrim(PrimOps.And, Seq(a, b), _, _) =>
      val aStates = guardStates(a, name, allStates)
      val bStates = guardStates(b, name, allStates)
      combineAnd(aStates, bStates)
    case ir.DoPrim(PrimOps.Not, Seq(a), _, _) =>
      val aStates = guardStates(a, name, allStates)
      aStates match {
        case Some(s) => Some(allStates -- s)
        case None => None
      }
    // try to analyze the following pattern
    // orr(cat(cat(eq(release_state, UInt(9)), eq(release_state, UInt(6))), eq(release_state, UInt(1))))
    case ir.DoPrim(PrimOps.Orr, Seq(s @ ir.DoPrim(PrimOps.Cat, _, _, _)), _, tpe) =>
      val bits = getCatBits(s)
      bits.foreach { b =>
        assert(firrtl.bitWidth(b.tpe) == 1, s"Cannot deal with concatenated value ${b.serialize}")
      }

      val sts = bits.map(guardStates(_, name, allStates))
      if(sts.length == 1) {
        sts.head
      } else {
        sts.reduce(combineOr)
      }
    case other =>
      val symbols = findSymbols(other)
      if(symbols.contains(name)) {
        // throw new RuntimeException(s"failed to analyze:\n" + other.serialize)
        // logger.warn("[FSM] over-approximating the states")
        Some(allStates)
      } else { None } // no states
  }

  private def combineOr(aStates: Option[Set[BigInt]], bStates: Option[Set[BigInt]]): Option[Set[BigInt]] = {
    (aStates, bStates) match {
      case (None, None) => None
      case (None, a) => a
      case (a, None) => a
      case (Some(a), Some(b)) => Some(a | b)
    }
  }

  private def combineAnd(aStates: Option[Set[BigInt]], bStates: Option[Set[BigInt]]): Option[Set[BigInt]] = {
    (aStates, bStates) match {
      case (None, None) => None
      case (None, a) => a
      case (a, None) => a
      case (Some(a), Some(b)) => Some(a & b)
    }
  }

  private def getCatBits(e: ir.Expression): List[ir.Expression] = e match {
    case ir.DoPrim(PrimOps.Cat, Seq(msb, lsb), _, _) =>
      getCatBits(msb) ++ getCatBits(lsb)
    case other => List(other)
  }

  private def findSymbols(e: ir.Expression): Seq[String] = e match {
    case r: ir.Reference => Seq(r.name)
    case ir.SubField(expr, _, _, _) => findSymbols(expr)
    case ir.SubIndex(expr, _, _, _) => findSymbols(expr)
    case ir.SubAccess(expr, index, _, _) => Seq(expr, index).flatMap(findSymbols)
    case ir.DoPrim(_, args, _, _) => args.flatMap(findSymbols)
    case ir.Mux(cond, tval, fval, _) => Seq(cond, tval, fval).flatMap(findSymbols)
    case ir.ValidIf(cond, value, _) => Seq(cond, value).flatMap(findSymbols)
    case _ => Seq()
  }

  private def toReferenceTarget(n: Named): ReferenceTarget = n match {
    case ComponentName(name, module) => module.toTarget.ref(name)
  }
}

/** searches for state machine registers */
private class ModuleScanner(localComponents: Map[String, EnumComponentAnnotation]) {
  private val regDefs = mutable.HashMap[String, ir.DefRegister]()
  private val connects = mutable.HashMap[String, ir.Expression]()

  def run(mod: ir.Module): Seq[EnumReg] = {
    mod.foreachStmt(onStmt)
    regDefs.keys.toSeq.map { key =>
      val (next, _) = inlineComb(connects(key), key)
      EnumReg(localComponents(key).enumTypeName, regDefs(key), next)
    }
  }

  /** resolves references to nodes (all wires should have been removed at this point)
   *  Ignores any subexpressions that do not actually contain references to the state register.
   * */
  private def inlineComb(e: ir.Expression, stateReg: String): (ir.Expression, Boolean) = e match {
    case r: ir.Reference if r.kind == firrtl.NodeKind =>
      val (e, shouldInline) = inlineComb(connects(r.name), stateReg)
      if(shouldInline) { (e, true) } else { (r, false) }
    case r: ir.Reference if r.name == stateReg => (r, true)
    // registers are always plain references, so any RefLikeExpression that gets here is not a state register
    case r: ir.RefLikeExpression => (r, false)
    case p : ir.DoPrim =>
      val c = p.args.map(inlineComb(_, stateReg))
      val shouldInline = c.exists(_._2)
      if(shouldInline) { (p.copy(args = c.map(_._1)), true) } else { (p, false) }
    case m @ ir.Mux(cond, tval, fval, tpe) =>
      val c = Seq(cond, tval, fval).map(inlineComb(_, stateReg))
      val shouldInline = c.exists(_._2)
      if(shouldInline) {
        (ir.Mux(c(0)._1, c(1)._1, c(2)._1, tpe), true)
      } else { (m, false) }
    case v@ ir.ValidIf(cond, value, tpe) =>
      val c = Seq(cond, value).map(inlineComb(_, stateReg))
      val shouldInline = c.exists(_._2)
      if(shouldInline) {
        (ir.ValidIf(c(0)._1, c(1)._1, tpe), true)
      } else { (v, false) }
    case l: ir.Literal => (l, false)
    case other => throw new RuntimeException(s"Unexpected expression $other")
  }
  private def onStmt(s: ir.Statement): Unit = s match {
    case r: ir.DefRegister if localComponents.contains(r.name) => regDefs(r.name) = r
    case ir.Connect(_, loc, expr) => connects(loc.serialize) = expr
    case ir.DefNode(_, name, expr) => connects(name) = expr
    case other => other.foreachStmt(onStmt)
  }
}
private case class EnumReg(enumTypeName: String, regDef: ir.DefRegister, next: ir.Expression)