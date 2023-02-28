// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt.random

import firrtl._
import firrtl.annotations.NoTargetAnnotation
import firrtl.ir._
import firrtl.passes._
import firrtl.options.Dependency
import firrtl.stage.Forms
import firrtl.transforms.RemoveWires

import scala.collection.mutable

/** Chooses how to model explicit and implicit invalid values in the circuit */
case class InvalidToRandomOptions(
  randomizeInvalidSignals: Boolean = true,
  randomizeDivisionByZero: Boolean = true)
    extends NoTargetAnnotation

/** Replaces all explicit and implicit "invalid" values with random values.
  * Explicit invalids are:
  * - signal is invalid
  * - signal <= valid(..., expr)
  * Implicit invalids are:
  * - a / b when eq(b, 0)
  */
object InvalidToRandomPass extends Transform with DependencyAPIMigration {
  override def prerequisites = Forms.LowForm
  // once ValidIf has been removed, we can no longer detect and randomize them
  override def optionalPrerequisiteOf = Seq(Dependency(RemoveValidIf))
  override def invalidates(a: Transform) = a match {
    // this pass might destroy SSA form, as we add a wire for the data field of every read port
    case _: RemoveWires => true
    // TODO: should we add some optimization passes here? we could be generating some dead code.
    case _ => false
  }

  override protected def execute(state: CircuitState): CircuitState = {
    val opts = state.annotations.collect { case o: InvalidToRandomOptions => o }
    require(opts.size < 2, s"Multiple options: $opts")
    val opt = opts.headOption.getOrElse(InvalidToRandomOptions())

    // quick exit if we just want to skip this pass
    if (!opt.randomizeDivisionByZero && !opt.randomizeInvalidSignals) {
      state
    } else {
      val c = state.circuit.mapModule(onModule(_, opt))
      state.copy(circuit = c)
    }
  }

  private def onModule(m: DefModule, opt: InvalidToRandomOptions): DefModule = m match {
    case d: DescribedMod =>
      throw new RuntimeException(s"CompilerError: Unexpected internal node: ${d.serialize}")
    case e:   ExtModule => e
    case mod: Module =>
      val namespace = Namespace(mod)
      mod.mapStmt(onStmt(namespace, opt, _))
  }

  private def onStmt(namespace: Namespace, opt: InvalidToRandomOptions, s: Statement): Statement = s match {
    case IsInvalid(info, loc: RefLikeExpression) if opt.randomizeInvalidSignals =>
      val name = namespace.newName(loc.serialize.replace('.', '_') + "_invalid")
      val rand = DefRandom(info, name, loc.tpe, None)
      Block(List(rand, Connect(info, loc, Reference(rand))))
    case other =>
      val info = other match {
        case h: HasInfo => h.info
        case _ => NoInfo
      }
      val prefix = other match {
        case c: Connect => c.loc.serialize.replace('.', '_')
        case h: HasName => h.name
        case _ => ""
      }
      val ctx = ExprCtx(namespace, opt, prefix, info, mutable.ListBuffer[Statement]())
      val stmt = other.mapExpr(onExpr(ctx, _)).mapStmt(onStmt(namespace, opt, _))
      if (ctx.rands.isEmpty) { stmt }
      else { Block(Block(ctx.rands.toList), stmt) }
  }

  private case class ExprCtx(
    namespace: Namespace,
    opt:       InvalidToRandomOptions,
    prefix:    String,
    info:      Info,
    rands:     mutable.ListBuffer[Statement])

  private def onExpr(ctx: ExprCtx, e: Expression): Expression =
    e.mapExpr(onExpr(ctx, _)) match {
      case ValidIf(_, value, tpe) if tpe == ClockType =>
        // we currently assume that clocks are always valid
        // TODO: is that a good assumption?
        value
      case ValidIf(cond, value, tpe) if ctx.opt.randomizeInvalidSignals =>
        makeRand(ctx, cond, tpe, value, invert = true)
      case d @ DoPrim(PrimOps.Div, Seq(_, den), _, tpe) if ctx.opt.randomizeDivisionByZero =>
        val denIsZero = Utils.eq(den, Utils.getGroundZero(den.tpe.asInstanceOf[GroundType]))
        makeRand(ctx, denIsZero, tpe, d, invert = false)
      case other => other
    }

  private def makeRand(
    ctx:    ExprCtx,
    cond:   Expression,
    tpe:    Type,
    value:  Expression,
    invert: Boolean
  ): Expression = {
    val name = ctx.namespace.newName(if (ctx.prefix.isEmpty) "invalid" else ctx.prefix + "_invalid")
    // create a condition node if the condition isn't a reference already
    val condRef = cond match {
      case r: RefLikeExpression => if (invert) Utils.not(r) else r
      case other =>
        val cond = if (invert) Utils.not(other) else other
        val condNode = DefNode(ctx.info, ctx.namespace.newName(name + "_cond"), cond)
        ctx.rands.append(condNode)
        Reference(condNode)
    }
    val rand = DefRandom(ctx.info, name, tpe, None, condRef)
    ctx.rands.append(rand)
    Utils.mux(condRef, Reference(rand), value)
  }
}
