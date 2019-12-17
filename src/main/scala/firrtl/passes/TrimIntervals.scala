// See LICENSE for license details.

package firrtl.passes

import firrtl.PrimOps._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.constraint.{IsFloor, IsKnown, IsMul}
import firrtl.options.{Dependency, PreservesAll}
import firrtl.Transform

/** Replaces IntervalType with SIntType, three AST walks:
  * 1) Align binary points
  *    - adds shift operators to primop args and connections
  *    - does not affect declaration- or inferred-types
  * 2) Replace declaration IntervalType's with SIntType's
  *    - for each declaration:
  *      a. remove non-zero binary points
  *      b. remove open bounds
  *      c. replace with SIntType
  * 3) Run InferTypes
  */
class TrimIntervals extends Pass with PreservesAll[Transform] {

  override val prerequisites =
    Seq( Dependency(ResolveKinds),
         Dependency(InferTypes),
         Dependency(Uniquify),
         Dependency(ResolveFlows),
         Dependency[InferBinaryPoints] )

  override val dependents = Seq.empty

  def run(c: Circuit): Circuit = {
    // Open -> closed
    val firstPass = InferTypes.run(c map replaceModuleInterval)
    // Align binary points and adjust range accordingly (loss of precision changes range)
    firstPass map alignModuleBP
  }

  /* Replace interval types */
  private def replaceModuleInterval(m: DefModule): DefModule = m map replaceStmtInterval map replacePortInterval

  private def replaceStmtInterval(s: Statement): Statement = s map replaceTypeInterval map replaceStmtInterval

  private def replacePortInterval(p: Port): Port = p map replaceTypeInterval

  private def replaceTypeInterval(t: Type): Type = t match {
    case i@IntervalType(l: IsKnown, u: IsKnown, IntWidth(p)) =>
      IntervalType(Closed(i.min.get), Closed(i.max.get), IntWidth(p))
    case i: IntervalType => i
    case v => v map replaceTypeInterval
  }

  /* Align interval binary points -- BINARY POINT ALIGNMENT AFFECTS RANGE INFERENCE! */
  private def alignModuleBP(m: DefModule): DefModule = m map alignStmtBP

  private def alignStmtBP(s: Statement): Statement = s map alignExpBP match {
    case c@Connect(info, loc, expr) => loc.tpe match {
      case IntervalType(_, _, p) => Connect(info, loc, fixBP(p)(expr))
      case _ => c
    }
    case c@PartialConnect(info, loc, expr) => loc.tpe match {
      case IntervalType(_, _, p) => PartialConnect(info, loc, fixBP(p)(expr))
      case _ => c
    }
    case other => other map alignStmtBP
  }

  // Note - wrap/clip/squeeze ignore the binary point of the second argument, thus not needed to be aligned
  // Note - Mul does not need its binary points aligned, because multiplication is cool like that
  private val opsToFix = Seq(Add, Sub, Lt, Leq, Gt, Geq, Eq, Neq/*, Wrap, Clip, Squeeze*/)

  private def alignExpBP(e: Expression): Expression = e map alignExpBP match {
    case DoPrim(SetP, Seq(arg), Seq(const), tpe: IntervalType) => fixBP(IntWidth(const))(arg)
    case DoPrim(o, args, consts, t) if opsToFix.contains(o) &&
      (args.map(_.tpe).collect { case x: IntervalType => x }).size == args.size =>
      val maxBP = args.map(_.tpe).collect { case IntervalType(_, _, p) => p }.reduce(_ max _)
      DoPrim(o, args.map { a => fixBP(maxBP)(a) }, consts, t)
    case Mux(cond, tval, fval, t: IntervalType) =>
      val maxBP = Seq(tval, fval).map(_.tpe).collect { case IntervalType(_, _, p) => p }.reduce(_ max _)
      Mux(cond, fixBP(maxBP)(tval), fixBP(maxBP)(fval), t)
    case other => other
  }
  private def fixBP(p: Width)(e: Expression): Expression = (p, e.tpe) match {
    case (IntWidth(desired), IntervalType(l, u, IntWidth(current))) if desired == current => e
    case (IntWidth(desired), IntervalType(l, u, IntWidth(current))) if desired > current  =>
      DoPrim(IncP, Seq(e), Seq(desired - current), IntervalType(l, u, IntWidth(desired)))
    case (IntWidth(desired), IntervalType(l, u, IntWidth(current))) if desired < current  =>
      val shiftAmt = current - desired
      val shiftGain = BigDecimal(BigInt(1) << shiftAmt.toInt)
      val shiftMul = Closed(BigDecimal(1) / shiftGain)
      val bpGain = BigDecimal(BigInt(1) << current.toInt)
      // BP is inferred at this point
      // y = floor(x * 2^(-amt + bp)) gets rid of precision --> y * 2^(-bp + amt)
      val newBPRes = Closed(shiftGain / bpGain)
      val bpResInv = Closed(bpGain)
      val newL = IsMul(IsFloor(IsMul(IsMul(l, shiftMul), bpResInv)), newBPRes)
      val newU = IsMul(IsFloor(IsMul(IsMul(u, shiftMul), bpResInv)), newBPRes)
      DoPrim(DecP, Seq(e), Seq(current - desired), IntervalType(CalcBound(newL), CalcBound(newU), IntWidth(desired)))
    case x => sys.error(s"Shouldn't be here: $x")
  }
}

// vim: set ts=4 sw=4 et:
