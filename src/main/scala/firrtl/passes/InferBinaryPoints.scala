// See LICENSE for license details.

package firrtl.passes

import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.annotations.{CircuitTarget, ModuleTarget, ReferenceTarget, Target}
import firrtl.constraint.ConstraintSolver
import firrtl.Transform
import firrtl.options.{Dependency, PreservesAll}

class InferBinaryPoints extends Pass with PreservesAll[Transform] {

  override val prerequisites =
    Seq( Dependency(ResolveKinds),
         Dependency(InferTypes),
         Dependency(Uniquify),
         Dependency(ResolveFlows) )

  override val dependents = Seq.empty

  private val constraintSolver = new ConstraintSolver()

  private def addTypeConstraints(r1: ReferenceTarget, r2: ReferenceTarget)(t1: Type, t2: Type): Unit = (t1,t2) match {
    case (UIntType(w1), UIntType(w2)) =>
    case (SIntType(w1), SIntType(w2)) =>
    case (ClockType, ClockType) =>
    case (ResetType, _) =>
    case (_, ResetType) =>
    case (AsyncResetType, AsyncResetType) =>
    case (FixedType(w1, p1), FixedType(w2, p2)) =>
      constraintSolver.addGeq(p1, p2, r1.prettyPrint(""), r2.prettyPrint(""))
    case (IntervalType(l1, u1, p1), IntervalType(l2, u2, p2)) =>
      constraintSolver.addGeq(p1, p2, r1.prettyPrint(""), r2.prettyPrint(""))
    case (AnalogType(w1), AnalogType(w2)) =>
    case (t1: BundleType, t2: BundleType) =>
      (t1.fields zip t2.fields) foreach { case (f1, f2) =>
        (f1.flip, f2.flip) match {
          case (Default, Default) => addTypeConstraints(r1.field(f1.name), r2.field(f2.name))(f1.tpe, f2.tpe)
          case (Flip, Flip) => addTypeConstraints(r2.field(f2.name), r1.field(f1.name))(f2.tpe, f1.tpe)
          case _ => sys.error("Shouldn't be here")
        }
      }
    case (t1: VectorType, t2: VectorType) => addTypeConstraints(r1.index(0), r2.index(0))(t1.tpe, t2.tpe)
    case other => throwInternalError(s"Illegal compiler state: cannot constraint different types - $other")
  }
  private def addDecConstraints(t: Type): Type = t map addDecConstraints
  private def addStmtConstraints(mt: ModuleTarget)(s: Statement): Statement = s map addDecConstraints match {
    case c: Connect =>
      val n = get_size(c.loc.tpe)
      val locs = create_exps(c.loc)
      val exps = create_exps(c.expr)
      (locs zip exps) foreach { case (loc, exp) =>
        to_flip(flow(loc)) match {
          case Default => addTypeConstraints(Target.asTarget(mt)(loc), Target.asTarget(mt)(exp))(loc.tpe, exp.tpe)
          case Flip => addTypeConstraints(Target.asTarget(mt)(exp), Target.asTarget(mt)(loc))(exp.tpe, loc.tpe)
        }
      }
      c
    case pc: PartialConnect =>
      val ls = get_valid_points(pc.loc.tpe, pc.expr.tpe, Default, Default)
      val locs = create_exps(pc.loc)
      val exps = create_exps(pc.expr)
      ls foreach { case (x, y) =>
        val loc = locs(x)
        val exp = exps(y)
        to_flip(flow(loc)) match {
          case Default => addTypeConstraints(Target.asTarget(mt)(loc), Target.asTarget(mt)(exp))(loc.tpe, exp.tpe)
          case Flip => addTypeConstraints(Target.asTarget(mt)(exp), Target.asTarget(mt)(loc))(exp.tpe, loc.tpe)
        }
      }
      pc
    case r: DefRegister =>
       addTypeConstraints(mt.ref(r.name), Target.asTarget(mt)(r.init))(r.tpe, r.init.tpe)
      r
    case x => x map addStmtConstraints(mt)
  }
  private def fixWidth(w: Width): Width = constraintSolver.get(w) match {
    case Some(Closed(x)) if trim(x).isWhole => IntWidth(x.toBigInt)
    case None => w
    case _ => sys.error("Shouldn't be here")
  }
  private def fixType(t: Type): Type = t map fixType map fixWidth match {
    case IntervalType(l, u, p) =>
      val px = constraintSolver.get(p) match {
        case Some(Closed(x)) if trim(x).isWhole => IntWidth(x.toBigInt)
        case None => p
        case _ => sys.error("Shouldn't be here")
      }
      IntervalType(l, u, px)
    case FixedType(w, p) =>
      val px = constraintSolver.get(p) match {
        case Some(Closed(x)) if trim(x).isWhole => IntWidth(x.toBigInt)
        case None => p
        case _ => sys.error("Shouldn't be here")
      }
      FixedType(w, px)
    case x => x
  }
  private def fixStmt(s: Statement): Statement = s map fixStmt map fixType
  private def fixPort(p: Port): Port = Port(p.info, p.name, p.direction, fixType(p.tpe))
  def run (c: Circuit): Circuit = {
    val ct = CircuitTarget(c.main)
    c.modules foreach (m => m map addStmtConstraints(ct.module(m.name)))
    c.modules foreach (_.ports foreach {p => addDecConstraints(p.tpe)})
    constraintSolver.solve()
    InferTypes.run(c.copy(modules = c.modules map (_
      map fixPort
      map fixStmt)))
  }
}
