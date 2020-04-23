// See LICENSE for license details.

package firrtl.checks

import firrtl._
import firrtl.options.{Dependency, PreservesAll}
import firrtl.passes.{Errors, PassException}
import firrtl.ir._
import firrtl.Utils.isCast
import firrtl.traversals.Foreachers._
import firrtl.WrappedExpression._

import scala.collection.mutable
import scala.annotation.tailrec

object CheckResets {
  class NonLiteralAsyncResetValueException(info: Info, mname: String, reg: String, init: String) extends PassException(
    s"$info: [module $mname] AsyncReset Reg '$reg' reset to non-literal '$init'")

  // Map of Initialization Expression to check
  private type RegCheckList = mutable.ListBuffer[(Expression, DefRegister)]
  // Record driving for literal propagation
  // Indicates *driven by*
  private type DirectDriverMap = mutable.HashMap[WrappedExpression, Expression]

}

// Must run after ExpandWhens
// Requires
//   - static single connections of ground types
class CheckResets extends Transform with DependencyAPIMigration with PreservesAll[Transform] {

  override def prerequisites =
    Seq( Dependency(passes.LowerTypes),
         Dependency(passes.Legalize),
         Dependency(firrtl.transforms.RemoveReset) ) ++ firrtl.stage.Forms.MidForm

  override def optionalPrerequisites = Seq(Dependency[firrtl.transforms.CheckCombLoops])

  override def optionalPrerequisiteOf = Seq.empty

  import CheckResets._

  private def onStmt(regCheck: RegCheckList, drivers: DirectDriverMap)(stmt: Statement): Unit = {
    stmt match {
      case DefNode(_, name, expr) => drivers += we(WRef(name)) -> expr
      case Connect(_, lhs, rhs) => drivers += we(lhs) -> rhs
      case reg @ DefRegister(_, name, _,_,_, init) if weq(WRef(name), init) => // Self-reset, allowed!
      case reg @ DefRegister(_,_,_,_, reset, init) if reset.tpe == AsyncResetType =>
        regCheck += init -> reg
      case _ => // Do nothing
    }
    stmt.foreach(onStmt(regCheck, drivers))
  }

  private def wireOrNode(kind: Kind) = (kind == WireKind || kind == NodeKind)

  @tailrec
  private def findDriver(drivers: DirectDriverMap)(expr: Expression): Expression = expr match {
    case lit: Literal => lit
    case DoPrim(op, args, _,_) if isCast(op) => findDriver(drivers)(args.head)
    case other => drivers.get(we(other)) match {
      case Some(e) if wireOrNode(Utils.kind(other)) => findDriver(drivers)(e)
      case _ => other
    }
  }

  private def onMod(errors: Errors)(mod: DefModule): Unit = {
    val regCheck = new RegCheckList()
    val drivers = new DirectDriverMap()
    mod.foreach(onStmt(regCheck, drivers))
    for ((init, reg) <- regCheck) {
      for (subInit <- Utils.create_exps(init)) {
        findDriver(drivers)(subInit) match {
          case lit: Literal => // All good
          case other =>
            val e = new NonLiteralAsyncResetValueException(reg.info, mod.name, reg.name, other.serialize)
            errors.append(e)
        }
      }
    }
  }

  def execute(state: CircuitState): CircuitState = {
    val errors = new Errors
    state.circuit.foreach(onMod(errors))
    errors.trigger()
    state
  }
}
