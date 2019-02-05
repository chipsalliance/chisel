package chisel3.transform

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations.ModuleTarget

import scala.collection._

class InlineTemporaries extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override protected def execute(state: CircuitState): CircuitState = {
    val (ret, renames) = InlineTemporaries.inlineTemps(state.circuit)
    state.copy(circuit = ret, renames = Some(renames))
  }

}

object InlineTemporaries {
  def recordNodes(temps: mutable.HashMap[String, DefNode], reads: mutable.HashMap[String, Int])
                 (s: Statement): Statement = {
    def onExpr(info: Info)(e: Expression): Expression = e match {
      case Reference(name, tpe) if temps.contains(name) && info == temps(name).info =>
        reads(name) = reads.getOrElse(name, 0) + 1
        e
      case WRef(name, _, _, _) if temps.contains(name) =>
        reads(name) = reads.getOrElse(name, 0) + 1
        e
      case other => other map onExpr(info)
    }
    s match {
      case h: HasInfo =>
        s.mapExpr(onExpr(h.info)) match {
          case n: DefNode =>
            temps(n.name) = n
            n
          case other => other.mapStmt(recordNodes(temps, reads))
        }
      case other => other.mapStmt(recordNodes(temps, reads))
    }
  }

  val isTemp = "_T([0-9]+)".r

  def canInline(temps: mutable.HashMap[String, DefNode], reads: mutable.HashMap[String, Int])
               (name: String, info: Info): Boolean = {
    name match {
      case isTemp(id) if temps.contains(name) && info == temps(name).info && reads.get(name) == Some(1) => true
      case _ => false
    }
  }

  def inlineNodes(temps: mutable.HashMap[String, DefNode],
                  reads: mutable.HashMap[String, Int],
                  renameMap: RenameMap,
                  mt: ModuleTarget
                 )(s: Statement): Statement = {

    def onExpr(info: Info)(e: Expression): Expression = e match {
      case Reference(name, tpe) if canInline(temps, reads)(name, info) =>
        onExpr(info)(temps(name).value)
      case WRef(name, _, _, _) if canInline(temps, reads)(name, info) =>
        onExpr(info)(temps(name).value)
      case other => other map onExpr(info)
    }

    s match {
      case h: HasInfo =>
        s.mapExpr(onExpr(h.info)) match {
          case n: DefNode if canInline(temps, reads)(n.name, n.info) =>
            renameMap.record(mt.ref(n.name), Nil)
            EmptyStmt
          case other => other.mapStmt(inlineNodes(temps, reads, renameMap, mt))
        }
      case other => other.mapStmt(inlineNodes(temps, reads, renameMap, mt))
    }
  }


  def inlineTemps(circuit: Circuit): (Circuit, RenameMap) = {
    val renameMap = RenameMap()
    val newModules = circuit.modules.map { m =>
      val temps = new mutable.HashMap[String, DefNode]()
      val reads = new mutable.HashMap[String, Int]()

      m.foreachStmt(recordNodes(temps, reads))
      m.mapStmt(inlineNodes(temps, reads, renameMap, ModuleTarget(circuit.main, m.name))).mapStmt(Utils.squashEmpty)
    }

    (circuit.copy(modules = newModules), renameMap)
  }
}
