// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.graph.MutableDiGraph
import scala.collection.mutable

case class State(sym: SMTSymbol, init: Option[SMTExpr], next: Option[SMTExpr]) {
  def name: String = sym.name
}
case class Signal(name: String, e: SMTExpr, lbl: SignalLabel = IsNode) {
  def toSymbol: SMTSymbol = SMTSymbol.fromExpr(name, e)
  def sym:      SMTSymbol = toSymbol
}
case class TransitionSystem(
  name:     String,
  inputs:   List[BVSymbol],
  states:   List[State],
  signals:  List[Signal],
  comments: Map[String, String] = Map(),
  header:   String = "") {
  def serialize: String = TransitionSystem.serialize(this)
}

sealed trait SignalLabel
case object IsNode extends SignalLabel
case object IsOutput extends SignalLabel
case object IsConstraint extends SignalLabel
case object IsBad extends SignalLabel
case object IsFair extends SignalLabel
case object IsNext extends SignalLabel
case object IsInit extends SignalLabel

object SignalLabel {
  private val labels = Seq(IsNode, IsOutput, IsConstraint, IsBad, IsFair, IsNext, IsInit)
  val labelStrings = Seq("node", "output", "constraint", "bad", "fair", "next", "init")
  val labelToString: SignalLabel => String = labels.zip(labelStrings).toMap
  val stringToLabel: String => SignalLabel = labelStrings.zip(labels).toMap
}

object TransitionSystem {
  def serialize(sys: TransitionSystem): String = {
    (Iterator(sys.name) ++
      sys.inputs.map(i => s"input ${i.name} : ${SMTExpr.serializeType(i)}") ++
      sys.signals.map(s => s"${SignalLabel.labelToString(s.lbl)} ${s.name} : ${SMTExpr.serializeType(s.e)} = ${s.e}") ++
      sys.states.map(serialize)).mkString("\n")
  }

  def serialize(s: State): String = {
    s"state ${s.sym.name} : ${SMTExpr.serializeType(s.sym)}" +
      s.init.map("\n  [init] " + _).getOrElse("") +
      s.next.map("\n  [next] " + _).getOrElse("")
  }

  def systemExpressions(sys: TransitionSystem): List[SMTExpr] =
    sys.signals.map(_.e) ++ sys.states.flatMap(s => s.init ++ s.next)

  def findUninterpretedFunctions(sys: TransitionSystem): List[DeclareFunction] = {
    val calls = systemExpressions(sys).flatMap(findUFCalls)
    // find unique functions
    calls.groupBy(_.sym.name).map(_._2.head).toList
  }

  private def findUFCalls(e: SMTExpr): List[DeclareFunction] = {
    val f = e match {
      case BVFunctionCall(name, args, width) =>
        Some(DeclareFunction(BVSymbol(name, width), args))
      case ArrayFunctionCall(name, args, indexWidth, dataWidth) =>
        Some(DeclareFunction(ArraySymbol(name, indexWidth, dataWidth), args))
      case _ => None
    }
    f.toList ++ e.children.flatMap(findUFCalls)
  }
}

private object TopologicalSort {

  /** Ensures that all signals in the resulting system are topologically sorted.
    * This is necessary because [[firrtl.transforms.RemoveWires]] does
    * not sort assignments to outputs, submodule inputs nor memory ports.
    */
  def run(sys: TransitionSystem): TransitionSystem = {
    val inputsAndStates = sys.inputs.map(_.name) ++ sys.states.map(_.sym.name)
    val signalOrder = sort(sys.signals.map(s => s.name -> s.e), inputsAndStates)
    // TODO: maybe sort init expressions of states (this should not be needed most of the time)
    signalOrder match {
      case None => sys
      case Some(order) =>
        val signalMap = sys.signals.map(s => s.name -> s).toMap
        // we flatMap over `get` in order to ignore inputs/states in the order
        sys.copy(signals = order.flatMap(signalMap.get).toList)
    }
  }

  private def sort(signals: Iterable[(String, SMTExpr)], globalSignals: Iterable[String]): Option[Iterable[String]] = {
    val known = new mutable.HashSet[String]() ++ globalSignals
    var needsReordering = false
    val digraph = new MutableDiGraph[String]
    signals.foreach {
      case (name, expr) =>
        digraph.addVertex(name)
        val uniqueDependencies = mutable.LinkedHashSet[String]() ++ findDependencies(expr)
        uniqueDependencies.foreach { d =>
          if (!known.contains(d)) { needsReordering = true }
          digraph.addPairWithEdge(name, d)
        }
        known.add(name)
    }
    if (needsReordering) {
      Some(digraph.linearize.reverse)
    } else { None }
  }

  private def findDependencies(expr: SMTExpr): List[String] = expr match {
    case BVSymbol(name, _)       => List(name)
    case ArraySymbol(name, _, _) => List(name)
    case other                   => other.children.flatMap(findDependencies)
  }
}
