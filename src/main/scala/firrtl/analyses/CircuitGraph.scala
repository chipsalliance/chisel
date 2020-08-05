// See LICENSE for license details.

package firrtl.analyses

import firrtl.Kind
import firrtl.analyses.InstanceKeyGraph.InstanceKey
import firrtl.annotations.TargetToken.{Instance, OfModule}
import firrtl.annotations._
import firrtl.ir.{Circuit, DefInstance}

/** Use to construct [[CircuitGraph]]
  * Also contains useful related functions
  */
object CircuitGraph {

  /** Build a CircuitGraph
    * [[firrtl.ir.Circuit]] must be of MiddleForm or lower
    * @param circuit
    * @return
    */
  def apply(circuit: Circuit): CircuitGraph = new CircuitGraph(ConnectionGraph(circuit))

  /** Return a nicely-formatted string of a path of [[firrtl.annotations.ReferenceTarget]]
    * @param connectionPath
    * @param tab
    * @return
    */
  def prettyToString(connectionPath: Seq[ReferenceTarget], tab: String = ""): String = {
    tab + connectionPath.mkString(s"\n$tab")
  }
}

/** Graph-representation of a FIRRTL Circuit
  *
  * Requires Middle FIRRTL
  * Useful for writing design-specific custom-transforms that require connectivity information
  *
  * @param connectionGraph Source-to-sink connectivity graph
  */
class CircuitGraph private[analyses] (connectionGraph: ConnectionGraph) {

  // Reverse (sink-to-source) connectivity graph
  private lazy val reverseConnectionGraph = connectionGraph.reverseConnectionGraph

  // AST Circuit
  private val circuit = connectionGraph.circuit

  // AST Information
  private val irLookup = connectionGraph.irLookup

  // Module/Instance Hierarchy information
  private lazy val instanceGraph = InstanceKeyGraph(circuit)

  // Per module, which modules does it instantiate
  private lazy val moduleChildren = instanceGraph.getChildInstances.toMap

  // Top-level module target
  private val main = ModuleTarget(circuit.main, circuit.main)

  /** Given a signal, return the signals that it drives
    * @param source
    * @return
    */
  def fanOutSignals(source: ReferenceTarget): Set[ReferenceTarget] = connectionGraph.getEdges(source).toSet

  /** Given a signal, return the signals that drive it
    * @param sink
    * @return
    */
  def fanInSignals(sink: ReferenceTarget): Set[ReferenceTarget] = reverseConnectionGraph.getEdges(sink).toSet

  /** Return the absolute paths of all instances of this module.
    *
    * For example:
    *   - Top instantiates a1 of A and a2 of A
    *   - A instantiates b1 of B and b2 of B
    * Then, absolutePaths of B will return:
    *   - Seq(~Top|Top/a1:A/b1:B, ~Top|Top/a1:A/b2:B, ~Top|Top/a2:A/b1:B, ~Top|Top/a2:A/b2:B)
    * @param mt
    * @return
    */
  def absolutePaths(mt: ModuleTarget): Seq[IsModule] = instanceGraph.findInstancesInHierarchy(mt.module).map {
    case seq if seq.nonEmpty => seq.foldLeft(CircuitTarget(circuit.main).module(circuit.main): IsModule) {
      case (it, InstanceKey(instance, ofModule)) => it.instOf(instance, ofModule)
    }
  }

  /** Return the sequence of nodes from source to sink, inclusive
    * @param source
    * @param sink
    * @return
    */
  def connectionPath(source: ReferenceTarget, sink: ReferenceTarget): Seq[ReferenceTarget] =
    connectionGraph.path(source, sink)

  /** Return a reference to all nodes of given kind, directly contained in the referenced module/instance
    * Path can be either a module, or an instance
    * @param path
    * @param kind
    * @return
    */
  def localReferences(path: IsModule, kind: Kind): Seq[ReferenceTarget] = {
    val leafModule = path.leafModule
    irLookup.kindFinder(ModuleTarget(circuit.main, leafModule), kind).map(_.setPathTarget(path))
  }

  /** Return a reference to all nodes of given kind, contained in the referenced module/instance or any child instance
    * Path can be either a module, or an instance
    * @param kind
    * @param path
    * @return
    */
  def deepReferences(kind: Kind, path: IsModule = ModuleTarget(circuit.main, circuit.main)): Seq[ReferenceTarget] = {
    val leafModule = path.leafModule
    val children = moduleChildren(leafModule)
    val localRefs = localReferences(path, kind)
    localRefs ++ children.flatMap { child => deepReferences(kind, path.instOf(child.name, child.module)) }
  }

  /** Return all absolute references to signals of the given kind directly contained in the module
    * @param moduleTarget
    * @param kind
    * @return
    */
  def absoluteReferences(moduleTarget: ModuleTarget, kind: Kind): Seq[ReferenceTarget] = {
    localReferences(moduleTarget, kind).flatMap(makeAbsolute)
  }

  /** Given a reference, return all instances of that reference (i.e. with absolute paths)
    * @param reference
    * @return
    */
  def makeAbsolute(reference: ReferenceTarget): Seq[ReferenceTarget] = {
    absolutePaths(reference.moduleTarget).map(abs => reference.setPathTarget(abs))
  }
}
