// See LICENSE for license details.

package firrtl.analyses

import scala.collection.mutable
import firrtl._
import firrtl.ir._
import firrtl.graph._
import firrtl.Utils._
import firrtl.traversals.Foreachers._
import firrtl.annotations.TargetToken._

/** A class representing the instance hierarchy of a working IR Circuit
  *
  * @constructor constructs an instance graph from a Circuit
  * @param c the Circuit to analyze
  * @note The current implementation has some performance problems, which is why [[InstanceKeyGraph]]
  *       exists and should be preferred for new use cases. Eventually the old class will be deprecated
  *       in favor of the new implementation.
  *       The performance problems in the old implementation stem from the fact that DefInstance is used as the
  *       key to the underlying Map. DefInstance contains the type of the module besides the module and instance names.
  *       This type is not needed as it can be inferred from the module name. If the module name is the same,
  *       the type will be the same and vice versa.
  *       Hashing and comparing deep bundle types however is inefficient which can manifest in slower then necessary
  *       lookups and insertions.
  */
@deprecated("Use InstanceKeyGraph instead.", "FIRRTL 1.4")
class InstanceGraph(c: Circuit) {

  @deprecated("Use InstanceKeyGraph.moduleMap instead.", "FIRRTL 1.4")
  val moduleMap = c.modules.map({ m => (m.name, m) }).toMap
  private val instantiated = new mutable.LinkedHashSet[String]
  private val childInstances =
    new mutable.LinkedHashMap[String, mutable.LinkedHashSet[DefInstance]]
  for (m <- c.modules) {
    childInstances(m.name) = new mutable.LinkedHashSet[DefInstance]
    m.foreach(InstanceGraph.collectInstances(childInstances(m.name)))
    instantiated ++= childInstances(m.name).map(i => i.module)
  }

  private val instanceGraph = new MutableDiGraph[DefInstance]
  private val instanceQueue = new mutable.Queue[DefInstance]

  for (subTop <- c.modules.view.map(_.name).filterNot(instantiated)) {
    val topInstance = DefInstance(subTop, subTop)
    instanceQueue.enqueue(topInstance)
    while (instanceQueue.nonEmpty) {
      val current = instanceQueue.dequeue
      instanceGraph.addVertex(current)
      for (child <- childInstances(current.module)) {
        if (!instanceGraph.contains(child)) {
          instanceQueue.enqueue(child)
          instanceGraph.addVertex(child)
        }
        instanceGraph.addEdge(current, child)
      }
    }
  }

  // The true top module (circuit main)
  private val trueTopInstance = DefInstance(c.main, c.main)

  /** A directed graph showing the instance dependencies among modules
    * in the circuit. Every DefInstance of a module has an edge to
    * every DefInstance arising from every instance statement in
    * that module.
    */
  @deprecated("Use InstanceKeyGraph.graph instead.", "FIRRTL 1.4")
  lazy val graph = DiGraph(instanceGraph)

  /** A list of absolute paths (each represented by a Seq of instances)
    * of all module instances in the Circuit.
    */
  @deprecated("Use InstanceKeyGraph.fullHierarchy instead.", "FIRRTL 1.4")
  lazy val fullHierarchy: mutable.LinkedHashMap[DefInstance, Seq[Seq[DefInstance]]] = graph.pathsInDAG(trueTopInstance)

  /** A count of the *static* number of instances of each module. For any module other than the top (main) module, this is
    * equivalent to the number of inst statements in the circuit instantiating each module, irrespective of the number
    * of times (if any) the enclosing module appears in the hierarchy. Note that top module of the circuit has an
    * associated count of one, even though it is never directly instantiated. Any modules *not* instantiated at all will
    * have a count of zero.
    */
  @deprecated("Use InstanceKeyGraph.staticInstanceCount instead.", "FIRRTL 1.4")
  lazy val staticInstanceCount: Map[OfModule, Int] = {
    val foo = mutable.LinkedHashMap.empty[OfModule, Int]
    childInstances.keys.foreach {
      case main if main == c.main => foo += main.OfModule -> 1
      case other                  => foo += other.OfModule -> 0
    }
    childInstances.values.flatten.map(_.OfModule).foreach {
      case mod => foo += mod -> (foo(mod) + 1)
    }
    foo.toMap
  }

  /** Finds the absolute paths (each represented by a Seq of instances
    * representing the chain of hierarchy) of all instances of a particular
    * module. Note that this includes one implicit instance of the top (main)
    * module of the circuit. If the module is not instantiated within the
    * hierarchy of the top module of the circuit, it will return Nil.
    *
    * @param module the name of the selected module
    * @return a Seq[ Seq[DefInstance] ] of absolute instance paths
    */
  @deprecated("Use InstanceKeyGraph.findInstancesInHierarchy instead (now with caching of vertices!).", "FIRRTL 1.4")
  def findInstancesInHierarchy(module: String): Seq[Seq[DefInstance]] = {
    val instances = graph.getVertices.filter(_.module == module).toSeq
    instances.flatMap { i => fullHierarchy.getOrElse(i, Nil) }
  }

  /** An [[firrtl.graph.EulerTour EulerTour]] representation of the [[firrtl.graph.DiGraph DiGraph]] */
  @deprecated("Should have been private. Do not use outside of InstanceGraph.", "FIRRTL 1.4")
  lazy val tour = EulerTour(graph, trueTopInstance)

  /** Finds the lowest common ancestor instances for two module names in
    * a design
    */
  @deprecated("Use InstanceKeyGraph and EulerTour(iGraph.graph, iGraph.top).rmq(moduleA, moduleB).", "FIRRTL 1.4")
  def lowestCommonAncestor(moduleA: Seq[DefInstance], moduleB: Seq[DefInstance]): Seq[DefInstance] = {
    tour.rmq(moduleA, moduleB)
  }

  /**
    * Module order from highest module to leaf module
    * @return sequence of modules in order from top to leaf
    */
  @deprecated("Use InstanceKeyGraph.moduleOrder instead.", "FIRRTL 1.4")
  def moduleOrder: Seq[DefModule] = {
    graph.transformNodes(_.module).linearize.map(moduleMap(_))
  }

  /** Given a circuit, returns a map from module name to children
    * instance/module definitions
    */
  @deprecated("Use InstanceKeyGraph.getChildInstances instead.", "FIRRTL 1.4")
  def getChildrenInstances: mutable.LinkedHashMap[String, mutable.LinkedHashSet[DefInstance]] = childInstances

  /** Given a circuit, returns a map from module name to children
    * instance/module [[firrtl.annotations.TargetToken]]s
    */
  @deprecated("Use InstanceKeyGraph.getChildInstances instead.", "FIRRTL 1.4")
  def getChildrenInstanceOfModule: mutable.LinkedHashMap[String, mutable.LinkedHashSet[(Instance, OfModule)]] =
    childInstances.map(kv => kv._1 -> kv._2.map(_.toTokens))

  // Transforms a TraversableOnce input into an order-preserving map
  // Iterates only once, no intermediate collections
  // Can possibly be replaced using LinkedHashMap.from(..) or better immutable map in Scala 2.13
  private def asOrderedMap[K1, K2, V](it: TraversableOnce[K1], f: (K1) => (K2, V)): collection.Map[K2, V] = {
    val lhmap = new mutable.LinkedHashMap[K2, V]
    it.foreach { lhmap += f(_) }
    lhmap
  }

  /** Given a circuit, returns a map from module name to a map
    * in turn mapping instances names to corresponding module names
    */
  @deprecated("Use InstanceKeyGraph.getChildInstanceMap instead.", "FIRRTL 1.4")
  def getChildrenInstanceMap: collection.Map[OfModule, collection.Map[Instance, OfModule]] =
    childInstances.map(kv => kv._1.OfModule -> asOrderedMap(kv._2, (i: DefInstance) => i.toTokens))

  /** The set of all modules in the circuit */
  @deprecated("Use InstanceKeyGraph instead.", "FIRRTL 1.4")
  lazy val modules: collection.Set[OfModule] = graph.getVertices.map(_.OfModule)

  /** The set of all modules in the circuit reachable from the top module */
  @deprecated("Use InstanceKeyGraph instead.", "FIRRTL 1.4")
  lazy val reachableModules: collection.Set[OfModule] =
    mutable.LinkedHashSet(trueTopInstance.OfModule) ++ graph.reachableFrom(trueTopInstance).map(_.OfModule)

  /** The set of all modules *not* reachable in the circuit */
  @deprecated("Use InstanceKeyGraph.unreachableModules instead.", "FIRRTL 1.4")
  lazy val unreachableModules: collection.Set[OfModule] = modules.diff(reachableModules)

}

@deprecated("Use InstanceKeyGraph instead.", "FIRRTL 1.4")
object InstanceGraph {

  /** Returns all DefInstances in a Statement
    *
    * @param insts mutable datastructure to append to
    * @param s statement to descend
    * @return
    */
  @deprecated("Use InstanceKeyGraph.collectInstances instead.", "FIRRTL 1.4")
  def collectInstances(insts: mutable.Set[DefInstance])(s: Statement): Unit = s match {
    case i: DefInstance           => insts += i
    case i: DefInstance           => throwInternalError("Expecting DefInstance, found a DefInstance!")
    case i: WDefInstanceConnector => throwInternalError("Expecting DefInstance, found a DefInstanceConnector!")
    case _ => s.foreach(collectInstances(insts))
  }
}
