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
class InstanceGraph(c: Circuit) {

  val moduleMap = c.modules.map({m => (m.name,m) }).toMap
  private val instantiated = new mutable.LinkedHashSet[String]
  private val childInstances =
    new mutable.LinkedHashMap[String, mutable.LinkedHashSet[WDefInstance]]
  for (m <- c.modules) {
    childInstances(m.name) = new mutable.LinkedHashSet[WDefInstance]
    m.foreach(InstanceGraph.collectInstances(childInstances(m.name)))
    instantiated ++= childInstances(m.name).map(i => i.module)
  }

  private val instanceGraph = new MutableDiGraph[WDefInstance]
  private val instanceQueue = new mutable.Queue[WDefInstance]

  for (subTop <- c.modules.view.map(_.name).filterNot(instantiated)) {
    val topInstance = WDefInstance(subTop,subTop)
    instanceQueue.enqueue(topInstance)
    while (instanceQueue.nonEmpty) {
      val current = instanceQueue.dequeue
      instanceGraph.addVertex(current)
      for (child <- childInstances(current.module)) {
        if (!instanceGraph.contains(child)) {
          instanceQueue.enqueue(child)
          instanceGraph.addVertex(child)
        }
        instanceGraph.addEdge(current,child)
      }
    }
  }

  // The true top module (circuit main)
  private val trueTopInstance = WDefInstance(c.main, c.main)

  /** A directed graph showing the instance dependencies among modules
    * in the circuit. Every WDefInstance of a module has an edge to
    * every WDefInstance arising from every instance statement in
    * that module.
    */
  lazy val graph = DiGraph(instanceGraph)

  /** A list of absolute paths (each represented by a Seq of instances)
    * of all module instances in the Circuit.
    */
  lazy val fullHierarchy: mutable.LinkedHashMap[WDefInstance,Seq[Seq[WDefInstance]]] = graph.pathsInDAG(trueTopInstance)

  /** A count of the *static* number of instances of each module. For any module other than the top (main) module, this is
    * equivalent to the number of inst statements in the circuit instantiating each module, irrespective of the number
    * of times (if any) the enclosing module appears in the hierarchy. Note that top module of the circuit has an
    * associated count of one, even though it is never directly instantiated. Any modules *not* instantiated at all will
    * have a count of zero.
    */
  lazy val staticInstanceCount: Map[OfModule, Int] = {
    val foo = mutable.LinkedHashMap.empty[OfModule, Int]
    childInstances.keys.foreach {
      case main if main == c.main => foo += main.OfModule  -> 1
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
    * @return a Seq[ Seq[WDefInstance] ] of absolute instance paths
    */
  def findInstancesInHierarchy(module: String): Seq[Seq[WDefInstance]] = {
    val instances = graph.getVertices.filter(_.module == module).toSeq
    instances flatMap { i => fullHierarchy.getOrElse(i, Nil) }
  }

  /** An [[firrtl.graph.EulerTour EulerTour]] representation of the [[firrtl.graph.DiGraph DiGraph]] */
  lazy val tour = EulerTour(graph, trueTopInstance)

  /** Finds the lowest common ancestor instances for two module names in
    * a design
    */
  def lowestCommonAncestor(moduleA: Seq[WDefInstance],
                           moduleB: Seq[WDefInstance]): Seq[WDefInstance] = {
    tour.rmq(moduleA, moduleB)
  }

  /**
    * Module order from highest module to leaf module
    * @return sequence of modules in order from top to leaf
    */
  def moduleOrder: Seq[DefModule] = {
    graph.transformNodes(_.module).linearize.map(moduleMap(_))
  }


  /** Given a circuit, returns a map from module name to children
     * instance/module definitions
     */
  def getChildrenInstances: mutable.LinkedHashMap[String, mutable.LinkedHashSet[WDefInstance]] = childInstances

  /** Given a circuit, returns a map from module name to children
    * instance/module [[firrtl.annotations.TargetToken]]s
    */
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
  def getChildrenInstanceMap: collection.Map[OfModule, collection.Map[Instance, OfModule]] =
    childInstances.map(kv => kv._1.OfModule -> asOrderedMap(kv._2, (i: WDefInstance) => i.toTokens))

  /** The set of all modules in the circuit */
  lazy val modules: collection.Set[OfModule] = graph.getVertices.map(_.OfModule)

  /** The set of all modules in the circuit reachable from the top module */
  lazy val reachableModules: collection.Set[OfModule] =
    mutable.LinkedHashSet(trueTopInstance.OfModule) ++ graph.reachableFrom(trueTopInstance).map(_.OfModule)

  /** The set of all modules *not* reachable in the circuit */
  lazy val unreachableModules: collection.Set[OfModule] = modules diff reachableModules

}

object InstanceGraph {

  /** Returns all WDefInstances in a Statement
    *
    * @param insts mutable datastructure to append to
    * @param s statement to descend
    * @return
    */
  def collectInstances(insts: mutable.Set[WDefInstance])
                      (s: Statement): Unit = s match {
    case i: WDefInstance => insts += i
    case i: DefInstance => throwInternalError("Expecting WDefInstance, found a DefInstance!")
    case i: WDefInstanceConnector => throwInternalError("Expecting WDefInstance, found a WDefInstanceConnector!")
    case _ => s.foreach(collectInstances(insts))
  }
}
