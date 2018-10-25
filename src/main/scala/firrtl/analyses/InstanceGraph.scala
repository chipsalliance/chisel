// See LICENSE for license details.

package firrtl.analyses

import scala.collection.mutable
import firrtl._
import firrtl.ir._
import firrtl.graph._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.annotations.TargetToken.{Instance, OfModule}


/** A class representing the instance hierarchy of a working IR Circuit
  *
  * @constructor constructs an instance graph from a Circuit
  * @param c the Circuit to analyze
  */
class InstanceGraph(c: Circuit) {

  val moduleMap = c.modules.map({m => (m.name,m) }).toMap
  private val instantiated = new mutable.LinkedHashSet[String]
  private val childInstances =
    new mutable.LinkedHashMap[String, mutable.LinkedHashSet[WDefInstance]]
  for (m <- c.modules) {
    childInstances(m.name) = new mutable.LinkedHashSet[WDefInstance]
    m map InstanceGraph.collectInstances(childInstances(m.name))
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

  /** Finds the absolute paths (each represented by a Seq of instances
    * representing the chain of hierarchy) of all instances of a
    * particular module.
    *
    * @param module the name of the selected module
    * @return a Seq[ Seq[WDefInstance] ] of absolute instance paths
    */
  def findInstancesInHierarchy(module: String): Seq[Seq[WDefInstance]] = {
    val instances = graph.getVertices.filter(_.module == module).toSeq
    instances flatMap { i => fullHierarchy(i) }
  }

  /** An `[[EulerTour]]` representation of the `[[DiGraph]]` */
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
    childInstances.map(kv => kv._1 -> kv._2.map(i => (Instance(i.name), OfModule(i.module))))


}

object InstanceGraph {

  /** Returns all WDefInstances in a Statement
    *
    * @param insts mutable datastructure to append to
    * @param s statement to descend
    * @return
    */
  def collectInstances(insts: mutable.Set[WDefInstance])
                      (s: Statement): Statement = s match {
    case i: WDefInstance =>
      insts += i
      i
    case i: DefInstance => throwInternalError("Expecting WDefInstance, found a DefInstance!")
    case i: WDefInstanceConnector => throwInternalError("Expecting WDefInstance, found a WDefInstanceConnector!")
    case _ => s map collectInstances(insts)
  }
}
