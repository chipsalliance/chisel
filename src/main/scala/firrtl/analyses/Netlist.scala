// See LICENSE for license details.

package firrtl.analyses

import scala.collection.mutable

import firrtl._
import firrtl.ir._
import firrtl.graph._
import firrtl.Utils._
import firrtl.Mappers._


/** A class representing the instance hierarchy of a working IR Circuit
  *
  * @constructor constructs an instance graph from a Circuit
  * @param c the Circuit to analyze
  */
class InstanceGraph(c: Circuit) {

  private def collectInstances(insts: mutable.Set[WDefInstance])
                              (s: Statement): Statement = s match {
    case i: WDefInstance =>
      insts += i
      i
    case _ =>
      s map collectInstances(insts)
  }

  private val moduleMap = c.modules.map({m => (m.name,m) }).toMap
  private val instantiated = new mutable.HashSet[String]
  private val childInstances =
    new mutable.HashMap[String,mutable.Set[WDefInstance]]
  for (m <- c.modules) {
    childInstances(m.name) = new mutable.HashSet[WDefInstance]
    m map collectInstances(childInstances(m.name))
    instantiated ++= childInstances(m.name).map(i => i.module)
  }

  private val uninstantiated = moduleMap.keySet -- instantiated
  private val instanceGraph = new MutableDiGraph[WDefInstance]
  private val instanceQueue = new mutable.Queue[WDefInstance]

  uninstantiated.foreach({ subTop =>
    val topInstance = WDefInstance(subTop,subTop)
    instanceQueue.enqueue(topInstance)
    while (!instanceQueue.isEmpty) {
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
  })

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
  lazy val fullHierarchy = graph.pathsInDAG(trueTopInstance)

  /** Finds the absolute paths (each represented by a Seq of instances
    * representing the chain of hierarchy) of all instances of a
    * particular module.
    *
    * @param module the name of the selected module
    * @return a Seq[Seq[WDefInstance]] of absolute instance paths
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
}
