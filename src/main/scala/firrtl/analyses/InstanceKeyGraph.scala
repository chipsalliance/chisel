// See LICENSE for license details.

package firrtl.analyses

import firrtl.annotations._
import firrtl.annotations.TargetToken._
import firrtl.graph.{DiGraph, EulerTour, MutableDiGraph}
import firrtl.ir

import scala.collection.mutable

/** A class representing the instance hierarchy of firrtl Circuit
  * This is a faster version of the old `InstanceGraph` which only uses
  * pairs of InstanceName and Module name as vertex keys instead of using WDefInstance
  * which will hash the instance type causing some performance issues.
  */
class InstanceKeyGraph private (c: ir.Circuit) {
  import InstanceKeyGraph._

  private val nameToModule: Map[String, ir.DefModule] = c.modules.map({ m => (m.name, m) }).toMap
  private val childInstances: Seq[(String, Seq[InstanceKey])] = c.modules.map { m =>
    m.name -> InstanceKeyGraph.collectInstances(m)
  }
  private val instantiated = childInstances.flatMap(_._2).map(_.module).toSet
  private val roots = c.modules.map(_.name).filterNot(instantiated)
  private val internalGraph = buildGraph(childInstances, roots)
  private val circuitTopInstance = topKey(c.main)
  // cache vertices to speed up repeat calls to findInstancesInHierarchy
  private lazy val vertices = internalGraph.getVertices

  /** A list of absolute paths (each represented by a Seq of instances) of all module instances in the Circuit. */
  private lazy val cachedFullHierarchy: mutable.LinkedHashMap[InstanceKey, Seq[Seq[InstanceKey]]] =
    internalGraph.pathsInDAG(circuitTopInstance)

  /** modules reachable from the main module as well as the main modules itself */
  private lazy val cachedReachableModules: Seq[OfModule] =
    circuitTopInstance.OfModule +: internalGraph.reachableFrom(circuitTopInstance).toSeq.map(_.OfModule)

  private lazy val cachedUnreachableModules: Seq[OfModule] = {
    val all = mutable.LinkedHashSet(childInstances.map(c => OfModule(c._1)): _*)
    val reachable = mutable.LinkedHashSet(cachedReachableModules: _*)
    all.diff(reachable).toSeq
  }

  /** returns the underlying graph */
  def graph: DiGraph[InstanceKey] = internalGraph

  /** returns the key to the top (main) module */
  def top: InstanceKey = circuitTopInstance

  /** maps module names to the DefModule node */
  def moduleMap: Map[String, ir.DefModule] = nameToModule

  /** Module order from highest module to leaf module */
  def moduleOrder: Seq[ir.DefModule] = internalGraph.transformNodes(_.module).linearize.map(nameToModule(_))

  /** Returns a sequence that can be turned into a map from module name to instances defined in said module. */
  def getChildInstances: Seq[(String, Seq[InstanceKey])] = childInstances

  /** A count of the *static* number of instances of each module. For any module other than the top (main) module,
    * this is equivalent to the number of inst statements in the circuit instantiating each module,
    * irrespective of the number of times (if any) the enclosing module appears in the hierarchy.
    * Note that top module of the circuit has an associated count of one, even though it is never directly instantiated.
    * Any modules *not* instantiated at all will have a count of zero.
    */
  def staticInstanceCount: Map[OfModule, Int] = cachedStaticInstanceCount

  private lazy val cachedStaticInstanceCount = {
    val foo = mutable.LinkedHashMap.empty[OfModule, Int]
    childInstances.foreach {
      case (main, _) if main == c.main => foo += main.OfModule -> 1
      case (other, _)                  => foo += other.OfModule -> 0
    }
    childInstances.flatMap(_._2).map(_.OfModule).foreach { mod =>
      foo += mod -> (foo(mod) + 1)
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
  def findInstancesInHierarchy(module: String): Seq[Seq[InstanceKey]] = {
    val instances = vertices.filter(_.module == module).toSeq
    instances.flatMap { i => cachedFullHierarchy.getOrElse(i, Nil) }
  }

  /** Given a circuit, returns a map from module name to a map
    * in turn mapping instances names to corresponding module names
    */
  def getChildInstanceMap: mutable.LinkedHashMap[OfModule, mutable.LinkedHashMap[Instance, OfModule]] =
    mutable.LinkedHashMap(childInstances.map {
      case (k, v) =>
        val moduleMap: mutable.LinkedHashMap[Instance, OfModule] = mutable.LinkedHashMap(v.map(_.toTokens): _*)
        TargetToken.OfModule(k) -> moduleMap
    }: _*)

  /** All modules in the circuit reachable from the top module */
  def reachableModules: Seq[OfModule] = cachedReachableModules

  /** All modules *not* reachable from the top module of the circuit */
  def unreachableModules: Seq[OfModule] = cachedUnreachableModules

  /** A list of absolute paths (each represented by a Seq of instances) of all module instances in the Circuit. */
  def fullHierarchy: mutable.LinkedHashMap[InstanceKey, Seq[Seq[InstanceKey]]] = cachedFullHierarchy
}

object InstanceKeyGraph {
  def apply(c: ir.Circuit): InstanceKeyGraph = new InstanceKeyGraph(c)

  /** We want to only use this untyped version as key because hashing bundle types is expensive
    * @param name the name of the instance
    * @param module the name of the module that is instantiated
    */
  case class InstanceKey(name: String, module: String) {
    def Instance: Instance = TargetToken.Instance(name)
    def OfModule: OfModule = TargetToken.OfModule(module)
    def toTokens: (Instance, OfModule) = (Instance, OfModule)
  }

  /** Finds all instance definitions in a firrtl Module. */
  def collectInstances(m: ir.DefModule): Seq[InstanceKey] = m match {
    case _: ir.ExtModule => Seq()
    case ir.Module(_, _, _, body) => {
      val instances = mutable.ArrayBuffer[InstanceKey]()
      def onStmt(s: ir.Statement): Unit = s match {
        case firrtl.WDefInstance(_, name, module, _) => instances += InstanceKey(name, module)
        case ir.DefInstance(_, name, module, _)      => instances += InstanceKey(name, module)
        case _: firrtl.WDefInstanceConnector =>
          firrtl.Utils.throwInternalError("Expecting WDefInstance, found a WDefInstanceConnector!")
        case other => other.foreachStmt(onStmt)
      }
      onStmt(body)
      instances.toSeq
    }
  }

  private def topKey(module: String): InstanceKey = InstanceKey(module, module)

  private def buildGraph(
    childInstances: Seq[(String, Seq[InstanceKey])],
    roots:          Iterable[String]
  ): DiGraph[InstanceKey] = {
    val instanceGraph = new MutableDiGraph[InstanceKey]
    val childInstanceMap = childInstances.toMap

    // iterate over all modules that are not instantiated and thus act as a root
    roots.foreach { subTop =>
      // create a root node
      val topInstance = topKey(subTop)
      // graph traversal
      val instanceQueue = new mutable.Queue[InstanceKey]
      instanceQueue.enqueue(topInstance)
      while (instanceQueue.nonEmpty) {
        val current = instanceQueue.dequeue
        instanceGraph.addVertex(current)
        for (child <- childInstanceMap(current.module)) {
          if (!instanceGraph.contains(child)) {
            instanceQueue.enqueue(child)
            instanceGraph.addVertex(child)
          }
          instanceGraph.addEdge(current, child)
        }
      }
    }
    instanceGraph
  }
}
