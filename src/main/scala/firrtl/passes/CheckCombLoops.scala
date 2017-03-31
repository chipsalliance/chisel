// See LICENSE for license details.

package firrtl.passes

import scala.collection.mutable
import scala.collection.immutable.HashSet
import scala.collection.immutable.HashMap
import annotation.tailrec

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils.throwInternalError
import firrtl.graph.{MutableDiGraph,DiGraph}
import firrtl.analyses.InstanceGraph

/** Finds and detects combinational logic loops in a circuit, if any
  * exist. Returns the input circuit with no modifications.
  * 
  * @throws a CombLoopException if a loop is found
  * @note Input form: Low FIRRTL
  * @note Output form: Low FIRRTL (identity transform)
  * @note The pass looks for loops through combinational-read memories
  * @note The pass cannot find loops that pass through ExtModules
  * @note The pass will throw exceptions on "false paths"
  */

object CheckCombLoops extends Pass {

  class CombLoopException(info: Info, mname: String, cycle: Seq[String]) extends PassException(
    s"$info: [module $mname] Combinational loop detected:\n" + cycle.mkString("\n"))

  /*
   * A case class that represents a net in the circuit. This is
   * necessary since combinational loop checking is an analysis on the
   * netlist of the circuit; the fields are specialized for low
   * FIRRTL. Since all wires are ground types, a given ground type net
   * may only be a subfield of an instance or a memory
   * port. Therefore, it is uniquely specified within its module
   * context by its name, its optional parent instance (a WDefInstance
   * or WDefMemory), and its optional memory port name.
   */
  private case class LogicNode(name: String, inst: Option[String] = None, memport: Option[String] = None)

  private def toLogicNode(e: Expression): LogicNode = e match {
    case r: WRef =>
      LogicNode(r.name)
    case s: WSubField =>
      s.exp match {
        case modref: WRef =>
          LogicNode(s.name,Some(modref.name))
        case memport: WSubField =>
          memport.exp match {
            case memref: WRef =>
              LogicNode(s.name,Some(memref.name),Some(memport.name))
            case _ => throwInternalError
          }
        case _ => throwInternalError
      }
  }

  private def getExprDeps(deps: mutable.Set[LogicNode])(e: Expression): Expression = e match {
    case r: WRef =>
      deps += toLogicNode(r)
      r
    case s: WSubField =>
      deps += toLogicNode(s)
      s
    case _ =>
      e map getExprDeps(deps)
  }

  private def getStmtDeps(
    simplifiedModules: mutable.Map[String,DiGraph[LogicNode]],
    deps: MutableDiGraph[LogicNode])(s: Statement): Statement = {
    s match {
      case Connect(_,loc,expr) =>
        val lhs = toLogicNode(loc)
        if (deps.contains(lhs)) {
          getExprDeps(deps.getEdges(lhs))(expr)
        }
      case w: DefWire =>
        deps.addVertex(LogicNode(w.name))
      case n: DefNode =>
        val lhs = LogicNode(n.name)
        deps.addVertex(lhs)
        getExprDeps(deps.getEdges(lhs))(n.value)
      case m: DefMemory if (m.readLatency == 0) =>
        for (rp <- m.readers) {
          val dataNode = deps.addVertex(LogicNode("data",Some(m.name),Some(rp)))
          deps.addEdge(dataNode, deps.addVertex(LogicNode("addr",Some(m.name),Some(rp))))
          deps.addEdge(dataNode, deps.addVertex(LogicNode("en",Some(m.name),Some(rp))))
        }
      case i: WDefInstance =>
        val iGraph = simplifiedModules(i.module).transformNodes(n => n.copy(inst = Some(i.name)))
        for (v <- iGraph.getVertices) {
          deps.addVertex(v)
          iGraph.getEdges(v).foreach { deps.addEdge(v,_) }
        }
      case _ =>
        s map getStmtDeps(simplifiedModules,deps)
    }
    s
  }

  /*
   * Recover the full path from a path passing through simplified
   * instances. Since edges may pass through simplified instances, the
   * hierarchy that the path passes through must be recursively
   * recovered.
   */
  private def expandInstancePaths(
    m: String,
    moduleGraphs: mutable.Map[String,DiGraph[LogicNode]],
    moduleDeps: Map[String, Map[String,String]], 
    prefix: Seq[String],
    path: Seq[LogicNode]): Seq[String] = {
    def absNodeName(prefix: Seq[String], n: LogicNode) =
      (prefix ++ n.inst ++ n.memport :+ n.name).mkString(".")
    val pathNodes = (path zip path.tail) map { case (a, b) =>
      if (a.inst.isDefined && !a.memport.isDefined && a.inst == b.inst) {
        val child = moduleDeps(m)(a.inst.get)
        val newprefix = prefix :+ a.inst.get
        val subpath = moduleGraphs(child).path(b.copy(inst=None),a.copy(inst=None)).tail.reverse
        expandInstancePaths(child,moduleGraphs,moduleDeps,newprefix,subpath)
      } else {
        Seq(absNodeName(prefix,a))
      }
    }
    pathNodes.flatten :+ absNodeName(prefix, path.last)
  }

  /*
   * An SCC may contain more than one loop. In this case, the sequence
   * of nodes forming the SCC cannot be interpreted as a simple
   * cycle. However, it is desirable to print an error consisting of a
   * loop rather than an arbitrary ordering of the SCC. This function
   * operates on a pruned subgraph composed only of the SCC and finds
   * a simple cycle by performing an arbitrary walk.
   */
  private def findCycleInSCC[T](sccGraph: DiGraph[T]): Seq[T] = {
    val walk = new mutable.ArrayBuffer[T]
    val visited = new mutable.HashSet[T]
    var current = sccGraph.getVertices.head
    while (!visited.contains(current)) {
      walk += current
      visited += current
      current = sccGraph.getEdges(current).head
    }
    walk.drop(walk.indexOf(current)).toSeq :+ current
  }

  /*
   * This implementation of combinational loop detection avoids ever
   * generating a full netlist from the FIRRTL circuit. Instead, each
   * module is converted to a netlist and analyzed locally, with its
   * subinstances represented by trivial, simplified subgraphs. The
   * overall outline of the process is:
   * 
   * 1. Create a graph of module instance dependances

   * 2. Linearize this acyclic graph
   * 
   * 3. Generate a local netlist; replace any instances with
   * simplified subgraphs representing connectivity of their IOs
   * 
   * 4. Check for nontrivial strongly connected components
   * 
   * 5. Create a reduced representation of the netlist with only the
   * module IOs as nodes, where output X (which must be a ground type,
   * as only low FIRRTL is supported) will have an edge to input Y if
   * and only if it combinationally depends on input Y. Associate this
   * reduced graph with the module for future use.
   */
  def run(c: Circuit): Circuit = {
    val errors = new Errors()
    /* TODO(magyar): deal with exmodules! No pass warnings currently
     *  exist. Maybe warn when iterating through modules.
     */
    val moduleMap = c.modules.map({m => (m.name,m) }).toMap
    val iGraph = new InstanceGraph(c)
    val moduleDeps = iGraph.graph.edges.map{ case (k,v) => (k.module, (v map { i => (i.name, i.module) }).toMap) }
    val topoSortedModules = iGraph.graph.transformNodes(_.module).linearize.reverse map { moduleMap(_) }
    val moduleGraphs = new mutable.HashMap[String,DiGraph[LogicNode]]
    val simplifiedModuleGraphs = new mutable.HashMap[String,DiGraph[LogicNode]]
    for (m <- topoSortedModules) {
      val internalDeps = new MutableDiGraph[LogicNode]
      m.ports.foreach({ p => internalDeps.addVertex(LogicNode(p.name)) })
      m map getStmtDeps(simplifiedModuleGraphs, internalDeps)
      val moduleGraph = DiGraph(internalDeps)
      moduleGraphs(m.name) = moduleGraph
      simplifiedModuleGraphs(m.name) = moduleGraphs(m.name).simplify((m.ports map { p => LogicNode(p.name) }).toSet)
      for (scc <- moduleGraphs(m.name).findSCCs.filter(_.length > 1)) {
        val sccSubgraph = moduleGraphs(m.name).subgraph(scc.toSet)
        val cycle = findCycleInSCC(sccSubgraph)
        (cycle zip cycle.tail).foreach({ case (a,b) => require(moduleGraph.getEdges(a).contains(b)) })
        val expandedCycle = expandInstancePaths(m.name,moduleGraphs,moduleDeps,Seq(m.name),cycle.reverse)
        errors.append(new CombLoopException(m.info, m.name, expandedCycle))
      }
    }
    errors.trigger()
    c
  }

}
