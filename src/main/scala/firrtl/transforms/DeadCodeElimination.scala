
package firrtl.transforms

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.annotations._
import firrtl.graph._
import firrtl.analyses.InstanceGraph
import firrtl.Mappers._
import firrtl.WrappedExpression._
import firrtl.Utils.{throwInternalError, toWrappedExpression, kind}
import firrtl.MemoizedHash._
import firrtl.options.RegisteredTransform
import scopt.OptionParser

import collection.mutable
import java.io.{File, FileWriter}

/** Dead Code Elimination (DCE)
  *
  * Performs DCE by constructing a global dependency graph starting with top-level outputs, external
  * module ports, and simulation constructs as circuit sinks. External modules can optionally be
  * eligible for DCE via the [[OptimizableExtModuleAnnotation]].
  *
  * Dead code is eliminated across module boundaries. Wires, ports, registers, and memories are all
  * eligible for removal. Components marked with a [[DontTouchAnnotation]] will be treated as a
  * circuit sink and thus anything that drives such a marked component will NOT be removed.
  *
  * This transform preserves deduplication. All instances of a given [[firrtl.ir.DefModule]] are treated as
  * the same individual module. Thus, while certain instances may have dead code due to the
  * circumstances of their instantiation in their parent module, they will still not be removed. To
  * remove such modules, use the [[NoDedupAnnotation]] to prevent deduplication.
  */
class DeadCodeElimination extends Transform with ResolvedAnnotationPaths with RegisteredTransform {
  def inputForm = LowForm
  def outputForm = LowForm

  def addOptions(parser: OptionParser[AnnotationSeq]): Unit = parser
    .opt[Unit]("no-dce")
    .action( (x, c) => c :+ NoDCEAnnotation )
    .maxOccurs(1)
    .text("Do NOT run dead code elimination")

  /** Based on LogicNode ins CheckCombLoops, currently kind of faking it */
  private type LogicNode = MemoizedHash[WrappedExpression]
  private object LogicNode {
    def apply(moduleName: String, expr: Expression): LogicNode =
      WrappedExpression(Utils.mergeRef(WRef(moduleName), expr))
    def apply(moduleName: String, name: String): LogicNode = apply(moduleName, WRef(name))
    def apply(component: ComponentName): LogicNode = {
      // Currently only leaf nodes are supported TODO implement
      val loweredName = LowerTypes.loweredName(component.name.split('.'))
      apply(component.module.name, WRef(loweredName))
    }
    /** External Modules are representated as a single node driven by all inputs and driving all
      * outputs
      */
    def apply(ext: ExtModule): LogicNode = LogicNode(ext.name, ext.name)
  }

  /** Expression used to represent outputs in the circuit (# is illegal in names) */
  private val circuitSink = LogicNode("#Top", "#Sink")

  /** Extract all References and SubFields from a possibly nested Expression */
  def extractRefs(expr: Expression): Seq[Expression] = {
    val refs = mutable.ArrayBuffer.empty[Expression]
    def rec(e: Expression): Expression = {
      e match {
        case ref @ (_: WRef | _: WSubField) => refs += ref
        case nested @ (_: Mux | _: DoPrim | _: ValidIf) => nested map rec
        case ignore @ (_: Literal) => // Do nothing
        case unexpected => throwInternalError()
      }
      e
    }
    rec(expr)
    refs
  }

  // Gets all dependencies and constructs LogicNodes from them
  private def getDepsImpl(mname: String,
                          instMap: collection.Map[String, String])
                         (expr: Expression): Seq[LogicNode] =
    extractRefs(expr).map { e =>
      if (kind(e) == InstanceKind) {
        val (inst, tail) = Utils.splitRef(e)
        LogicNode(instMap(inst.name), tail)
      } else {
        LogicNode(mname, e)
      }
    }


  /** Construct the dependency graph within this module */
  private def setupDepGraph(depGraph: MutableDiGraph[LogicNode],
                            instMap: collection.Map[String, String])
                           (mod: Module): Unit = {
    def getDeps(expr: Expression): Seq[LogicNode] = getDepsImpl(mod.name, instMap)(expr)

    def onStmt(stmt: Statement): Unit = stmt match {
      case DefRegister(_, name, _, clock, reset, init) =>
        val node = LogicNode(mod.name, name)
        depGraph.addVertex(node)
        Seq(clock, reset, init).flatMap(getDeps(_)).foreach(ref => depGraph.addPairWithEdge(node, ref))
      case DefNode(_, name, value) =>
        val node = LogicNode(mod.name, name)
        depGraph.addVertex(node)
        getDeps(value).foreach(ref => depGraph.addPairWithEdge(node, ref))
      case DefWire(_, name, _) =>
        depGraph.addVertex(LogicNode(mod.name, name))
      case mem: DefMemory =>
        // Treat DefMems as a node with outputs depending on the node and node depending on inputs
        // From perpsective of the module or instance, MALE expressions are inputs, FEMALE are outputs
        val memRef = WRef(mem.name, MemPortUtils.memType(mem), ExpKind, FEMALE)
        val exprs = Utils.create_exps(memRef).groupBy(Utils.gender(_))
        val sources = exprs.getOrElse(MALE, List.empty).flatMap(getDeps(_))
        val sinks = exprs.getOrElse(FEMALE, List.empty).flatMap(getDeps(_))
        val memNode = getDeps(memRef) match { case Seq(node) => node }
        depGraph.addVertex(memNode)
        sinks.foreach(sink => depGraph.addPairWithEdge(sink, memNode))
        sources.foreach(source => depGraph.addPairWithEdge(memNode, source))
      case Attach(_, exprs) => // Add edge between each expression
        exprs.flatMap(getDeps(_)).toSet.subsets(2).map(_.toList).foreach {
          case Seq(a, b) =>
            depGraph.addPairWithEdge(a, b)
            depGraph.addPairWithEdge(b, a)
        }
      case Connect(_, loc, expr) =>
        // This match enforces the low Firrtl requirement of expanded connections
        val node = getDeps(loc) match { case Seq(elt) => elt }
        getDeps(expr).foreach(ref => depGraph.addPairWithEdge(node, ref))
      // Simulation constructs are treated as top-level outputs
      case Stop(_,_, clk, en) =>
        Seq(clk, en).flatMap(getDeps(_)).foreach(ref => depGraph.addPairWithEdge(circuitSink, ref))
      case Print(_, _, args, clk, en) =>
        (args :+ clk :+ en).flatMap(getDeps(_)).foreach(ref => depGraph.addPairWithEdge(circuitSink, ref))
      case Block(stmts) => stmts.foreach(onStmt(_))
      case ignore @ (_: IsInvalid | _: WDefInstance | EmptyStmt) => // do nothing
      case other => throw new Exception(s"Unexpected Statement $other")
    }

    // Add all ports as vertices
    mod.ports.foreach {
      case Port(_, name, _, _: GroundType) => depGraph.addVertex(LogicNode(mod.name, name))
      case other => throwInternalError()
    }
    onStmt(mod.body)
  }

  // TODO Make immutable?
  private def createDependencyGraph(instMaps: collection.Map[String, collection.Map[String, String]],
                                    doTouchExtMods: Set[String],
                                    c: Circuit): MutableDiGraph[LogicNode] = {
    val depGraph = new MutableDiGraph[LogicNode]
    c.modules.foreach {
      case mod: Module => setupDepGraph(depGraph, instMaps(mod.name))(mod)
      case ext: ExtModule =>
        // Connect all inputs to all outputs
        val node = LogicNode(ext)
        // Don't touch external modules *unless* they are specifically marked as doTouch
        // Simply marking the extmodule itself is sufficient to prevent inputs from being removed
        if (!doTouchExtMods.contains(ext.name)) depGraph.addPairWithEdge(circuitSink, node)
        ext.ports.foreach {
          case Port(_, pname, _, AnalogType(_)) =>
            depGraph.addPairWithEdge(LogicNode(ext.name, pname), node)
            depGraph.addPairWithEdge(node, LogicNode(ext.name, pname))
          case Port(_, pname, Output, _) =>
            val portNode = LogicNode(ext.name, pname)
            depGraph.addPairWithEdge(portNode, node)
            // Also mark all outputs as circuit sinks (unless marked doTouch obviously)
            if (!doTouchExtMods.contains(ext.name)) depGraph.addPairWithEdge(circuitSink, portNode)
          case Port(_, pname, Input, _) => depGraph.addPairWithEdge(node, LogicNode(ext.name, pname))
        }
    }
    // Connect circuitSink to ALL top-level ports (we don't want to change the top-level interface)
    val topModule = c.modules.find(_.name == c.main).get
    val topOutputs = topModule.ports.foreach { port =>
      depGraph.addPairWithEdge(circuitSink, LogicNode(c.main, port.name))
    }

    depGraph
  }

  private def deleteDeadCode(instMap: collection.Map[String, String],
                             deadNodes: collection.Set[LogicNode],
                             moduleMap: collection.Map[String, DefModule],
                             renames: RenameMap,
                             topName: String,
                             doTouchExtMods: Set[String])
                            (mod: DefModule): Option[DefModule] = {
    // For log-level debug
    def deleteMsg(decl: IsDeclaration): String = {
      val tpe = decl match {
        case _: DefNode => "node"
        case _: DefRegister => "reg"
        case _: DefWire => "wire"
        case _: Port => "port"
        case _: DefMemory => "mem"
        case (_: DefInstance | _: WDefInstance) => "inst"
        case _: Module => "module"
        case _: ExtModule => "extmodule"
      }
      val ref = decl match {
        case (_: Module | _: ExtModule) => mod.name
        case _ => s"${mod.name}.${decl.name}"
      }
      s"[DCE] $tpe $ref"
    }
    def getDeps(expr: Expression): Seq[LogicNode] = getDepsImpl(mod.name, instMap)(expr)

    var emptyBody = true
    renames.setModule(mod.name)

    def onStmt(stmt: Statement): Statement = {
      val stmtx = stmt match {
        case inst: WDefInstance =>
          moduleMap.get(inst.module) match {
            case Some(instMod) => inst.copy(tpe = Utils.module_type(instMod))
            case None =>
              logger.debug(deleteMsg(inst))
              renames.delete(inst.name)
              EmptyStmt
          }
        case decl: IsDeclaration =>
          val node = LogicNode(mod.name, decl.name)
          if (deadNodes.contains(node)) {
            logger.debug(deleteMsg(decl))
            renames.delete(decl.name)
            EmptyStmt
          }
          else decl
        case con: Connect =>
          val node = getDeps(con.loc) match { case Seq(elt) => elt }
          if (deadNodes.contains(node)) EmptyStmt else con
        case Attach(info, exprs) => // If any exprs are dead then all are
          val dead = exprs.flatMap(getDeps(_)).forall(deadNodes.contains(_))
          if (dead) EmptyStmt else Attach(info, exprs)
        case IsInvalid(info, expr) =>
          val node = getDeps(expr) match { case Seq(elt) => elt }
          if (deadNodes.contains(node)) EmptyStmt else IsInvalid(info, expr)
        case block: Block => block map onStmt
        case other => other
      }
      stmtx match { // Check if module empty
        case EmptyStmt | _: Block =>
        case other => emptyBody = false
      }
      stmtx
    }

    val (deadPorts, portsx) = mod.ports.partition(p => deadNodes.contains(LogicNode(mod.name, p.name)))
    deadPorts.foreach { p =>
      logger.debug(deleteMsg(p))
      renames.delete(p.name)
    }

    mod match {
      case Module(info, name, _, body) =>
        val bodyx = onStmt(body)
        // We don't delete the top module, even if it's empty
        if (emptyBody && portsx.isEmpty && name != topName) {
          logger.debug(deleteMsg(mod))
          None
        } else {
          Some(Module(info, name, portsx, bodyx))
        }
      case ext: ExtModule =>
        if (portsx.isEmpty && doTouchExtMods.contains(ext.name)) {
          logger.debug(deleteMsg(mod))
          None
        }
        else {
          if (ext.ports != portsx) throwInternalError() // Sanity check
          Some(ext.copy(ports = portsx))
        }
    }

  }

  def run(state: CircuitState,
          dontTouches: Seq[LogicNode],
          doTouchExtMods: Set[String]): CircuitState = {
    val c = state.circuit
    val moduleMap = c.modules.map(m => m.name -> m).toMap
    val iGraph = new InstanceGraph(c)
    val moduleDeps = iGraph.graph.getEdgeMap.map({ case (k,v) =>
      k.module -> v.map(i => i.name -> i.module).toMap
    })
    val topoSortedModules = iGraph.graph.transformNodes(_.module).linearize.reverse.map(moduleMap(_))

    val depGraph = {
      val dGraph = createDependencyGraph(moduleDeps, doTouchExtMods, c)

      val vertices = dGraph.getVertices
      dontTouches.foreach { dontTouch =>
        // Ensure that they are actually found
        if (vertices.contains(dontTouch)) {
          dGraph.addPairWithEdge(circuitSink, dontTouch)
        } else {
          val (root, tail) = Utils.splitRef(dontTouch.e1)
          DontTouchAnnotation.errorNotFound(root.serialize, tail.serialize)
        }
      }

      // Check for dont touches that are not found
      DiGraph(dGraph)
    }

    val liveNodes = depGraph.reachableFrom(circuitSink) + circuitSink
    val deadNodes = depGraph.getVertices -- liveNodes
    val renames = RenameMap()
    renames.setCircuit(c.main)

    // As we delete deadCode, we will delete ports from Modules and somtimes complete modules
    // themselves. We iterate over the modules in a topological order from leaves to the top. The
    // current status of the modulesxMap is used to either delete instances or update their types
    val modulesxMap = mutable.HashMap.empty[String, DefModule]
    topoSortedModules.foreach { case mod =>
      deleteDeadCode(moduleDeps(mod.name), deadNodes, modulesxMap, renames, c.main, doTouchExtMods)(mod) match {
        case Some(m) => modulesxMap += m.name -> m
        case None => renames.delete(ModuleName(mod.name, CircuitName(c.main)))
      }
    }

    // Preserve original module order
    val newCircuit = c.copy(modules = c.modules.flatMap(m => modulesxMap.get(m.name)))

    state.copy(circuit = newCircuit, renames = Some(renames))
  }

  override val annotationClasses: Traversable[Class[_]] =
    Seq(classOf[DontTouchAnnotation], classOf[OptimizableExtModuleAnnotation])

  def execute(state: CircuitState): CircuitState = {
    val dontTouches: Seq[LogicNode] = state.annotations.collect {
      case DontTouchAnnotation(component: ReferenceTarget) if component.isLocal => LogicNode(component)
    }
    val doTouchExtMods: Seq[String] = state.annotations.collect {
      case OptimizableExtModuleAnnotation(ModuleName(name, _)) => name
    }
    val noDCE = state.annotations.contains(NoDCEAnnotation)
    if (noDCE) {
      logger.info("Skipping DCE")
      state
    } else {
      run(state, dontTouches, doTouchExtMods.toSet)
    }
  }
}
