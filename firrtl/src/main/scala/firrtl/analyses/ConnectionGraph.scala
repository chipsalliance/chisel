// SPDX-License-Identifier: Apache-2.0

package firrtl.analyses

import firrtl.Mappers._
import firrtl.annotations.{TargetToken, _}
import firrtl.graph.{CyclicException, DiGraph, MutableDiGraph}
import firrtl.ir._
import firrtl.passes.MemPortUtils
import firrtl.{InstanceKind, PortKind, SinkFlow, SourceFlow, Utils, WInvalid}

import scala.collection.mutable

/** Class to represent circuit connection.
  *
  * @param circuit firrtl AST of this graph.
  * @param digraph Directed graph of ReferenceTarget in the AST.
  * @param irLookup [[IRLookup]] instance of circuit graph.
  */
class ConnectionGraph protected (val circuit: Circuit, val digraph: DiGraph[ReferenceTarget], val irLookup: IRLookup)
    extends DiGraph[ReferenceTarget](
      digraph.getEdgeMap.asInstanceOf[mutable.LinkedHashMap[ReferenceTarget, mutable.LinkedHashSet[ReferenceTarget]]]
    ) {

  lazy val serialize: String = s"""{
                                  |${getEdgeMap.map {
    case (k, vs) =>
      s"""  "$k": {
                                  |    "kind": "${irLookup.kind(k)}",
                                  |    "type": "${irLookup.tpe(k)}",
                                  |    "expr": "${irLookup.expr(k, irLookup.flow(k))}",
                                  |    "sinks": [${vs.map { v => s""""$v"""" }.mkString(", ")}],
                                  |    "declaration": "${irLookup.declaration(k)}"
                                  |  }""".stripMargin
  }.mkString(",\n")}
                                  |}""".stripMargin

  /** Used by BFS to map each visited node to the list of instance inputs visited thus far
    *
    * When BFS descends into a child instance, the child instance port is prepended to the list
    * When BFS ascends into a parent instance, the head of the list is removed
    * In essence, the list is a stack that you push when descending, and pop when ascending
    *
    * Because the search is BFS not DFS, we must record the state of the stack for each edge node, so
    * when that edge node is finally visited, we know the state of the stack
    *
    * For example:
    * circuit Top:
    *   module Top:
    *     input in: UInt
    *     output out: UInt
    *     inst a of A
    *       a.in <= in
    *       out <= a.out
    *   module A:
    *     input in: UInt
    *     output out: UInt
    *     inst b of B
    *       b.in <= in
    *     out <= b.out
    *   module B:
    *     input in: UInt
    *     output out: UInt
    *     out <= in
    *
    * We perform BFS starting at `Top>in`,
    * Node                [[portConnectivityStack]]
    *
    * Top>in              List()
    * Top>a.in            List()
    * Top/a:A>in          List(Top>a.in)
    * Top/a:A>b.in        List(Top>a.in)
    * Top/a:A/b:B/in      List(Top/a:A>b.in, Top>a.in)
    * Top/a:A/b:B/out     List(Top/a:A>b.in, Top>a.in)
    * Top/a:A>b.out       List(Top>a.in)
    * Top/a:A>out         List(Top>a.in)
    * Top>a.out           List()
    * Top>out             List()
    * when we reach `Top/a:A>`, `Top>a.in` will be pushed into [[portConnectivityStack]];
    * when we reach `Top/a:A/b:B>`, `Top/a:A>b.in` will be pushed into [[portConnectivityStack]];
    * when we leave `Top/a:A/b:B>`, `Top/a:A>b.in` will be popped from [[portConnectivityStack]];
    * when we leave `Top/a:A>`, `Top/a:A>b.in` will be popped from [[portConnectivityStack]].
    */
  private val portConnectivityStack: mutable.HashMap[ReferenceTarget, List[ReferenceTarget]] =
    mutable.HashMap.empty[ReferenceTarget, List[ReferenceTarget]]

  /** Records connectivities found while BFS is executing, from a module's source port to sink ports of a module
    *
    * All keys and values are local references.
    *
    * A BFS search will first query this map. If the query fails, then it continues and populates the map. If the query
    * succeeds, then the BFS shortcuts with the values provided by the query.
    *
    * Because this BFS implementation uses a priority queue which prioritizes exploring deeper instances first, a
    * successful query during BFS will only occur after all paths which leave the module from that reference have
    * already been searched.
    */
  private val bfsShortCuts: mutable.HashMap[ReferenceTarget, mutable.HashSet[ReferenceTarget]] =
    mutable.HashMap.empty[ReferenceTarget, mutable.HashSet[ReferenceTarget]]

  /** Records connectivities found after BFS is completed, from a module's source port to sink ports of a module
    *
    * All keys and values are local references.
    *
    * If its keys contain a reference, then the value will be complete, in that all paths from the reference out of
    * the module will have been explored
    *
    * For example, if Top>in connects to Top>out1 and Top>out2, then foundShortCuts(Top>in) will contain
    * Set(Top>out1, Top>out2), not Set(Top>out1) or Set(Top>out2)
    */
  private val foundShortCuts: mutable.HashMap[ReferenceTarget, mutable.HashSet[ReferenceTarget]] =
    mutable.HashMap.empty[ReferenceTarget, mutable.HashSet[ReferenceTarget]]

  /** Returns whether a previous BFS search has found a shortcut out of a module, starting from target
    *
    * @param target first target to find shortcut.
    * @return true if find a shortcut.
    */
  def hasShortCut(target: ReferenceTarget): Boolean = getShortCut(target).nonEmpty

  /** Optionally returns the shortcut a previous BFS search may have found out of a module, starting from target
    *
    * @param target first target to find shortcut.
    * @return [[firrtl.annotations.ReferenceTarget]] of short cut.
    */
  def getShortCut(target: ReferenceTarget): Option[Set[ReferenceTarget]] =
    foundShortCuts.get(target.pathlessTarget).map(set => set.map(_.setPathTarget(target.pathTarget)).toSet)

  /** Returns the shortcut a previous BFS search may have found out of a module, starting from target
    *
    * @param target first target to find shortcut.
    * @return [[firrtl.annotations.ReferenceTarget]] of short cut.
    */
  def shortCut(target: ReferenceTarget): Set[ReferenceTarget] = getShortCut(target).get

  /** @return a new, reversed connection graph where edges point from sinks to sources. */
  def reverseConnectionGraph: ConnectionGraph = new ConnectionGraph(circuit, digraph.reverse, irLookup)

  override def BFS(
    root:      ReferenceTarget,
    blacklist: collection.Set[ReferenceTarget]
  ): collection.Map[ReferenceTarget, ReferenceTarget] = {
    val prev = new mutable.LinkedHashMap[ReferenceTarget, ReferenceTarget]()
    val ordering = new Ordering[ReferenceTarget] {
      override def compare(x: ReferenceTarget, y: ReferenceTarget): Int = x.path.size - y.path.size
    }
    val bfsQueue = new mutable.PriorityQueue[ReferenceTarget]()(ordering)
    bfsQueue.enqueue(root)
    while (bfsQueue.nonEmpty) {
      val u = bfsQueue.dequeue()
      for (v <- getEdges(u)) {
        if (!prev.contains(v) && !blacklist.contains(v)) {
          prev(v) = u
          bfsQueue.enqueue(v)
        }
      }
    }

    foundShortCuts ++= bfsShortCuts
    bfsShortCuts.clear()
    portConnectivityStack.clear()

    prev
  }

  /** Linearizes (topologically sorts) a DAG
    *
    * @throws firrtl.graph.CyclicException if the graph is cyclic
    * @return a Seq[T] describing the topological order of the DAG
    *         traversal
    */
  override def linearize: Seq[ReferenceTarget] = {
    // permanently marked nodes are implicitly held in order
    val order = new mutable.ArrayBuffer[ReferenceTarget]
    // invariant: no intersection between unmarked and tempMarked
    val unmarked = new mutable.LinkedHashSet[ReferenceTarget]
    val tempMarked = new mutable.LinkedHashSet[ReferenceTarget]
    val finished = new mutable.LinkedHashSet[ReferenceTarget]

    case class LinearizeFrame[A](v: A, expanded: Boolean)
    val callStack = mutable.Stack[LinearizeFrame[ReferenceTarget]]()

    unmarked ++= getVertices
    while (unmarked.nonEmpty) {
      callStack.push(LinearizeFrame(unmarked.head, false))
      while (callStack.nonEmpty) {
        val LinearizeFrame(n, expanded) = callStack.pop()
        if (!expanded) {
          if (tempMarked.contains(n)) {
            throw new CyclicException(n)
          }
          if (unmarked.contains(n)) {
            tempMarked += n
            unmarked -= n
            callStack.push(LinearizeFrame(n, true))
            // We want to visit the first edge first (so push it last)
            for (m <- getEdges(n).toSeq.reverse) {
              if (!unmarked.contains(m) && !tempMarked.contains(m) && !finished.contains(m)) {
                unmarked += m
              }
              callStack.push(LinearizeFrame(m, false))
            }
          }
        } else {
          tempMarked -= n
          finished += n
          order.append(n)
        }
      }
    }

    // visited nodes are in post-traversal order, so must be reversed
    order.toSeq.reverse
  }

  override def getEdges(source: ReferenceTarget): collection.Set[ReferenceTarget] = {
    import ConnectionGraph._

    val localSource = source.pathlessTarget

    bfsShortCuts.get(localSource) match {
      case Some(set) => set.map { x => x.setPathTarget(source.pathTarget) }
      case None =>
        val pathlessEdges = super.getEdges(localSource)

        val ret = pathlessEdges.flatMap {

          case localSink if withinSameInstance(source)(localSink) =>
            portConnectivityStack(localSink) = portConnectivityStack.getOrElse(localSource, Nil)
            Set[ReferenceTarget](localSink.setPathTarget(source.pathTarget))

          case localSink if enteringParentInstance(source)(localSink) =>
            val currentStack = portConnectivityStack.getOrElse(localSource, Nil)
            if (currentStack.nonEmpty && currentStack.head.module == localSink.module) {
              // Exiting back to parent module
              // Update shortcut path from entrance from parent to new exit to parent
              val instancePort = currentStack.head
              val modulePort = ReferenceTarget(
                localSource.circuit,
                localSource.module,
                Nil,
                instancePort.component.head.value.toString,
                instancePort.component.tail
              )
              val destinations = bfsShortCuts.getOrElse(modulePort, mutable.HashSet.empty[ReferenceTarget])
              bfsShortCuts(modulePort) = destinations + localSource
              // Remove entrance from parent from stack
              portConnectivityStack(localSink) = currentStack.tail
            } else {
              // Exiting to parent, but had unresolved trip through child, so don't update shortcut
              portConnectivityStack(localSink) = localSource +: currentStack
            }
            Set[ReferenceTarget](
              localSink.setPathTarget(source.noComponents.targetParent.asInstanceOf[IsComponent].pathTarget)
            )

          case localSink if enteringChildInstance(source)(localSink) =>
            portConnectivityStack(localSink) = localSource +: portConnectivityStack.getOrElse(localSource, Nil)
            val x = localSink.setPathTarget(source.pathTarget.instOf(source.ref, localSink.module))
            Set[ReferenceTarget](x)

          case localSink if leavingRootInstance(source)(localSink) => Set[ReferenceTarget]()

          case localSink if enteringNonParentInstance(source)(localSink) => Set[ReferenceTarget]()

          case other => Utils.throwInternalError(s"BAD? $source -> $other")

        }
        ret
    }

  }

  override def path(
    start:     ReferenceTarget,
    end:       ReferenceTarget,
    blacklist: collection.Set[ReferenceTarget]
  ): Seq[ReferenceTarget] = {
    insertShortCuts(super.path(start, end, blacklist))
  }

  private def insertShortCuts(path: Seq[ReferenceTarget]): Seq[ReferenceTarget] = {
    val soFar = mutable.HashSet[ReferenceTarget]()
    if (path.size > 1) {
      path.head +: path
        .sliding(2)
        .flatMap {
          case Seq(from, to) =>
            getShortCut(from) match {
              case Some(set) if set.contains(to) && soFar.contains(from.pathlessTarget) =>
                soFar += from.pathlessTarget
                Seq(from.pathTarget.ref("..."), to)
              case _ =>
                soFar += from.pathlessTarget
                Seq(to)
            }
        }
        .toSeq
    } else path
  }

  /** Finds all paths starting at a particular node in a DAG
    *
    * WARNING: This is an exponential time algorithm (as any algorithm
    * must be for this problem), but is useful for flattening circuit
    * graph hierarchies. Each path is represented by a Seq[T] of nodes
    * in a traversable order.
    *
    * @param start the node to start at
    * @return a `Map[T,Seq[Seq[T]]]` where the value associated with v is the Seq of all paths from start to v
    */
  override def pathsInDAG(start: ReferenceTarget): mutable.LinkedHashMap[ReferenceTarget, Seq[Seq[ReferenceTarget]]] = {
    val linkedMap = super.pathsInDAG(start)
    linkedMap.keysIterator.foreach { key =>
      linkedMap(key) = linkedMap(key).map(insertShortCuts)
    }
    linkedMap
  }

  override def findSCCs: Seq[Seq[ReferenceTarget]] = Utils.throwInternalError("Cannot call findSCCs on ConnectionGraph")
}

object ConnectionGraph {

  /** Returns a [[firrtl.graph.DiGraph]] of [[firrtl.annotations.Target]] and corresponding [[IRLookup]]
    * Represents the directed connectivity of a FIRRTL circuit
    *
    * @param circuit firrtl AST of graph to be constructed.
    * @return [[ConnectionGraph]] of this `circuit`.
    */
  def apply(circuit: Circuit): ConnectionGraph = buildCircuitGraph(circuit)

  /** Within a module, given an [[firrtl.ir.Expression]] inside a module, return a corresponding [[firrtl.annotations.ReferenceTarget]]
    * @todo why no subaccess.
    *
    * @param m      Target of module containing the expression
    * @param e
    * @return
    */
  def asTarget(m: ModuleTarget, tagger: TokenTagger)(e: FirrtlNode): ReferenceTarget = e match {
    case l: Literal   => m.ref(tagger.getRef(l.value.toString))
    case r: Reference => m.ref(r.name)
    case s: SubIndex  => asTarget(m, tagger)(s.expr).index(s.value)
    case s: SubField  => asTarget(m, tagger)(s.expr).field(s.name)
    case d: DoPrim    => m.ref(tagger.getRef(d.op.serialize))
    case _: Mux       => m.ref(tagger.getRef("mux"))
    case _: ValidIf   => m.ref(tagger.getRef("validif"))
    case WInvalid => m.ref(tagger.getRef("invalid"))
    case _: Print => m.ref(tagger.getRef("print"))
    case _: Stop  => m.ref(tagger.getRef("print"))
    case other => sys.error(s"Unsupported: $other")
  }

  def withinSameInstance(source: ReferenceTarget)(localSink: ReferenceTarget): Boolean = {
    source.encapsulatingModule == localSink.encapsulatingModule
  }

  def enteringParentInstance(source: ReferenceTarget)(localSink: ReferenceTarget): Boolean = {
    def b1 = source.path.nonEmpty

    def b2 = source.noComponents.targetParent.asInstanceOf[InstanceTarget].encapsulatingModule == localSink.module

    def b3 = localSink.ref == source.path.last._1.value

    b1 && b2 && b3
  }

  def enteringNonParentInstance(source: ReferenceTarget)(localSink: ReferenceTarget): Boolean = {
    source.path.nonEmpty &&
    (source.noComponents.targetParent.asInstanceOf[InstanceTarget].encapsulatingModule != localSink.module ||
    localSink.ref != source.path.last._1.value)
  }

  def enteringChildInstance(source: ReferenceTarget)(localSink: ReferenceTarget): Boolean = source match {
    case ReferenceTarget(_, _, _, _, TargetToken.Field(port) +: comps)
        if port == localSink.ref && comps == localSink.component =>
      true
    case _ => false
  }

  def leavingRootInstance(source: ReferenceTarget)(localSink: ReferenceTarget): Boolean = source match {
    case ReferenceTarget(_, _, Seq(), port, comps)
        if port == localSink.component.head.value && comps == localSink.component.tail =>
      true
    case _ => false
  }

  private def buildCircuitGraph(circuit: Circuit): ConnectionGraph = {
    val mdg = new MutableDiGraph[ReferenceTarget]()
    val declarations = mutable.LinkedHashMap[ModuleTarget, mutable.LinkedHashMap[ReferenceTarget, FirrtlNode]]()
    val circuitTarget = CircuitTarget(circuit.main)
    val moduleMap = circuit.modules.map { m => circuitTarget.module(m.name) -> m }.toMap

    circuit.map(buildModule(circuitTarget))

    def addLabeledVertex(v: ReferenceTarget, f: FirrtlNode): Unit = {
      mdg.addVertex(v)
      declarations.getOrElseUpdate(v.moduleTarget, mutable.LinkedHashMap.empty[ReferenceTarget, FirrtlNode])(v) = f
    }

    def buildModule(c: CircuitTarget)(module: DefModule): DefModule = {
      val m = c.module(module.name)
      module.map(buildPort(m)).map(buildStatement(m, new TokenTagger()))
    }

    def buildPort(m: ModuleTarget)(port: Port): Port = {
      val p = m.ref(port.name)
      addLabeledVertex(p, port)
      port
    }

    def buildInstance(m: ModuleTarget, tagger: TokenTagger, name: String, ofModule: String, tpe: Type): Unit = {
      val instPorts = Utils.create_exps(Reference(name, tpe, InstanceKind, SinkFlow))
      val modulePorts = tpe.asInstanceOf[BundleType].fields.flatMap {
        // Module output
        case firrtl.ir.Field(name, Default, tpe) => Utils.create_exps(Reference(name, tpe, PortKind, SourceFlow))
        // Module input
        case firrtl.ir.Field(name, Flip, tpe) => Utils.create_exps(Reference(name, tpe, PortKind, SinkFlow))
        case x                                => Utils.error(s"Unexpected flip: ${x.flip}")
      }
      assert(instPorts.size == modulePorts.size)
      val o = m.circuitTarget.module(ofModule)
      instPorts.zip(modulePorts).foreach { x =>
        val (instExp, modExp) = x
        val it = asTarget(m, tagger)(instExp)
        val mt = asTarget(o, tagger)(modExp)
        (Utils.flow(instExp), Utils.flow(modExp)) match {
          case (SourceFlow, SinkFlow) => mdg.addPairWithEdge(it, mt)
          case (SinkFlow, SourceFlow) => mdg.addPairWithEdge(mt, it)
          case _                      => sys.error("Something went wrong...")
        }
      }
    }

    def buildMemory(mt: ModuleTarget, d: DefMemory): Unit = {
      val readers = d.readers.toSet
      val readwriters = d.readwriters.toSet
      val mem = mt.ref(d.name)
      MemPortUtils.memType(d).fields.foreach {
        case Field(name, _, _: BundleType) if readers.contains(name) || readwriters.contains(name) =>
          val port = mem.field(name)
          val sources = Seq(
            port.field("clk"),
            port.field("en"),
            port.field("addr")
          ) ++ (if (readwriters.contains(name)) Seq(port.field("wmode")) else Nil)

          val data = if (readers.contains(name)) port.field("data") else port.field("rdata")
          val sinks = data.leafSubTargets(d.dataType)

          sources.foreach {
            mdg.addVertex
          }
          sinks.foreach { sink =>
            mdg.addVertex(sink)
            sources.foreach { source => mdg.addEdge(source, sink) }
          }
        case _ =>
      }
    }

    def buildRegister(m: ModuleTarget, tagger: TokenTagger, d: DefRegister): Unit = {
      val regTarget = m.ref(d.name)
      val clockTarget = regTarget.clock
      val resetTarget = regTarget.reset
      val initTarget = regTarget.init

      // Build clock expression
      mdg.addVertex(clockTarget)
      buildExpression(m, tagger, clockTarget)(d.clock)

      // Build reset expression
      mdg.addVertex(resetTarget)
      buildExpression(m, tagger, resetTarget)(d.reset)

      // Connect each subTarget to the corresponding init subTarget
      val allRegTargets = regTarget.leafSubTargets(d.tpe)
      val allInitTargets = initTarget.leafSubTargets(d.tpe).zip(Utils.create_exps(d.init))
      allRegTargets.zip(allInitTargets).foreach {
        case (r, (i, e)) =>
          mdg.addVertex(i)
          mdg.addVertex(r)
          mdg.addEdge(clockTarget, r)
          mdg.addEdge(resetTarget, r)
          mdg.addEdge(i, r)
          buildExpression(m, tagger, i)(e)
      }
    }

    def buildStatement(m: ModuleTarget, tagger: TokenTagger)(stmt: Statement): Statement = {
      stmt match {
        case d: DefWire =>
          addLabeledVertex(m.ref(d.name), stmt)

        case d: DefNode =>
          val sinkTarget = m.ref(d.name)
          addLabeledVertex(sinkTarget, stmt)
          val nodeTargets = sinkTarget.leafSubTargets(d.value.tpe)
          nodeTargets.zip(Utils.create_exps(d.value)).foreach {
            case (n, e) =>
              mdg.addVertex(n)
              buildExpression(m, tagger, n)(e)
          }

        case c: Connect =>
          val sinkTarget = asTarget(m, tagger)(c.loc)
          mdg.addVertex(sinkTarget)
          buildExpression(m, tagger, sinkTarget)(c.expr)

        case i: IsInvalid =>
          val sourceTarget = asTarget(m, tagger)(WInvalid)
          addLabeledVertex(sourceTarget, stmt)
          mdg.addVertex(sourceTarget)
          val sinkTarget = asTarget(m, tagger)(i.expr)
          sinkTarget.allSubTargets(i.expr.tpe).foreach { st =>
            mdg.addVertex(st)
            mdg.addEdge(sourceTarget, st)
          }

        case DefInstance(_, name, ofModule, tpe) =>
          addLabeledVertex(m.ref(name), stmt)
          buildInstance(m, tagger, name, ofModule, tpe)

        case d: DefRegister =>
          addLabeledVertex(m.ref(d.name), d)
          buildRegister(m, tagger, d)

        case d: DefMemory =>
          addLabeledVertex(m.ref(d.name), d)
          buildMemory(m, d)

        /** @todo [[firrtl.Transform.prerequisites]] ++ [[firrtl.passes.ExpandWhensAndCheck]] */
        case _: Conditionally => sys.error("Unsupported! Only works on Middle Firrtl")

        case s: Block => s.map(buildStatement(m, tagger))

        case a: Attach =>
          val attachTargets = a.exprs.map { r =>
            val at = asTarget(m, tagger)(r)
            mdg.addVertex(at)
            at
          }
          attachTargets.combinations(2).foreach {
            case Seq(l, r) =>
              mdg.addEdge(l, r)
              mdg.addEdge(r, l)
          }
        case p: Print => addLabeledVertex(asTarget(m, tagger)(p), p)
        case s: Stop  => addLabeledVertex(asTarget(m, tagger)(s), s)
        case EmptyStmt =>
      }
      stmt
    }

    def buildExpression(
      m:          ModuleTarget,
      tagger:     TokenTagger,
      sinkTarget: ReferenceTarget
    )(expr:       Expression
    ): Expression = {

      /** @todo [[firrtl.Transform.prerequisites]] ++ [[firrtl.stage.Forms.Resolved]]. */
      val sourceTarget = asTarget(m, tagger)(expr)
      mdg.addVertex(sourceTarget)
      mdg.addEdge(sourceTarget, sinkTarget)
      expr match {
        case _: DoPrim | _: Mux | _: ValidIf | _: Literal =>
          addLabeledVertex(sourceTarget, expr)
          expr.map(buildExpression(m, tagger, sourceTarget))
        case _ =>
      }
      expr
    }

    new ConnectionGraph(circuit, DiGraph(mdg), new IRLookup(declarations.mapValues(_.toMap).toMap, moduleMap))
  }
}

/** Used for obtaining a tag for a given label unnamed Target. */
class TokenTagger {
  private val counterMap = mutable.HashMap[String, Int]()

  def getTag(label: String): Int = {
    val tag = counterMap.getOrElse(label, 0)
    counterMap(label) = tag + 1
    tag
  }

  def getRef(label: String): String = {
    "@" + label + "#" + getTag(label)
  }
}

object TokenTagger {
  val literalRegex = "@([-]?[0-9]+)#[0-9]+".r
}
