package firrtl.transforms

import firrtl._
import firrtl.Mappers._
import firrtl.ir._
import firrtl.annotations.{Annotation, ComponentName}
import firrtl.passes.{InferTypes, LowerTypes, MemPortUtils}
import firrtl.Utils.kind
import firrtl.graph.{DiGraph, MutableDiGraph}

import scala.collection.mutable


/**
  * Specifies a group of components, within a module, to pull out into their own module
  * Components that are only connected to a group's components will also be included
  *
  * @param components components in this group
  * @param newModule suggested name of the new module
  * @param newInstance suggested name of the instance of the new module
  * @param outputSuffix suggested suffix of any output ports of the new module
  * @param inputSuffix suggested suffix of any input ports of the new module
  */
case class GroupAnnotation(components: Seq[ComponentName], newModule: String, newInstance: String, outputSuffix: Option[String] = None, inputSuffix: Option[String] = None) extends Annotation {
  if(components.nonEmpty) {
    require(components.forall(_.module == components.head.module), "All components must be in the same module.")
    require(components.forall(!_.name.contains('.')), "No components can be a subcomponent.")
  }

  /**
    * The module that all components are located in
    * @return
    */
  def currentModule: String = components.head.module.name

  /* Only keeps components renamed to components */
  def update(renames: RenameMap): Seq[Annotation] = {
    val newComponents = components.flatMap{c => renames.get(c).getOrElse(Seq(c))}.collect {
      case c: ComponentName => c
    }
    Seq(GroupAnnotation(newComponents, newModule, newInstance, outputSuffix, inputSuffix))
  }
}

/**
  * Splits a module into multiple modules by grouping its components via [[GroupAnnotation]]'s
  */
class GroupComponents extends firrtl.Transform {
  type MSet[T] = mutable.Set[T]

  def inputForm: CircuitForm = MidForm
  def outputForm: CircuitForm = MidForm

  override def execute(state: CircuitState): CircuitState = {
    val groups = state.annotations.collect {case g: GroupAnnotation => g}
    val module2group = groups.groupBy(_.currentModule)
    val mnamespace = Namespace(state.circuit)
    val newModules = state.circuit.modules.flatMap {
      case m: Module if module2group.contains(m.name) =>
        // do stuff
        groupModule(m, module2group(m.name).filter(_.components.nonEmpty), mnamespace)
      case other => Seq(other)
    }
    val cs = state.copy(circuit = state.circuit.copy(modules = newModules))
    val csx = InferTypes.execute(cs)
    csx
  }

  def groupModule(m: Module, groups: Seq[GroupAnnotation], mnamespace: Namespace): Seq[Module] = {
    val namespace = Namespace(m)
    val groupRoots = groups.map(_.components.map(_.name))
    val totalSum = groupRoots.map(_.size).sum
    val union = groupRoots.foldLeft(Set.empty[String]){(all, set) => all.union(set.toSet)}

    require(groupRoots.forall{_.forall{namespace.contains}}, "All names should be in this module")
    require(totalSum == union.size, "No name can be in more than one group")
    require(groupRoots.forall(_.nonEmpty), "All groupRoots must by non-empty")


    // Order of groups, according to their label. The label is the first root in the group
    val labelOrder = groups.collect({ case g: GroupAnnotation => g.components.head.name })

    // Annotations, by label
    val label2annotation = groups.collect({ case g: GroupAnnotation => g.components.head.name -> g }).toMap

    // Group roots, by label
    // The label "" indicates the original module, and components belonging to that group will remain
    //   in the original module (not get moved into a new module)
    val label2group: Map[String, MSet[String]] = groups.collect{
      case GroupAnnotation(set, module, instance, _, _) => set.head.name -> mutable.Set(set.map(_.name):_*)
    }.toMap + ("" -> mutable.Set(""))

    // Name of new module containing each group, by label
    val label2module: Map[String, String] =
      groups.map(a => a.components.head.name -> mnamespace.newName(a.newModule)).toMap

    // Name of instance of new module, by label
    val label2instance: Map[String, String] =
      groups.map(a => a.components.head.name -> namespace.newName(a.newInstance)).toMap

    // Build set of components not in set
    val notSet = label2group.map { case (key, value) => key -> union.diff(value) }


    // Get all dependencies between components
    val deps = getComponentConnectivity(m)

    // For each node not in the set, which group (by label) can reach it
    val reachableNodes = new mutable.HashMap[String, MSet[String]]()

    // For each group (by label), add connectivity between nodes in set
    // Populate reachableNodes with reachability, where blacklist is their notSet
    label2group.foreach { case (label, set) =>
      set.foreach { x =>
        deps.addPairWithEdge(label, x)
      }
      deps.reachableFrom(label, notSet(label)) foreach { node =>
        reachableNodes.getOrElseUpdate(node, mutable.Set.empty[String]) += label
      }
    }

    // Add nodes who are reached by a single group, to that group
    reachableNodes.foreach { case (node, membership) =>
      if(membership.size == 1) {
        label2group(membership.head) += node
      } else {
        label2group("") += node
      }
    }

    applyGrouping(m, labelOrder, label2group, label2module, label2instance, label2annotation)
  }

  /**
    * Applies datastructures to a module, to group its components into distinct modules
    * @param m module to split apart
    * @param labelOrder order of groups in SeqAnnotation, to make the grouping more deterministic
    * @param label2group group components, by label
    * @param label2module module name, by label
    * @param label2instance instance name of the group's module, by label
    * @param label2annotation annotation specifying the group, by label
    * @return new modules, including each group's module and the new split module
    */
  def applyGrouping( m: Module,
                     labelOrder: Seq[String],
                     label2group: Map[String, MSet[String]],
                     label2module: Map[String, String],
                     label2instance: Map[String, String],
                     label2annotation: Map[String, GroupAnnotation]
                   ): Seq[Module] = {
    // Maps node to group
    val byNode = mutable.HashMap[String, String]()
    label2group.foreach { case (group, nodes) =>
      nodes.foreach { node =>
        byNode(node) = group
      }
    }
    val groupNamespace = label2group.map { case (head, set) => head -> Namespace(set.toSeq) }

    val groupStatements = mutable.HashMap[String, mutable.ArrayBuffer[Statement]]()
    val groupPorts = mutable.HashMap[String, mutable.ArrayBuffer[Port]]()
    val groupPortNames = mutable.HashMap[String, mutable.HashMap[String, String]]()
    label2group.keys.foreach { group =>
      groupStatements(group) = new mutable.ArrayBuffer[Statement]()
      groupPorts(group) = new mutable.ArrayBuffer[Port]()
      groupPortNames(group) = new mutable.HashMap[String, String]()
    }

    def addPort(group: String, exp: Expression, d: Direction): String = {
      val source = LowerTypes.loweredName(exp)
      val portNames = groupPortNames(group)
      val suffix = d match {
        case Output => label2annotation(group).outputSuffix.getOrElse("")
        case Input => label2annotation(group).inputSuffix.getOrElse("")
      }
      val newName = groupNamespace(group).newName(source + suffix)
      val portName = portNames.getOrElseUpdate(source, newName)
      groupPorts(group) += Port(NoInfo, portName, d, exp.tpe)
      portName
    }

    def punchSignalOut(group: String, exp: Expression): String = {
      val portName = addPort(group, exp, Output)
      val connectStatement = exp.tpe match {
        case AnalogType(_) => Attach(NoInfo, Seq(WRef(portName), exp))
        case _ => Connect(NoInfo, WRef(portName), exp)
      }
      groupStatements(group) += connectStatement
      portName
    }

    // Given the sink is in a group, tidy up source references
    def inGroupFixExps(group: String, added: mutable.ArrayBuffer[Statement])(e: Expression): Expression = e match {
      case _: Literal => e
      case _: DoPrim | _: Mux | _: ValidIf => e map inGroupFixExps(group, added)
      case otherExp: Expression =>
        val wref = getWRef(otherExp)
        val source = wref.name
        byNode(source) match {
          // case 1: source in the same group as sink
          case `group` => otherExp //do nothing

          // case 2: source in top
          case "" =>
            // Add port to group's Module
            val toPort = addPort(group, otherExp, Input)

            // Add connection in Top to group's Module port
            added += Connect(NoInfo, WSubField(WRef(label2instance(group)), toPort), otherExp)

            // Return WRef with new kind (its inside the group Module now)
            WRef(toPort, otherExp.tpe, PortKind, MALE)

          // case 3: source in different group
          case otherGroup =>
            // Add port to otherGroup's Module
            val fromPort = punchSignalOut(otherGroup, otherExp)
            val toPort = addPort(group, otherExp, Input)

            // Add connection in Top from otherGroup's port to group's port
            val groupInst = label2instance(group)
            val otherInst = label2instance(otherGroup)
            added += Connect(NoInfo, WSubField(WRef(groupInst), toPort), WSubField(WRef(otherInst), fromPort))

            // Return WRef with new kind (its inside the group Module now)
            WRef(toPort, otherExp.tpe, PortKind, MALE)
        }
    }

    // Given the sink is in the parent module, tidy up source references belonging to groups
    def inTopFixExps(e: Expression): Expression = e match {
      case _: DoPrim | _: Mux | _: ValidIf => e map inTopFixExps
      case otherExp: Expression =>
        val wref = getWRef(otherExp)
        if(byNode(wref.name) != "") {
          // Get the name of source's group
          val otherGroup = byNode(wref.name)

          // Add port to otherGroup's Module
          val otherPortName = punchSignalOut(otherGroup, otherExp)

          // Return WSubField (its inside the top Module still)
          WSubField(WRef(label2instance(otherGroup)), otherPortName)

        } else otherExp
    }

    def onStmt(s: Statement): Statement = {
      s match {
        // Sink is in a group
        case r: IsDeclaration if byNode(r.name) != "" =>
          val topStmts = mutable.ArrayBuffer[Statement]()
          val group = byNode(r.name)
          groupStatements(group) += r mapExpr inGroupFixExps(group, topStmts)
          Block(topStmts)
        case c: Connect if byNode(getWRef(c.loc).name) != "" =>
          // Sink is in a group
          val topStmts = mutable.ArrayBuffer[Statement]()
          val group = byNode(getWRef(c.loc).name)
          groupStatements(group) += Connect(c.info, c.loc, inGroupFixExps(group, topStmts)(c.expr))
          Block(topStmts)
        // TODO Attach if all are in a group?
        case _: IsDeclaration | _: Connect | _: Attach =>
          // Sink is in Top
          val ret = s mapExpr inTopFixExps
          ret
        case other => other map onStmt
      }
    }


    // Build datastructures
    val newTopBody = Block(labelOrder.map(g => WDefInstance(NoInfo, label2instance(g), label2module(g), UnknownType)) ++ Seq(onStmt(m.body)))
    val finalTopBody = Block(Utils.squashEmpty(newTopBody).asInstanceOf[Block].stmts.distinct)

    // For all group labels (not including the original module label), return a new Module.
    val newModules = labelOrder.filter(_ != "") map { group =>
      Module(NoInfo, label2module(group), groupPorts(group).distinct, Block(groupStatements(group).distinct))
    }
    Seq(m.copy(body = finalTopBody)) ++ newModules
  }

  def getWRef(e: Expression): WRef = e match {
    case w: WRef => w
    case other =>
      var w = WRef("")
      other mapExpr { e => w = getWRef(e); e}
      w
  }

  /**
    * Compute how each component connects to each other component
    * It is non-directioned; there is an edge from source to sink and from sink to souce
    * @param m module to compute connectivity
    * @return a bi-directional representation of component connectivity
    */
  def getComponentConnectivity(m: Module): MutableDiGraph[String] = {
    val bidirGraph = new MutableDiGraph[String]
    val simNamespace = Namespace()
    val simulations = new mutable.HashMap[String, Statement]
    def onExpr(sink: WRef)(e: Expression): Expression = e match {
      case w @ WRef(name, _, _, _) =>
        bidirGraph.addPairWithEdge(sink.name, name)
        bidirGraph.addPairWithEdge(name, sink.name)
        w
      case other => other map onExpr(sink)
    }
    def onStmt(stmt: Statement): Unit = stmt match {
      case w: WDefInstance =>
      case h: IsDeclaration => h map onExpr(WRef(h.name))
      case Attach(_, exprs) => // Add edge between each expression
        exprs.tail map onExpr(getWRef(exprs.head))
      case Connect(_, loc, expr) =>
        onExpr(getWRef(loc))(expr)
      case q @ Stop(_,_, clk, en) =>
        val simName = simNamespace.newTemp
        simulations(simName) = q
        Seq(clk, en) map onExpr(WRef(simName))
      case q @ Print(_, _, args, clk, en) =>
        val simName = simNamespace.newTemp
        simulations(simName) = q
        (args :+ clk :+ en) map onExpr(WRef(simName))
      case Block(stmts) => stmts.foreach(onStmt)
      case ignore @ (_: IsInvalid | EmptyStmt) => // do nothing
      case other => throw new Exception(s"Unexpected Statement $other")
    }

    onStmt(m.body)
    m.ports.foreach { p =>
      bidirGraph.addPairWithEdge("", p.name)
      bidirGraph.addPairWithEdge(p.name, "")
    }
    bidirGraph
  }
}

/**
  * Splits a module into multiple modules by grouping its components via [[GroupAnnotation]]'s
  * Tries to deduplicate the resulting circuit
  */
class GroupAndDedup extends Transform {
  def inputForm: CircuitForm = MidForm
  def outputForm: CircuitForm = MidForm

  override def execute(state: CircuitState): CircuitState = {
    val cs = new GroupComponents().execute(state)
    val csx = new DedupModules().execute(cs)
    csx
  }
}
