// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._
import firrtl.annotations.TargetToken.{Instance, OfModule}
import firrtl.analyses.InstanceKeyGraph
import firrtl.graph.{DiGraph, MutableDiGraph}
import firrtl.stage.{Forms, RunFirrtlTransformAnnotation}
import firrtl.options.{RegisteredTransform, ShellOption}
import firrtl.renamemap.MutableRenameMap

// Datastructures
import scala.collection.mutable

/** Indicates that something should be inlined */
case class InlineAnnotation(target: Named) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named) = InlineAnnotation(n)
}

object InlineInstances {

  /** Enumerates all possible names for a given type. For example:
    * {{{
    * foo : { bar : { a, b }[2], c }
    *   => foo, foo bar, foo bar 0, foo bar 1, foo bar 0 a, foo bar 0 b, foo bar 1 a, foo bar 1 b, foo c
    * }}}
    */
  private def enumerateNames(tpe: Type): Seq[Seq[String]] = tpe match {
    case t: BundleType =>
      t.fields.flatMap { f =>
        (enumerateNames(f.tpe).map(f.name +: _)) ++ Seq(Seq(f.name))
      }
    case t: VectorType =>
      ((0 until t.size).map(i => Seq(i.toString))) ++
        ((0 until t.size).flatMap { i =>
          enumerateNames(t.tpe).map(i.toString +: _)
        })
    case _ => Seq()
  }

}

/** Inline instances as indicated by existing [[InlineAnnotation]]s
  * @note Only use on legal Firrtl. Specifically, the restriction of instance loops must have been checked, or else this
  * pass can infinitely recurse.
  */
class InlineInstances extends Transform with DependencyAPIMigration with RegisteredTransform {
  import InlineInstances._

  override def prerequisites = Forms.LowForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.LowEmitters

  override def invalidates(a: Transform): Boolean = a == ResolveKinds

  private[firrtl] val inlineDelim: String = "_"

  val options = Seq(
    new ShellOption[Seq[String]](
      longOption = "inline",
      toAnnotationSeq = (a: Seq[String]) =>
        a.map { value =>
          value.split('.') match {
            case Array(circuit) =>
              InlineAnnotation(CircuitName(circuit))
            case Array(circuit, module) =>
              InlineAnnotation(ModuleName(module, CircuitName(circuit)))
            case Array(circuit, module, inst) =>
              InlineAnnotation(ComponentName(inst, ModuleName(module, CircuitName(circuit))))
          }
        } :+ RunFirrtlTransformAnnotation(new InlineInstances),
      helpText = "Inline selected modules",
      shortOption = Some("fil"),
      helpValueName = Some("<circuit>[.<module>[.<instance>]][,...]")
    )
  )

  private def collectAnns(circuit: Circuit, anns: Iterable[Annotation]): (Set[ModuleName], Set[ComponentName]) =
    anns.foldLeft((Set.empty[ModuleName], Set.empty[ComponentName])) {
      case ((modNames, instNames), ann) =>
        ann match {
          case InlineAnnotation(CircuitName(c)) =>
            (
              circuit.modules.collect {
                case Module(_, name, _, _) if name != circuit.main => ModuleName(name, CircuitName(c))
              }.toSet,
              instNames
            )
          case InlineAnnotation(ModuleName(mod, cir))    => (modNames + ModuleName(mod, cir), instNames)
          case InlineAnnotation(ComponentName(com, mod)) => (modNames, instNames + ComponentName(com, mod))
          case _                                         => (modNames, instNames)
        }
    }

  def execute(state: CircuitState): CircuitState = {
    // TODO Add error check for more than one annotation for inlining
    val (modNames, instNames) = collectAnns(state.circuit, state.annotations)
    if (modNames.nonEmpty || instNames.nonEmpty) {
      run(state.circuit, modNames, instNames, state.annotations)
    } else {
      state
    }
  }

  // Checks the following properties:
  // 1) All annotated modules exist
  // 2) All annotated modules are InModules (can be inlined)
  // 3) All annotated instances exist, and their modules can be inline
  def check(c: Circuit, moduleNames: Set[ModuleName], instanceNames: Set[ComponentName]): Unit = {
    val errors = mutable.ArrayBuffer[PassException]()
    val moduleMap = InstanceKeyGraph(c).moduleMap
    def checkExists(name: String): Unit =
      if (!moduleMap.contains(name))
        errors += new PassException(s"Annotated module does not exist: $name")
    def checkExternal(name: String): Unit = moduleMap(name) match {
      case m: ExtModule => errors += new PassException(s"Annotated module cannot be an external module: $name")
      case _ =>
    }
    def checkInstance(cn: ComponentName): Unit = {
      var containsCN = false
      def onStmt(name: String)(s: Statement): Statement = {
        s match {
          case WDefInstance(_, inst_name, module_name, tpe) =>
            if (name == inst_name) {
              containsCN = true
              checkExternal(module_name)
            }
          case _ =>
        }
        s.map(onStmt(name))
      }
      onStmt(cn.name)(moduleMap(cn.module.name).asInstanceOf[Module].body)
      if (!containsCN) errors += new PassException(s"Annotated instance does not exist: ${cn.module.name}.${cn.name}")
    }

    moduleNames.foreach { mn => checkExists(mn.name) }
    if (errors.nonEmpty) throw new PassExceptions(errors.toSeq)
    moduleNames.foreach { mn => checkExternal(mn.name) }
    if (errors.nonEmpty) throw new PassExceptions(errors.toSeq)
    instanceNames.foreach { cn => checkInstance(cn) }
    if (errors.nonEmpty) throw new PassExceptions(errors.toSeq)
  }

  def run(
    c:             Circuit,
    modsToInline:  Set[ModuleName],
    instsToInline: Set[ComponentName],
    annos:         AnnotationSeq
  ): CircuitState = {
    def getInstancesOf(c: Circuit, modules: Set[String]): Set[(OfModule, Instance)] =
      c.modules.foldLeft(Set[(OfModule, Instance)]()) { (set, d) =>
        d match {
          case e: ExtModule => set
          case m: Module =>
            val instances = mutable.HashSet[(OfModule, Instance)]()
            def findInstances(s: Statement): Statement = s match {
              case WDefInstance(info, instName, moduleName, instTpe) if modules.contains(moduleName) =>
                instances += (OfModule(m.name) -> Instance(instName))
                s
              case sx => sx.map(findInstances)
            }
            findInstances(m.body)
            instances.toSet ++ set
        }
      }

    // Check annotations and circuit match up
    check(c, modsToInline, instsToInline)
    val flatModules = modsToInline.map(m => m.name)
    val flatInstances: Set[(OfModule, Instance)] =
      instsToInline.map(i => OfModule(i.module.name) -> Instance(i.name)) ++ getInstancesOf(c, flatModules)
    val iGraph = InstanceKeyGraph(c)
    val namespaceMap = collection.mutable.Map[String, Namespace]()
    // Map of Module name to Map of instance name to Module name
    val instMaps = iGraph.getChildInstanceMap

    /** Add a prefix to all declarations updating a [[Namespace]] and appending to a [[RenameMap]] */
    def appendNamePrefix(
      currentModule: InstanceTarget,
      nextModule:    IsModule,
      prefix:        String,
      ns:            Namespace,
      renames:       mutable.HashMap[String, String],
      renameMap:     MutableRenameMap
    )(s:             Statement
    ): Statement = {
      def onName(ofModuleOpt: Option[String])(name: String): String = {
        // Empty names are allowed for backwards compatibility reasons and
        // indicate that the entity has essentially no name and thus cannot be prefixed.
        if (name.isEmpty) { name }
        else {
          if (prefix.nonEmpty && !ns.tryName(prefix + name)) {
            throw new Exception(s"Inlining failed. Inlined name '${prefix + name}' already exists")
          }
          ofModuleOpt match {
            case None =>
              renameMap.record(currentModule.ofModuleTarget.ref(name), nextModule.ref(prefix + name))
              renameMap.record(currentModule.ref(name), nextModule.ref(prefix + name))
            case Some(ofModule) =>
              renameMap.record(
                currentModule.ofModuleTarget.instOf(name, ofModule),
                nextModule.instOf(prefix + name, ofModule)
              )
              renameMap.record(currentModule.instOf(name, ofModule), nextModule.instOf(prefix + name, ofModule))
          }
          renames(name) = prefix + name
          prefix + name
        }
      }

      s match {
        case s: WDefInstance =>
          s.map(onName(Some(s.module))).map(appendNamePrefix(currentModule, nextModule, prefix, ns, renames, renameMap))
        case other =>
          s.map(onName(None)).map(appendNamePrefix(currentModule, nextModule, prefix, ns, renames, renameMap))
      }
    }

    /** Modify all references */
    def appendRefPrefix(
      currentModule: IsModule,
      renames:       mutable.HashMap[String, String]
    )(s:             Statement
    ): Statement = {
      def onExpr(e: Expression): Expression = e match {
        case wr @ WRef(name, _, _, _) =>
          renames.get(name) match {
            case Some(prefixedName) => wr.copy(name = prefixedName)
            case None               => wr
          }
        case ex => ex.map(onExpr)
      }
      s.map(onExpr).map(appendRefPrefix(currentModule, renames))
    }

    val cache = mutable.HashMap.empty[ModuleTarget, Statement]

    /** renamesMap is a map of instances to [[RenameMap]].
      *  The keys are pairs of enclosing [[OfModule]] and [[Instance]]
      *  The [[RenameMap]]s in renamesMap are appear in renamesSeq
      *  in the order that they should be applied
      */
    val (renamesMap, renamesSeq) = {
      val mutableDiGraph = new MutableDiGraph[(OfModule, Instance)]
      // compute instance graph
      instMaps.foreach {
        case (grandParentOfMod, parents) =>
          parents.foreach {
            case (parentInst, parentOfMod) =>
              val from = grandParentOfMod -> parentInst
              mutableDiGraph.addVertex(from)
              instMaps(parentOfMod).foreach {
                case (childInst, _) =>
                  val to = parentOfMod -> childInst
                  mutableDiGraph.addVertex(to)
                  mutableDiGraph.addEdge(from, to)
              }
          }
      }

      val diGraph = DiGraph(mutableDiGraph)
      val subgraph = diGraph.simplify(flatInstances)
      val edges = subgraph.getEdgeMap

      // calculate which [[RenameMap]] should be associated with each instance
      val indexMap = new mutable.HashMap[(OfModule, Instance), Int]
      flatInstances.foreach(v => indexMap(v) = 0)
      subgraph.linearize.foreach { parent =>
        edges(parent).foreach { child =>
          indexMap(child) = indexMap(parent) + 1
        }
      }

      indexMap match {
        case a if a.isEmpty =>
          (Map.empty[(OfModule, Instance), MutableRenameMap], Seq.empty[MutableRenameMap])
        case a =>
          val maxIdx = indexMap.values.max
          val resultSeq = Seq.fill(maxIdx + 1)(MutableRenameMap())
          val resultMap = indexMap.mapValues(idx => resultSeq(maxIdx - idx))
          (resultMap, resultSeq)
      }
    }

    def fixupRefs(
      instMap:       collection.Map[Instance, OfModule],
      currentModule: IsModule
    )(e:             Expression
    ): Expression = {
      e match {
        case wsf @ WSubField(wr @ WRef(ref, _, InstanceKind, _), field, tpe, gen) =>
          val inst = currentModule.instOf(ref, instMap(Instance(ref)).value)
          val renamesOpt = renamesMap.get(OfModule(currentModule.module) -> Instance(inst.instance))
          val port = inst.ref(field)
          renamesOpt.flatMap(_.get(port)) match {
            case Some(Seq(p)) =>
              p match {
                case ReferenceTarget(_, _, Seq((TargetToken.Instance(r), TargetToken.OfModule(_))), f, Nil) =>
                  wsf.copy(expr = wr.copy(name = r), name = f)
                case ReferenceTarget(_, _, Nil, r, Nil) => WRef(r, tpe, WireKind, gen)
              }
            case None => wsf
          }
        case wr @ WRef(name, _, InstanceKind, _) =>
          val inst = currentModule.instOf(name, instMap(Instance(name)).value)
          val renamesOpt = renamesMap.get(OfModule(currentModule.module) -> Instance(inst.instance))
          val comp = currentModule.ref(name)
          renamesOpt.flatMap(_.get(comp)).getOrElse(Seq(comp)) match {
            case Seq(car: ReferenceTarget) => wr.copy(name = car.ref)
          }
        case ex => ex.map(fixupRefs(instMap, currentModule))
      }
    }

    def onStmt(currentModule: ModuleTarget)(s: Statement): Statement = {
      val currentModuleName = currentModule.module
      val ns = namespaceMap.getOrElseUpdate(currentModuleName, Namespace(iGraph.moduleMap(currentModuleName)))
      val instMap = instMaps(OfModule(currentModuleName))
      s match {
        case wDef @ WDefInstance(_, instName, modName, _)
            if flatInstances.contains(OfModule(currentModuleName) -> Instance(instName)) =>
          val renames = renamesMap(OfModule(currentModuleName) -> Instance(instName))
          val toInline = iGraph.moduleMap(modName) match {
            case m: ExtModule => throw new PassException(s"Cannot inline external module ${m.name}")
            case m: Module    => m
          }

          val ports = toInline.ports.map(p => DefWire(p.info, p.name, p.tpe))

          val bodyx = {
            val module = currentModule.copy(module = modName)
            cache.getOrElseUpdate(module, Block(ports :+ toInline.body).map(onStmt(module)))
          }

          val names = "" +:
            enumerateNames(Utils.stmtToType(bodyx))
              .map(_.mkString("_"))

          /** The returned prefix will not be "prefix unique". It may be the same as other existing prefixes in the namespace.
            * However, prepending this prefix to all inlined components is guaranteed to not conflict with this module's
            * namespace. To make it prefix unique, this requires expanding all names in the namespace to include their
            * prefixes before calling findValidPrefix.
            */
          val safePrefix = Namespace.findValidPrefix(instName + inlineDelim, names, ns.cloneUnderlying - instName)

          val prefixMap = mutable.HashMap.empty[String, String]
          val inlineTarget = currentModule.instOf(instName, modName)
          val renamedBody = bodyx
            .map(appendNamePrefix(inlineTarget, currentModule, safePrefix, ns, prefixMap, renames))
            .map(appendRefPrefix(inlineTarget, prefixMap))

          renames.record(inlineTarget, currentModule)
          renamedBody
        case sx =>
          sx
            .map(fixupRefs(instMap, currentModule))
            .map(onStmt(currentModule))
      }
    }

    val flatCircuit = c.copy(modules = c.modules.flatMap {
      case m if flatModules.contains(m.name) => None
      case m =>
        Some(m.map(onStmt(ModuleName(m.name, CircuitName(c.main)))))
    })

    // Upcast so reduce works (andThen returns RenameMap)
    val renames = (renamesSeq: Seq[RenameMap]).reduceLeftOption(_ andThen _)

    val cleanedAnnos = annos.filterNot {
      case InlineAnnotation(_) => true
      case _                   => false
    }

    CircuitState(flatCircuit, LowForm, cleanedAnnos, renames)
  }
}
