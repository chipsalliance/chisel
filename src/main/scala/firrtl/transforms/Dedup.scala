// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.analyses.InstanceKeyGraph
import firrtl.annotations._
import firrtl.passes.{InferTypes, MemPortUtils}
import firrtl.Utils.{kind, splitRef, throwInternalError}
import firrtl.annotations.transforms.DupedResult
import firrtl.annotations.TargetToken.{Instance, OfModule}
import firrtl.options.{HasShellOptions, ShellOption}
import logger.LazyLogging

import scala.annotation.tailrec

// Datastructures
import scala.collection.mutable


/** A component, e.g. register etc. Must be declared only once under the TopAnnotation */
case class NoDedupAnnotation(target: ModuleTarget) extends SingleTargetAnnotation[ModuleTarget] {
  def duplicate(n: ModuleTarget): NoDedupAnnotation = NoDedupAnnotation(n)
}

/** If this [[firrtl.annotations.Annotation Annotation]] exists in an [[firrtl.AnnotationSeq AnnotationSeq]],
  * then the [[firrtl.transforms.DedupModules]] transform will *NOT* be run on the circuit.
  *  - set with '--no-dedup'
  */
case object NoCircuitDedupAnnotation extends NoTargetAnnotation with HasShellOptions {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-dedup",
      toAnnotationSeq = _ => Seq(NoCircuitDedupAnnotation),
      helpText = "Do NOT dedup modules" ) )

}

/** Holds the mapping from original module to the instances the original module pointed to
  * The original module target is unaffected by renaming
  * @param duplicate Instance target of what the original module now points to
  * @param original Original module
  * @param index the normalized position of the original module in the original module list, fraction between 0 and 1
  */
case class DedupedResult(original: ModuleTarget, duplicate: Option[IsModule], index: Double) extends MultiTargetAnnotation {
  override val targets: Seq[Seq[Target]] = Seq(Seq(original), duplicate.toList)
  override def duplicate(n: Seq[Seq[Target]]): Annotation = {
    n.toList match {
      case Seq(_, List(dup: IsModule)) => DedupedResult(original, Some(dup), index)
      case _                           => DedupedResult(original, None, -1)
    }
  }
}

/** Only use on legal Firrtl.
  *
  * Specifically, the restriction of instance loops must have been checked, or else this pass can
  * infinitely recurse.
  *
  * Deduped modules are renamed using a chain of 3 [[RenameMap]]s. The first
  * [[RenameMap]] renames the original [[annotations.ModuleTarget]]s and relative
  * [[annotations.InstanceTarget]]s to the groups of absolute [[annotations.InstanceTarget]]s that they
  * target. These renames only affect instance names and paths and use the old
  * module names. During this rename, modules will also have their instance
  * names renamed if they dedup with a module that has different instance
  * names.
  * The second [[RenameMap]] renames all component names within modules that
  * dedup with another module that has different component names.
  * The third [[RenameMap]] renames original [[annotations.ModuleTarget]]s to their deduped
  * [[annotations.ModuleTarget]].
  *
  * This transform will also emit [[DedupedResult]] for deduped modules that
  * only have one instance.
  */
class DedupModules extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.Resolved

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  /** Deduplicate a Circuit
    * @param state Input Firrtl AST
    * @return A transformed Firrtl AST
    */
  def execute(state: CircuitState): CircuitState = {
    if (state.annotations.contains(NoCircuitDedupAnnotation)) {
      state
    } else {
      // Don't try deduping the main module of the circuit
      val noDedups = state.circuit.main +: state.annotations.collect { case NoDedupAnnotation(ModuleTarget(_, m)) => m }
      val (remainingAnnotations, dupResults) = state.annotations.partition {
        case _: DupedResult => false
        case _                => true
      }
      val previouslyDupedMap = dupResults.flatMap {
        case DupedResult(newModules, original) =>
          newModules.collect {
            case m: ModuleTarget => m.module -> original.module
          }
      }.toMap
      val (newC, renameMap, newAnnos) = run(state.circuit, noDedups, previouslyDupedMap)
      state.copy(circuit = newC, renames = Some(renameMap), annotations = newAnnos ++ remainingAnnotations)
    }
  }

  /** Deduplicates a circuit, and records renaming
    * @param c Circuit to dedup
    * @param noDedups Modules not to dedup
    * @return Deduped Circuit and corresponding RenameMap
    */
  def run(c: Circuit,
          noDedups: Seq[String],
          previouslyDupedMap: Map[String, String]): (Circuit, RenameMap, AnnotationSeq) = {

    // RenameMap
    val componentRenameMap = RenameMap()
    componentRenameMap.setCircuit(c.main)

    // Maps module name to corresponding dedup module
    val dedupMap = DedupModules.deduplicate(c, noDedups.toSet, previouslyDupedMap, componentRenameMap)
    val dedupCliques = dedupMap.foldLeft(Map.empty[String, Set[String]]) {
      case (dedupCliqueMap, (orig: String, dupMod: DefModule)) =>
        val set = dedupCliqueMap.getOrElse(dupMod.name, Set.empty[String]) + dupMod.name + orig
        dedupCliqueMap + (dupMod.name -> set)
    }.flatMap { case (dedupName, set) =>
      set.map { _ -> set }
    }

    // Use old module list to preserve ordering
    // Lookup what a module deduped to, if its a duplicate, remove it
    val dedupedModules = {
      val seen = mutable.Set[String]()
      c.modules.flatMap { m =>
        val dedupMod = dedupMap(m.name)
        if (!seen(dedupMod.name)) {
          seen += dedupMod.name
          Some(dedupMod)
        } else {
          None
        }
      }
    }

    val ct = CircuitTarget(c.main)

    val map = dedupMap.map { case (from, to) =>
      logger.debug(s"[Dedup] $from -> ${to.name}")
      ct.module(from).asInstanceOf[CompleteTarget] -> Seq(ct.module(to.name))
    }
    val moduleRenameMap = RenameMap()
    moduleRenameMap.recordAll(map)

    // Build instanceify renaming map
    val instanceGraph = InstanceKeyGraph(c)
    val instanceify = RenameMap()
    val moduleName2Index = c.modules.map(_.name).zipWithIndex.map { case (n, i) =>
      {
        c.modules.size match {
          case 0 => (n, 0.0)
          case 1 => (n, 1.0)
          case d => (n, i.toDouble / (d - 1))
        }
      }
    }.toMap

    // get the ordered set of instances a module, includes new Deduped modules
    val getChildrenInstances = {
      val childrenMap = instanceGraph.getChildInstances.toMap
      val newModsMap = dedupMap.map {
        case (_, m: Module) =>
          m.name -> InstanceKeyGraph.collectInstances(m)
        case (_, m: DefModule) =>
          m.name -> List()
      }
      (mod: String) => childrenMap.getOrElse(mod, newModsMap(mod))
    }

    val instanceNameMap: Map[OfModule, Map[Instance, Instance]] = {
      dedupMap.map { case (oldName, dedupedMod) =>
        val key = OfModule(oldName)
        val value = getChildrenInstances(oldName).zip(getChildrenInstances(dedupedMod.name)).map {
          case (oldInst, newInst) => Instance(oldInst.name) -> Instance(newInst.name)
        }.toMap
        key -> value
      }.toMap
    }
    val dedupAnnotations = c.modules.map(_.name).map(ct.module).flatMap { case mt@ModuleTarget(c, m) if dedupCliques(m).size > 1 =>
      dedupMap.get(m) match {
        case None => Nil
        case Some(module: DefModule) =>
          val paths = instanceGraph.findInstancesInHierarchy(m)
          // If dedupedAnnos is exactly annos, contains is because dedupedAnnos is type Option
          val newTargets = paths.map { path =>
            val root: IsModule = ct.module(c)
            path.foldLeft(root -> root) { case ((oldRelPath, newRelPath), InstanceKeyGraph.InstanceKey(name, mod)) =>
              if(mod == c) {
                val mod = CircuitTarget(c).module(c)
                mod -> mod
              } else {
                val enclosingMod = oldRelPath match {
                  case i: InstanceTarget => i.ofModule
                  case m: ModuleTarget => m.module
                }
                val instMap = instanceNameMap(OfModule(enclosingMod))
                val newInstName = instMap(Instance(name)).value
                val old = oldRelPath.instOf(name, mod)
                old -> newRelPath.instOf(newInstName, mod)
              }
            }
          }

          // Add all relative paths to referredModule to map to new instances
          def addRecord(from: IsMember, to: IsMember): Unit = from match {
            case x: ModuleTarget =>
              instanceify.record(x, to)
            case x: IsComponent =>
              instanceify.record(x, to)
              addRecord(x.stripHierarchy(1), to)
          }
          // Instanceify deduped Modules!
          if (dedupCliques(module.name).size > 1) {
            newTargets.foreach { case (from, to) => addRecord(from, to) }
          }
          // Return Deduped Results
          if (newTargets.size == 1) {
            Seq(DedupedResult(mt, newTargets.headOption.map(_._1), moduleName2Index(m)))
          } else Nil
      }
      case noDedups => Nil
    }

    val finalRenameMap = instanceify.andThen(componentRenameMap).andThen(moduleRenameMap)
    (InferTypes.run(c.copy(modules = dedupedModules)), finalRenameMap, dedupAnnotations.toList)
  }
}

/** Utility functions for [[DedupModules]] */
object DedupModules extends LazyLogging {
  /** Change's a module's internal signal names, types, infos, and modules.
    * @param rename Function to rename a signal. Called on declaration and references.
    * @param retype Function to retype a signal. Called on declaration, references, and subfields
    * @param reinfo Function to re-info a statement
    * @param renameOfModule Function to rename an instance's module
    * @param module Module to change internals
    * @return Changed Module
    */
  def changeInternals(rename: String=>String,
                      retype: String=>Type=>Type,
                      reinfo: Info=>Info,
                      renameOfModule: (String, String)=>String,
                      renameExps: Boolean = true
                     )(module: DefModule): DefModule = {
    def onPort(p: Port): Port = Port(reinfo(p.info), rename(p.name), p.direction, retype(p.name)(p.tpe))
    def onExp(e: Expression): Expression = e match {
      case WRef(n, t, k, g) => WRef(rename(n), retype(n)(t), k, g)
      case WSubField(expr, n, tpe, kind) =>
        val fieldIndex = expr.tpe.asInstanceOf[BundleType].fields.indexWhere(f => f.name == n)
        val newExpr = onExp(expr)
        val newField = newExpr.tpe.asInstanceOf[BundleType].fields(fieldIndex)
        val finalExpr = WSubField(newExpr, newField.name, newField.tpe, kind)
        //TODO: renameMap.rename(e.serialize, finalExpr.serialize)
        finalExpr
      case other => other map onExp
    }
    def onStmt(s: Statement): Statement = s match {
      case DefNode(info, name, value) =>
        retype(name)(value.tpe)
        if(renameExps) DefNode(reinfo(info), rename(name), onExp(value))
        else DefNode(reinfo(info), rename(name), value)
      case WDefInstance(i, n, m, t) =>
        val newmod = renameOfModule(n, m)
        WDefInstance(reinfo(i), rename(n), newmod, retype(n)(t))
      case DefInstance(i, n, m, t) =>
        val newmod = renameOfModule(n, m)
        WDefInstance(reinfo(i), rename(n), newmod, retype(n)(t))
      case d: DefMemory =>
        val oldType = MemPortUtils.memType(d)
        val newType = retype(d.name)(oldType)
        val index = oldType
          .asInstanceOf[BundleType].fields.headOption
          .map(_.tpe.asInstanceOf[BundleType].fields.indexWhere(
            {
              case Field("data" | "wdata" | "rdata", _, _) => true
              case _ => false
            }))
        val newDataType = index match {
          case Some(i) =>
            //If index nonempty, then there exists a port
            newType.asInstanceOf[BundleType].fields.head.tpe.asInstanceOf[BundleType].fields(i).tpe
          case None =>
            //If index is empty, this mem has no ports, and so we don't need to record the dataType
            // Thus, call retype with an illegal name, so we can retype the memory's datatype, but not
            // associate it with the type of the memory (as the memory type is different than the datatype)
            retype(d.name + ";&*^$")(d.dataType)
        }
        d.copy(dataType = newDataType) map rename map reinfo
      case h: IsDeclaration =>
        val temp = h map rename map retype(h.name) map reinfo
        if(renameExps) temp map onExp else temp
      case other =>
        val temp = other map reinfo map onStmt
        if(renameExps) temp map onExp else temp
    }
    module map onPort map onStmt
  }

  /** Dedup a module's instances based on dedup map
    *
    * Will fixes up module if deduped instance's ports are differently named
    *
    * @param top CircuitTarget of circuit
    * @param originalModule Module name who's instances will be deduped
    * @param moduleMap Map of module name to its original module
    * @param name2name Map of module name to the module deduping it. Not mutated in this function.
    * @param renameMap Will be modified to keep track of renames in this function
    * @return fixed up module deduped instances
    */
  def dedupInstances(top: CircuitTarget,
                     originalModule: String,
                     moduleMap: Map[String, DefModule],
                     name2name: Map[String, String],
                     renameMap: RenameMap): DefModule = {
    val module = moduleMap(originalModule)

    // If black box, return it (it has no instances)
    if (module.isInstanceOf[ExtModule]) return module

    // Get all instances to know what to rename in the module s
    val instances = InstanceKeyGraph.collectInstances(module)
    val instanceModuleMap = instances.map(i => i.name -> i.module).toMap

    def getNewModule(old: String): DefModule = {
      moduleMap(name2name(old))
    }
    val typeMap = mutable.HashMap[String, Type]()
    def retype(name: String)(tpe: Type): Type = {
      if (typeMap.contains(name)) typeMap(name) else {
        if (instanceModuleMap.contains(name)) {
          val newType = Utils.module_type(getNewModule(instanceModuleMap(name)))
          typeMap(name) = newType
          getAffectedExpressions(WRef(name, tpe)).zip(getAffectedExpressions(WRef(name, newType))).foreach {
            case (old, nuu) => renameMap.rename(old.serialize, nuu.serialize)
          }
          newType
        } else {
          tpe
        }
      }
    }

    renameMap.setModule(module.name)
    // Change module internals
    // Define rename functions
    def renameOfModule(instance: String, ofModule: String): String = {
      name2name(ofModule)
    }
    changeInternals({n => n}, retype, {i => i}, renameOfModule)(module)
  }

  @tailrec
  private def hasBundleType(tpe: Type): Boolean = tpe match {
    case _: BundleType => true
    case _: GroundType => false
    case VectorType(t, _) => hasBundleType(t)
  }

  // Find modules that should not have their ports agnostified to avoid bug in
  // https://github.com/freechipsproject/firrtl/issues/1703
  // Marks modules that have a port of BundleType that are connected via an aggregate connect or
  // partial connect in an instantiating parent
  // Order of modules does not matter
  private def modsToNotAgnostifyPorts(modules: Seq[DefModule]): Set[String] = {
    val dontDedup = mutable.HashSet.empty[String]
    def onModule(mod: DefModule): Unit = {
      val instToModule = mutable.HashMap.empty[String, String]
      def markAggregatePorts(expr: Expression): Unit = {
        if (kind(expr) == InstanceKind && hasBundleType(expr.tpe)) {
          val (WRef(inst, _, _, _), _) = splitRef(expr)
          dontDedup += instToModule(inst)
        }
      }
      def onStmt(stmt: Statement): Unit = {
        stmt.foreach(onStmt)
        stmt match {
          case inst: DefInstance =>
            instToModule(inst.name) = inst.module
          case Connect(_, lhs, rhs) =>
            markAggregatePorts(lhs)
            markAggregatePorts(rhs)
          case PartialConnect(_, lhs, rhs) =>
            markAggregatePorts(lhs)
            markAggregatePorts(rhs)
          case _ =>
        }
      }
      mod.foreach(onStmt)
    }
    modules.foreach(onModule)
    dontDedup.toSet
  }

  /** Visits every module in the circuit, starting at the leaf nodes.
    * Every module is hashed in order to find ones that have the exact
    * same structure and are thus functionally equivalent.
    * Every unique hash is mapped to a human-readable tag which starts with `Dedup#`.
    * @param top CircuitTarget
    * @param moduleLinearization Sequence of modules from leaf to top
    * @param noDedups names of modules that should not be deduped
    * @return A map from tag to names of modules with the same structure and
    *         a RenameMap which maps Module names to their Tag.
    */
  def buildRTLTags(top: CircuitTarget,
                   moduleLinearization: Seq[DefModule],
                   noDedups: Set[String]
                  ): (collection.Map[String, collection.Set[String]], RenameMap) = {
    // maps hash code to human readable tag
    val hashToTag = mutable.HashMap[ir.HashCode, String]()

    // remembers all modules with the same hash
    val hashToNames = mutable.HashMap[ir.HashCode, List[String]]()

    // rename modules that we have already visited to their hash-derived tag name
    val moduleNameToTag = mutable.HashMap[String, String]()

    val dontAgnostifyPorts = modsToNotAgnostifyPorts(moduleLinearization)

    moduleLinearization.foreach { originalModule =>
      val hash = if (noDedups.contains(originalModule.name)) {
        // if we do not want to dedup we just hash the name of the module which is guaranteed to be unique
        StructuralHash.sha256(originalModule.name)
      } else if (dontAgnostifyPorts(originalModule.name)) {
        StructuralHash.sha256WithSignificantPortNames(originalModule, moduleNameToTag)
      } else {
        StructuralHash.sha256(originalModule, moduleNameToTag)
      }

      if (hashToTag.contains(hash)) {
        hashToNames(hash) = hashToNames(hash) :+ originalModule.name
      } else {
        hashToTag(hash) = "Dedup#" + originalModule.name
        hashToNames(hash) = List(originalModule.name)
      }
      moduleNameToTag(originalModule.name) = hashToTag(hash)
    }

    val tag2all = hashToNames.map{ case (hash, names) => hashToTag(hash) -> names.toSet }
    val tagMap = RenameMap()
    moduleNameToTag.foreach{ case (name, tag) => tagMap.record(top.module(name), top.module(tag)) }
    (tag2all, tagMap)
  }

  /** Deduplicate
    * @param circuit Circuit
    * @param noDedups list of modules to not dedup
    * @param renameMap rename map to populate when deduping
    * @return Map of original Module name -> Deduped Module
    */
  def deduplicate(circuit: Circuit,
                  noDedups: Set[String],
                  previousDupResults: Map[String, String],
                  renameMap: RenameMap): Map[String, DefModule] = {

    val (moduleMap, moduleLinearization) = {
      val iGraph = InstanceKeyGraph(circuit)
      (iGraph.moduleMap, iGraph.moduleOrder.reverse)
    }
    val main = circuit.main
    val top = CircuitTarget(main)

    // Maps a module name to its agnostic name
    // tagMap is a RenameMap containing ModuleTarget renames of original name to tag name
    // tag2all is a Map of tag to original names of all modules with that tag
    val (tag2all, tagMap) = buildRTLTags(top, moduleLinearization, noDedups)

    // Set tag2name to be the best dedup module name
    val moduleIndex = circuit.modules.zipWithIndex.map{case (m, i) => m.name -> i}.toMap

    // returns the module matching the circuit name or the module with lower index otherwise
    def order(l: String, r: String): String = {
      if (l == main) l
      else if (r == main) r
      else if (moduleIndex(l) < moduleIndex(r)) l else r
    }

    // Maps a module's tag to its deduplicated module
    val tag2name = mutable.HashMap.empty[String, String]

    // Maps a deduped module name to its original Module (its instance names need to be updated)
    val moduleMapWithOldNames = tag2all.map {
      case (tag, all: collection.Set[String]) =>
        val dedupWithoutOldName = all.reduce(order)
        val dedupName = previousDupResults.getOrElse(dedupWithoutOldName, dedupWithoutOldName)
        tag2name(tag) = dedupName
        val dedupModule = moduleMap(dedupWithoutOldName) match {
          case e: ExtModule => e.copy(name = dedupName)
          case e: Module => e.copy(name = dedupName)
        }
        dedupName -> dedupModule
    }.toMap

    // Create map from original to dedup name
    val name2name = moduleMap.keysIterator.map { originalModule =>
      tagMap.get(top.module(originalModule)) match {
        case Some(Seq(Target(_, Some(tag), Nil))) => originalModule -> tag2name(tag)
        case None => originalModule -> originalModule
        case other => throwInternalError(other.toString)
      }
    }.toMap

    // Build Remap for modules with deduped module references
    val dedupedName2module = tag2name.map {
      case (tag, name) => name -> DedupModules.dedupInstances(
          top, name, moduleMapWithOldNames, name2name, renameMap)
    }

    // Build map from original name to corresponding deduped module
    // It is important to flatMap before looking up the DefModules so that they aren't hashed
    val name2module: Map[String, DefModule] =
      tag2all.flatMap { case (tag, names) => names.map(_ -> tag) }
             .mapValues(tag => dedupedName2module(tag2name(tag)))
             .toMap

    // Build renameMap
    val indexedTargets = mutable.HashMap[String, IndexedSeq[ReferenceTarget]]()
    name2module.foreach { case (originalName, depModule) =>
      if(originalName != depModule.name) {
        val toSeq = indexedTargets.getOrElseUpdate(depModule.name, computeIndexedNames(circuit.main, depModule))
        val fromSeq = computeIndexedNames(circuit.main, moduleMap(originalName))
        computeRenameMap(fromSeq, toSeq, renameMap)
      }
    }

    name2module
  }

  def computeIndexedNames(main: String, m: DefModule): IndexedSeq[ReferenceTarget] = {
    val refs = mutable.ArrayBuffer[ReferenceTarget]()
    def rename(name: String): String = name

    def retype(name: String)(tpe: Type): Type = {
      val exps = Utils.expandRef(WRef(name, tpe, ExpKind, UnknownFlow))
      refs ++= exps.map(Utils.toTarget(main, m.name))
      tpe
    }

    changeInternals(rename, retype, {i => i}, {(x, y) => x}, renameExps = false)(m)
    refs.toIndexedSeq
  }

  def computeRenameMap(originalNames: IndexedSeq[ReferenceTarget],
                       dedupedNames: IndexedSeq[ReferenceTarget],
                       renameMap: RenameMap): Unit = {

    originalNames.zip(dedupedNames).foreach {
      case (o, d) => if (o.component != d.component || o.ref != d.ref) {
        renameMap.record(o, d.copy(module = o.module))
      }
    }

  }

  def getAffectedExpressions(root: Expression): Seq[Expression] = {
    val all = mutable.ArrayBuffer[Expression]()

    def onExp(expr: Expression): Unit = {
      expr.tpe match {
        case _: GroundType =>
        case b: BundleType => b.fields.foreach { f => onExp(WSubField(expr, f.name, f.tpe)) }
        case v: VectorType => (0 until v.size).foreach { i => onExp(WSubIndex(expr, i, v.tpe, UnknownFlow)) }
      }
      all += expr
    }

    onExp(root)
    all.toSeq
  }
}
