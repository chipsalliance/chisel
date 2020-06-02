// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.analyses.InstanceGraph
import firrtl.annotations._
import firrtl.passes.{InferTypes, MemPortUtils}
import firrtl.Utils.throwInternalError
import firrtl.annotations.transforms.DupedResult
import firrtl.annotations.TargetToken.{OfModule, Instance}
import firrtl.options.{HasShellOptions, PreservesAll, ShellOption}
import logger.LazyLogging

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
class DedupModules extends Transform with DependencyAPIMigration with PreservesAll[Transform] {

  override def prerequisites = firrtl.stage.Forms.Resolved

  override def optionalPrerequisiteOf = Seq.empty

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
    val instanceGraph = new InstanceGraph(c)
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
      val childrenMap = instanceGraph.getChildrenInstances
      val newModsMap: Map[String, mutable.LinkedHashSet[WDefInstance]] = dedupMap.map {
        case (name, m: Module) =>
          val set = new mutable.LinkedHashSet[WDefInstance]
          InstanceGraph.collectInstances(set)(m.body)
          m.name -> set
        case (name, m: DefModule) =>
          m.name -> mutable.LinkedHashSet.empty[WDefInstance]
      }.toMap
      (mod: String) => childrenMap.get(mod).getOrElse(newModsMap(mod))
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
            path.foldLeft(root -> root) { case ((oldRelPath, newRelPath), WDefInstance(_, name, mod, _)) =>
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
  def fastSerializedHash(s: Statement): Int ={
    def serialize(builder: StringBuilder, nindent: Int)(s: Statement): Unit = s match {
      case Block(stmts) => stmts.map {
        val x = serialize(builder, nindent)(_)
        builder ++= "\n"
        x
      }
      case Conditionally(info, pred, conseq, alt) =>
        builder ++= ("  " * nindent)
        builder ++= s"when ${pred.serialize} :"
        builder ++= info.serialize
        serialize(builder, nindent + 1)(conseq)
        builder ++= "\n" + ("  " * nindent)
        builder ++= "else :\n"
        serialize(builder, nindent + 1)(alt)
      case Print(info, string, args, clk, en) =>
        builder ++= ("  " * nindent)
        val strs = Seq(clk.serialize, en.serialize, string.string) ++
          (args map (_.serialize))
        builder ++= "printf(" + (strs mkString ", ") + ")" + info.serialize
      case other: Statement =>
        builder ++= ("  " * nindent)
        builder ++= other.serialize
    }
    val builder = new mutable.StringBuilder()
    serialize(builder, 0)(s)
    builder.hashCode()
  }

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

  def uniquifyField(ref: String, depth: Int, field: String): String = ref + depth + field

  /** Turns a module into a name-agnostic module
    * @param module module to change
    * @return name-agnostic module
    */
  def agnostify(top: CircuitTarget,
                module: DefModule,
                renameMap: RenameMap,
                agnosticModuleName: String
               ): DefModule = {


    val namespace = Namespace()
    val typeMap = mutable.HashMap[String, Type]()
    val nameMap = mutable.HashMap[String, String]()

    val mod = top.module(module.name)
    val agnosticMod = top.module(agnosticModuleName)

    def rename(name: String): String = {
      nameMap.getOrElseUpdate(name, {
        val newName = namespace.newTemp
        renameMap.record(mod.ref(name), agnosticMod.ref(newName))
        newName
      })
    }

    def retype(name: String)(tpe: Type): Type = {
      if (typeMap.contains(name)) typeMap(name) else {
        def onType(depth: Int)(tpe: Type): Type = tpe map onType(depth + 1) match {
          //TODO bugfix: ref.data.data and ref.datax.data will not rename to the right tags, even if they should be
          case BundleType(fields) =>
            BundleType(fields.map(f => Field(rename(uniquifyField(name, depth, f.name)), f.flip, f.tpe)))
          case other => other
        }
        val newType = onType(0)(tpe)
        typeMap(name) = newType
        newType
      }
    }

    def reOfModule(instance: String, ofModule: String): String = {
      renameMap.get(top.module(ofModule)) match {
        case Some(Seq(Target(_, Some(ofModuleTag), Nil))) => ofModuleTag
        case None => ofModule
        case other => throwInternalError(other.toString)
      }
    }

    val renamedModule = changeInternals(rename, retype, {i: Info => NoInfo}, reOfModule)(module)
    renamedModule
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

    // Get all instances to know what to rename in the module
    val instances = mutable.Set[WDefInstance]()
    InstanceGraph.collectInstances(instances)(module.asInstanceOf[Module].body)
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

  //scalastyle:off
  /** Returns
    *  1) map of tag to all matching module names,
    *  2) renameMap of module name to tag (agnostic name)
    *  3) maps module name to agnostic renameMap
    * @param top CircuitTarget
    * @param moduleLinearization Sequence of modules from leaf to top
    * @param noDedups Set of modules to not dedup
    * @return
    */
  def buildRTLTags(top: CircuitTarget,
                   moduleLinearization: Seq[DefModule],
                   noDedups: Set[String]
                  ): (collection.Map[String, collection.Set[String]], RenameMap) = {


    // Maps a module name to its agnostic name
    val tagMap = RenameMap()

    // Maps a tag to all matching module names
    val tag2all = mutable.HashMap.empty[String, mutable.HashSet[String]]

    val agnosticRename = RenameMap()

    moduleLinearization.foreach { originalModule =>
      // Replace instance references to new deduped modules
      val dontcare = RenameMap()
      dontcare.setCircuit("dontcare")

      if (noDedups.contains(originalModule.name)) {
        // Don't dedup. Set dedup module to be the same as fixed module
        tag2all(originalModule.name) = mutable.HashSet(originalModule.name)
      } else { // Try to dedup

        // Build name-agnostic module
        val agnosticModule = DedupModules.agnostify(top, originalModule, agnosticRename, "thisModule")
        agnosticRename.record(top.module(originalModule.name), top.module("thisModule"))
        agnosticRename.delete(top.module(originalModule.name))

        // Build tag
        val builder = new mutable.ArrayBuffer[Any]()
        agnosticModule.ports.foreach { builder ++= _.serialize }

        agnosticModule match {
          case Module(i, n, ps, b) => builder ++= fastSerializedHash(b).toString()//.serialize
          case ExtModule(i, n, ps, dn, p) =>
            builder ++= dn
            p.foreach { builder ++= _.serialize }
        }
        val tag = builder.hashCode().toString

        // Match old module name to its tag
        agnosticRename.record(top.module(originalModule.name), top.module(tag))
        tagMap.record(top.module(originalModule.name), top.module(tag))

        // Set tag's module to be the first matching module
        val all = tag2all.getOrElseUpdate(tag, mutable.HashSet.empty[String])
        all += originalModule.name
      }
    }
    (tag2all, tagMap)
  }
  //scalastyle:on

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
      val iGraph = new InstanceGraph(circuit)
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
    refs
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
    all
  }
}
