// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.analyses.{InstanceKeyGraph, SymbolTable}
import firrtl.annotations.{
  CircuitTarget,
  MemoryInitAnnotation,
  MemoryRandomInitAnnotation,
  ModuleTarget,
  ReferenceTarget,
  TargetToken
}
import TargetToken.{Instance, OfModule}
import firrtl.{
  CircuitForm,
  CircuitState,
  DependencyAPIMigration,
  InstanceKind,
  Kind,
  MemKind,
  PortKind,
  RenameMap,
  Transform,
  UnknownForm,
  Utils
}
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.stage.TransformManager.TransformDependency

import scala.annotation.tailrec
import scala.collection.mutable

/** Flattens Bundles and Vecs.
  * - Some implicit bundle types remain, but with a limited depth:
  *   - the type of a memory is still a bundle with depth 2 (mem -> port -> field), see [[MemPortUtils.memType]]
  *   - the type of a module instance is still a bundle with depth 1 (instance -> port)
  */
object LowerTypes extends Transform with DependencyAPIMigration {
  override def prerequisites: Seq[TransformDependency] = Seq(
    Dependency(RemoveAccesses), // we require all SubAccess nodes to have been removed
    Dependency(CheckTypes), // we require all types to be correct
    Dependency(InferTypes), // we require instance types to be resolved (i.e., DefInstance.tpe != UnknownType)
    Dependency(ExpandConnects) // we require all PartialConnect nodes to have been expanded
  )
  override def optionalPrerequisiteOf: Seq[TransformDependency] = Seq.empty
  override def invalidates(a: Transform): Boolean = a match {
    case ResolveFlows => true // we generate UnknownFlow for now (could be fixed)
    case _            => false
  }

  /** Delimiter used in lowering names */
  val delim = "_"

  /** Expands a chain of referential [[firrtl.ir.Expression]]s into the equivalent lowered name
    * @param e [[firrtl.ir.Expression]] made up of _only_ [[firrtl.WRef]], [[firrtl.WSubField]], and [[firrtl.WSubIndex]]
    * @return Lowered name of e
    * @note Please make sure that there will be no name collisions when you use this outside of the context of LowerTypes!
    */
  def loweredName(e: Expression): String = e match {
    case e: Reference => e.name
    case e: SubField  => s"${loweredName(e.expr)}$delim${e.name}"
    case e: SubIndex  => s"${loweredName(e.expr)}$delim${e.value}"
  }
  def loweredName(s: Seq[String]): String = s.mkString(delim)

  override def execute(state: CircuitState): CircuitState = {
    // When memories are lowered to ground type, we have to fix the init annotation or error on it.
    val (memInitAnnos, otherAnnos) = state.annotations.partition {
      case _: MemoryRandomInitAnnotation => false
      case _: MemoryInitAnnotation       => true
      case _ => false
    }
    val memInitByModule = memInitAnnos.map(_.asInstanceOf[MemoryInitAnnotation]).groupBy(_.target.encapsulatingModule)

    val c = CircuitTarget(state.circuit.main)
    // By default, the RenameMap enforces a .distinct invariant for renames. This helps transform
    // writers not mess up because violating that invariant can cause problems for transform
    // writers. Unfortunately, when you have lots of renames, this is very expensive
    // performance-wise. We use a private internal API that does not run .distinct to improve
    // performance, but we must be careful to not insert any duplicates.
    val refRenameMap = RenameMap.noDistinct()
    val resultAndRenames =
      state.circuit.modules.map(m => onModule(c, m, memInitByModule.getOrElse(m.name, Seq()), refRenameMap))
    val result = state.circuit.copy(modules = resultAndRenames.map(_._1))

    // memory init annotations could have been modified
    val newAnnos = otherAnnos ++ resultAndRenames.flatMap(_._3)

    // Build RenameMap for instances
    val moduleRenames = resultAndRenames.map { case (m, r, _) => OfModule(m.name) -> r }.toMap
    val instRenameMap = RenameMap.fromInstanceRenames(InstanceKeyGraph(state.circuit), moduleRenames)
    val renames = instRenameMap.andThen(refRenameMap)

    state.copy(circuit = result, renames = Some(renames), annotations = newAnnos)
  }

  private def onModule(
    c:          CircuitTarget,
    m:          DefModule,
    memoryInit: Seq[MemoryInitAnnotation],
    renameMap:  RenameMap
  ): (DefModule, Map[Instance, Instance], Seq[MemoryInitAnnotation]) = {
    val ref = c.module(m.name)

    // first we lower the ports in order to ensure that their names are independent of the module body
    val (mLoweredPorts, portRefs) = lowerPorts(ref, m, renameMap)

    // scan modules to find all references
    val scan = SymbolTable.scanModule(mLoweredPorts, new LoweringSymbolTable)
    // replace all declarations and references with the destructed types
    implicit val symbols: LoweringTable = new LoweringTable(scan, renameMap, ref, portRefs)
    implicit val memInit: Seq[MemoryInitAnnotation] = memoryInit
    val newMod = mLoweredPorts.mapStmt(onStatement)

    val instRenames = symbols.getInstanceRenames.toMap

    (newMod, instRenames, memInit)
  }

  // We lower ports in a separate pass in order to ensure that statements inside the module do not influence port names.
  private def lowerPorts(
    ref:       ModuleTarget,
    m:         DefModule,
    renameMap: RenameMap
  ): (DefModule, Seq[(String, Seq[Reference])]) = {
    val namespace = mutable.HashSet[String]() ++ m.ports.map(_.name)
    val loweredPortsAndRefs = m.ports.flatMap { p =>
      val fieldsAndRefs =
        DestructTypes.destruct(ref, Field(p.name, Utils.to_flip(p.direction), p.tpe), namespace, renameMap, Set())
      fieldsAndRefs.map {
        case (f, ref) =>
          (Port(p.info, f.name, Utils.to_dir(f.flip), f.tpe), ref -> Seq(Reference(f.name, f.tpe, PortKind)))
      }
    }
    val newM = m match {
      case e:   ExtModule => e.copy(ports = loweredPortsAndRefs.map(_._1))
      case mod: Module    => mod.copy(ports = loweredPortsAndRefs.map(_._1))
    }
    (newM, loweredPortsAndRefs.map(_._2))
  }

  private def onStatement(
    s: Statement
  )(
    implicit symbols: LoweringTable,
    memInit:          Seq[MemoryInitAnnotation]
  ): Statement = s match {
    // declarations
    case d: DefWire =>
      Block(symbols.lower(d.name, d.tpe, firrtl.WireKind).map { case (name, tpe, _) => d.copy(name = name, tpe = tpe) })
    case d @ DefRegister(info, _, _, clock, reset, _) =>
      // clock and reset are always of ground type
      val loweredClock = onExpression(clock)
      val loweredReset = onExpression(reset)
      // It is important to first lower the declaration, because the reset can refer to the register itself!
      val loweredRegs = symbols.lower(d.name, d.tpe, firrtl.RegKind)
      val inits = Utils.create_exps(d.init).map(onExpression)
      Block(loweredRegs.zip(inits).map {
        case ((name, tpe, _), init) =>
          DefRegister(info, name, tpe, loweredClock, loweredReset, init)
      })
    case d: DefNode =>
      val values = Utils.create_exps(d.value).map(onExpression)
      Block(symbols.lower(d.name, d.value.tpe, firrtl.NodeKind).zip(values).map {
        case ((name, tpe, _), value) =>
          assert(tpe == value.tpe)
          DefNode(d.info, name, value)
      })
    case d: DefMemory =>
      // TODO: as an optimization, we could just skip ground type memories here.
      //       This would require that we don't error in getReferences() but instead return the old reference.
      val mems = symbols.lower(d)
      if (mems.length > 1 && memInit.exists(_.target.ref == d.name)) {
        val mod = memInit.find(_.target.ref == d.name).get.target.encapsulatingModule
        val msg = s"[module $mod] Cannot initialize memory ${d.name} of non ground type ${d.dataType.serialize}"
        throw new RuntimeException(msg)
      }
      Block(mems)
    case d: DefInstance => symbols.lower(d)
    // connections
    case Connect(info, loc, expr) =>
      if (!expr.tpe.isInstanceOf[GroundType]) {
        throw new RuntimeException(s"LowerTypes expects Connects to have been expanded! ${expr.tpe.serialize}")
      }
      val rhs = onExpression(expr)
      // We can get multiple refs on the lhs because of ground-type memory ports like "clk" which can get duplicated.
      val lhs = symbols.getReferences(loc.asInstanceOf[RefLikeExpression])
      Block(lhs.map(loc => Connect(info, loc, rhs)))
    case p: PartialConnect =>
      throw new RuntimeException(s"LowerTypes expects PartialConnects to be resolved! $p")
    case IsInvalid(info, expr) =>
      if (!expr.tpe.isInstanceOf[GroundType]) {
        throw new RuntimeException(s"LowerTypes expects IsInvalids to have been expanded! ${expr.tpe.serialize}")
      }
      // We can get multiple refs on the lhs because of ground-type memory ports like "clk" which can get duplicated.
      val lhs = symbols.getReferences(expr.asInstanceOf[RefLikeExpression])
      Block(lhs.map(loc => IsInvalid(info, loc)))
    // others
    case other => other.mapExpr(onExpression).mapStmt(onStatement)
  }

  /** Replaces all Reference, SubIndex and SubField nodes with the updated references */
  private def onExpression(e: Expression)(implicit symbols: LoweringTable): Expression = e match {
    case r: RefLikeExpression =>
      // When reading (and not assigning to) an expression, we can always just pick the first one.
      // Only very few ground-type references are duplicated and they are all related to lowered memories.
      // e.g., the `clk` field of a memory port gets duplicated when the memory is split into ground-types.
      // We ensure that all of these references carry the same value when they are expanded in onStatement.
      symbols.getReferences(r).head
    case other => other.mapExpr(onExpression)
  }
}

// Holds the first level of the module-level namespace.
// (i.e. everything that can be addressed directly by a Reference node)
private class LoweringSymbolTable extends SymbolTable {
  def declare(name:         String, tpe:    Type, kind: Kind): Unit = symbols.append(name)
  def declareInstance(name: String, module: String): Unit = symbols.append(name)
  private val symbols = mutable.ArrayBuffer[String]()
  def getSymbolNames: Iterable[String] = symbols
}

// Lowers types and keeps track of references to lowered types.
private class LoweringTable(
  table:           LoweringSymbolTable,
  renameMap:       RenameMap,
  m:               ModuleTarget,
  portNameToExprs: Seq[(String, Seq[Reference])]) {
  private val portNames: Set[String] = portNameToExprs.map(_._2.head.name).toSet
  private val namespace = mutable.HashSet[String]() ++ table.getSymbolNames
  // Serialized old access string to new ground type reference.
  private val nameToExprs = mutable.HashMap[String, Seq[RefLikeExpression]]() ++ portNameToExprs
  private val instRenames = mutable.ListBuffer[(Instance, Instance)]()

  def lower(mem: DefMemory): Seq[DefMemory] = {
    val (mems, refs) = DestructTypes.destructMemory(m, mem, namespace, renameMap, portNames)
    nameToExprs ++= refs.groupBy(_._1).mapValues(_.map(_._2))
    mems
  }
  def lower(inst: DefInstance): DefInstance = {
    val (newInst, refs) = DestructTypes.destructInstance(m, inst, namespace, instRenames, portNames)
    nameToExprs ++= refs.map { case (name, r) => name -> List(r) }
    newInst
  }

  /** used to lower nodes, registers and wires */
  def lower(name: String, tpe: Type, kind: Kind, flip: Orientation = Default): Seq[(String, Type, Orientation)] = {
    val fieldsAndRefs = DestructTypes.destruct(m, Field(name, flip, tpe), namespace, renameMap, portNames)
    nameToExprs ++= fieldsAndRefs.map { case (f, ref) => ref -> List(Reference(f.name, f.tpe, kind)) }
    fieldsAndRefs.map { case (f, _) => (f.name, f.tpe, f.flip) }
  }
  def lower(p: Port): Seq[Port] = {
    val fields = lower(p.name, p.tpe, PortKind, Utils.to_flip(p.direction))
    fields.map { case (name, tpe, flip) => Port(p.info, name, Utils.to_dir(flip), tpe) }
  }

  def getReferences(expr: RefLikeExpression): Seq[RefLikeExpression] = nameToExprs(serialize(expr))
  def getInstanceRenames: List[(Instance, Instance)] = instRenames.toList

  // We could just use FirrtlNode.serialize here, but we want to make sure there are not SubAccess nodes left.
  private def serialize(expr: RefLikeExpression): String = expr match {
    case Reference(name, _, _, _)    => name
    case SubField(expr, name, _, _)  => serialize(expr.asInstanceOf[RefLikeExpression]) + "." + name
    case SubIndex(expr, index, _, _) => serialize(expr.asInstanceOf[RefLikeExpression]) + "[" + index.toString + "]"
    case a: SubAccess =>
      throw new RuntimeException(s"LowerTypes expects all SubAccesses to have been expanded! ${a.serialize}")
  }
}

/** Calculate new type layouts and names. */
private object DestructTypes {
  type Namespace = mutable.HashSet[String]

  /** Does the following with a reference:
    * - rename reference and any bundle fields to avoid name collisions after destruction
    * - updates rename map with new targets
    * - generates all ground type fields
    * - generates a list of all old reference name that now refer to the particular ground type field
    * - updates namespace with all possibly conflicting names
    */
  def destruct(
    m:         ModuleTarget,
    ref:       Field,
    namespace: Namespace,
    renameMap: RenameMap,
    reserved:  Set[String]
  ): Seq[(Field, String)] = {
    // field renames (uniquify) are computed bottom up
    val (rename, _) = uniquify(ref, namespace, reserved)

    // early exit for ground types that do not need renaming
    if (ref.tpe.isInstanceOf[GroundType] && rename.isEmpty) {
      return List((ref, ref.name))
    }

    // the reference renames are computed top down since they do need the full path
    val res = destruct(m, ref, rename)
    recordRenames(res, renameMap, ModuleParentRef(m))

    res.map { case (c, r) => c -> extractGroundTypeRefString(r) }
  }

  /** instances are special because they remain a 1-deep bundle
    * @note this relies on the ports of the module having been properly renamed.
    * @return The potentially renamed instance with newly flattened type.
    *         Note that the list of fields is only of the child fields, and needs a SubField node
    *         instead of a flat Reference when turning them into access expressions.
    */
  def destructInstance(
    m:           ModuleTarget,
    instance:    DefInstance,
    namespace:   Namespace,
    instRenames: mutable.ListBuffer[(Instance, Instance)],
    reserved:    Set[String]
  ): (DefInstance, Seq[(String, SubField)]) = {
    val (rename, _) = uniquify(Field(instance.name, Default, instance.tpe), namespace, reserved)
    val newName = rename.map(_.name).getOrElse(instance.name)

    // only destruct the sub-fields (aka ports)
    val oldParent = RefParentRef(m.ref(instance.name))
    val children = instance.tpe.asInstanceOf[BundleType].fields.flatMap { f =>
      val childRename = rename.flatMap(_.children.get(f.name))
      destruct("", oldParent, f, isVecField = false, rename = childRename)
    }

    // rename all references to the instance if necessary
    if (newName != instance.name) {
      instRenames += Instance(instance.name) -> Instance(newName)
    }
    // The ports do not need to be explicitly renamed here. They are renamed when the module ports are lowered.

    val newInstance = instance.copy(name = newName, tpe = BundleType(children.map(_._1)))
    val instanceRef = Reference(newName, newInstance.tpe, InstanceKind)
    val refs = children.map { case (c, r) => extractGroundTypeRefString(r) -> SubField(instanceRef, c.name, c.tpe) }

    (newInstance, refs)
  }

  private val BoolType = UIntType(IntWidth(1))

  /** memories are special because they end up a 2-deep bundle.
    * @note That a single old ground type reference could be replaced with multiple new ground type reference.
    *       e.g. ("mem_a.r.clk", "mem.r.clk") and ("mem_b.r.clk", "mem.r.clk")
    *       Thus it is appropriate to groupBy old reference string instead of just inserting into a hash table.
    */
  def destructMemory(
    m:         ModuleTarget,
    mem:       DefMemory,
    namespace: Namespace,
    renameMap: RenameMap,
    reserved:  Set[String]
  ): (Seq[DefMemory], Seq[(String, SubField)]) = {
    // Uniquify the lowered memory names: When memories get split up into ground types, the access order is changes.
    // E.g. `mem.r.data.x` becomes `mem_x.r.data`.
    // This is why we need to create the new bundle structure before we can resolve any name clashes.
    val bundle = memBundle(mem)
    val (dataTypeRenames, _) = uniquify(bundle, namespace, reserved)
    val res = destruct(m, Field(mem.name, Default, mem.dataType), dataTypeRenames)

    // Renames are now of the form `mem.a.b` --> `mem_a_b`.
    // We want to turn them into `mem.r.data.a.b` --> `mem_a_b.r.data`, etc. (for all readers, writers and for all ports)
    val oldMemRef = m.ref(mem.name)

    // the "old dummy field" is used as a template for the new memory port types
    val oldDummyField = Field("dummy", Default, MemPortUtils.memType(mem.copy(dataType = BoolType)))

    val newMemAndSubFields = res.map {
      case (field, refs) =>
        val newMem = mem.copy(name = field.name, dataType = field.tpe)
        val newMemRef = m.ref(field.name)
        val memWasRenamed = field.name != mem.name // false iff the dataType was a GroundType
        if (memWasRenamed) { renameMap.record(oldMemRef, newMemRef) }

        val newMemReference = Reference(field.name, MemPortUtils.memType(newMem), MemKind)
        val refSuffixes = refs.map(_.component).filterNot(_.isEmpty)

        val subFields = oldDummyField.tpe.asInstanceOf[BundleType].fields.flatMap { port =>
          val oldPortRef = oldMemRef.field(port.name)
          val newPortRef = newMemRef.field(port.name)

          val newPortType = newMemReference.tpe.asInstanceOf[BundleType].fields.find(_.name == port.name).get.tpe
          val newPortAccess = SubField(newMemReference, port.name, newPortType)

          port.tpe.asInstanceOf[BundleType].fields.map { portField =>
            val isDataField = portField.name == "data" || portField.name == "wdata" || portField.name == "rdata"
            val isMaskField = portField.name == "mask" || portField.name == "wmask"
            val isDataOrMaskField = isDataField || isMaskField
            val oldFieldRefs = if (memWasRenamed && isDataOrMaskField) {
              // there might have been multiple different fields which now alias to the same lowered field.
              val oldPortFieldBaseRef = oldPortRef.field(portField.name)
              refSuffixes.map(s => oldPortFieldBaseRef.copy(component = oldPortFieldBaseRef.component ++ s))
            } else {
              List(oldPortRef.field(portField.name))
            }

            val newPortType = if (isDataField) { newMem.dataType }
            else { portField.tpe }
            val newPortFieldAccess = SubField(newPortAccess, portField.name, newPortType)

            // record renames only for the data field which is the only port field of non-ground type
            val newPortFieldRef = newPortRef.field(portField.name)
            if (memWasRenamed && isDataOrMaskField) {
              oldFieldRefs.foreach { o => renameMap.record(o, newPortFieldRef) }
            }

            val oldFieldStringRef = extractGroundTypeRefString(oldFieldRefs)
            (oldFieldStringRef, newPortFieldAccess)
          }
        }
        (newMem, subFields)
    }

    (newMemAndSubFields.map(_._1), newMemAndSubFields.flatMap(_._2))
  }

  private def memBundle(mem: DefMemory): Field = mem.dataType match {
    case _: GroundType => Field(mem.name, Default, mem.dataType)
    case _: BundleType | _: VectorType =>
      val subMems = getFields(mem.dataType).map(f => mem.copy(name = f.name, dataType = f.tpe))
      val fields = subMems.map(memBundle)
      Field(mem.name, Default, BundleType(fields))
  }

  private def recordRenames(
    fieldToRefs: Seq[(Field, Seq[ReferenceTarget])],
    renameMap:   RenameMap,
    parent:      ParentRef
  ): Unit = {
    // TODO: if we group by ReferenceTarget, we could reduce the number of calls to `record`. Is it worth it?
    fieldToRefs.foreach {
      case (field, refs) =>
        val fieldRef = parent.ref(field.name)
        refs.foreach { r => renameMap.record(r, fieldRef) }
    }
  }

  private def extractGroundTypeRefString(refs: Seq[ReferenceTarget]): String = {
    if (refs.isEmpty) { "" }
    else {
      // Since we depend on ExpandConnects any reference we encounter will be of ground type
      // and thus the one with the longest access path.
      refs
        .reduceLeft((x, y) => if (x.component.length > y.component.length) x else y)
        // convert references to strings relative to the module
        .serialize
        .dropWhile(_ != '>')
        .tail
    }
  }

  private def destruct(m: ModuleTarget, field: Field, rename: Option[RenameNode]): Seq[(Field, Seq[ReferenceTarget])] =
    destruct(prefix = "", oldParent = ModuleParentRef(m), oldField = field, isVecField = false, rename = rename)

  /** Lowers a field into its ground type fields.
    * @param prefix carries the prefix of the new ground type name
    * @param isVecField is used to generate an appropriate old (field/index) reference
    * @param rename The information from the `uniquify` function is consumed to appropriately rename generated fields.
    * @return a sequence of ground type fields with new names and, for each field,
    *         a sequence of old references that should to be renamed to point to the particular field
    */
  private def destruct(
    prefix:     String,
    oldParent:  ParentRef,
    oldField:   Field,
    isVecField: Boolean,
    rename:     Option[RenameNode]
  ): Seq[(Field, Seq[ReferenceTarget])] = {
    val newName = rename.map(_.name).getOrElse(oldField.name)
    val oldRef = oldParent.ref(oldField.name, isVecField)

    oldField.tpe match {
      case _: GroundType => List((oldField.copy(name = prefix + newName), List(oldRef)))
      case _: BundleType | _: VectorType =>
        val newPrefix = prefix + newName + LowerTypes.delim
        val isVecField = oldField.tpe.isInstanceOf[VectorType]
        val fields = getFields(oldField.tpe)
        val fieldsWithCorrectOrientation = fields.map(f => f.copy(flip = Utils.times(f.flip, oldField.flip)))
        val children = fieldsWithCorrectOrientation.flatMap { f =>
          destruct(newPrefix, RefParentRef(oldRef), f, isVecField, rename.flatMap(_.children.get(f.name)))
        }
        // the bundle/vec reference refers to all children
        children.map { case (c, r) => (c, r :+ oldRef) }
    }
  }

  private case class RenameNode(name: String, children: Map[String, RenameNode])

  /** Implements the core functionality of the old Uniquify pass: rename bundle fields and top-level references
    * where necessary in order to avoid name clashes when lowering aggregate type with the `_` delimiter.
    * We don't actually do the rename here but just calculate a rename tree.
    */
  private def uniquify(ref: Field, namespace: Namespace, reserved: Set[String]): (Option[RenameNode], Seq[String]) = {
    // ensure that there are no name clashes with the list of reserved (port) names
    val newRefName = findValidPrefix(ref.name, reserved.contains)
    ref.tpe match {
      case BundleType(fields) =>
        // we rename bottom-up
        val localNamespace = new Namespace() ++ fields.map(_.name)
        val renamedFields = fields.map(f => uniquify(f, localNamespace, Set()))

        // Need leading _ for findValidPrefix, it doesn't add _ for checks
        val renamedFieldNames = renamedFields.flatMap(_._2)
        val suffixNames: Seq[String] = renamedFieldNames.map(f => LowerTypes.delim + f)
        val prefix = findValidPrefix(newRefName, namespace.contains, suffixNames)
        // We added f.name in previous map, delete if we change it
        val renamed = prefix != ref.name
        if (renamed) {
          if (!reserved.contains(ref.name)) namespace -= ref.name
          namespace += prefix
        }
        val suffixes = renamedFieldNames.map(f => prefix + LowerTypes.delim + f)

        val anyChildRenamed = renamedFields.exists(_._1.isDefined)
        val rename = if (renamed || anyChildRenamed) {
          val children = renamedFields.map(_._1).zip(fields).collect { case (Some(r), f) => f.name -> r }.toMap
          Some(RenameNode(prefix, children))
        } else { None }

        (rename, suffixes :+ prefix)
      case v: VectorType =>
        // if Vecs are to be lowered, we can just treat them like a bundle
        uniquify(ref.copy(tpe = vecToBundle(v)), namespace, reserved)
      case _: GroundType =>
        if (newRefName == ref.name) {
          (None, List(ref.name))
        } else {
          (Some(RenameNode(newRefName, Map())), List(newRefName))
        }
      case UnknownType => throw new RuntimeException(s"Cannot uniquify field of unknown type: $ref")
    }
  }

  /** Appends delim to prefix until no collisions of prefix + elts in names We don't add an _ in the collision check
    * because elts could be Seq("") In this case, we're just really checking if prefix itself collides
    */
  @tailrec
  private def findValidPrefix(prefix: String, inNamespace: String => Boolean, elts: Seq[String] = List("")): String = {
    elts.find(elt => inNamespace(prefix + elt)) match {
      case Some(_) => findValidPrefix(prefix + "_", inNamespace, elts)
      case None    => prefix
    }
  }

  private def getFields(tpe: Type): Seq[Field] = tpe match {
    case BundleType(fields) => fields
    case v: VectorType => vecToBundle(v).fields
  }

  private def vecToBundle(v: VectorType): BundleType = {
    BundleType((0 until v.size).map(i => Field(i.toString, Default, v.tpe)))
  }

  /** Used to abstract over module and reference parents.
    * This helps us simplify the `destruct` method as it does not need to distinguish between
    * a module (in the initial call) or a bundle/vector (in the recursive call) reference as parent.
    */
  private trait ParentRef { def ref(name: String, asVecField: Boolean = false): ReferenceTarget }
  private case class ModuleParentRef(m: ModuleTarget) extends ParentRef {
    override def ref(name: String, asVecField: Boolean): ReferenceTarget = m.ref(name)
  }
  private case class RefParentRef(r: ReferenceTarget) extends ParentRef {
    override def ref(name: String, asVecField: Boolean): ReferenceTarget =
      if (asVecField) { r.index(name.toInt) }
      else { r.field(name) }
  }
}
