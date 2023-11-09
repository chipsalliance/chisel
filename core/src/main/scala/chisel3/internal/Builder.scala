// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.util.DynamicVariable
import scala.collection.mutable.ArrayBuffer
import chisel3._
import chisel3.experimental._
import chisel3.experimental.hierarchy.core.{Clone, ImportDefinitionAnnotation, Instance}
import chisel3.properties.Class
import chisel3.internal.firrtl._
import chisel3.internal.naming._
import _root_.firrtl.annotations.{CircuitName, ComponentName, IsMember, ModuleName, Named, ReferenceTarget}
import _root_.firrtl.annotations.AnnotationUtils.validComponentName
import _root_.firrtl.AnnotationSeq
import _root_.firrtl.renamemap.MutableRenameMap
import _root_.firrtl.util.BackendCompilationUtilities._
import _root_.firrtl.{ir => fir}
import chisel3.experimental.dataview.{reify, reifySingleData}
import chisel3.internal.Builder.Prefix
import logger.LazyLogging

import scala.collection.mutable
import scala.annotation.tailrec
import java.io.File
import scala.util.control.NonFatal

private[chisel3] class Namespace(keywords: Set[String], separator: Char = '_') {
  // This HashMap is compressed, not every name in the namespace is present here.
  // If the same name is requested multiple times, it only takes 1 entry in the HashMap and the
  // value is incremented for each time the name is requested.
  // Names can be requested that collide with compressed sets of names, thus the algorithm for
  // checking if a name is present in the Namespace is more complex than just checking the HashMap,
  // see getIndex below.
  private val names = collection.mutable.HashMap[String, Long]()
  def copyTo(other: Namespace): Unit = names.foreach { case (s: String, l: Long) => other.names(s) = l }
  for (keyword <- keywords)
    names(keyword) = 1

  @tailrec
  private def rename(n: String, index: Long): String = {
    val tryName = s"${n}${separator}${index}"
    if (names.contains(tryName)) {
      rename(n, index + 1)
    } else {
      names(n) = index + 1
      tryName
    }
  }

  /** Checks if `n` ends in `_\d+` and returns the substring before `_` if so, null otherwise */
  // TODO can and should this be folded in to sanitize? Same iteration as the forall?
  private def prefix(n: String): Int = {
    // This is micro-optimized because it runs on every single name
    var i = n.size - 1
    while (i > 0 && n(i).isDigit) {
      i -= 1
    }
    // Will get i == 0 for all digits or _\d+ with empty prefix, those have no prefix so returning 0 is correct
    if (i >= (n.size - 1)) 0 // no digits
    else if (n(i) != separator) 0 // no _
    else i
  }

  // Gets the current index for this name, None means it is not contained in the Namespace
  private def getIndex(elem: String): Option[Long] =
    names.get(elem).orElse {
      // This exact name isn't contained, but if we end in _<idx>, we need to check our prefix
      val maybePrefix = prefix(elem)
      if (maybePrefix == 0) None
      else {
        // If we get a prefix collision and our index is taken, we start disambiguating with _<idx>_1
        names
          .get(elem.take(maybePrefix))
          .filter { prefixIdx =>
            val ourIdx = elem.drop(maybePrefix + 1).toInt
            // The namespace starts disambiguating at _1 so _0 is a false collision case
            ourIdx != 0 && prefixIdx > ourIdx
          }
          .map(_ => 1)
      }
    }

  def contains(elem: String): Boolean = getIndex(elem).isDefined

  // leadingDigitOk is for use in fields of Records
  def name(elem: String, leadingDigitOk: Boolean = false): String = {
    val sanitized = sanitize(elem, leadingDigitOk)
    getIndex(sanitized) match {
      case Some(idx) => rename(sanitized, idx)
      case None =>
        names(sanitized) = 1
        sanitized
    }
  }
}

private[chisel3] object Namespace {

  /** Constructs an empty Namespace */
  def empty(separator: Char): Namespace = new Namespace(Set.empty[String], separator)
  def empty: Namespace = new Namespace(Set.empty[String])
}

private[chisel3] class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
  def value: Long = counter
}

private[chisel3] trait HasId extends chisel3.InstanceId {
  // using nullable var for better memory usage
  private var _parentVar:       BaseModule = Builder.currentModule.getOrElse(null)
  private[chisel3] def _parent: Option[BaseModule] = Option(_parentVar)
  private[chisel3] def _parent_=(target: Option[BaseModule]): Unit = {
    _parentVar = target.getOrElse(null)
  }

  // Set if the returned top-level module of a nested call to the Chisel Builder, see Definition.apply
  private var _circuitVar:       BaseModule = null // using nullable var for better memory usage
  private[chisel3] def _circuit: Option[BaseModule] = Option(_circuitVar)
  private[chisel3] def _circuit_=(target: Option[BaseModule]): Unit = {
    _circuitVar = target.getOrElse(null)
  }

  private[chisel3] val _id: Long = Builder.idGen.next

  // TODO: remove this, but its removal seems to cause a nasty Scala compiler crash.
  override def hashCode: Int = super.hashCode()
  override def equals(that: Any): Boolean = super.equals(that)

  // Contains suggested seed (user-decided seed)
  private var suggested_seedVar: String = null // using nullable var for better memory usage
  private def suggested_seed:    Option[String] = Option(suggested_seedVar)

  // Contains the seed computed automatically by the compiler plugin
  private var auto_seedVar: String = null // using nullable var for better memory usage
  private def auto_seed:    Option[String] = Option(auto_seedVar)

  // Prefix for use in naming
  // - Defaults to prefix at time when object is created
  // - Overridden when [[suggestSeed]] or [[autoSeed]] is called
  private var naming_prefix: Prefix = Builder.getPrefix

  /** Takes the last seed suggested. Multiple calls to this function will take the last given seed, unless
    * this HasId is a module port (see overridden method in Data.scala).
    *
    * If the final computed name conflicts with the final name of another signal, the final name may get uniquified by
    * appending a digit at the end of the name.
    *
    * Is a lower priority than [[suggestName]], in that regardless of whether [[autoSeed]]
    * was called, [[suggestName]] will always take precedence if it was called.
    *
    * @param seed Seed for the name of this component
    * @return this object
    */
  private[chisel3] def autoSeed(seed: String): this.type = forceAutoSeed(seed)
  // Bypass the overridden behavior of autoSeed in [[Data]], apply autoSeed even to ports
  private[chisel3] def forceAutoSeed(seed: String): this.type = {
    auto_seedVar = seed
    naming_prefix = Builder.getPrefix
    this
  }

  // Private internal version of suggestName that tells you if the name changed
  // Returns Some(old name, old prefix) if name changed, None otherwise
  private[chisel3] def _suggestNameCheck(seed: => String): Option[(String, Prefix)] = {
    val oldSeed = this.seedOpt
    val oldPrefix = this.naming_prefix
    suggestName(seed)
    if (oldSeed.nonEmpty && (oldSeed != this.seedOpt || oldPrefix != this.naming_prefix)) {
      Some(oldSeed.get -> oldPrefix)
    } else None
  }

  /** Takes the first seed suggested. Multiple calls to this function will be ignored.
    * If the final computed name conflicts with another name, it may get uniquified by appending
    * a digit at the end.
    *
    * Is a higher priority than `autoSeed`, in that regardless of whether `autoSeed`
    * was called, [[suggestName]] will always take precedence.
    *
    * @param seed The seed for the name of this component
    * @return this object
    */
  def suggestName(seed: => String): this.type = {
    if (suggested_seed.isEmpty) {
      suggested_seedVar = seed
      // Only set the prefix if a seed hasn't been suggested
      naming_prefix = Builder.getPrefix
    }
    this
  }

  /** Computes the name of this HasId, if one exists
    * @param defaultSeed Optionally provide default seed for computing the name
    * @return the name, if it can be computed
    */
  private[chisel3] def _computeName(defaultSeed: Option[String]): Option[String] = {
    seedOpt
      .orElse(defaultSeed)
      .map(name => buildName(name, naming_prefix.reverse))
  }

  /** This resolves the precedence of [[autoSeed]] and [[suggestName]]
    *
    * @return the current calculation of a name, if it exists
    */
  private[chisel3] def seedOpt: Option[String] = suggested_seed.orElse(auto_seed)

  /** @return Whether either autoName or suggestName has been called */
  def hasSeed: Boolean = seedOpt.isDefined

  private[chisel3] def hasAutoSeed: Boolean = auto_seed.isDefined

  // Uses a namespace to convert suggestion into a true name
  // Will not do any naming if the reference already assigned.
  // (e.g. tried to suggest a name to part of a Record)
  private[chisel3] def forceName(
    default:    => String,
    namespace:  Namespace,
    errorIfDup: Boolean = false,
    refBuilder: String => Arg = Ref(_)
  ): Unit =
    if (_ref.isEmpty) {
      val candidate_name = _computeName(Some(default).filterNot(_ => errorIfDup)).getOrElse {
        throwException(
          s"Attempted to name a nameless IO port ($this): this is usually caused by instantiating an IO but not assigning it to a val.\n" +
            s"Assign $this to a val, or explicitly call suggestName to seed a unique name"
        )
      }

      val sanitized = sanitize(candidate_name)
      val available_name = namespace.name(candidate_name)

      // Check for both cases of name duplication
      if (errorIfDup && (available_name != sanitized)) {
        // If sanitization occurred, then the sanitized name duplicate an existing name
        if ((candidate_name != sanitized)) {
          Builder.error(
            s"Attempted to name $this with an unsanitary name '$candidate_name': sanitization results in a duplicated name '$sanitized'. Please seed a more unique name"
          )(UnlocatableSourceInfo)
        } else {
          // Otherwise the candidate name duplicates an existing name
          Builder.error(
            s"Attempted to name $this with a duplicated name '$candidate_name'. Use suggestName to seed a unique name"
          )(UnlocatableSourceInfo)
        }
      }

      setRef(refBuilder(available_name))
      // Clear naming prefix to free memory
      naming_prefix = Nil
    }

  private var _refVar: Arg = null // using nullable var for better memory usage
  private def _ref:    Option[Arg] = Option(_refVar)
  private[chisel3] def setRef(imm: Arg): Unit = setRef(imm, false)
  private[chisel3] def setRef(imm: Arg, force: Boolean): Unit = {
    if (_ref.isEmpty || force) {
      _refVar = imm
    }
  }
  private[chisel3] def setRef(parent: HasId, name: String, opaque: Boolean = false): Unit = {
    if (!opaque) setRef(Slot(Node(parent), name))
    else setRef(OpaqueSlot(Node(parent)))
  }

  private[chisel3] def setRef(parent: HasId, index: Int):  Unit = setRef(Index(Node(parent), ILit(index)))
  private[chisel3] def setRef(parent: HasId, index: UInt): Unit = setRef(Index(Node(parent), index.ref))
  private[chisel3] def getRef:       Arg = _ref.get
  private[chisel3] def getOptionRef: Option[Arg] = _ref

  private def refName(c: Component): String = _ref match {
    case Some(arg) => arg.fullName(c)
    case None => {
      throwException(
        "You cannot access the .instanceName or .toTarget of non-hardware Data" + _errorContext
      )
    }
  }

  private[chisel3] def _errorContext: String = {
    val nameGuess: String = _computeName(None) match {
      case Some(name) => s": '$name'"
      case None       => ""
    }

    val parentGuess: String = _parent match {
      case Some(ViewParent) => s", in module '${reifyParent.pathName}'"
      case Some(p)          => s", in module '${p.pathName}'"
      case None             => ""
    }

    nameGuess + parentGuess
  }

  // Helper for reifying views if they map to a single Target
  private[chisel3] def reifyTarget: Option[Data] = this match {
    case d: Data => reifySingleData(d) // Only Data can be views
    case bad => throwException(s"This shouldn't be possible - got $bad with ${_parent}")
  }

  // Helper for reifying the parent of a view if the view maps to a single Target
  private[chisel3] def reifyParent: BaseModule = reifyTarget.flatMap(_._parent).getOrElse(ViewParent)

  // Implementation of public methods.
  def instanceName: String = _parent match {
    case Some(ViewParent) => reifyTarget.map(_.instanceName).getOrElse(this.refName(ViewParent.fakeComponent))
    case Some(p) =>
      (p._component, this) match {
        case (Some(c), _) => refName(c)
        case (None, d: Data) if d.topBindingOpt == Some(CrossModuleBinding) => _ref.get.localName
        case (None, _: MemBase[_]) => _ref.get.localName
        case (None, _) if _ref.isDefined => {
          // Support instance names for HasIds that don't have a _parent set yet, but do have a _ref set.
          // This allows HasIds to be named in atModuleBodyEnd, for example.
          // In this case, we directly use the localName. This is valid, because the only time names are
          // context-dependent is on ports. If a port doesn't have a _parent set yet, the port must be within
          // the currently elaborating module, and should be named by its localName.
          _ref.get.localName
        }
        case (None, _) =>
          throwException(s"signalName/pathName should be called after circuit elaboration: $this, ${_parent}")
      }
    case None => throwException("this cannot happen")
  }
  def pathName: String = _parent match {
    case None             => instanceName
    case Some(ViewParent) => s"${reifyParent.pathName}.$instanceName"
    case Some(p)          => s"${p.pathName}.$instanceName"
  }
  def parentPathName: String = _parent match {
    case Some(ViewParent) => reifyParent.pathName
    case Some(p)          => p.pathName
    case None             => throwException(s"$instanceName doesn't have a parent")
  }
  def parentModName: String = _parent match {
    case Some(ViewParent) => reifyParent.name
    case Some(p)          => p.name
    case None             => throwException(s"$instanceName doesn't have a parent")
  }
  // TODO Should this be public?
  protected def circuitName: String = _parent match {
    case None =>
      _circuit match {
        case None    => instanceName
        case Some(o) => o.circuitName
      }
    case Some(ViewParent) => reifyParent.circuitName
    case Some(p)          => p.circuitName
  }
}

/** Holds the implementation of toNamed for Data and MemBase */
private[chisel3] trait NamedComponent extends HasId {

  /** Returns a FIRRTL ComponentName that references this object
    * @note Should not be called until circuit elaboration is complete
    */
  final def toNamed: ComponentName = {
    assertValidTarget()
    ComponentName(this.instanceName, ModuleName(this.parentModName, CircuitName(this.circuitName)))
  }

  /** Returns a FIRRTL ReferenceTarget that references this object
    * @note Should not be called until circuit elaboration is complete
    */
  final def toTarget: ReferenceTarget = {
    assertValidTarget()
    val name = this.instanceName
    if (!validComponentName(name)) throwException(s"Illegal component name: $name (note: literals are illegal)")
    import _root_.firrtl.annotations.{Target, TargetToken}
    val root = _parent.map {
      case ViewParent => reifyParent
      case other      => other
    }.get.getTarget // All NamedComponents will have a parent, only the top module can have None here
    Target.toTargetTokens(name).toList match {
      case TargetToken.Ref(r) :: components => root.ref(r).copy(component = components)
      case other =>
        throw _root_.firrtl.annotations.Target.NamedException(s"Cannot convert $name into [[ReferenceTarget]]: $other")
    }
  }

  final def toAbsoluteTarget: ReferenceTarget = {
    val localTarget = toTarget
    def makeTarget(p: BaseModule) = p.toAbsoluteTarget.ref(localTarget.ref).copy(component = localTarget.component)
    _parent match {
      case Some(ViewParent) => makeTarget(reifyParent)
      case Some(parent)     => makeTarget(parent)
      case None             => localTarget
    }
  }

  /** Returns a FIRRTL ReferenceTarget that references this object, relative to an optional root.
    *
    * If `root` is defined, the target is a hierarchical path starting from `root`.
    *
    * If `root` is not defined, the target is a hierarchical path equivalent to `toAbsoluteTarget`.
    *
    * @note If `root` is defined, and has not finished elaboration, this must be called within `atModuleBodyEnd`.
    * @note The NamedComponent must be a descendant of `root`, if it is defined.
    * @note This doesn't have special handling for Views.
    */
  final def toRelativeTarget(root: Option[BaseModule]): ReferenceTarget = {
    val localTarget = toTarget
    def makeTarget(p: BaseModule) =
      p.toRelativeTarget(root).ref(localTarget.ref).copy(component = localTarget.component)
    _parent match {
      case Some(ViewParent) => makeTarget(reifyParent)
      case Some(parent)     => makeTarget(parent)
      case None             => localTarget
    }
  }

  private def assertValidTarget(): Unit = {
    val isVecSubaccess = getOptionRef.map {
      case Index(_, _: ULit) => true // Vec literal indexing
      case Index(_, _: Node) => true // Vec dynamic indexing
      case _ => false
    }.getOrElse(false)

    if (isVecSubaccess) {
      throwException(
        s"You cannot target Vec subaccess" + _errorContext +
          ". Instead, assign it to a temporary (for example, with WireInit) and target the temporary."
      )
    }
  }
}

// Mutable global state for chisel that can appear outside a Builder context
private[chisel3] class ChiselContext() {
  val idGen = new IdGen

  // Records the different prefixes which have been scoped at this point in time
  var prefixStack: Prefix = Nil

  // Views belong to a separate namespace (for renaming)
  // The namespace outside of Builder context is useless, but it ensures that views can still be created
  // and the resulting .toTarget is very clearly useless (_$$View$$_...)
  val viewNamespace = Namespace.empty
}

private[chisel3] class DynamicContext(
  val annotationSeq:     AnnotationSeq,
  val throwOnFirstError: Boolean,
  val warningFilters:    Seq[WarningFilter],
  val sourceRoots:       Seq[File]) {
  val importedDefinitionAnnos = annotationSeq.collect { case a: ImportDefinitionAnnotation[_] => a }

  // Map from proto module name to ext-module name
  // Pick the definition name by default in case not overridden
  // 1. Ensure there are no repeated names for imported Definitions - both Proto Names as well as ExtMod Names
  // 2. Return the distinct definition / extMod names
  val importedDefinitionMap = importedDefinitionAnnos
    .map(a => a.definition.proto.name -> a.overrideDefName.getOrElse(a.definition.proto.name))
    .toMap

  private def checkAndGetDistinctProtoExtModNames() = {
    val allProtoNames = importedDefinitionAnnos.map { a => a.definition.proto.name }
    val distinctProtoNames = importedDefinitionMap.keys.toSeq
    val allExtModNames = importedDefinitionMap.toSeq.map(_._2)
    val distinctExtModNames = allExtModNames.distinct

    if (distinctProtoNames.length < allProtoNames.length) {
      val duplicates = allProtoNames.diff(distinctProtoNames).mkString(", ")
      throwException(s"Expected distinct imported Definition names but found duplicates for: $duplicates")
    }
    if (distinctExtModNames.length < allExtModNames.length) {
      val duplicates = allExtModNames.diff(distinctExtModNames).mkString(", ")
      throwException(s"Expected distinct overrideDef names but found duplicates for: $duplicates")
    }
    (
      (allProtoNames ++ allExtModNames).distinct,
      importedDefinitionAnnos.map(a => a.definition.proto.definitionIdentifier)
    )
  }

  val globalNamespace = Namespace.empty
  val globalIdentifierNamespace = Namespace.empty('$')

  // A mapping from previously named bundles to their hashed structural/FIRRTL types, for
  // disambiguation purposes when emitting type aliases
  // Records are used as the key for this map to both represent their alias name and preserve
  // the chisel Bundle structure when passing everything off to the Converter
  private[chisel3] val aliasMap: mutable.LinkedHashMap[String, (fir.Type, SourceInfo)] =
    mutable.LinkedHashMap.empty[String, (fir.Type, SourceInfo)]

  // Ensure imported Definitions emit as ExtModules with the correct name so
  // that instantiations will also use the correct name and prevent any name
  // conflicts with Modules/Definitions in this elaboration
  checkAndGetDistinctProtoExtModNames() match {
    case (names, identifiers) =>
      names.foreach(globalNamespace.name(_))
      identifiers.foreach(globalIdentifierNamespace.name(_))
  }

  val components = ArrayBuffer[Component]()
  val annotations = ArrayBuffer[ChiselAnnotation]()
  val newAnnotations = ArrayBuffer[ChiselMultiAnnotation]()
  val groups = mutable.LinkedHashSet[group.Declaration]()
  var currentModule: Option[BaseModule] = None

  /** Contains a mapping from a elaborated module to their aspect
    * Set by [[ModuleAspect]]
    */
  val aspectModule: mutable.HashMap[BaseModule, BaseModule] = mutable.HashMap.empty[BaseModule, BaseModule]

  // Views that do not correspond to a single ReferenceTarget and thus require renaming
  val unnamedViews: ArrayBuffer[Data] = ArrayBuffer.empty

  val contextCache: BuilderContextCache = BuilderContextCache.empty

  // Set by object Module.apply before calling class Module constructor
  // Used to distinguish between no Module() wrapping, multiple wrappings, and rewrapping
  var readyForModuleConstr: Boolean = false
  var whenStack:            List[WhenContext] = Nil
  var currentClock:         Option[Clock] = None
  var currentReset:         Option[Reset] = None
  var currentDisable:       Disable.Type = Disable.BeforeReset
  var groupStack:           List[group.Declaration] = group.Declaration.rootDeclaration :: Nil
  val errors = new ErrorLog(warningFilters, sourceRoots, throwOnFirstError)
  val namingStack = new NamingStack

  // Used to indicate if this is the top-level module of full elaboration, or from a Definition
  var inDefinition: Boolean = false
}

private[chisel3] object Builder extends LazyLogging {

  // Represents the current state of the prefixes given
  type Prefix = List[String]

  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)
  private def dynamicContext: DynamicContext = {
    require(dynamicContextVar.value.isDefined, "must be inside Builder context")
    dynamicContextVar.value.get
  }

  // Used to suppress warnings when casting from a UInt to an Enum
  var suppressEnumCastWarning: Boolean = false

  // Returns the current dynamic context
  def captureContext(): DynamicContext = dynamicContext
  // Sets the current dynamic contents
  def restoreContext(dc: DynamicContext) = dynamicContextVar.value = Some(dc)

  // Ensure we have a thread-specific ChiselContext
  private val chiselContext = new ThreadLocal[ChiselContext] {
    override def initialValue: ChiselContext = {
      new ChiselContext
    }
  }

  // Initialize any singleton objects before user code inadvertently inherits them.
  private def initializeSingletons(): Unit = {
    // This used to contain:
    //    val dummy = core.DontCare
    //  but this would occasionally produce hangs due to static initialization deadlock
    //  when Builder initialization collided with chisel3.package initialization of the DontCare value.
    // See:
    //  http://ternarysearch.blogspot.com/2013/07/static-initialization-deadlock.html
    //  https://bugs.openjdk.java.net/browse/JDK-8037567
    //  https://stackoverflow.com/questions/28631656/runnable-thread-state-but-in-object-wait
  }

  def namingStackOption: Option[NamingStack] = dynamicContextVar.value.map(_.namingStack)

  def idGen: IdGen = chiselContext.get.idGen

  def globalNamespace:           Namespace = dynamicContext.globalNamespace
  def globalIdentifierNamespace: Namespace = dynamicContext.globalIdentifierNamespace

  def aliasMap: mutable.LinkedHashMap[String, (fir.Type, SourceInfo)] =
    dynamicContext.aliasMap

  def components:  ArrayBuffer[Component] = dynamicContext.components
  def annotations: ArrayBuffer[ChiselAnnotation] = dynamicContext.annotations

  def groups: mutable.LinkedHashSet[group.Declaration] = dynamicContext.groups

  def contextCache: BuilderContextCache = dynamicContext.contextCache

  // TODO : Unify this with annotations in the future - done this way for backward compatability
  def newAnnotations: ArrayBuffer[ChiselMultiAnnotation] = dynamicContext.newAnnotations

  def annotationSeq:         AnnotationSeq = dynamicContext.annotationSeq
  def namingStack:           NamingStack = dynamicContext.namingStack
  def importedDefinitionMap: Map[String, String] = dynamicContext.importedDefinitionMap

  def unnamedViews:  ArrayBuffer[Data] = dynamicContext.unnamedViews
  def viewNamespace: Namespace = chiselContext.get.viewNamespace

  // Puts a prefix string onto the prefix stack
  def pushPrefix(d: String): Unit = {
    val context = chiselContext.get()
    context.prefixStack = d :: context.prefixStack
  }

  /** Pushes the current name of a data onto the prefix stack
    *
    * @param d data to derive prefix from
    * @return whether anything was pushed to the stack
    */
  def pushPrefix(d: HasId): Boolean = {
    def buildAggName(id: HasId): Option[String] = {
      def getSubName(field: Data): Option[String] = field.getOptionRef.flatMap {
        case Slot(_, field)       => Some(field) // Record
        case OpaqueSlot(_)        => None // OpaqueSlots don't contribute to the name
        case Index(_, ILit(n))    => Some(n.toString) // Vec static indexing
        case Index(_, ULit(n, _)) => Some(n.toString) // Vec lit indexing
        case Index(_, _: Node) => None // Vec dynamic indexing
        case ModuleIO(_, n) => Some(n) // BlackBox port
        case f =>
          throw new InternalErrorException(s"Match Error: field=$f")
      }
      def map2[A, B](a: Option[A], b: Option[A])(f: (A, A) => B): Option[B] =
        a.flatMap(ax => b.map(f(ax, _)))
      // If the binding is None, this is an illegal connection and later logic will error
      def recData(data: Data): Option[String] = data.binding.flatMap {
        case (_: WireBinding | _: RegBinding | _: MemoryPortBinding | _: OpBinding) => data.seedOpt
        case ChildBinding(parent) =>
          recData(parent).map { p =>
            // And name of the field if we have one, we don't for dynamic indexing of Vecs
            getSubName(data).map(p + "_" + _).getOrElse(p)
          }
        case SampleElementBinding(parent)                            => recData(parent)
        case PortBinding(mod) if Builder.currentModule.contains(mod) => data.seedOpt
        case PortBinding(mod)                                        => map2(mod.seedOpt, data.seedOpt)(_ + "_" + _)
        case (_: LitBinding | _: DontCareBinding) => None
        case _ => Some("view_") // TODO implement
      }
      id match {
        case d: Data => recData(d)
        case _ => None
      }
    }
    buildAggName(d).map { name =>
      if (isTemp(name)) {
        pushPrefix(name.tail)
      } else {
        pushPrefix(name)
      }
    }.isDefined
  }

  // Remove a prefix from top of the stack
  def popPrefix(): List[String] = {
    val context = chiselContext.get()
    val tail = context.prefixStack.tail
    context.prefixStack = tail
    tail
  }

  // Removes all prefixes from the prefix stack
  def clearPrefix(): Unit = {
    chiselContext.get().prefixStack = Nil
  }

  // Clears existing prefixes and sets to new prefix stack
  def setPrefix(prefix: Prefix): Unit = {
    chiselContext.get().prefixStack = prefix
  }

  // Returns the prefix stack at this moment
  def getPrefix: Prefix = chiselContext.get().prefixStack

  def currentModule: Option[BaseModule] = dynamicContextVar.value match {
    case Some(dynamicContext) => dynamicContext.currentModule
    case _                    => None
  }
  def currentModule_=(target: Option[BaseModule]): Unit = {
    dynamicContext.currentModule = target
  }
  def aspectModule(module: BaseModule): Option[BaseModule] = dynamicContextVar.value match {
    case Some(dynamicContext) => dynamicContext.aspectModule.get(module)
    case _                    => None
  }

  /** Retrieves the parent of a module based on the elaboration context
    *
    * @param module the module to get the parent of
    * @param context the context the parent should be evaluated in
    * @return the parent of the module provided
    */
  def retrieveParent(module: BaseModule, context: BaseModule): Option[BaseModule] = {
    module._parent match {
      case Some(parentModule) => { //if a parent exists investigate, otherwise return None
        context match {
          case aspect: ModuleAspect => { //if aspect context, do the translation
            Builder.aspectModule(parentModule) match {
              case Some(parentAspect) => Some(parentAspect) //we've found a translation
              case _                  => Some(parentModule) //no translation found
            }
          } //otherwise just return our parent
          case _ => Some(parentModule)
        }
      }
      case _ => None
    }
  }
  def addAspect(module: BaseModule, aspect: BaseModule): Unit = {
    dynamicContext.aspectModule += ((module, aspect))
  }
  def forcedModule: BaseModule = currentModule match {
    case Some(module) => module
    case None =>
      throwException(
        "Error: Not in a Module. Likely cause: Missed Module() wrap or bare chisel API call."
        // A bare api call is, e.g. calling Wire() from the scala console).
      )
  }
  def referenceUserModule: RawModule = {
    currentModule match {
      case Some(module: RawModule) =>
        aspectModule(module) match {
          case Some(aspect: RawModule) => aspect
          case other => module
        }
      case _ =>
        throwException(
          "Error: Not in a RawModule. Likely cause: Missed Module() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox."
          // A bare api call is, e.g. calling Wire() from the scala console).
        )
    }
  }
  def referenceUserContainer: BaseModule = {
    currentModule match {
      case Some(module: RawModule) =>
        aspectModule(module) match {
          case Some(aspect: RawModule) => aspect
          case other => module
        }
      case Some(cls: Class) => cls
      case _ =>
        throwException(
          "Error: Not in a RawModule or Class. Likely cause: Missed Module() or Definition() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox."
          // A bare api call is, e.g. calling Wire() from the scala console).
        )
    }
  }
  def forcedUserModule: RawModule = currentModule match {
    case Some(module: RawModule) => module
    case _ =>
      throwException(
        "Error: Not in a UserModule. Likely cause: Missed Module() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox."
        // A bare api call is, e.g. calling Wire() from the scala console).
      )
  }
  def hasDynamicContext: Boolean = dynamicContextVar.value.isDefined

  def readyForModuleConstr: Boolean = dynamicContext.readyForModuleConstr
  def readyForModuleConstr_=(target: Boolean): Unit = {
    dynamicContext.readyForModuleConstr = target
  }

  def whenDepth: Int = dynamicContext.whenStack.length

  def pushWhen(wc: WhenContext): Unit = {
    dynamicContext.whenStack = wc :: dynamicContext.whenStack
  }

  def popWhen(): WhenContext = {
    val lastWhen = dynamicContext.whenStack.head
    dynamicContext.whenStack = dynamicContext.whenStack.tail
    lastWhen
  }

  def whenStack: List[WhenContext] = dynamicContext.whenStack
  def whenStack_=(s: List[WhenContext]): Unit = {
    dynamicContext.whenStack = s
  }

  def currentWhen: Option[WhenContext] = dynamicContext.whenStack.headOption

  def currentClock: Option[Clock] = dynamicContext.currentClock
  def currentClock_=(newClock: Option[Clock]): Unit = {
    dynamicContext.currentClock = newClock
  }

  def currentReset: Option[Reset] = dynamicContext.currentReset
  def currentReset_=(newReset: Option[Reset]): Unit = {
    dynamicContext.currentReset = newReset
  }

  def currentDisable: Disable.Type = dynamicContext.currentDisable
  def currentDisable_=(newDisable: Disable.Type): Unit = {
    dynamicContext.currentDisable = newDisable
  }

  def groupStack: List[group.Declaration] = dynamicContext.groupStack
  def groupStack_=(s: List[group.Declaration]): Unit = {
    dynamicContext.groupStack = s
  }

  def inDefinition: Boolean = {
    dynamicContextVar.value
      .map(_.inDefinition)
      .getOrElse(false)
  }

  def forcedClock: Clock = currentClock.getOrElse(
    throwException("Error: No implicit clock.")
  )
  def forcedReset: Reset = currentReset.getOrElse(
    throwException("Error: No implicit reset.")
  )

  // TODO(twigg): Ideally, binding checks and new bindings would all occur here
  // However, rest of frontend can't support this yet.
  def pushCommand[T <: Command](c: T): T = {
    forcedUserModule.addCommand(c)
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = {
    // Bind each element of the returned Data to being a Op
    cmd.id.bind(OpBinding(forcedUserModule, currentWhen))
    pushCommand(cmd).id
  }

  /** Recursively suggests names to supported "container" classes
    * Arbitrary nestings of supported classes are allowed so long as the
    * innermost element is of type HasId
    * (Note: Map is Iterable[Tuple2[_,_]] and thus excluded)
    */
  def nameRecursively(prefix: String, nameMe: Any, namer: (HasId, String) => Unit): Unit = nameMe match {
    case (id: Instance[_]) =>
      id.underlying match {
        case Clone(m: experimental.hierarchy.ModuleClone[_]) => namer(m.getPorts, prefix)
        case _ =>
      }
    case (id: HasId) => namer(id, prefix)
    case Some(elt) => nameRecursively(prefix, elt, namer)
    case (iter: Iterable[_]) if iter.hasDefiniteSize =>
      for ((elt, i) <- iter.zipWithIndex) {
        nameRecursively(s"${prefix}_${i}", elt, namer)
      }
    case product: Product =>
      product.productIterator.zip(product.productElementNames).foreach {
        case (elt, fullName) =>
          val name = fullName.stripPrefix("_")
          nameRecursively(s"${prefix}_${name}", elt, namer)
      }
    case disable: Disable => nameRecursively(prefix, disable.value, namer)
    case _ => // Do nothing
  }

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String)(implicit sourceInfo: SourceInfo): Unit = {
    // If --throw-on-first-error is requested, throw an exception instead of aggregating errors
    if (dynamicContextVar.value.isDefined) {
      errors.error(m, sourceInfo)
    } else {
      throwException(m)
    }
  }
  def warning(warning: Warning): Unit =
    if (dynamicContextVar.value.isDefined) errors.warning(warning)

  def deprecated(m: => String, location: Option[String] = None): Unit =
    if (dynamicContextVar.value.isDefined) errors.deprecated(m, location)

  /** Record an exception as an error, and throw it.
    *
    * @param m exception message
    */
  @throws(classOf[chisel3.ChiselException])
  def exception(m: => String)(implicit sourceInfo: SourceInfo): Nothing = {
    error(m)
    throwException(m)
  }

  def getScalaMajorVersion: Int = {
    val "2" :: major :: _ :: Nil = chisel3.BuildInfo.scalaVersion.split("\\.").toList
    major.toInt
  }

  // Builds a RenameMap for all Views that do not correspond to a single Data
  // These Data give a fake ReferenceTarget for .toTarget and .toReferenceTarget that the returned
  // RenameMap can split into the constituent parts
  private[chisel3] def makeViewRenameMap: MutableRenameMap = {
    val renames = MutableRenameMap()
    for (view <- unnamedViews) {
      val localTarget = view.toTarget
      val absTarget = view.toAbsoluteTarget
      val elts = getRecursiveFields.lazily(view, "").collect { case (elt: Element, _) => elt }
      for (elt <- elts) {
        // This is a hack to not crash when .viewAs is called on non-hardware
        // It can be removed in Chisel 6.0.0 when it becomes illegal to call .viewAs on non-hardware
        val targetOfViewOpt =
          try {
            Some(reify(elt))
          } catch {
            case _: NoSuchElementException => None
          }
        targetOfViewOpt.foreach { targetOfView =>
          renames.record(localTarget, targetOfView.toTarget)
          renames.record(absTarget, targetOfView.toAbsoluteTarget)
        }
      }
    }
    renames
  }

  def setRecordAlias(record: Record with HasTypeAlias, parentDirection: SpecifiedDirection): Unit = {
    val finalizedAlias: Option[String] = {
      val sourceInfo = record.aliasName.info

      // If the aliased bundle is coerced and it has flipped signals, then they must be stripped
      val isCoerced = parentDirection match {
        case SpecifiedDirection.Input | SpecifiedDirection.Output => true
        case other                                                => false
      }
      val isStripped = isCoerced && record.containsAFlipped

      // The true alias, after sanitization and (TODO) disambiguation
      val alias = sanitize(s"${record.aliasName.id}${if (isStripped) record.aliasName.strippedSuffix else ""}")
      // Filter out (TODO: disambiguate) FIRRTL keywords that cause parser errors if used
      if (illegalTypeAliases.contains(alias)) {
        Builder.error(
          s"Attempted to use an illegal word '$alias' for a type alias. Chisel does not automatically disambiguate these aliases at this time."
        )(sourceInfo)

        None
      } else {
        val tpe = Converter.extractType(record, isCoerced, sourceInfo, true, true, aliasMap.keys.toSeq)
        // If the name is already taken, check if there exists a *structurally equivalent* bundle with the same name, and
        // simply error (TODO: disambiguate that name)
        if (
          Builder.aliasMap.contains(alias) &&
          Builder.aliasMap.get(alias).exists(_._1 != tpe)
        ) {
          // Get full structural map value
          val recordValue = Builder.aliasMap.get(alias).get
          // Conflict found:
          error(
            s"Attempted to redeclare an existing type alias '$alias' with a new Record structure:\n'$tpe'.\n\nThe alias was previously defined as:\n'${recordValue._1}${recordValue._2
              .makeMessage(" " + _)}"
          )(sourceInfo)

          None
        } else {
          if (!Builder.aliasMap.contains(alias)) {
            Builder.aliasMap.put(alias, (tpe, sourceInfo))
          }

          Some(alias)
        }
      }
    }
    record.finalizedAlias = finalizedAlias
  }

  private[chisel3] def build[T <: BaseModule](
    f:              => T,
    dynamicContext: DynamicContext,
    forceModName:   Boolean = true
  ): (Circuit, T) = {
    dynamicContextVar.withValue(Some(dynamicContext)) {
      ViewParent: Unit // Must initialize the singleton in a Builder context or weird things can happen
      // in tiny designs/testcases that never access anything in chisel3.internal
      logger.info("Elaborating design...")
      val mod =
        try {
          val m = f
          if (forceModName) { // This avoids definition name index skipping with D/I
            m.forceName(m.name, globalNamespace)
          }
          m
        } catch {
          case NonFatal(e) =>
            // Make sure to report any aggregated errors in case the Exception is due to a tainted return value
            // Note that errors.checkpoint may throw an Exception which will suppress e
            errors.checkpoint(logger)
            throw e
        }
      errors.checkpoint(logger)
      logger.info("Done elaborating.")

      val typeAliases = aliasMap.flatMap {
        case (name, (underlying: fir.Type, info: SourceInfo)) => Some(DefTypeAlias(info, underlying, name))
        case _ => None
      }.toSeq

      /** Stores an adjacency list representation of groups.  Connections indicating children. */
      val groupAdjacencyList = mutable
        .LinkedHashMap[group.Declaration, mutable.LinkedHashSet[group.Declaration]]()
        .withDefault(_ => mutable.LinkedHashSet[group.Declaration]())

      // Populate the adjacency list.
      groups.foreach { group =>
        groupAdjacencyList(group.parent) = groupAdjacencyList(group.parent) += group
      }

      /** For a `group.Declaration`, walk all its children and fold them into a
        * `GroupDecl`.  This "folding" creates one `GroupDecl` for each child
        * nested under each parent `GroupDecl`.
        */
      def foldGroupDecls(decl: group.Declaration): GroupDecl = {
        val children = groupAdjacencyList(decl)
        val convention = decl.convention match {
          case group.Convention.Bind => GroupConvention.Bind
          case _                     => ???
        }
        GroupDecl(decl.sourceInfo, decl.name, convention, children.map(foldGroupDecls).toSeq)
      }

      val rootGroups = groupAdjacencyList(group.Declaration.Root)

      (
        Circuit(
          components.last.name,
          components.toSeq,
          annotations.toSeq,
          makeViewRenameMap,
          newAnnotations.toSeq,
          typeAliases,
          rootGroups.map(foldGroupDecls).toSeq
        ),
        mod
      )
    }
  }
  initializeSingletons()
}

/** Allows public access to the naming stack in Builder / DynamicContext, and handles invocations
  * outside a Builder context.
  * Necessary because naming macros expand in user code and don't have access into private[chisel3]
  * objects.
  */
object DynamicNamingStack {
  def pushContext(): NamingContextInterface = {
    Builder.namingStackOption match {
      case Some(namingStack) => namingStack.pushContext()
      case None              => DummyNamer
    }
  }

  def popReturnContext[T <: Any](prefixRef: T, until: NamingContextInterface): T = {
    until match {
      case DummyNamer =>
        require(
          Builder.namingStackOption.isEmpty,
          "Builder context must remain stable throughout a chiselName-annotated function invocation"
        )
      case context: NamingContext =>
        require(
          Builder.namingStackOption.isDefined,
          "Builder context must remain stable throughout a chiselName-annotated function invocation"
        )
        Builder.namingStackOption.get.popContext(prefixRef, context)
    }
    prefixRef
  }

  def length(): Int = Builder.namingStackOption.get.length()
}

/** Casts BigInt to Int, issuing an error when the input isn't representable. */
private[chisel3] object castToInt {
  def apply(x: BigInt, msg: String): Int = {
    val res = x.toInt
    require(x == res, s"$msg $x is too large to be represented as Int")
    res
  }
}
