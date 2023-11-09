// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.sourceinfo.{InstTransform}
import chisel3.properties.Class
import chisel3.reflect.DataMirror
import _root_.firrtl.annotations.{IsModule, ModuleName, ModuleTarget}
import _root_.firrtl.AnnotationSeq
import chisel3.internal.plugin.autoNameRecursively
import chisel3.util.simpleClassName

object Module extends SourceInfoDoc {

  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: => T): T = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](bc: => T)(implicit sourceInfo: SourceInfo): T = {
    if (Builder.readyForModuleConstr) {
      throwException(
        "Error: Called Module() twice without instantiating a Module." +
          sourceInfo.makeMessage(" See " + _)
      )
    }
    Builder.readyForModuleConstr = true

    val parent = Builder.currentModule
    val parentWhenStack = Builder.whenStack

    // Save then clear clock and reset to prevent leaking scope, must be set again in the Module
    // Note that Disable is a function of whatever the current reset is, so it does not need a port
    //   and thus does not change when we cross module boundaries
    val (saveClock, saveReset) = (Builder.currentClock, Builder.currentReset)
    val savePrefix = Builder.getPrefix
    Builder.clearPrefix()
    Builder.currentClock = None
    Builder.currentReset = None

    // Execute the module, this has the following side effects:
    //   - set currentModule
    //   - unset readyForModuleConstr
    //   - reset whenStack to be empty
    //   - set currentClockAndReset
    val module: T = bc // bc is actually evaluated here

    if (Builder.whenDepth != 0) {
      throwException("Internal Error! when() scope depth is != 0, this should have been caught!")
    }
    if (Builder.readyForModuleConstr) {
      throwException(
        "Error: attempted to instantiate a Module, but nothing happened. " +
          "This is probably due to rewrapping a Module instance with Module()." +
          sourceInfo.makeMessage(" See " + _)
      )
    }

    // Only add the component if the module generates one
    val componentOpt = module.generateComponent()
    for (component <- componentOpt) {
      Builder.components += component
    }

    // Reset Builder state *after* generating the component, so any atModuleBodyEnd generators are still within the
    // scope of the current Module.
    Builder.currentModule = parent // Back to parent!
    Builder.whenStack = parentWhenStack
    Builder.currentClock = saveClock // Back to clock and reset scope
    Builder.currentReset = saveReset
    Builder.setPrefix(savePrefix)

    // Handle connections at enclosing scope
    // We use _component because Modules that don't generate them may still have one
    if (Builder.currentModule.isDefined && module._component.isDefined) {
      // Class only uses the Definition API, and is not allowed here.
      module match {
        case _: Class => throwException("Module() cannot be called on a Class. Please use Definition().")
        case _ => ()
      }

      val component = module._component.get
      pushCommand(DefInstance(sourceInfo, module, component.ports))
      module.initializeInParent()
    }
    module
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock

  /** Returns the implicit Clock, if it is defined */
  def clockOption: Option[Clock] = Builder.currentClock

  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset

  /** Returns the implicit Reset, if it is defined */
  def resetOption: Option[Reset] = Builder.currentReset

  /** Returns the implicit Disable
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disable(implicit sourceInfo: SourceInfo): Disable =
    disableOption.getOrElse(throwException("Error: No implicit disable."))

  /** Returns the current implicit [[Disable]], if one is defined
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disableOption(implicit sourceInfo: SourceInfo): Option[Disable] = {
    Builder.currentDisable match {
      case Disable.Never       => None
      case Disable.BeforeReset => hasBeenReset.map(x => autoNameRecursively("disable")(!x))
    }
  }

  // Should this be public or should users just go through .disable?
  // Note that having a reset but not clock means hasBeenReset is None, should we default to just !reset?
  private def hasBeenReset(implicit sourceInfo: SourceInfo): Option[Disable] = {
    // TODO memoize this
    (Builder.currentClock, Builder.currentReset) match {
      case (Some(clock), Some(reset)) =>
        val hasBeenReset = Module(new HasBeenResetIntrinsic)
        hasBeenReset.clock := clock
        hasBeenReset.reset := reset
        Some(new Disable(hasBeenReset.out))
      case _ => None
    }
  }

  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule

  private[chisel3] def do_pseudo_apply[T <: BaseModule](
    bc: => T
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    val parent = Builder.currentModule
    val module: T = bc // bc is actually evaluated here
    if (!parent.isEmpty) { Builder.currentModule = parent }

    module
  }

  /**  Assign directionality on any IOs that are still Unspecified/Flipped
    *
    *  Chisel2 did not require explicit direction on nodes
    *  (unspecified treated as output, and flip on nothing was input).
    *  As of 3.6, chisel3 is now also using these semantics, so we need to make it work
    *  even for chisel3 code.
    *  This assigns the explicit directions required by both semantics on all Bundles.
    * This recursively walks the tree, and assigns directions if no explicit
    *   direction given by upper-levels (override Input / Output)
    */
  private[chisel3] def assignCompatDir(data: Data): Unit =
    // Collect all leaf elements of the data which have an unspecified or flipped
    // direction, and assign explicit directions to them
    DataMirror
      .collectMembers(data) {
        case x: Element
            if x.specifiedDirection == SpecifiedDirection.Unspecified || x.specifiedDirection == SpecifiedDirection.Flip =>
          x
      }
      .foreach { x => x._assignCompatibilityExplicitDirection }

  /** Allowed values for the types of Module.reset */
  object ResetType {

    /** Allowed values for the types of Module.reset */
    sealed trait Type

    /** The default reset type. This is Uninferred, unless it is the top Module, in which case it is Bool */
    case object Default extends Type

    /** Explicitly Uninferred Reset, even if this is the top Module */
    case object Uninferred extends Type

    /** Explicitly Bool (Synchronous) Reset */
    case object Synchronous extends Type

    /** Explicitly Asynchronous Reset */
    case object Asynchronous extends Type
  }
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  * This abstract base class includes an implicit clock and reset.
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module extends RawModule {

  /** Override this to explicitly set the type of reset you want on this module , before any reset inference */
  def resetType: Module.ResetType.Type = Module.ResetType.Default

  // Implicit clock and reset pins
  final val clock: Clock = IO(Input(Clock()))(UnlocatableSourceInfo).suggestName("clock")
  final val reset: Reset = IO(Input(mkReset))(UnlocatableSourceInfo).suggestName("reset")
  // TODO add a way to memoize hasBeenReset iff it is used

  // TODO It's hard to remove these deprecated override methods because they're used by
  //   Chisel.QueueCompatibility which extends chisel3.Queue which extends chisel3.Module
  private var _override_clock: Option[Clock] = None
  private var _override_reset: Option[Bool] = None
  @deprecated("Use withClock at Module instantiation", "Chisel 3.5")
  protected def override_clock: Option[Clock] = _override_clock
  @deprecated("Use withClock at Module instantiation", "Chisel 3.5")
  protected def override_reset: Option[Bool] = _override_reset
  @deprecated("Use withClock at Module instantiation", "Chisel 3.5")
  protected def override_clock_=(rhs: Option[Clock]): Unit = {
    _override_clock = rhs
  }
  @deprecated("Use withClock at Module instantiation", "Chisel 3.5")
  protected def override_reset_=(rhs: Option[Bool]): Unit = {
    _override_reset = rhs
  }

  private[chisel3] def mkReset: Reset = {
    // Top module and compatibility mode use Bool for reset
    // Note that a Definition elaboration will lack a parent, but still not be a Top module
    resetType match {
      case Module.ResetType.Default => {
        val inferReset = (_parent.isDefined || Builder.inDefinition)
        if (inferReset) Reset() else Bool()
      }
      case Module.ResetType.Uninferred   => Reset()
      case Module.ResetType.Synchronous  => Bool()
      case Module.ResetType.Asynchronous => AsyncReset()
    }
  }

  // Setup ClockAndReset
  Builder.currentClock = Some(clock)
  Builder.currentReset = Some(reset)
  // Note that we do no such setup for disable, it will default to hasBeenReset of the currentReset
  Builder.clearPrefix()

  private[chisel3] override def initializeInParent(): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    super.initializeInParent()
    clock := _override_clock.getOrElse(Builder.forcedClock)
    reset := _override_reset.getOrElse(Builder.forcedReset)
  }
}

package internal {

  object BaseModule {

    import chisel3.experimental.hierarchy._

    /** Record type returned by CloneModuleAsRecord
      *
      * @note These are not true Data (the Record doesn't correspond to anything in the emitted
      * FIRRTL yet its elements *do*) so have some very specialized behavior.
      */
    private[chisel3] class ClonePorts(elts: (String, Data)*) extends Record {
      val elements = ListMap(elts.map { case (name, d) => name -> d.cloneTypeFull }: _*)
      def apply(field: String) = elements(field)
      override def cloneType = (new ClonePorts(elts: _*)).asInstanceOf[this.type]
    }

    private[chisel3] def cloneIORecord(
      proto: BaseModule
    )(
      implicit sourceInfo: SourceInfo
    ): ClonePorts = {
      require(proto.isClosed, "Can't clone a module before module close")
      // Fake Module to serve as the _parent of the cloned ports
      // We make this before clonePorts because we want it to come up first in naming in
      // currentModule
      val cloneParent = Module(new ModuleClone(proto))
      require(proto.isClosed, "Can't clone a module before module close")
      require(cloneParent.getOptionRef.isEmpty, "Can't have ref set already!")
      // Chisel ports can be Data or Property, but to clone as a Record, we can only return Data.
      val dataPorts = proto.getChiselPorts.collect { case (name, data: Data) => (name, data) }
      // Fake Module to serve as the _parent of the cloned ports
      // We don't create this inside the ModuleClone because we need the ref to be set by the
      // currentModule (and not clonePorts)
      val clonePorts = proto match {
        // BlackBox needs special handling for its pseduo-io Bundle
        case b: BlackBox =>
          new ClonePorts(dataPorts :+ ("io" -> b._io.get): _*)
        case _ => new ClonePorts(dataPorts: _*)
      }
      // getChiselPorts (nor cloneTypeFull in general)
      // does not recursively copy the right specifiedDirection,
      // still need to fix it up here.
      Module.assignCompatDir(clonePorts)
      clonePorts.bind(PortBinding(cloneParent))
      clonePorts.setAllParents(Some(cloneParent))
      cloneParent._portsRecord = clonePorts
      if (proto.isInstanceOf[Module]) {
        clonePorts("clock") := Module.clock
        clonePorts("reset") := Module.reset
      }
      clonePorts
    }
  }
}

package experimental {

  import chisel3.experimental.hierarchy.core.{IsInstantiable, Proto}
  import scala.annotation.nowarn

  object BaseModule {
    implicit class BaseModuleExtensions[T <: BaseModule](b: T)(implicit si: SourceInfo) {
      import chisel3.experimental.hierarchy.core.{Definition, Instance}
      def toInstance: Instance[T] = new Instance(Proto(b))
      def toDefinition: Definition[T] = {
        b.toDefinitionCalled = Some(si)
        new Definition(Proto(b))
      }
    }
  }

  /** Abstract base class for Modules, an instantiable organizational unit for RTL.
    */
  // TODO: seal this?
  @nowarn("msg=class Port") // delete when Port becomes private
  abstract class BaseModule extends HasId with IsInstantiable {
    _parent.foreach(_.addId(this))

    // Protected so it can be overridden by the compiler plugin
    protected def _sourceInfo: SourceInfo = UnlocatableSourceInfo

    // Accessor for Chisels internals
    private[chisel3] final def _getSourceLocator: SourceInfo = _sourceInfo

    // Used with chisel3.naming.fixTraitIdentifier
    protected def _traitModuleDefinitionIdentifierProposal: Option[String] = None

    protected def _moduleDefinitionIdentifierProposal = {
      val baseName = _traitModuleDefinitionIdentifierProposal.getOrElse(this.getClass.getName)

      /* A sequence of string filters applied to the name */
      val filters: Seq[String => String] =
        Seq(((a: String) => raw"\$$+anon".r.replaceAllIn(a, "_Anon")) // Merge the "$$anon" name with previous name
        )

      filters
        .foldLeft(baseName) { case (str, filter) => filter(str) } // 1. Apply filters to baseName
        .split("\\.|\\$") // 2. Split string at '.' or '$'
        .filterNot(_.forall(_.isDigit)) // 3. Drop purely numeric names
        .last // 4. Use the last name
    }
    // Needed this to override identifier for DefinitionClone
    private[chisel3] def _definitionIdentifier = {
      val madeProposal = chisel3.naming.IdentifierProposer.makeProposal(this._moduleDefinitionIdentifierProposal)
      Builder.globalIdentifierNamespace.name(madeProposal)
    }

    /** Represents an eagerly-determined unique and descriptive identifier for this module */
    final val definitionIdentifier = _definitionIdentifier
    //
    // Builder Internals - this tracks which Module RTL construction belongs to.
    //
    this match {
      case _: PseudoModule =>
      case other =>
        if (!Builder.readyForModuleConstr) {
          throwException("Error: attempted to instantiate a Module without wrapping it in Module().")
        }
    }
    if (Builder.hasDynamicContext) {
      readyForModuleConstr = false

      Builder.currentModule = Some(this)
      Builder.whenStack = Nil
    }

    //
    // Module Construction Internals
    //
    protected var _closed = false

    /** Internal check if a Module's constructor has finished executing */
    private[chisel3] def isClosed = _closed

    /** Mutable state that indicates if IO is allowed to be created for this module.
      *  - List.empty: IO creation is allowed
      *  - List.nonEmpty: IO creation is not allowed at contained location(s)
      * This can be used for advanced Chisel library APIs that want to limit
      * what IO is allowed to be created for a module.
      */
    private[chisel3] var _whereIOCreationIsDisallowed: List[SourceInfo] = Nil

    /** If true, then this module is allowed to have user-created IO. */
    private[chisel3] def isIOCreationAllowed = _whereIOCreationIsDisallowed.isEmpty

    /** Disallow any more IO creation for this module. */
    private def disallowIOCreation()(implicit si: SourceInfo): Unit = {
      _whereIOCreationIsDisallowed = si +: _whereIOCreationIsDisallowed
    }

    /** Remove one layer of disallowed IO creation
      * Note that IO creation is only legal if _whereIOCreationIsDisallowed is empty
      */
    private def allowIOCreation(): Unit = {
      if (_whereIOCreationIsDisallowed.nonEmpty) {
        _whereIOCreationIsDisallowed = _whereIOCreationIsDisallowed.tail
      }
    }

    /** Disallow any more IO creation for this module. */
    def endIOCreation()(implicit si: SourceInfo): Unit = disallowIOCreation()

    /** Disallow any more IO creation for this module. */
    private[chisel3] def disallowIO[T](thunk: => T)(implicit si: SourceInfo): T = {
      disallowIOCreation()
      val ret = thunk
      allowIOCreation()
      ret
    }

    private[chisel3] var toDefinitionCalled:  Option[SourceInfo] = None
    private[chisel3] var modulePortsAskedFor: Option[SourceInfo] = None

    /** Where a Module becomes fully closed (no secret ports drilled afterwards) */
    private[chisel3] def isFullyClosed = fullyClosedErrorMessages.nonEmpty
    private[chisel3] def fullyClosedErrorMessages: Iterable[(SourceInfo, String)] = {
      toDefinitionCalled.map(si =>
        (si, s"Calling .toDefinition fully closes ${name}, but it is later bored through!")
      ) ++
        modulePortsAskedFor.map(si =>
          (si, s"Reflecting on all io's fully closes ${name}, but it is later bored through!")
        )
    }

    // Fresh Namespace because in Firrtl, Modules namespaces are disjoint with the global namespace
    private[chisel3] val _namespace = Namespace.empty

    // Expose _ids in Chisel. The ids should almost always be accessed through getIds, but there is a use-case to access
    // the ids directly in generateComponent.
    private[chisel3] val _ids = ArrayBuffer[HasId]()

    private[chisel3] def addId(d: HasId): Unit = {
      if (Builder.aspectModule(this).isDefined) {
        aspectModule(this).get.addId(d)
      } else {
        require(!_closed, "Can't write to module after module close")
        _ids += d
      }
    }

    // Returns the last id contained within a Module
    private[chisel3] def _lastId: Long = _ids.last match {
      case mod: BaseModule => mod._lastId
      case agg: Aggregate  =>
        // Ideally we could just take .last._id, but Records store their elements in reverse order
        getRecursiveFields.lazily(agg, "").map(_._1._id).max
      case other => other._id
    }

    private[chisel3] def getIds: Iterable[HasId] = {
      require(_closed, "Can't get ids before module close")
      _ids
    }

    private val _ports = new ArrayBuffer[(Data, SourceInfo)]()

    // getPorts unfortunately already used for tester compatibility
    protected[chisel3] def getModulePorts: Seq[Data] = {
      require(_closed, "Can't get ports before module close")
      _ports.iterator.collect { case (d: Data, _) => d }.toSeq
    }

    // gets Ports along with there source locators
    private[chisel3] def getModulePortsAndLocators: Seq[(Data, SourceInfo)] = {
      require(_closed, "Can't get ports before module close")
      _ports.toSeq
    }

    // These methods allow checking some properties of ports before the module is closed,
    // mainly for compatibility purposes.
    protected def portsContains(elem: Data): Boolean = {
      _ports.exists { port => port._1 == elem }
    }

    // This is dangerous because it can be called before the module is closed and thus there could
    // be more ports and names have not yet been finalized.
    // This should only to be used during the process of closing when it is safe to do so.
    private[chisel3] def findPort(name: String): Option[Data] =
      _ports.collectFirst { case (data, _) if data.seedOpt.contains(name) => data }

    protected def portsSize: Int = _ports.size

    /** Generates the FIRRTL Component (Module or Blackbox) of this Module.
      * Also closes the module so no more construction can happen inside.
      */
    private[chisel3] def generateComponent(): Option[Component]

    /** Sets up this module in the parent context
      */
    private[chisel3] def initializeInParent(): Unit

    private[chisel3] def namePorts(): Unit = {
      for ((port, source) <- getModulePortsAndLocators) {
        port._computeName(None) match {
          case Some(name) =>
            if (_namespace.contains(name)) {
              Builder.error(
                s"""Unable to name port $port to "$name" in $this,""" +
                  s" name is already taken by another port! ${source.makeMessage(x => x)}"
              )(UnlocatableSourceInfo)
            }
            port.setRef(ModuleIO(this, _namespace.name(name)))
          case None =>
            Builder.error(
              s"Unable to name port $port in $this, " +
                s"try making it a public field of the Module ${source.makeMessage(x => x)}"
            )(UnlocatableSourceInfo)
            port.setRef(ModuleIO(this, "<UNNAMED>"))
        }
      }
    }

    //
    // Chisel Internals
    //

    /** The desired name of this module (which will be used in generated FIRRTL IR or Verilog).
      *
      * The name of a module approximates the behavior of the Java Reflection `getSimpleName` method
      * https://docs.oracle.com/javase/8/docs/api/java/lang/Class.html#getSimpleName-- with some modifications:
      *
      * - Anonymous modules will get an `"_Anon"` tag
      * - Modules defined in functions will use their class name and not a numeric name
      *
      * @note If you want a custom or parametric name, override this method.
      */
    def desiredName: String = simpleClassName(this.getClass)

    /** Legalized name of this module. */
    final lazy val name =
      try {
        // PseudoModules are not "true modules" and thus should share
        // their original modules names without uniquification
        this match {
          case _: PseudoModule => desiredName
          case _ => Builder.globalNamespace.name(desiredName)
        }
      } catch {
        case e: NullPointerException =>
          throwException(
            s"Error: desiredName of ${this.getClass.getName} is null. Did you evaluate 'name' before all values needed by desiredName were available?",
            e
          )
        case t: Throwable => throw t
      }

    /** Returns a FIRRTL ModuleName that references this object
      *
      * @note Should not be called until circuit elaboration is complete
      */
    final def toNamed: ModuleName = ModuleTarget(this.circuitName, this.name).toNamed

    /** Returns a FIRRTL ModuleTarget that references this object
      *
      * @note Should not be called until circuit elaboration is complete
      */
    final def toTarget: ModuleTarget = this match {
      case m: experimental.hierarchy.InstanceClone[_] =>
        throwException(s"Internal Error! It's not legal to call .toTarget on an InstanceClone. $m")
      case m: experimental.hierarchy.DefinitionClone[_] =>
        throwException(s"Internal Error! It's not legal to call .toTarget on an DefinitionClone. $m")
      case _ => ModuleTarget(this.circuitName, this.name)
    }

    /** Returns the real target of a Module which may be an [[InstanceTarget]]
      *
      * BaseModule.toTarget returns a ModuleTarget because the classic Module(new MyModule) API elaborates
      * Modules in a way that there is a 1:1 relationship between instances and elaborated definitions
      *
      * Instance/Definition introduced special internal modules [[InstanceClone]] and [[ModuleClone]] that
      * do not have this 1:1 relationship so need the ability to return [[InstanceTarget]]s.
      * Because users can never actually get references to these underlying objects, we can maintain
      * BaseModule.toTarget's API returning [[ModuleTarget]] while providing an internal API for getting
      * the correct [[InstanceTarget]]s whenever using the Definition/Instance API.
      */
    private[chisel3] def getTarget: IsModule = this match {
      case m: experimental.hierarchy.InstanceClone[_] if m._parent.nonEmpty =>
        m._parent.get.getTarget.instOf(instanceName, name)
      case m: experimental.hierarchy.ModuleClone[_] if m._madeFromDefinition =>
        m._parent.get.getTarget.instOf(instanceName, name)
      // Without this, we get the wrong CircuitName for the Definition
      case m: experimental.hierarchy.DefinitionClone[_] if m._circuit.nonEmpty =>
        ModuleTarget(this._circuit.get.circuitName, this.name)
      case _ => this.toTarget
    }

    /** Returns a FIRRTL ModuleTarget that references this object
      *
      * @note Should not be called until circuit elaboration is complete
      */
    final def toAbsoluteTarget: IsModule = {
      _parent match {
        case Some(parent) => parent.toAbsoluteTarget.instOf(this.instanceName, name)
        case None         =>
          // FIXME Special handling for Views - evidence of "weirdness" of .toAbsoluteTarget
          // In theory, .toAbsoluteTarget should not be necessary, .toTarget combined with the
          // target disambiguation in FIRRTL's deduplication transform should ensure that .toTarget
          // is always unambigous. However, legacy workarounds for Chisel's lack of an instance API
          // have lead some to use .toAbsoluteTarget as a workaround. A proper instance API will make
          // it possible to deprecate and remove .toAbsoluteTarget
          if (this == ViewParent) ViewParent.absoluteTarget else getTarget
      }
    }

    /** Returns a FIRRTL ModuleTarget that references this object, relative to an optional root.
      *
      * If `root` is defined, the target is a hierarchical path starting from `root`.
      *
      * If `root` is not defined, the target is a hierarchical path equivalent to `toAbsoluteTarget`.
      *
      * @note If `root` is defined, and has not finished elaboration, this must be called within `atModuleBodyEnd`.
      * @note The BaseModule must be a descendant of `root`, if it is defined.
      * @note This doesn't have special handling for Views.
      */
    final def toRelativeTarget(root: Option[BaseModule]): IsModule = {
      // If root was defined, and we are it, return this.
      if (root.contains(this)) getTarget
      // If we are a ViewParent, use its absolute target.
      else if (this == ViewParent) ViewParent.absoluteTarget
      // Otherwise check if root and _parent are defined.
      else
        (root, _parent) match {
          // If root was defined, and we are not there yet, recurse up.
          case (_, Some(parent)) => parent.toRelativeTarget(root).instOf(this.instanceName, name)
          // If root was defined, and there is no parent, the root was not an ancestor.
          case (Some(definedRoot), None) =>
            throwException(s"Requested .toRelativeTarget relative to ${definedRoot.name}, but it is not an ancestor")
          // If root was not defined, and there is no parent, return this.
          case (None, None) => getTarget
        }
    }

    /**
      * Internal API. Returns a list of this module's generated top-level ports as a map of a String
      * (FIRRTL name) to the IO object. Only valid after the module is closed.
      *
      * Note: for BlackBoxes (but not ExtModules), this returns the contents of the top-level io
      * object, consistent with what is emitted in FIRRTL.
      *
      * TODO: Use SeqMap/VectorMap when those data structures become available.
      */
    private[chisel3] def getChiselPorts(implicit si: SourceInfo): Seq[(String, Data)] = {
      require(_closed, "Can't get ports before module close")
      modulePortsAskedFor = Some(si) // super-lock down the module
      (_component.get.ports ++ _component.get.secretPorts).map { port =>
        (port.id.getRef.asInstanceOf[ModuleIO].name, port.id)
      }
    }

    /** Chisel2 code didn't require the IO(...) wrapper and would assign a Chisel type directly to
      * io, then do operations on it. This binds a Chisel type in-place (mutably) as an IO.
      */
    protected def _bindIoInPlace(iodef: Data)(implicit sourceInfo: SourceInfo): Unit = {

      // Assign any signals (Chisel or chisel3) with Unspecified/Flipped directions to Output/Input.
      // This is only required for Data, not all Datas in general.
      iodef match {
        case (data: Data) => Module.assignCompatDir(data)
        case _ => ()
      }

      iodef.bind(PortBinding(this))
      _ports += iodef -> sourceInfo
    }

    /** Private accessor for _bindIoInPlace */
    private[chisel3] def bindIoInPlace(
      iodef: Data
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = _bindIoInPlace(iodef)

    // Must have separate createSecretIO from addSecretIO to get plugin to name it
    // data must be a fresh Chisel type
    private[chisel3] def createSecretIO[A <: Data](data: => A)(implicit sourceInfo: SourceInfo): A = {
      val iodef = data
      internal.requireIsChiselType(iodef, "io type")
      require(!isFullyClosed, "Cannot create secret ports if module is fully closed")

      Module.assignCompatDir(iodef)
      iodef.bind(internal.SecretPortBinding(this), iodef.specifiedDirection)
      iodef
    }

    private[chisel3] val secretPorts: ArrayBuffer[Port] = ArrayBuffer.empty

    // Must have separate createSecretIO from addSecretIO to get plugin to name it
    private[chisel3] def addSecretIO[A <: Data](iodef: A)(implicit sourceInfo: SourceInfo): A = {
      val name = iodef._computeName(None).getOrElse("secret")
      iodef.setRef(ModuleIO(this, _namespace.name(name)))
      val newPort = new Port(iodef, iodef.specifiedDirection, sourceInfo)
      if (_closed) {
        _component.get.secretPorts += newPort
      } else secretPorts += newPort
      iodef
    }

    /**
      * This must wrap the datatype used to set the io field of any Module.
      * i.e. All concrete modules must have defined io in this form:
      * [lazy] val io[: io type] = IO(...[: io type])
      *
      * Items in [] are optional.
      *
      * The granted iodef must be a chisel type and not be bound to hardware.
      *
      * Also registers an Data as a port, also performing bindings. Cannot be called once ports are
      * requested (so that all calls to ports will return the same information).
      * Internal API.
      *
      * TODO(twigg): Specifically walk the Data definition to call out which nodes
      * are problematic.
      */
    protected def IO[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
      chisel3.IO.apply(iodef)
    }

    //
    // Internal Functions
    //

    /** Keep component for signal names */
    private[chisel3] var _component: Option[Component] = None

    /** Signal name (for simulation). */
    override def instanceName: String =
      if (_parent == None) name
      else
        _component match {
          case None    => getRef.name
          case Some(c) => getRef.fullName(c)
        }

  }
}
