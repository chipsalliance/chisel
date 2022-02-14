// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.{IsModule, ModuleName, ModuleTarget}
import _root_.firrtl.AnnotationSeq

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
  def do_apply[T <: BaseModule](bc: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
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
    Builder.currentModule = parent // Back to parent!
    Builder.whenStack = parentWhenStack
    Builder.currentClock = saveClock // Back to clock and reset scope
    Builder.currentReset = saveReset

    // Only add the component if the module generates one
    val componentOpt = module.generateComponent()
    for (component <- componentOpt) {
      Builder.components += component
    }

    Builder.setPrefix(savePrefix)

    // Handle connections at enclosing scope
    // We use _component because Modules that don't generate them may still have one
    if (Builder.currentModule.isDefined && module._component.isDefined) {
      val component = module._component.get
      pushCommand(DefInstance(sourceInfo, module, component.ports))
      module.initializeInParent(compileOptions)
    }
    module
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock

  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset

  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule

  private[chisel3] def do_pseudo_apply[T <: BaseModule](
    bc: => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    val parent = Builder.currentModule
    val module: T = bc // bc is actually evaluated here
    if (!parent.isEmpty) { Builder.currentModule = parent }

    module
  }
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  * This abstract base class includes an implicit clock and reset.
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module(implicit moduleCompileOptions: CompileOptions) extends RawModule {
  // Implicit clock and reset pins
  final val clock: Clock = IO(Input(Clock())).suggestName("clock")
  final val reset: Reset = IO(Input(mkReset)).suggestName("reset")

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
    val inferReset = (_parent.isDefined || Builder.inDefinition) && moduleCompileOptions.inferModuleReset
    if (inferReset) Reset() else Bool()
  }

  // Setup ClockAndReset
  Builder.currentClock = Some(clock)
  Builder.currentReset = Some(reset)
  Builder.clearPrefix()

  private[chisel3] override def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    super.initializeInParent(parentCompileOptions)
    clock := _override_clock.getOrElse(Builder.forcedClock)
    reset := _override_reset.getOrElse(Builder.forcedReset)
  }
}

package experimental {

  import chisel3.internal.requireIsChiselType // Fix ambiguous import

  object IO {

    /** Constructs a port for the current Module
      *
      * This must wrap the datatype used to set the io field of any Module.
      * i.e. All concrete modules must have defined io in this form:
      * [lazy] val io[: io type] = IO(...[: io type])
      *
      * Items in [] are optional.
      *
      * The granted iodef must be a chisel type and not be bound to hardware.
      *
      * Also registers a Data as a port, also performing bindings. Cannot be called once ports are
      * requested (so that all calls to ports will return the same information).
      * Internal API.
      */
    def apply[T <: Data](iodef: T): T = {
      val module = Module.currentModule.get // Impossible to fail
      require(!module.isClosed, "Can't add more ports after module close")
      requireIsChiselType(iodef, "io type")

      // Clone the IO so we preserve immutability of data types
      val iodefClone =
        try {
          iodef.cloneTypeFull
        } catch {
          // For now this is going to be just a deprecation so we don't suddenly break everyone's code
          case e: AutoClonetypeException =>
            Builder.deprecated(e.getMessage, Some(s"${iodef.getClass}"))
            iodef
        }
      module.bindIoInPlace(iodefClone)
      iodefClone
    }
  }
}

package internal {
  import chisel3.experimental.BaseModule
  import chisel3.experimental.hierarchy.{Clone, IsInstantiable, Proto}

  object BaseModule {

    /** Represents a clone of an underlying object. This is used to support CloneModuleAsRecord and Instance/Definition.
      *
      * @note We don't actually "clone" anything in the traditional sense but is a placeholder so we lazily clone internal state
      */
    trait IsClone[+T] {
      // Underlying object of which this is a clone of
      private[chisel3] def getProto: T

      /** Determines whether another object is a clone of the same underlying proto
        *
        * @param a
        */
      def hasSameProto(a: Any): Boolean = {
        val aProto = a match {
          case x: IsClone[BaseModule] => x.getProto
          case o => o
        }
        this == aProto || getProto == aProto
      }
    }

    // Private internal class to serve as a _parent for Data in cloned ports
    private[chisel3] class ModuleClone[T <: BaseModule](val getProto: T) extends PseudoModule with IsClone[T] {
      override def toString = s"ModuleClone(${getProto})"
      // Do not call default addId function, which may modify a module that is already "closed"
      override def addId(d: HasId): Unit = ()
      def getPorts = _portsRecord
      // ClonePorts that hold the bound ports for this module
      // Used for setting the refs of both this module and the Record
      private[BaseModule] var _portsRecord: Record = _
      // This is necessary for correctly supporting .toTarget on a Module Clone. If it is made from the
      // Instance/Definition API, it should return an instanceTarget. If made from CMAR, it should return a
      // ModuleTarget.
      private[chisel3] var _madeFromDefinition: Boolean = false
      // Don't generate a component, but point to the one for the cloned Module
      private[chisel3] def generateComponent(): Option[Component] = {
        require(!_closed, "Can't generate module more than once")
        _closed = true
        _component = getProto._component
        None
      }
      // Maps proto ports to module clone's ports
      private[chisel3] lazy val ioMap: Map[Data, Data] = {
        val name2Port = getPorts.elements
        getProto.getChiselPorts.map { case (name, data) => data -> name2Port(name) }.toMap
      }
      // This module doesn't actually exist in the FIRRTL so no initialization to do
      private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()

      // Name of this instance's module is the same as the proto's name
      override def desiredName: String = getProto.name

      private[chisel3] def setRefAndPortsRef(namespace: Namespace): Unit = {
        val record = _portsRecord
        // Use .forceName to re-use default name resolving behavior
        record.forceName(None, default = this.desiredName, namespace)
        // Now take the Ref that forceName set and convert it to the correct Arg
        val instName = record.getRef match {
          case Ref(name)              => name
          case ModuleCloneIO(_, name) => namespace.name(name)
          case bad                    => throwException(s"Internal Error! Cloned-module Record $record has unexpected ref $bad")
        }
        // Set both the record and the module to have the same instance name
        record.setRef(ModuleCloneIO(getProto, instName), force = true) // force because we did .forceName first
        this.setRef(Ref(instName))
      }

      def suggestName(name: String): Unit = {
        _portsRecord.setRef(ModuleCloneIO(getProto, name)) // force because we did .forceName first
        this.setRef(Ref(name))
      }
    }

    /** Represents a module viewed from a different instance context.
      *
      * @note Why do we need both ModuleClone and InstanceClone? If we are annotating a reference in a module-clone,
      * all submodules must be also be 'cloned' so the toTarget can be computed properly. However, we don't need separate
      * connectable ports for this instance; all that's different from the proto is the parent.
      *
      * @note In addition, the instance name of an InstanceClone is going to be the SAME as the proto, but this is not true
      * for ModuleClone.
      */
    private[chisel3] final class InstanceClone[T <: BaseModule](val getProto: T, val instName: () => String)
        extends PseudoModule
        with IsClone[T] {
      override def toString = s"InstanceClone(${getProto})"
      // No addition components are generated
      private[chisel3] def generateComponent(): Option[Component] = None
      // Necessary for toTarget to work
      private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(instName())) }
      // This module doesn't acutally exist in the FIRRTL so no initialization to do
      private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
      // Instance name is the same as proto's instance name
      override def instanceName = instName()
      // Module name is the same as proto's module name
      override def desiredName: String = getProto.name
    }

    /** Represents a Definition root module, when accessing something from a definition
      *
      * @note This is necessary to distinguish between the toTarget behavior for a Module returned from a Definition,
      * versus a normal Module. A normal Module.toTarget will always return a local target. If calling toTarget
      * on a Module returned from a Definition (and thus wrapped in an Instance), we need to return the non-local
      * target whose root is the Definition. This DefinitionClone is used to represent the root parent of the
      * InstanceClone (which represents the returned module).
      */
    private[chisel3] class DefinitionClone[T <: BaseModule](val getProto: T) extends PseudoModule with IsClone[T] {
      override def toString = s"DefinitionClone(${getProto})"
      // No addition components are generated
      private[chisel3] def generateComponent(): Option[Component] = None
      // Do not call default addId function, which may modify a module that is already "closed"
      override def addId(d: HasId): Unit = ()
      // Necessary for toTarget to work
      private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
      // Module name is the same as proto's module name
      override def desiredName: String = getProto.name
    }

    /** @note If we are cloning a non-module, we need another object which has the proper _parent set!
      */
    trait InstantiableClone[T <: IsInstantiable] extends IsClone[T] {
      private[chisel3] def _innerContext: experimental.hierarchy.Hierarchy[_]
      private[chisel3] def getInnerContext: Option[BaseModule] = _innerContext.getInnerDataContext
    }

    /** Record type returned by CloneModuleAsRecord
      *
      * @note These are not true Data (the Record doesn't correspond to anything in the emitted
      * FIRRTL yet its elements *do*) so have some very specialized behavior.
      */
    private[chisel3] class ClonePorts(elts: Data*)(implicit compileOptions: CompileOptions) extends Record {
      val elements = ListMap(elts.map(d => d.instanceName -> d.cloneTypeFull): _*)
      def apply(field: String) = elements(field)
      override def cloneType = (new ClonePorts(elts: _*)).asInstanceOf[this.type]
    }

    private[chisel3] def cloneIORecord(
      proto: BaseModule
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): ClonePorts = {
      require(proto.isClosed, "Can't clone a module before module close")
      // Fake Module to serve as the _parent of the cloned ports
      // We make this before clonePorts because we want it to come up first in naming in
      // currentModule
      val cloneParent = Module(new ModuleClone(proto))
      require(proto.isClosed, "Can't clone a module before module close")
      require(cloneParent.getOptionRef.isEmpty, "Can't have ref set already!")
      // Fake Module to serve as the _parent of the cloned ports
      // We don't create this inside the ModuleClone because we need the ref to be set by the
      // currentModule (and not clonePorts)
      val clonePorts = new ClonePorts(proto.getModulePorts: _*)
      clonePorts.bind(PortBinding(cloneParent))
      clonePorts.setAllParents(Some(cloneParent))
      cloneParent._portsRecord = clonePorts
      // Normally handled during Module construction but ClonePorts really lives in its parent's parent
      if (!compileOptions.explicitInvalidate) {
        pushCommand(DefInvalid(sourceInfo, clonePorts.ref))
      }
      if (proto.isInstanceOf[Module]) {
        clonePorts("clock") := Module.clock
        clonePorts("reset") := Module.reset
      }
      clonePorts
    }
  }
}

package experimental {

  import chisel3.experimental.hierarchy.{IsInstantiable, Proto}

  object BaseModule {
    implicit class BaseModuleExtensions[T <: BaseModule](b: T) {
      import chisel3.experimental.hierarchy.{Definition, Instance}
      def toInstance:   Instance[T] = new Instance(Proto(b))
      def toDefinition: Definition[T] = new Definition(Proto(b))
    }
  }

  /** Abstract base class for Modules, an instantiable organizational unit for RTL.
    */
  // TODO: seal this?
  abstract class BaseModule extends HasId with IsInstantiable {
    _parent.foreach(_.addId(this))

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

    /** Internal check if a Module is closed */
    private[chisel3] def isClosed = _closed

    // Fresh Namespace because in Firrtl, Modules namespaces are disjoint with the global namespace
    private[chisel3] val _namespace = Namespace.empty
    private val _ids = ArrayBuffer[HasId]()
    private[chisel3] def addId(d: HasId) {
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
      case _ =>
        // Ideally we could just take last._id, but Records store and thus bind their Data in reverse order
        _ids.maxBy(_._id)._id
    }

    private[chisel3] def getIds = {
      require(_closed, "Can't get ids before module close")
      _ids.toSeq
    }

    private val _ports = new ArrayBuffer[Data]()

    // getPorts unfortunately already used for tester compatibility
    protected[chisel3] def getModulePorts = {
      require(_closed, "Can't get ports before module close")
      _ports.toSeq
    }

    // These methods allow checking some properties of ports before the module is closed,
    // mainly for compatibility purposes.
    protected def portsContains(elem: Data): Boolean = _ports contains elem

    // This is dangerous because it can be called before the module is closed and thus there could
    // be more ports and names have not yet been finalized.
    // This should only to be used during the process of closing when it is safe to do so.
    private[chisel3] def findPort(name: String): Option[Data] = _ports.find(_.seedOpt.contains(name))

    protected def portsSize: Int = _ports.size

    /** Generates the FIRRTL Component (Module or Blackbox) of this Module.
      * Also closes the module so no more construction can happen inside.
      */
    private[chisel3] def generateComponent(): Option[Component]

    /** Sets up this module in the parent context
      */
    private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit

    //
    // Chisel Internals
    //

    /** The desired name of this module (which will be used in generated FIRRTL IR or Verilog).
      *
      * The name of a module approximates the behavior of the Java Reflection [[`getSimpleName` method
      * https://docs.oracle.com/javase/8/docs/api/java/lang/Class.html#getSimpleName--]] with some modifications:
      *
      * - Anonymous modules will get an `"_Anon"` tag
      * - Modules defined in functions will use their class name and not a numeric name
      *
      * @note If you want a custom or parametric name, override this method.
      */
    def desiredName: String = {
      /* The default module name is derived from the Java reflection derived class name. */
      val baseName = this.getClass.getName

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
      case m: internal.BaseModule.InstanceClone[_] =>
        throwException(s"Internal Error! It's not legal to call .toTarget on an InstanceClone. $m")
      case m: internal.BaseModule.DefinitionClone[_] =>
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
      case m: internal.BaseModule.InstanceClone[_] if m._parent.nonEmpty =>
        m._parent.get.getTarget.instOf(instanceName, name)
      case m: internal.BaseModule.ModuleClone[_] if m._madeFromDefinition =>
        m._parent.get.getTarget.instOf(instanceName, name)
      // Without this, we get the wrong CircuitName for the Definition
      case m: internal.BaseModule.DefinitionClone[_] if m._circuit.nonEmpty =>
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

    /**
      * Internal API. Returns a list of this module's generated top-level ports as a map of a String
      * (FIRRTL name) to the IO object. Only valid after the module is closed.
      *
      * Note: for BlackBoxes (but not ExtModules), this returns the contents of the top-level io
      * object, consistent with what is emitted in FIRRTL.
      *
      * TODO: Use SeqMap/VectorMap when those data structures become available.
      */
    private[chisel3] def getChiselPorts: Seq[(String, Data)] = {
      require(_closed, "Can't get ports before module close")
      _component.get.ports.map { port =>
        (port.id.getRef.asInstanceOf[ModuleIO].name, port.id)
      }
    }

    /** Called at the Module.apply(...) level after this Module has finished elaborating.
      * Returns a map of nodes -> names, for named nodes.
      *
      * Helper method.
      */
    protected def nameIds(rootClass: Class[_]): HashMap[HasId, String] = {
      val names = new HashMap[HasId, String]()

      def name(node: HasId, name: String) {
        // First name takes priority, like suggestName
        // TODO: DRYify with suggestName
        if (!names.contains(node)) {
          names.put(node, name)
        }
      }

      /** Scala generates names like chisel3$util$Queue$$ram for private vals
        * This extracts the part after $$ for names like this and leaves names
        * without $$ unchanged
        */
      def cleanName(name: String): String = name.split("""\$\$""").lastOption.getOrElse(name)

      for (m <- getPublicFields(rootClass)) {
        Builder.nameRecursively(cleanName(m.getName), m.invoke(this), name)
      }

      names
    }

    /** Invokes _onModuleClose on HasIds found via reflection but not bound to hardware
      * (thus not part of _ids)
      * This maintains old naming behavior for non-hardware Data
      */
    private[chisel3] def closeUnboundIds(names: HashMap[HasId, String]): Unit = {
      val idLookup = _ids.toSet
      for ((id, _) <- names if !idLookup(id)) {
        id._onModuleClose
      }
    }

    /** Compatibility function. Allows Chisel2 code which had ports without the IO wrapper to
      * compile under Bindings checks. Does nothing in non-compatibility mode.
      *
      * Should NOT be used elsewhere. This API will NOT last.
      *
      * TODO: remove this, perhaps by removing Bindings checks in compatibility mode.
      */
    def _compatAutoWrapPorts() {}

    /** Chisel2 code didn't require the IO(...) wrapper and would assign a Chisel type directly to
      * io, then do operations on it. This binds a Chisel type in-place (mutably) as an IO.
      */
    protected def _bindIoInPlace(iodef: Data): Unit = {
      // Compatibility code: Chisel2 did not require explicit direction on nodes
      // (unspecified treated as output, and flip on nothing was input).
      // This sets assigns the explicit directions required by newer semantics on
      // Bundles defined in compatibility mode.
      // This recursively walks the tree, and assigns directions if no explicit
      // direction given by upper-levels (override Input / Output) AND element is
      // directly inside a compatibility Bundle determined by compile options.
      def assignCompatDir(data: Data, insideCompat: Boolean): Unit = {
        data match {
          case data: Element if insideCompat => data._assignCompatibilityExplicitDirection
          case data: Element => // Not inside a compatibility Bundle, nothing to be done
          case data: Aggregate =>
            data.specifiedDirection match {
              // Recurse into children to ensure explicit direction set somewhere
              case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip =>
                data match {
                  case record: Record =>
                    val compatRecord = !record.compileOptions.dontAssumeDirectionality
                    record.getElements.foreach(assignCompatDir(_, compatRecord))
                  case vec: Vec[_] =>
                    vec.getElements.foreach(assignCompatDir(_, insideCompat))
                }
              case SpecifiedDirection.Input | SpecifiedDirection.Output => // forced assign, nothing to do
            }
        }
      }

      assignCompatDir(iodef, false)

      iodef.bind(PortBinding(this))
      _ports += iodef
    }

    /** Private accessor for _bindIoInPlace */
    private[chisel3] def bindIoInPlace(iodef: Data): Unit = _bindIoInPlace(iodef)

    /**
      * This must wrap the datatype used to set the io field of any Module.
      * i.e. All concrete modules must have defined io in this form:
      * [lazy] val io[: io type] = IO(...[: io type])
      *
      * Items in [] are optional.
      *
      * The granted iodef must be a chisel type and not be bound to hardware.
      *
      * Also registers a Data as a port, also performing bindings. Cannot be called once ports are
      * requested (so that all calls to ports will return the same information).
      * Internal API.
      *
      * TODO(twigg): Specifically walk the Data definition to call out which nodes
      * are problematic.
      */
    protected def IO[T <: Data](iodef: T): T = chisel3.experimental.IO.apply(iodef)

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
