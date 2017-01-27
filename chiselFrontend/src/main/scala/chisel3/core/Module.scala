// See LICENSE for license details.

package chisel3.core

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.JavaConversions._
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.firrtl.{Command => _, _}
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, UnlocatableSourceInfo}

object Module {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: => T): T = macro InstTransform.apply[T]

  def do_apply[T <: BaseModule](bc: => T)(implicit sourceInfo: SourceInfo): T = {
    if (Builder.readyForModuleConstr) {
      throwException("Error: Called Module() twice without instantiating a Module." +
                     sourceInfo.makeMessage(" See " + _))
    }
    Builder.readyForModuleConstr = true

    val parent: Option[BaseModule] = Builder.currentModule
    val whenDepth: Int = Builder.whenDepth
    val clockAndReset: Option[ClockAndReset] = Builder.currentClockAndReset

    // Execute the module, this has the following side effects:
    //   - set currentModule
    //   - unset readyForModuleConstr
    //   - reset whenDepth to 0
    //   - set currentClockAndReset
    val module: T = bc  // Module is actually invoked here

    if (Builder.whenDepth != 0) {
      throwException("Internal Error! When depth is != 0, this should not be possible")
    }
    if (Builder.readyForModuleConstr) {
      throwException("Error: attempted to instantiate a Module, but nothing happened. " +
                     "This is probably due to rewrapping a Module instance with Module()." +
                     sourceInfo.makeMessage(" See " + _))
    }
    Builder.currentModule = parent // Back to parent!
    Builder.whenDepth = whenDepth
    Builder.currentClockAndReset = clockAndReset // Back to clock and reset scope

    val component = module.generateComponent()
    Builder.components += component

    // Handle connections at enclosing scope
    if(!Builder.currentModule.isEmpty) {
      pushCommand(DefInstance(sourceInfo, module, component.ports))
      module.initializeInParent()
    }
    module
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  def reset: Bool = Builder.forcedReset
}

/** Abstract base class for Modules, an instantiable organizational unit for RTL.
  */
// TODO: seal this?
abstract class BaseModule extends HasId {
  //
  // Builder Internals - this tracks which Module RTL construction belongs to.
  //
  if (!Builder.readyForModuleConstr) {
    throwException("Error: attempted to instantiate a Module without wrapping it in Module().")
  }
  readyForModuleConstr = false

  Builder.currentModule = Some(this)
  Builder.whenDepth = 0

  //
  // Module Construction Internals
  //
  protected var _closed = false

  // Fresh Namespace because in Firrtl, Modules namespaces are disjoint with the global namespace
  private[core] val _namespace = Namespace.empty
  protected val _ids = ArrayBuffer[HasId]()
  private[chisel3] def addId(d: HasId) {
    require(!_closed, "Can't write to module after module close")
    _ids += d
  }

  protected val _ports = new ArrayBuffer[Data]()

  /** Generates the FIRRTL Component (Module or Blackbox) of this Module.
    * Also closes the module so no more construction can happen inside.
    */
  private[core] def generateComponent(): Component

  /** Sets up this module in the parent context
    */
  private[core] def initializeInParent()

  //
  // Chisel Internals
  //
  /** Desired name of this module. Override this to give this module a custom, perhaps parametric,
    * name.
    */
  def desiredName = this.getClass.getName.split('.').last

  /** Legalized name of this module. */
  final val name = Builder.globalNamespace.name(desiredName)

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

    /** Recursively suggests names to supported "container" classes
      * Arbitrary nestings of supported classes are allowed so long as the
      * innermost element is of type HasId
      * (Note: Map is Iterable[Tuple2[_,_]] and thus excluded)
      */
    def nameRecursively(prefix: String, nameMe: Any): Unit =
      nameMe match {
        case (id: HasId) => name(id, prefix)
        case Some(elt) => nameRecursively(prefix, elt)
        case (iter: Iterable[_]) if iter.hasDefiniteSize =>
          for ((elt, i) <- iter.zipWithIndex) {
            nameRecursively(s"${prefix}_${i}", elt)
          }
        case _ => // Do nothing
      }

    /** Scala generates names like chisel3$util$Queue$$ram for private vals
      * This extracts the part after $$ for names like this and leaves names
      * without $$ unchanged
      */
    def cleanName(name: String): String = name.split("""\$\$""").lastOption.getOrElse(name)

    for (m <- getPublicFields(rootClass)) {
      nameRecursively(cleanName(m.getName), m.invoke(this))
    }

    // For Module instances we haven't named, suggest the name of the Module
    _ids foreach {
      case m: BaseModule => name(m, m.desiredName)
      case _ =>
    }

    names
  }

  /** Compatibility function. Allows Chisel2 code which had ports without the IO wrapper to
    * compile under Bindings checks. Does nothing in non-compatibility mode.
    *
    * Should NOT be used elsewhere. This API will NOT last.
    *
    * TODO: remove this, perhaps by removing Bindings checks in compatibility mode.
    */
  def _autoWrapPorts() {}

  //
  // BaseModule User API functions
  //
  protected def annotate(annotation: ChiselAnnotation): Unit = {
    Builder.annotations += annotation
  }

  /**
   * This must wrap the datatype used to set the io field of any Module.
   * i.e. All concrete modules must have defined io in this form:
   * [lazy] val io[: io type] = IO(...[: io type])
   *
   * Items in [] are optional.
   *
   * The granted iodef WILL NOT be cloned (to allow for more seamless use of
   * anonymous Bundles in the IO) and thus CANNOT have been bound to any logic.
   * This will error if any node is bound (e.g. due to logic in a Bundle
   * constructor, which is considered improper).
   *
   * Also registers a Data as a port, also performing bindings. Cannot be called once ports are
   * requested (so that all calls to ports will return the same information).
   * Internal API.
   *
   * TODO(twigg): Specifically walk the Data definition to call out which nodes
   * are problematic.
   */
  protected def IO[T<:Data](iodef: T): iodef.type = {
    require(!_closed, "Can't add more ports after module close")
    // Bind each element of the iodef to being a Port
    Binding.bind(iodef, PortBinder(this), "Error: iodef")
    _ports += iodef
    iodef
  }

  //
  // Internal Functions
  //

  /** Keep component for signal names */
  private[chisel3] var _component: Option[Component] = None

  /** Signal name (for simulation). */
  override def instanceName =
    if (_parent == None) name else _component match {
      case None => getRef.name
      case Some(c) => getRef fullName c
    }

}

/** Abstract base class for Modules that contain Chisel RTL.
  */
abstract class UserModule(implicit moduleCompileOptions: CompileOptions)
    extends BaseModule {
  //
  // RTL construction internals
  //
  protected val _commands = ArrayBuffer[Command]()
  private[chisel3] def addCommand(c: Command) {
    require(!_closed, "Can't write to module after module close")
    _commands += c
   }

  //
  // Other Internal Functions
  //
  // For debuggers/testers, TODO: refactor out into proper public API
  private var _firrtlPorts: Option[Seq[firrtl.Port]] = None
  lazy val getPorts = _firrtlPorts.get

  val compileOptions = moduleCompileOptions

  private[core] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val names = nameIds(classOf[UserModule])

    // Ports get first naming priority, since they are part of a Module's IO spec
    for (port <- _ports) {
      require(names.contains(port), s"Unable to name port $port in $this")
      port.setRef(ModuleIO(this, _namespace.name(names(port))))
      // Initialize output as unused
      _commands.prepend(DefInvalid(UnlocatableSourceInfo, port.ref))
    }

    // Then everything else gets named
    for ((node, name) <- names) {
      node.suggestName(name)
    }

    // All suggestions are in, force names to every node.
    for (id <- _ids) {
      id.forceName(default="_T", _namespace)
      id._onModuleClose
    }

    val firrtlPorts = for (port <- _ports) yield {
      // Port definitions need to know input or output at top-level. 'flipped' means Input.
      val direction = if(Data.isFirrtlFlipped(port)) Direction.Input else Direction.Output
      firrtl.Port(port, direction)
    }
    _firrtlPorts = Some(firrtlPorts)

    val component = DefModule(this, name, firrtlPorts, _commands)
    _component = Some(component)
    component
  }
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class ImplicitModule()(implicit moduleCompileOptions: CompileOptions)
    extends UserModule {
  // Implicit clock and reset pins
  val clock = IO(Input(Clock()))
  val reset = IO(Input(Bool()))

  // Setup ClockAndReset
  Builder.currentClockAndReset = Some(ClockAndReset(clock, reset))

  private[core] def initializeInParent() {
    // Don't generate source info referencing parents inside a module, since this interferes with
    // module de-duplication in FIRRTL emission.
    implicit val sourceInfo = UnlocatableSourceInfo

    for (port <- _ports) {
      pushCommand(DefInvalid(sourceInfo, port.ref))
    }

    clock := Builder.forcedClock
    reset := Builder.forcedReset
  }
}

/** Legacy Module class that restricts IOs to just io, clock, and reset, and provides a constructor
  * for threading through explicit clock and reset.
  *
  * While this class isn't planned to be removed anytime soon (there are benefits to restricting
  * IO), the clock and reset constructors will be phased out. Recommendation is to wrap the module
  * in a withClock/withReset/withClockAndReset block, or directly hook up clock or reset IO pins.
  */
abstract class LegacyModule(
    override_clock: Option[Clock]=None, override_reset: Option[Bool]=None)
    (implicit moduleCompileOptions: CompileOptions)
    extends ImplicitModule {
  // _clock and _reset can be clock and reset in these 2ary constructors
  // once chisel2 compatibility issues are resolved
  def this(_clock: Clock)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), None)(moduleCompileOptions)
  def this(_reset: Bool)(implicit moduleCompileOptions: CompileOptions)  = this(None, Option(_reset))(moduleCompileOptions)
  def this(_clock: Clock, _reset: Bool)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), Option(_reset))(moduleCompileOptions)

  // IO for this Module. At the Scala level (pre-FIRRTL transformations),
  // connections in and out of a Module may only go through `io` elements.
  def io: Record

  // Allow access to bindings from the compatibility package
  protected def _ioPortBound() = _ports contains io

  protected override def nameIds(rootClass: Class[_]): HashMap[HasId, String] = {
    val names = super.nameIds(rootClass)

    // Allow IO naming without reflection
    names.put(io, "io")
    names.put(clock, "clock")
    names.put(reset, "reset")

    names
  }

  private[core] override def generateComponent(): Component = {
    _autoWrapPorts()  // pre-IO(...) compatibility hack

    // Restrict IO to just io, clock, and reset
    require(io != null, "Module must have io")
    require(_ports contains io, "Module must have io wrapped in IO(...)")
    require((_ports contains clock) && (_ports contains reset), "Internal error, module did not have clock or reset as IO")
    require(_ports.size == 3, "Module must only have io, clock, and reset as IO")

    super.generateComponent()
  }

  private[core] override def initializeInParent() {
    // Don't generate source info referencing parents inside a module, since this interferes with
    // module de-duplication in FIRRTL emission.
    implicit val sourceInfo = UnlocatableSourceInfo

    pushCommand(DefInvalid(sourceInfo, io.ref))

    override_clock match {
      case Some(override_clock) => clock := override_clock
      case _ => clock := Builder.forcedClock
    }

    override_reset match {
      case Some(override_reset) => reset := override_reset
      case _ => reset := Builder.forcedReset
    }
  }
}

