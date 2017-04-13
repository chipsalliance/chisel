// See LICENSE for license details.

package chisel3.core

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.JavaConversions._
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.firrtl.{Command => _, _}
import chisel3.internal.sourceinfo.UnlocatableSourceInfo

/** Abstract base class for Modules that contain Chisel RTL.
  */
abstract class UserModule(implicit moduleCompileOptions: CompileOptions)
    extends BaseModule {
  //
  // RTL construction internals
  //
  private val _commands = ArrayBuffer[Command]()
  private[chisel3] def addCommand(c: Command) {
    require(!_closed, "Can't write to module after module close")
    _commands += c
  }
  protected def getCommands = {
    require(_closed, "Can't get commands before module close")
    _commands.toSeq
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
    for (port <- getModulePorts) {
      require(names.contains(port), s"Unable to name port $port in $this")
      port.setRef(ModuleIO(this, _namespace.name(names(port))))
    }

    // Then everything else gets named
    for ((node, name) <- names) {
      node.suggestName(name)
    }

    // All suggestions are in, force names to every node.
    for (id <- getIds) {
      id match {
        case id: BaseModule => id.forceName(default=id.desiredName, _namespace)
        case id => id.forceName(default="_T", _namespace)
      }
      id._onModuleClose
    }

    val firrtlPorts = for (port <- getModulePorts) yield {
      // Port definitions need to know input or output at top-level. 'flipped' means Input.
      val direction = if(Data.isFirrtlFlipped(port)) Direction.Input else Direction.Output
      firrtl.Port(port, direction)
    }
    _firrtlPorts = Some(firrtlPorts)

    // Generate IO invalidation commands to initialize outputs as unused
    val invalidateCommands = getModulePorts map {port => DefInvalid(UnlocatableSourceInfo, port.ref)}
    
    val component = DefModule(this, name, firrtlPorts, invalidateCommands ++ getCommands)
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
abstract class ImplicitModule(implicit moduleCompileOptions: CompileOptions)
    extends UserModule {
  // Implicit clock and reset pins
  val clock = IO(Input(Clock()))
  val reset = IO(Input(Bool()))

  // Setup ClockAndReset
  Builder.currentClockAndReset = Some(ClockAndReset(clock, reset))

  private[core] def initializeInParent() {
    implicit val sourceInfo = UnlocatableSourceInfo
        
    for (port <- getModulePorts) {
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
  protected def _ioPortBound() = portsContains(io)

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
    require(portsContains(io), "Module must have io wrapped in IO(...)")
    require((portsContains(clock)) && (portsContains(reset)), "Internal error, module did not have clock or reset as IO")
    require(portsSize == 3, "Module must only have io, clock, and reset as IO")

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
