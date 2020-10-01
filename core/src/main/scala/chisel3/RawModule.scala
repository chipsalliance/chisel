// See LICENSE for license details.

package chisel3

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.JavaConversions._
import scala.language.experimental.macros

import chisel3.experimental.BaseModule
import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.UnlocatableSourceInfo

/** Abstract base class for Modules that contain Chisel RTL.
  * This abstract base class is a user-defined module which does not include implicit clock and reset and supports
  * multiple IO() declarations.
  */
abstract class RawModule(implicit moduleCompileOptions: CompileOptions)
    extends BaseModule
    with CommandMemoization {
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

  private[chisel3] def namePorts(names: HashMap[HasId, String]): Unit = {
    for (port <- getModulePorts) {
      port.suggestedName.orElse(names.get(port)) match {
        case Some(name) =>
          if (_namespace.contains(name)) {
            Builder.error(s"""Unable to name port $port to "$name" in $this,""" +
              " name is already taken by another port!")
          }
          port.setRef(ModuleIO(this, _namespace.name(name)))
        case None =>
          Builder.error(s"Unable to name port $port in $this, " +
            "try making it a public field of the Module")
          port.setRef(ModuleIO(this, "<UNNAMED>"))
      }
    }
  }


  private[chisel3] override def generateComponent(): Component = { // scalastyle:ignore cyclomatic.complexity
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val names = nameIds(classOf[RawModule])

    // Ports get first naming priority, since they are part of a Module's IO spec
    namePorts(names)

    // Then everything else gets named
    for ((node, name) <- names) {
      node.suggestName(name)
    }

    // All suggestions are in, force names to every node.
    for (id <- getIds) {
      id match {
        case id: BaseModule => id.forceName(default=id.desiredName, _namespace)
        case id: MemBase[_] => id.forceName(default="_T", _namespace)
        case id: Data  =>
          if (id.isSynthesizable) {
            id.topBinding match {
              case OpBinding(_) | MemoryPortBinding(_) | PortBinding(_) | RegBinding(_) | WireBinding(_) =>
                id.forceName(default="_T", _namespace)
              case _ =>  // don't name literals
            }
          } // else, don't name unbound types
      }
      id._onModuleClose
    }

    val firrtlPorts = getModulePorts map { port: Data =>
      // Special case Vec to make FIRRTL emit the direction of its
      // element.
      // Just taking the Vec's specifiedDirection is a bug in cases like
      // Vec(Flipped()), since the Vec's specifiedDirection is
      // Unspecified.
      val direction = port match {
        case v: Vec[_] => v.specifiedDirection match {
          case SpecifiedDirection.Input => SpecifiedDirection.Input
          case SpecifiedDirection.Output => SpecifiedDirection.Output
          case SpecifiedDirection.Flip => SpecifiedDirection.flip(v.sample_element.specifiedDirection)
          case SpecifiedDirection.Unspecified => v.sample_element.specifiedDirection
        }
        case _ => port.specifiedDirection
      }

      Port(port, direction)
    }
    _firrtlPorts = Some(firrtlPorts)

    // Generate IO invalidation commands to initialize outputs as unused,
    //  unless the client wants explicit control over their generation.
    val invalidateCommands = {
      if (!compileOptions.explicitInvalidate) {
        getModulePorts map { port => DefInvalid(UnlocatableSourceInfo, port.ref) }
      } else {
        Seq()
      }
    }
    val component = DefModule(this, name, firrtlPorts, invalidateCommands ++ getCommands)
    _component = Some(component)
    component
  }

  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    if (!parentCompileOptions.explicitInvalidate) {
      for (port <- getModulePorts) {
        pushCommand(DefInvalid(sourceInfo, port.ref))
      }
    }
  }
}

trait RequireAsyncReset extends MultiIOModule {
  override private[chisel3] def mkReset: AsyncReset = AsyncReset()
}

trait RequireSyncReset extends MultiIOModule {
  override private[chisel3] def mkReset: Bool = Bool()
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  * This abstract base class includes an implicit clock and reset.
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class MultiIOModule(implicit moduleCompileOptions: CompileOptions)
    extends RawModule {
  // Implicit clock and reset pins
  final val clock: Clock = IO(Input(Clock()))
  final val reset: Reset = IO(Input(mkReset))

  private[chisel3] def mkReset: Reset = {
    // Top module and compatibility mode use Bool for reset
    val inferReset = _parent.isDefined && moduleCompileOptions.inferModuleReset
    if (inferReset) Reset() else Bool()
  }

  // Setup ClockAndReset
  Builder.currentClock = Some(clock)
  Builder.currentReset = Some(reset)

  private[chisel3] override def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    super.initializeInParent(parentCompileOptions)
    clock := Builder.forcedClock
    reset := Builder.forcedReset
  }
}

package internal {

  /** Legacy Module class that restricts IOs to just io, clock, and reset, and provides a constructor
    * for threading through explicit clock and reset.
    *
    * While this class isn't planned to be removed anytime soon (there are benefits to restricting
    * IO), the clock and reset constructors will be phased out. Recommendation is to wrap the module
    * in a withClock/withReset/withClockAndReset block, or directly hook up clock or reset IO pins.
    */
  abstract class LegacyModule(implicit moduleCompileOptions: CompileOptions)
    extends MultiIOModule {
    // These are to be phased out
    protected var override_clock: Option[Clock] = None
    protected var override_reset: Option[Bool] = None

    // IO for this Module. At the Scala level (pre-FIRRTL transformations),
    // connections in and out of a Module may only go through `io` elements.
    def io: Record

    // Allow access to bindings from the compatibility package
    protected def _compatIoPortBound() = portsContains(io)// scalastyle:ignore method.name

    private[chisel3] override def namePorts(names: HashMap[HasId, String]): Unit = {
      for (port <- getModulePorts) {
        // This should already have been caught
        if (!names.contains(port)) throwException(s"Unable to name port $port in $this")
        val name = names(port)
        port.setRef(ModuleIO(this, _namespace.name(name)))
      }
    }

    private[chisel3] override def generateComponent(): Component = {
      _compatAutoWrapPorts()  // pre-IO(...) compatibility hack

      // Restrict IO to just io, clock, and reset
      require(io != null, "Module must have io")
      require(portsContains(io), "Module must have io wrapped in IO(...)")
      require((portsContains(clock)) && (portsContains(reset)), "Internal error, module did not have clock or reset as IO") // scalastyle:ignore line.size.limit
      require(portsSize == 3, "Module must only have io, clock, and reset as IO")

      super.generateComponent()
    }

    private[chisel3] override def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
      // Don't generate source info referencing parents inside a module, since this interferes with
      // module de-duplication in FIRRTL emission.
      implicit val sourceInfo = UnlocatableSourceInfo

      if (!parentCompileOptions.explicitInvalidate) {
        pushCommand(DefInvalid(sourceInfo, io.ref))
      }

      clock := override_clock.getOrElse(Builder.forcedClock)
      reset := override_reset.getOrElse(Builder.forcedReset)
    }
  }
}
