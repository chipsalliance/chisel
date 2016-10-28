// See LICENSE for license details.

package chisel3.core

import scala.collection.mutable.ArrayBuffer
import scala.language.experimental.macros
import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.firrtl.{Command => _, _}
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo, UnlocatableSourceInfo}

object Module {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param m the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: Module](bc: => T): T = macro InstTransform.apply[T]

  def do_apply[T <: Module](bc: => T)(implicit sourceInfo: SourceInfo): T = {
    // Don't generate source info referencing parents inside a module, sincce this interferes with
    // module de-duplication in FIRRTL emission.
    val childSourceInfo = UnlocatableSourceInfo

    val parent: Option[Module] = Builder.currentModule
    val m = bc.setRefs() // This will set currentModule!
    m._commands.prepend(DefInvalid(childSourceInfo, m.io.ref)) // init module outputs
    Builder.currentModule = parent // Back to parent!
    val ports = m.computePorts
    val component = Component(m, m.name, ports, m._commands)
    m._component = Some(component)
    Builder.components += component
    // Avoid referencing 'parent' in top module
    if(!Builder.currentModule.isEmpty) {
      pushCommand(DefInstance(sourceInfo, m, ports))
      m.setupInParent(childSourceInfo)
    }
    m
  }
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module(
  override_clock: Option[Clock]=None, override_reset: Option[Bool]=None)
                     (implicit moduleCompileOptions: CompileOptions)
extends HasId {
  // _clock and _reset can be clock and reset in these 2ary constructors
  // once chisel2 compatibility issues are resolved
  def this(_clock: Clock)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), None)(moduleCompileOptions)
  def this(_reset: Bool)(implicit moduleCompileOptions: CompileOptions)  = this(None, Option(_reset))(moduleCompileOptions)
  def this(_clock: Clock, _reset: Bool)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), Option(_reset))(moduleCompileOptions)

  // This function binds the iodef as a port in the hardware graph
  private[chisel3] def Port[T<:Data](iodef: T): iodef.type = {
    // Bind each element of the iodef to being a Port
    Binding.bind(iodef, PortBinder(this), "Error: iodef")
    iodef
  }

  private[core] var ioDefined: Boolean = false

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
   * TODO(twigg): Specifically walk the Data definition to call out which nodes
   * are problematic.
   */
  def IO[T<:Data](iodef: T): iodef.type = {
    require(!ioDefined, "Another IO definition for this module was already declared!")
    ioDefined = true

    Port(iodef)
  }

  private[core] val _namespace = Builder.globalNamespace.child
  private[chisel3] val _commands = ArrayBuffer[Command]()
  private[core] val _ids = ArrayBuffer[HasId]()
  Builder.currentModule = Some(this)

  /** Desired name of this module. */
  def desiredName = this.getClass.getName.split('.').last

  /** Legalized name of this module. */
  final val name = Builder.globalNamespace.name(desiredName)

  /** FIRRTL Module name */
  def modName = name

  /** Keep component for signal names */
  private[chisel3] var _component: Option[Component] = None

  /** Signal name (for simulation). */
  override def instanceName =
    if (_parent == None) name else _component match {
      case None => getRef.name
      case Some(c) => getRef fullName c
    }

  /** IO for this Module. At the Scala level (pre-FIRRTL transformations),
    * connections in and out of a Module may only go through `io` elements.
    */
  def io: Bundle
  val clock = Port(Input(Clock()))
  val reset = Port(Input(Bool()))

  private[chisel3] def addId(d: HasId) { _ids += d }

  private[core] def ports: Seq[(String,Data)] = Vector(
    ("clock", clock), ("reset", reset), ("io", io)
  )

  private[core] def computePorts: Seq[firrtl.Port] = {
    // If we're auto-wrapping IO definitions, do so now.
    if (!(compileOptions.requireIOWrap || ioDefined)) {
      IO(io)
    }
    for ((name, port) <- ports) yield {
      // Port definitions need to know input or output at top-level.
      // By FIRRTL semantics, 'flipped' becomes an Input
      val direction = if(Data.isFirrtlFlipped(port)) Direction.Input else Direction.Output
      firrtl.Port(port, direction)
    }
  }

  private[core] def setupInParent(implicit sourceInfo: SourceInfo): this.type = {
    _parent match {
      case Some(p) => {
        pushCommand(DefInvalid(sourceInfo, io.ref)) // init instance inputs
        clock := override_clock.getOrElse(p.clock)
        reset := override_reset.getOrElse(p.reset)
        this
      }
      case None => this
    }
  }

  private[core] def setRefs(): this.type = {
    for ((name, port) <- ports) {
      port.setRef(ModuleIO(this, _namespace.name(name)))
    }

    /** Recursively suggests names to supported "container" classes
      * Arbitrary nestings of supported classes are allowed so long as the
      * innermost element is of type HasId
      * Currently supported:
      *   - Iterable
      *   - Option
      * (Note that Map is Iterable[Tuple2[_,_]] and thus excluded)
      */
    def nameRecursively(prefix: String, nameMe: Any): Unit =
      nameMe match {
        case (id: HasId) => id.suggestName(prefix)
        case Some(elt) => nameRecursively(prefix, elt)
        case (iter: Iterable[_]) if iter.hasDefiniteSize =>
          for ((elt, i) <- iter.zipWithIndex) {
            nameRecursively(s"${prefix}_${i}", elt)
          }
        case _ => // Do nothing
      }
    for (m <- getPublicFields(classOf[Module])) {
      nameRecursively(m.getName, m.invoke(this))
    }

    // For Module instances we haven't named, suggest the name of the Module
    _ids foreach {
      case m: Module => m.suggestName(m.name)
      case _ =>
    }

    // All suggestions are in, force names to every node.
    _ids.foreach(_.forceName(default="T", _namespace))
    _ids.foreach(_._onModuleClose)
    this
  }
  // For debuggers/testers
  lazy val getPorts = computePorts
  val compileOptions = moduleCompileOptions
}
