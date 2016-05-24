// See LICENSE for license details.

package Chisel

import scala.collection.mutable.{ArrayBuffer, HashSet}

import internal._
import internal.Builder.pushCommand
import internal.firrtl
import internal.firrtl.{Command, Component, DefInstance, DefInvalid, ModuleIO}

object Module {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param m the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: Module](bc: => T): T = {
    val parent: Option[Module] = Builder.currentModule
    val m = bc.setRefs() // This will set currentModule!
    m._commands.prepend(DefInvalid(m.io.ref)) // init module outputs
    Builder.currentModule = parent // Back to parent!

    val ports = m.computePorts
    Builder.components += Component(m, m.name, ports, m._commands)
    // Avoid referencing 'parent' in top module
    if(!Builder.currentModule.isEmpty) {
      pushCommand(DefInstance(m, ports))
      m.setupInParent()
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
extends HasId {
  def this(clock: Clock) = this(Option(clock), None)
  def this(reset: Bool)  = this(None, Option(reset))
  def this(clock: Clock, reset: Bool) = this(Option(clock), Option(reset))

  // This function binds the iodef as a port in the hardware graph
  private[Chisel] def Port[T<:Data](iodef: T): iodef.type = {
    // Bind each element of the iodef to being a Port
    Binding.bind(iodef, PortBinder(this), "Error: iodef")
    iodef
  }

  private[this] var ioDefined: Boolean = false

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

  private[Chisel] val _namespace = Builder.globalNamespace.child
  private[Chisel] val _commands = ArrayBuffer[Command]()
  private[Chisel] val _ids = ArrayBuffer[HasId]()
  Builder.currentModule = Some(this)

  /** Name of the instance. */
  val name = Builder.globalNamespace.name(getClass.getName.split('.').last)

  /** IO for this Module. At the Scala level (pre-FIRRTL transformations),
    * connections in and out of a Module may only go through `io` elements.
    */
  def io: Bundle
  val clock = Port(Input(Clock()))
  val reset = Port(Input(Bool()))

  private[Chisel] def addId(d: HasId) { _ids += d }

  private[Chisel] def ports: Seq[(String,Data)] = Vector(
    ("clk", clock), ("reset", reset), ("io", io)
  )

  private[Chisel] def computePorts: Seq[firrtl.Port] =
    for((name, port) <- ports) yield {
      // Port definitions need to know input or output at top-level.
      // By FIRRTL semantics, 'flipped' becomes an Input
      val direction = if(Data.isFlipped(port)) Direction.Input else Direction.Output
      firrtl.Port(port, direction)
    }

  private[Chisel] def setupInParent(): this.type = _parent match {
    case Some(p) => {
      pushCommand(DefInvalid(io.ref)) // init instance inputs
      clock := override_clock.getOrElse(p.clock)
      reset := override_reset.getOrElse(p.reset)
      this
    }
    case None => this
  }

  private[Chisel] def setRefs(): this.type = {
    for ((name, port) <- ports) {
      port.setRef(ModuleIO(this, _namespace.name(name)))
    }

    // Suggest names to nodes using runtime reflection
    val valNames = HashSet[String](getClass.getDeclaredFields.map(_.getName):_*)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName)
    val methods = getClass.getMethods.sortWith(_.getName > _.getName)
    for (m <- methods; if isPublicVal(m)) m.invoke(this) match {
      case (id: HasId) => id.suggestName(m.getName)
      case _ =>
    }

    // All suggestions are in, force names to every node.
    _ids.foreach(_.forceName(default="T", _namespace))
    _ids.foreach(_._onModuleClose)
    this
  }
}
