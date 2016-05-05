// See LICENSE for license details.

package Chisel

import scala.collection.mutable.{ArrayBuffer, HashSet}
import scala.language.experimental.macros

import internal._
import internal.Builder.pushCommand
import internal.Builder.dynamicContext
import internal.firrtl._
import internal.sourceinfo.{SourceInfo, InstTransform, UnlocatableSourceInfo}

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

    val parent = dynamicContext.currentModule
    val m = bc.setRefs()
    m._commands.prepend(DefInvalid(childSourceInfo, m.io.ref)) // init module outputs
    dynamicContext.currentModule = parent
    val ports = m.computePorts
    Builder.components += Component(m, m.name, ports, m._commands)
    pushCommand(DefInstance(sourceInfo, m, ports))
    m.setupInParent(childSourceInfo)
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
  // _clock and _reset can be clock and reset in these 2ary constructors
  // once chisel2 compatibility issues are resolved
  def this(_clock: Clock) = this(Some(_clock), None)
  def this(_reset: Bool)  = this(None, Some(_reset))
  def this(_clock: Clock, _reset: Bool) = this(Some(_clock), Some(_reset))

  private[Chisel] val _namespace = Builder.globalNamespace.child
  private[Chisel] val _commands = ArrayBuffer[Command]()
  private[Chisel] val _ids = ArrayBuffer[HasId]()
  dynamicContext.currentModule = Some(this)

  /** Name of the instance. */
  val name = Builder.globalNamespace.name(getClass.getName.split('.').last)

  /** IO for this Module. At the Scala level (pre-FIRRTL transformations),
    * connections in and out of a Module may only go through `io` elements.
    */
  def io: Bundle
  val clock = Clock(INPUT)
  val reset = Bool(INPUT)

  private[Chisel] def addId(d: HasId) { _ids += d }

  private[Chisel] def ports: Seq[(String,Data)] = Vector(
    ("clk", clock), ("reset", reset), ("io", io)
  )

  private[Chisel] def computePorts = for((name, port) <- ports) yield {
    val bundleDir = if (port.isFlip) INPUT else OUTPUT
    Port(port, if (port.dir == NO_DIR) bundleDir else port.dir)
  }

  private[Chisel] def setupInParent(implicit sourceInfo: SourceInfo): this.type = {
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
