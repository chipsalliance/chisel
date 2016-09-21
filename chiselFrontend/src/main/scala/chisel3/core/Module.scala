// See LICENSE for license details.

package chisel3.core

import scala.collection.mutable.ArrayBuffer
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.Builder.dynamicContext
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, InstTransform, UnlocatableSourceInfo}

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
    val component = Component(m, m.name, ports, m._commands)
    m._component = Some(component)
    Builder.components += component
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
  def this(_clock: Clock) = this(Option(_clock), None)
  def this(_reset: Bool)  = this(None, Option(_reset))
  def this(_clock: Clock, _reset: Bool) = this(Option(_clock), Option(_reset))

  private[core] val _namespace = Builder.globalNamespace.child
  private[chisel3] val _commands = ArrayBuffer[Command]()
  private[core] val _ids = ArrayBuffer[HasId]()
  dynamicContext.currentModule = Some(this)

  /** Desired name of this module. */
  def desiredName = this.getClass.getName.split('.').last

  /** Legalized name of this module. */
  final val name = Builder.globalNamespace.name(desiredName)

  /** FIRRTL Module name */
  private var _modName: Option[String] = None
  private[chisel3] def setModName(name: String) = _modName = Some(name)
  def modName = _modName match {
    case Some(name) => name
    case None => throwException("modName should be called after circuit elaboration")
  }

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
  val clock = Clock(INPUT)
  val reset = Bool(INPUT)

  private[chisel3] def addId(d: HasId) { _ids += d }

  private[core] def ports: Seq[(String,Data)] = Vector(
    ("clock", clock), ("reset", reset), ("io", io)
  )

  private[core] def computePorts = for((name, port) <- ports) yield {
    val bundleDir = if (port.isFlip) INPUT else OUTPUT
    Port(port, if (port.dir == NO_DIR) bundleDir else port.dir)
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

    // Suggest names to nodes using runtime reflection
    def getValNames(c: Class[_]): Set[String] = {
      if (c == classOf[Module]) Set()
      else getValNames(c.getSuperclass) ++ c.getDeclaredFields.map(_.getName)
    }
    val valNames = getValNames(this.getClass)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName)

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
    val methods = getClass.getMethods.sortWith(_.getName > _.getName)
    for (m <- methods if isPublicVal(m)) {
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
}
