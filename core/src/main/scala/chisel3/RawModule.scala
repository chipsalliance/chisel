// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.util.Try
import scala.language.experimental.macros
import scala.annotation.nowarn
import chisel3.experimental.{BaseModule, UnlocatableSourceInfo}
import chisel3.internal._
import chisel3.experimental.hierarchy.{InstanceClone, ModuleClone}
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import _root_.firrtl.annotations.{IsModule, ModuleTarget}
import scala.collection.immutable.VectorBuilder

/** Abstract base class for Modules that contain Chisel RTL.
  * This abstract base class is a user-defined module which does not include implicit clock and reset and supports
  * multiple IO() declarations.
  */
@nowarn("msg=class Port") // delete when Port becomes private
abstract class RawModule extends BaseModule {
  //
  // RTL construction internals
  //
  // Perhaps this should be an ArrayBuffer (or ArrayBuilder), but DefModule is public and has Seq[Command]
  // so our best option is to share a single Seq datastructure with that
  private val _commands = new VectorBuilder[Command]()
  private[chisel3] def addCommand(c: Command): Unit = {
    require(!_closed, "Can't write to module after module close")
    _commands += c
  }
  protected def getCommands: Seq[Command] = {
    require(_closed, "Can't get commands before module close")
    // Unsafe cast but we know that any RawModule uses a DefModule
    // _component is defined as a var on BaseModule and we cannot override mutable vars
    _component.get.asInstanceOf[DefModule].commands
  }

  //
  // Other Internal Functions
  //
  private var _firrtlPorts: Option[Seq[firrtl.Port]] = None

  private[chisel3] def checkPorts(): Unit = {
    for ((port, source) <- getModulePortsAndLocators) {
      if (port._computeName(None).isEmpty) {
        Builder.error(
          s"Unable to name port $port in $this, " +
            s"try making it a public field of the Module ${source.makeMessage(x => x)}"
        )(UnlocatableSourceInfo)
      }
    }
  }

  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    // Check to make sure that all ports can be named
    checkPorts()

    // Now that elaboration is complete for this Module, we can finalize names
    for (id <- getIds) {
      id match {
        case id: ModuleClone[_]   => id.setRefAndPortsRef(_namespace) // special handling
        case id: InstanceClone[_] => id.setAsInstanceRef()
        case id: BaseModule       => id.forceName(default = id.desiredName, _namespace)
        case id: MemBase[_]       => id.forceName(default = "MEM", _namespace)
        case id: stop.Stop        => id.forceName(default = "stop", _namespace)
        case id: assert.Assert    => id.forceName(default = "assert", _namespace)
        case id: assume.Assume    => id.forceName(default = "assume", _namespace)
        case id: cover.Cover      => id.forceName(default = "cover", _namespace)
        case id: printf.Printf => id.forceName(default = "printf", _namespace)
        case id: Data =>
          if (id.isSynthesizable) {
            id.topBinding match {
              case OpBinding(_, _) =>
                id.forceName(default = "_T", _namespace)
              case MemoryPortBinding(_, _) =>
                id.forceName(default = "MPORT", _namespace)
              case PortBinding(_) =>
                id.forceName(default = "PORT", _namespace, true, x => ModuleIO(this, x))
              case RegBinding(_, _) =>
                id.forceName(default = "REG", _namespace)
              case WireBinding(_, _) =>
                id.forceName(default = "_WIRE", _namespace)
              case _ => // don't name literals
            }
          } // else, don't name unbound types
      }
    }

    val firrtlPorts = getModulePortsAndLocators.map {
      case (port, sourceInfo) =>
        Port(port, port.specifiedDirection, sourceInfo)
    }
    _firrtlPorts = Some(firrtlPorts)

    // Generate IO invalidation commands to initialize outputs as unused,
    //  unless the client wants explicit control over their generation.
    val invalidateCommands = {
      if (this.isInstanceOf[ImplicitInvalidate]) {
        getModulePortsAndLocators.map { case (port, sourceInfo) => DefInvalid(sourceInfo, port.ref) }
      } else {
        Seq()
      }
    }
    val component = DefModule(this, name, firrtlPorts, invalidateCommands ++: _commands.result())
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    if (Builder.currentModule.get.isInstanceOf[ImplicitInvalidate]) {
      for ((port, sourceInfo) <- getModulePortsAndLocators) {
        pushCommand(DefInvalid(sourceInfo, port.ref))
      }
    }
  }
}

trait RequireAsyncReset extends Module {
  override private[chisel3] def mkReset: AsyncReset = AsyncReset()
}

trait RequireSyncReset extends Module {
  override private[chisel3] def mkReset: Bool = Bool()
}

/** Mix with a [[RawModule]] to automatically connect DontCare to the module's ports, wires, and children instance IOs. */
trait ImplicitInvalidate { self: RawModule => }
