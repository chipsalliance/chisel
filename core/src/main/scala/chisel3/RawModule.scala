// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.util.Try
import scala.language.experimental.macros
import scala.annotation.nowarn
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
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
              // probes have their refs set eagerly
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
    val component = DefModule(this, name, firrtlPorts, _commands.result())

    // Secret connections can be staged if user bored into children modules
    component.secretCommands ++= stagedSecretCommands
    _component = Some(component)
    _component
  }
  private[chisel3] val stagedSecretCommands = collection.mutable.ArrayBuffer[Command]()

  private[chisel3] def secretConnection(left: Data, right: Data)(implicit si: SourceInfo): Unit = {
    val rhs = (left.probeInfo.nonEmpty, right.probeInfo.nonEmpty) match {
      case (true, true)                                 => ProbeDefine(si, left.lref, Node(right))
      case (true, false) if left.probeInfo.get.writable => ProbeDefine(si, left.lref, RWProbeExpr(Node(right)))
      case (true, false)                                => ProbeDefine(si, left.lref, ProbeExpr(Node(right)))
      case (false, true)                                => Connect(si, left.lref, ProbeRead(Node(right)))
      case (false, false)                               => Connect(si, left.lref, Node(right))
    }
    val secretCommands = if (_closed) {
      _component.get.asInstanceOf[DefModule].secretCommands
    } else {
      stagedSecretCommands
    }
    secretCommands += rhs
  }

  private[chisel3] def initializeInParent(): Unit = {}
}

/** Enforce that the Module.reset be Asynchronous (AsyncReset) */
trait RequireAsyncReset extends Module {
  override final def resetType = Module.ResetType.Asynchronous
}

/** Enforce that the Module.reset be Synchronous (Bool) */
trait RequireSyncReset extends Module {
  override final def resetType = Module.ResetType.Synchronous
}
