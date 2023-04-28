// SPDX-License-Identifier: Apache-2.0
package chisel3.interface

import chisel3.{BlackBox => _, Module => _, _}
import chisel3.experimental.{BaseModule, FlatIO}
import chisel3.experimental.dataview._
import scala.annotation.implicitNotFound

@implicitNotFound(
  "this method requires information from the separable compilation implementation, please bring one into scope as an `implicit val`. You can also consult the team that owns the implementation to refer to which one you should use!"
)
trait ConformsTo[Intf <: Interface, Mod <: BaseModule] {

  /** Return the module that conforms to a port-level interface. */
  private[interface] def genModule(): Mod

  /** Define how this module hooks up to the port-level interface. */
  private[interface] def portMap: Seq[(Mod, Intf#Ports) => (Data, Data)]

}

/** Functionality which is common */
sealed trait InterfaceCommon {

  private[interface] type Ports <: Record

  /** Returns the Record that is the port-level interface. */
  private[interface] val ports: Ports

}

/** A generator of different Interfaces. Currently, this is just an Interface
  * that is not a Singleton.
  */
trait InterfaceGenerator extends InterfaceCommon

/** Utilities related to Interface infastructure */
object Interface {

  /** An exception that indicates that the component-side conformance failed. This
    * is used to wrap other exceptions which may hide the cause of the invalid
    * conformance, e.g., a view was non-total.
    * @param message the message for this exception
    * @param cause an optional exception that caused this exception to occur
    */
  private case class InvalidConformance(message: String, cause: Throwable = null)
      extends ChiselException(message, cause)

}

/** An interface between hardware units. Any module that implements this
  * interface may be separately compiled from any module that instantiates this
  * interface.
  */
trait Interface extends InterfaceCommon { self: Singleton =>

  /** This types represents the type of a valid conformance to this Interface.
    */
  private type Conformance[Mod <: BaseModule] = ConformsTo[this.type, Mod]

  /** The name of this interface. This will be used as the name of any module
    * that implements this interface.
    * I.e., this is the name of the `BlackBox` and `Module` that are provided
    * below. The implementation of this method is just coming up with a good
    * name derived from the class name. (The name of the Interface in FIRRTL
    * will be the name of the Scala class that extends the Interface.)
    */
  private[interface] def interfaceName: String = {
    val className = getClass().getName()
    className
      .drop(className.lastIndexOf('.') + 1)
      .split('$')
      .filterNot(_.forall(_.isDigit))
      .last
  }

  sealed trait Entity { this: BaseModule =>
    val io: Ports

    override final def desiredName = interfaceName
  }

  object Wrapper {

    /** The black box that has the same ports as this interface. This is what is
      * instantiated by any user of this interface, i.e., a test harness.
      */
    final class BlackBox extends chisel3.BlackBox with Entity {
      final val io = IO(ports)
    }

    /** The module that wraps any module which conforms to this Interface.
      */
    final class Module[B <: BaseModule](
    )(
      implicit conformance: Conformance[B])
        extends RawModule
        with Entity {
      final val io = FlatIO(ports)

      // Use a dummy clock and reset connection when constructing the module.
      // This is fine as we rely on DataView to catch missing connections to
      // clock and reset.  The dummy clock and reset will never be used.
      private val internal = withClockAndReset(
        WireInit(Clock(), DontCare),
        WireInit(Reset(), DontCare)
      ) {
        chisel3.Module(conformance.genModule())
      }

      private implicit val pm = PartialDataView[B, Ports](
        _ => ports.cloneType,
        conformance.portMap: _*
      )

      // If the view fails, report this with a slightly better error message.
      try {
        io :<>= internal.viewAs[Ports]
      } catch {
        case e: InvalidViewException =>
          throw Interface.InvalidConformance(
            s"unable to conform module '${internal.name}' to interface '$interfaceName' (see InvalidViewException below for more information)",
            e
          )
      }
    }

    /** A stub module that implements the interface. All IO of this module are
      * just tied off.
      */
    final class Stub extends RawModule with Entity {
      final val io = FlatIO(ports)
      io := DontCare
      dontTouch(io)
    }

  }

}
