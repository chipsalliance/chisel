package chisel3

import chisel3.internal.requireIsChiselType // Fix ambiguous import
import chisel3.internal.{throwException, Builder}
import chisel3.experimental.SourceInfo
import chisel3.properties.{Class, Property}

object IO {

  /** Constructs a port for the current Module
    *
    * This must wrap the datatype used to create an io port of any Module.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    *
    * Also registers a Data as a port, also performing bindings. Cannot be called once ports are
    * requested (so that all calls to ports will return the same information).
    * Internal API.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    val module = Module.currentModule.get // Impossible to fail
    if (!module.isIOCreationAllowed)
      Builder.error(
        s"This module cannot have IOs instantiated after disallowing IOs: ${module._whereIOCreationIsDisallowed
          .map(_.makeMessage { s: String => s })
          .mkString(",")}"
      )
    require(!module.isClosed, "Can't add more ports after module close")
    val prevId = Builder.idGen.value
    val data = iodef // evaluate once (passed by name)
    requireIsChiselType(data, "io type")

    // Fail if the module is a Class, and the type is Data.
    module match {
      case _: Class => {
        data match {
          case _: Property[_] => ()
          case _ => Builder.error(s"Class ports must be Property type, but found ${data._localErrorContext}.")
        }
      }
      case _ => ()
    }

    // Clone the IO so we preserve immutability of data types
    // Note: we don't clone if the data is fresh (to avoid unnecessary clones)
    val iodefClone =
      if (!data.mustClone(prevId)) data
      else
        try {
          data.cloneTypeFull
        } catch {
          // For now this is going to be just a deprecation so we don't suddenly break everyone's code
          case e: AutoClonetypeException =>
            Builder.deprecated(e.getMessage, Some(s"${data.getClass}"))
            data
        }
    module.bindIoInPlace(iodefClone)
    iodefClone
  }
}

object Incoming {

  /** Constructs an incoming port of the provided `iodef` chisel type for the current Module
    *
    * The following example creates a input (and mixed-alignment) aggregate port in the current module
    *
    * ```scala
    *   val i = Incoming(new Decoupled(UInt(32.W)))
    * ```
    * In this example, the sub-elements of `i` are ports of the following direction:
    *  * `i.bits` is an input port
    *  * `i.valid` is an input port
    *  * `i.ready` is an output port
    *
    * Note that `Incoming(foo)` is equivalent to `IO(Flipped(foo))`
    *
    * This must wrap the datatype used to create an io port of any Module.
    *
    * The granted iodef must be a non-flipped chisel type and not be bound to hardware.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    if (SpecifiedDirection.isFlipped(iodef.specifiedDirection)) Builder.error("Incoming(..) cannot accept a chisel typed which is flipped")
    IO(Flipped(iodef))
  }
}

object Outgoing {

  /** Constructs an outgoing port of the provided `iodef` chisel type for the current Module
    *
    * The following example creates a output (and mixed-alignment) aggregate port in the current module
    *
    * ```scala
    *   val i = Outgoing(new Decoupled(UInt(32.W)))
    * ```
    * In this example, the sub-elements of `i` are ports of the following direction:
    *  * `i.bits` is an output port
    *  * `i.valid` is an output port
    *  * `i.ready` is an input port
    *
    * Note that `Outgoing(foo)` is equivalent to `IO(foo)`
    *
    * This must wrap the datatype used to create an io port of any Module.
    *
    * The granted iodef must be a non-flipped chisel type and not be bound to hardware.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    if (SpecifiedDirection.isFlipped(iodef.specifiedDirection)) Builder.error("Outgoing(..) cannot accept a chisel typed which is flipped")
    IO(iodef)
  }
}
