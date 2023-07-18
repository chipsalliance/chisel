package chisel3

import chisel3.internal.requireIsChiselType // Fix ambiguous import
import chisel3.internal.Builder
import chisel3.experimental.SourceInfo

object IO {

  /** Constructs a port for the current Module
    *
    * This must wrap the datatype used to set the io field of any Module.
    * i.e. All concrete modules must have defined io in this form:
    * [lazy] val io[: io type] = IO(...[: io type])
    *
    * Items in [] are optional.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    *
    * Also registers a BaseType as a port, also performing bindings. Cannot be called once ports are
    * requested (so that all calls to ports will return the same information).
    * Internal API.
    */
  def apply[T <: BaseType](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    val module = Module.currentModule.get // Impossible to fail
    require(!module.isClosed, "Can't add more ports after module close")
    val prevId = Builder.idGen.value
    val data = iodef // evaluate once (passed by name)
    requireIsChiselType(data, "io type")

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
