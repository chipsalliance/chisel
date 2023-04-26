package chisel3

import chisel3.internal.requireIsChiselType // Fix ambiguous import
import chisel3.internal.Builder
import chisel3.experimental.SourceInfo

object IO {

  /** Constructs a port for the current Module
    * 
    * Will build an incoming port if the iodef has an outer Flipped or Input
    * e.g. val io = IO(Flipped(Bool()))
    * e.g. val io = IO(Input(Bool()))
    *
    * Will build an outgoming port if the iodef is not flipped, or marked as Output
    * e.g. val io = IO(Bool())
    * e.g. val io = IO(Output(Bool()))
    * 
    * i.e. All concrete modules must have defined ios in this form:
    * val io[: io type] = IO(...[: io type])
    *
    * Items in [] are optional.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
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

object Incoming {

  /** Constructs an incoming port for the current Module
    * 
    * Requires iodef to not have an outer Flipped, Input, or Output
    * 
    * e.g. val io = Incoming(Bool())
    * e.g. val io = Incoming(new Bundle( val x = Bool(); val y = Flipped(Bool()))
    * e.g. val io = Incoming(Passive(new Bundle( val x = Bool(); val y = Flipped(Bool())))
    * e.g. ERROR: val io = Incoming(Input(Bool()))
    * e.g. ERROR: val io = Incoming(Output(Bool()))
    * e.g. ERROR: val io = Incoming(Flipped(Bool()))
    *
    * i.e. All concrete modules must have defined ios in this form:
    * val io[: io type] = IO(...[: io type])
    *
    * Items in [] are optional.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    IO(Flipped(iodef))
  }
}

object Outgoing {

  /** Constructs an outgoing port for the current Module
    * 
    * Requires iodef to not have an outer Flipped, Input, or Output
    * 
    * e.g. val io = Outgoing(Bool())
    * e.g. val io = Outgoing(new Bundle( val x = Bool(); val y = Flipped(Bool()))
    * e.g. val io = Outgoing(Passive(new Bundle( val x = Bool(); val y = Flipped(Bool())))
    * e.g. ERROR: val io = Outgoing(Input(Bool()))
    * e.g. ERROR: val io = Outgoing(Output(Bool()))
    * e.g. ERROR: val io = Outgoing(Flipped(Bool()))
    *
    * i.e. All concrete modules must have defined ios in this form:
    * val io[: io type] = IO(...[: io type])
    *
    * Items in [] are optional.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = {
    IO(iodef)
  }
}