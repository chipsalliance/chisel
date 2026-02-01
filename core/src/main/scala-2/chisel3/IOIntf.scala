// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait IO$Intf { self: IO.type =>

  /** Constructs a port for the current Module.
    *
    * This must wrap the datatype used to set the io field of any Module.
    * i.e. All concrete modules must have defined io in this form:
    * [lazy] val io[: io type] = IO(...[: io type])
    *
    * Items in [] are optional.
    *
    * The granted iodef must be a chisel type and not be bound to hardware.
    *
    * Also registers a Data as a port, also performing bindings. Cannot be called once ports are
    * requested (so that all calls to ports will return the same information).
    */
  def apply[T <: Data](iodef: => T)(implicit sourceInfo: SourceInfo): T = _applyImpl(iodef)
}

private[chisel3] trait FlatIO$Intf { self: FlatIO.type =>

  /** The same as [[IO]] except there is no prefix when given a [[Record]] or
    * [[Bundle]].  For [[Element]] ([[UInt]], etc.) or [[Vec]] types, this is
    * the same as [[IO]]. It is also the same as [[IO]] for [[chisel3.probe.Probe]] types.
    */
  def apply[T <: Data](gen: => T)(implicit sourceInfo: SourceInfo): T = _applyImpl(gen)
}
