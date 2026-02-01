// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait BoringUtils$Intf { self: BoringUtils.type =>

  /** Access a source [[Data]] that may or may not be in the current module.  If
    * this is in a child module, then create ports to allow access to the
    * requested source.
    */
  def bore[A <: Data](source: A)(using si: SourceInfo): A = _boreImpl(source)

  /** Access a sink [[Data]] for driving that may or may not be in the current module.
    *
    * If the sink is in a child module, than create input ports to allow driving the requested sink.
    *
    * Note that the sink may not be a probe, and [[rwTap]] should be used instead.
    */
  def drive[A <: Data](sink: A)(using si: SourceInfo): A = _driveImpl(sink)

  /** Access a source [[Data]] that may or may not be in the current module.  If
    * this is in a child module, then create read-only probe ports to allow
    * access to the requested source.
    *
    * Returns a probe Data type.
    */
  def tap[A <: Data](source: A)(using si: SourceInfo): A = _tapImpl(source)

  /** Access a source [[Data]] that may or may not be in the current module.  If
    * this is in a child module, then create write-only probe ports to allow
    * access to the requested source. Supports downward accesses only.
    *
    * Returns a probe Data type.
    */
  def rwTap[A <: Data](source: A)(using si: SourceInfo): A = _rwTapImpl(source)

  /** Access a source [[Data]] that may or may not be in the current module.  If
    * this is in a child module, then create read-only probe ports to allow
    * access to the requested source.
    *
    * Returns a non-probe Data type.
    */
  def tapAndRead[A <: Data](source: A)(using si: SourceInfo): A = _tapAndReadImpl(source)
}
