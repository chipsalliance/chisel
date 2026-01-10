// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

object Probe extends ProbeBase {

  /** Mark a Chisel type with a probe modifier.
    */
  def apply[T <: Data](using SourceInfo)(source: => T): T =
    super.apply(source, false, None)

  /** Mark a Chisel type with a probe modifier and layer color.
    */
  def apply[T <: Data](using SourceInfo)(source: => T, color: layer.Layer): T =
    super.apply(source, false, Some(color))
}

object RWProbe extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type with a writable probe modifier.
    */
  def apply[T <: Data](using SourceInfo)(source: => T): T =
    super.apply(source, true, None)

  /** Mark a Chisel type with a wirtable probe modifier and layer color.
    */
  def apply[T <: Data](using SourceInfo)(source: => T, color: layer.Layer): T =
    super.apply(source, true, Some(color))
}
