// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

object Mux extends MuxImpl with SourceInfoDoc {

  /** Creates a mux, whose output is one of the inputs depending on the
    * value of the condition.
    *
    * @param cond condition determining the input to choose
    * @param con the value chosen when `cond` is true
    * @param alt the value chosen when `cond` is false
    * @example
    * {{{
    * val muxOut = Mux(data_in === 3.U, 3.U(4.W), 0.U(4.W))
    * }}}
    */
  def apply[T <: Data](
    cond: Bool,
    con:  T,
    alt:  T
  )(
    using sourceInfo: SourceInfo
  ): T = _applyImpl(cond, con, alt)
}
