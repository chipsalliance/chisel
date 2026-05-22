// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.LogicalEachTransform

private[chisel3] trait LogicalEachIntf extends SourceInfoDoc { self: LogicalEach =>

  /** Wide logical operator. Every bit of data is combined with the Boolean operand.
    *
    * @param cond Bool operand to be applied to each bit of the input data
    * @param data the data to be qualified by the control signal
    * @example
    * {{{
    * val dataQualified = AndEach(enable, value)
    * }}}
    */
  def apply[T <: Data](cond: Bool, data: T): T = macro LogicalEachTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    cond: Bool,
    data: T
  )(
    implicit sourceInfo: SourceInfo
  ): T = _applyImpl(cond, data)
}
