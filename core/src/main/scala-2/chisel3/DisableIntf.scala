// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.experimental.{OpaqueType, SourceInfo}
import chisel3.internal.sourceinfo.SourceInfoTransform

import scala.language.experimental.macros
import scala.collection.immutable.ListMap

private[chisel3] trait DisableIntf { self: Disable =>

  /** Logical not
    *
    * @return invert the logical value of this `Disable`
    * @group Bitwise
    */
  final def unary_! : Disable = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_!(implicit sourceInfo: SourceInfo): Disable = _impl_unary_!
}
