// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait Intrinsic$Intf { self: Intrinsic.type =>

  /** Create an intrinsic statement.
    *
    * @param intrinsic name of the intrinsic
    * @param params parameter name/value pairs, if any.  Parameter names must be unique.
    * @param data inputs
    *
    * @example {{{
    * Intrinsic("test", "Foo" -> 5)(f, g)
    * }}}
    */
  def apply(intrinsic: String, params: (String, Param)*)(data: Data*)(implicit sourceInfo: SourceInfo): Unit =
    _applyImpl(intrinsic, params: _*)(data: _*)
}

private[chisel3] trait IntrinsicExpr$Intf { self: IntrinsicExpr.type =>

  /** Create an intrinsic expression.
    *
    * @param intrinsic name of the intrinsic
    * @param ret return type of the expression
    * @param params parameter name/value pairs, if any.  Parameter names must be unique.
    * @param data inputs
    * @return intrinsic expression that returns the specified return type
    *
    * @example {{{
    * val test = IntrinsicExpr("test", UInt(32.W), "Foo" -> 5)(f, g) + 3.U
    * }}}
    */
  def apply[T <: Data](
    intrinsic: String,
    ret:       => T,
    params:    (String, Param)*
  )(data: Data*)(
    implicit sourceInfo: SourceInfo
  ): T = _applyImpl(intrinsic, ret, params: _*)(data: _*)
}
