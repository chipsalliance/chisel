// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.experimental.{requireIsChiselType, SourceInfo}
import chisel3.internal.firrtl.ir._
import chisel3.internal.Builder
import chisel3.internal.binding.OpBinding
import chisel3.internal.Builder.pushCommand

object Intrinsic {

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
  def apply(intrinsic: String, params: (String, Param)*)(data: Data*)(implicit sourceInfo: SourceInfo): Unit = {
    require(params.map(_._1).distinct.size == params.size, "parameter names must be unique")
    pushCommand(DefIntrinsic(sourceInfo, intrinsic, data.map(_.ref), params))
  }
}

object IntrinsicExpr {

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
  ): T = {
    val prevId = Builder.idGen.value
    val t = ret // evaluate once (passed by name)
    requireIsChiselType(t, "intrinsic type")
    val int = if (!t.mustClone(prevId)) t else t.cloneTypeFull

    int.bind(OpBinding(Builder.forcedUserModule, Builder.currentBlock))
    require(params.map(_._1).distinct.size == params.size, "parameter names must be unique")
    pushCommand(DefIntrinsicExpr(sourceInfo, intrinsic, int, data.map(_.ref), params))
    int
  }
}
