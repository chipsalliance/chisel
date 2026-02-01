// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.experimental.{requireIsChiselType, SourceInfo}
import chisel3.internal.firrtl.ir._
import chisel3.internal.Builder
import chisel3.internal.binding.OpBinding
import chisel3.internal.Builder.pushCommand

object Intrinsic extends Intrinsic$Intf {

  private[chisel3] def _applyImpl(
    intrinsic: String,
    params:    (String, Param)*
  )(data: Data*)(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    require(params.map(_._1).distinct.size == params.size, "parameter names must be unique")
    pushCommand(DefIntrinsic(sourceInfo, intrinsic, data.map(_.ref), params))
  }
}

object IntrinsicExpr extends IntrinsicExpr$Intf {

  private[chisel3] def _applyImpl[T <: Data](
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
