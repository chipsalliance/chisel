// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, Param}
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir._
import chisel3.internal.throwException
import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}

object FormalTest {
  def apply(
    module: BaseModule,
    params: MapTestParam = MapTestParam(Map.empty),
    name:   String = ""
  )(implicit sourceInfo: SourceInfo): Unit = {
    val proposedName = if (name != "") {
      name
    } else {
      module._proposedName
    }
    val sanitizedName = Builder.globalNamespace.name(proposedName)
    Builder.components += DefFormalTest(sanitizedName, module, params, sourceInfo)
  }
}

/** Parameters for test declarations. */
sealed abstract class TestParam
case class IntTestParam(value: BigInt) extends TestParam
case class DoubleTestParam(value: Double) extends TestParam
case class StringTestParam(value: String) extends TestParam
case class ArrayTestParam(value: Seq[TestParam]) extends TestParam
case class MapTestParam(value: Map[String, TestParam]) extends TestParam
