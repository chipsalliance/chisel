// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.layer.Layer


@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
abstract class Param extends chisel3.Param

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
class IntParam(value: BigInt) extends chisel3.IntParam(value)
@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
object IntParam {
  def apply(value:   BigInt):   IntParam = new IntParam(value)
  def unapply(param: IntParam): Option[BigInt] = Some(param.value)
}

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
class DoubleParam(value: Double) extends chisel3.DoubleParam(value)
@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
object DoubleParam {
  def apply(value:   Double):      DoubleParam = new DoubleParam(value)
  def unapply(param: DoubleParam): Option[Double] = Some(param.value)
}

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
class StringParam(value: String) extends chisel3.StringParam(value)
@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
object StringParam {
  def apply(value:   String):      StringParam = new StringParam(value)
  def unapply(param: StringParam): Option[String] = Some(param.value)
}

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
class PrintableParam(value: chisel3.Printable, context: BaseModule) extends chisel3.PrintableParam(value, context)
@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
object PrintableParam {
  def apply(value: chisel3.Printable, context: BaseModule): PrintableParam = new PrintableParam(value, context)
  def unapply(param: PrintableParam): Option[(chisel3.Printable, BaseModule)] = Some((param.value, param.context))
}

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
class RawParam(value: String) extends chisel3.RawParam(value)
@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
object RawParam {
  def apply(value:   String):   RawParam = new RawParam(value)
  def unapply(param: RawParam): Option[String] = Some(param.value)
}

@deprecated("moved from chisel3.experimental to chisel3", "7.5.0")
abstract class ExtModule(
  params:      Map[String, chisel3.Param] = Map.empty[String, chisel3.Param],
  knownLayers: Seq[Layer] = Seq.empty[Layer]
) extends chisel3.ExtModule(params, knownLayers)
