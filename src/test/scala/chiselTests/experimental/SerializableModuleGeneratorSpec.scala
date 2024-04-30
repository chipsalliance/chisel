// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental
import chisel3._
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import upickle.default

// generate at compile time, but not implement it.
object GCDSerializableModuleParameter {
  implicit def rw: default.ReadWriter[GCDSerializableModuleParameter] = upickle.default.macroRW
}

case class GCDSerializableModuleParameter(width: Int) extends SerializableModuleParameter

class GCDSerializableModule(val parameter: GCDSerializableModuleParameter)
    extends Module
    with SerializableModule[GCDSerializableModuleParameter]{
  val a = 4
}

object GCDSerializableModule extends chisel3.experimental.SerializableModuleMain[GCDSerializableModuleParameter, GCDSerializableModule] {
  import scala.reflect.runtime.universe
  // this should be
  implicit val pRW: default.ReadWriter[GCDSerializableModuleParameter] = GCDSerializableModuleParameter.rw
  implicit val mTypeTag: universe.TypeTag[GCDSerializableModule] = implicitly[universe.TypeTag[GCDSerializableModule]]
  implicit val pTypeTag: universe.TypeTag[GCDSerializableModuleParameter] = implicitly[universe.TypeTag[GCDSerializableModuleParameter]]
}
