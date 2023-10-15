// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental
import chisel3._
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator}
import upickle.default
import upickle.default._

case class GCDSerializableModuleParameter(width: Int)

class GCDSerializableModule(val parameter: GCDSerializableModuleParameter) extends Module with SerializableModule {
  type SerializableModuleParameter = GCDSerializableModuleParameter
  val io = IO(new Bundle {
    val a = Input(UInt(parameter.width.W))
    val b = Input(UInt(parameter.width.W))
    val e = Input(Bool())
    val z = Output(UInt(parameter.width.W))
  })
  val x = Reg(UInt(parameter.width.W))
  val y = Reg(UInt(parameter.width.W))
  val z = Reg(UInt(parameter.width.W))
  val e = Reg(Bool())
  when(e) {
    x := io.a
    y := io.b
    z := 0.U
  }
  when(x =/= y) {
    when(x > y) {
      x := x - y
    }.otherwise {
      y := y - x
    }
  }.otherwise {
    z := x
  }
  io.z := z
}

// TODO: this should be constructed by plugin
class GCDSerializableGenerator$Auto(
  val parameter: GCDSerializableModuleParameter
)(
  implicit val parameterRW: ReadWriter[GCDSerializableModuleParameter])
    extends SerializableModuleGenerator {
  override type M = GCDSerializableModule
  override val moduleClass = classOf[M]
}

class SerializableModuleGeneratorSpec extends ChiselFlatSpec with Utils {
  "SerializableModuleGenerator" should "be serialized" in {
    // barely construct a SerializableModuleGenerator
    val g = new GCDSerializableGenerator$Auto(GCDSerializableModuleParameter(16))(
      macroRW[GCDSerializableModuleParameter]
    )

    upickle.default.write(g.asInstanceOf[SerializableModuleGenerator]) should be(
      """{"parameter":{"width":16},"module":"chiselTests.experimental.GCDSerializableModule"}"""
    )
  }

  "SerializableModuleGenerator" should "be able to construct with upickle reader" in {
    implicit val rwP: upickle.default.ReadWriter[GCDSerializableModuleParameter] = macroRW
    upickle.default.read[SerializableModuleGenerator](
      """{"parameter":{"width":16},"module":"chiselTests.experimental.GCDSerializableModule"}"""
    )
  }

}
