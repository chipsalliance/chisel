// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental
import chisel3._
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator, SerializableModuleParameter}
import upickle.default._
object GCDSerializableModuleParameter {
  implicit def rwP: ReadWriter[GCDSerializableModuleParameter] = macroRW
}

case class GCDSerializableModuleParameter(width: Int) extends SerializableModuleParameter

class GCDSerializableModule(val parameter: GCDSerializableModuleParameter)
    extends Module
    with SerializableModule[GCDSerializableModuleParameter] {
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

class SerializableModuleGeneratorSpec extends ChiselFlatSpec with Utils {
  val g = SerializableModuleGenerator(
    classOf[GCDSerializableModule],
    GCDSerializableModuleParameter(32)
  )

  "SerializableModuleGenerator" should "be serialized and deserialized" in {
    assert(
      g == upickle.default.read[SerializableModuleGenerator[GCDSerializableModule, GCDSerializableModuleParameter]](
        upickle.default.write(g)
      )
    )
  }

  "SerializableModuleGenerator" should "be able to elaborate" in {
    circt.stage.ChiselStage.emitCHIRRTL(g.module())
  }

  case class FooParameter() extends SerializableModuleParameter

  class InnerFoo(val parameter: FooParameter) extends RawModule with SerializableModule[FooParameter]

  "InnerClass" should "not be able to serialize" in {
    assert(
      intercept[java.lang.IllegalArgumentException](
        circt.stage.ChiselStage.emitCHIRRTL(
          {
            SerializableModuleGenerator(
              classOf[InnerFoo],
              FooParameter()
            ).module()
          }
        )
      ).getMessage.contains(
        "You define your class chiselTests.experimental.SerializableModuleGeneratorSpec$FooParameter inside other class."
      )
    )
  }
}
