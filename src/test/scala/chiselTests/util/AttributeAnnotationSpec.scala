package chiselTests.util

import _root_.circt.stage.ChiselStage
import chisel3._
import chisel3.util.addAttribute
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AttributeAnnotationSpec extends AnyFlatSpec with Matchers {
  class AttributeExample extends Module {
    val io = IO(new Bundle {
      val input = Input(UInt(8.W))
      val output = Output(UInt(8.W))
    })

    val reg = RegNext(io.input)

    addAttribute(reg, "synthesis translate_off")

    io.output := reg
  }

  it should "generate corresponding SystemVerilog attributes" in {
    val verilog = ChiselStage.emitSystemVerilog(new AttributeExample)

    verilog should include("(* synthesis translate_off *)")
  }

}
