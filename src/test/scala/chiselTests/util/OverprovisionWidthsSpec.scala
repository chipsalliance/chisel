// See LICENSE for license details.

package chiselTests.util

import chisel3._
import chisel3.util.experimental.OverprovisionWidths.OverprovisionedWidthException
import chisel3.util.experimental.overprovision
import chiselTests.ChiselFlatSpec

class OverprovisionWidthsSpec extends ChiselFlatSpec {

  class Top(param: String) extends Module {
    val io = IO(new Bundle{
      val in = Input(UInt(3.W))
      val out = Output(UInt())
    })
    val mid = Module(new Middle(param))
    mid.io.middlein := io.in
    io.out := mid.io.middleout
    // Overprovisioning from the parent works as well
    if(param == "Top") {
      overprovision(mid.io.middlein, 10.W)
      overprovision(mid.io.middleout, 10.W)
    }
  }

  class Middle(param: String) extends Module {
    val io = IO(new Bundle{
      val middlein = Input(UInt(3.W))
      val middleout = Output(UInt())
    })
    val bot = Module(new Bottom)
    bot.io.botin := io.middlein

    val intermediate = Wire(UInt(3.W))

    intermediate := bot.io.botout
    io.middleout := intermediate

    if(param == "Middle") {
      overprovision(io.middlein, 10.W)
      overprovision(io.middleout, 10.W)
    }
    if(param == "wire") {
      overprovision(intermediate, 10.W)
      dontTouch(intermediate)
    }
  }

  class Bottom extends Module {
    val io = IO(new Bundle{
      val botin = Input(UInt(3.W))
      val botout = Output(UInt())
    })
    io.botout := io.botin
  }

  "Overprovisioning" should "affect port declarations, even with inferred widths" in {
    val verilog = compile(new Top("Middle"))
    // Update ports to new overprovisioned widths
    verilog should include ("input  [9:0] io_middlein")
    verilog should include ("output [9:0] io_middleout")
    // Assignments within the module to/from ports are padded/trimmed
    verilog should include ("assign io_middleout = {{7'd0}, bot_io_botout};")
    verilog should include ("assign bot_io_botin = io_middlein[2:0];")
    // Assignments outside the module to/from ports are padded/trimmed
    verilog should include ("assign io_out = mid_io_middleout[2:0];")
    verilog should include ("assign mid_io_middlein = {{7'd0}, io_in};")
    // Overprovision should not affect width inference
    verilog should include ("input  [2:0] io_in")
    verilog should include ("output [2:0] io_out")
  }

  "Overprovisioning" should "affect port declarations even when called from parent" in {
    val verilog = compile(new Top("Top"))
    // Update ports to new overprovisioned widths
    verilog should include ("input  [9:0] io_middlein")
    verilog should include ("output [9:0] io_middleout")
    // Assignments within the module to/from ports are padded/trimmed
    verilog should include ("assign io_middleout = {{7'd0}, bot_io_botout};")
    verilog should include ("assign bot_io_botin = io_middlein[2:0];")
    // Assignments outside the module to/from ports are padded/trimmed
    verilog should include ("assign io_out = mid_io_middleout[2:0];")
    verilog should include ("assign mid_io_middlein = {{7'd0}, io_in};")
  }

  "Overprovisioning" should "error if applied on non-ports" in {
    intercept[OverprovisionedWidthException] {
      compile(new Top("wire"))
    }
  }
}
