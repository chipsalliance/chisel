// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.{HasBlackBoxInline, HasBlackBoxResource}
import firrtl.FirrtlExecutionSuccess
import org.scalacheck.Test.Failed
import org.scalatest.{FreeSpec, Matchers, Succeeded}

//scalastyle:off magic.number

class BlackBoxAdd(n : Int) extends HasBlackBoxInline {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  //scalastyle:off regex
  setInline("BlackBoxAdd.v",
    s"""
      |module BlackBoxAdd(
      |    input  [15:0] in,
      |    output [15:0] out
      |);
      |  assign out = in + $n;
      |endmodule
    """.stripMargin)
}

class UsesBlackBoxAddViaInline extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val blackBoxAdd = Module(new BlackBoxAdd(5))
  blackBoxAdd.io.in := io.in
  io.out := blackBoxAdd.io.out
}

class BlackBoxMinus extends HasBlackBoxResource {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })
  setResource("/chisel3/BlackBoxTest.v")
}

class UsesBlackBoxMinusViaResource extends Module {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val mod0 = Module(new BlackBoxMinus)

  mod0.io.in1 := io.in1
  mod0.io.in2 := io.in2
  io.out := mod0.io.out
}

class BlackBoxImplSpec extends FreeSpec with Matchers {
  val targetDir = "test_run_dir"
  "BlackBox can have verilator source implementation" - {
    "Implementations can be contained in-line" in {
      Driver.execute(Array("-X", "verilog", "--target-dir", targetDir), () => new UsesBlackBoxAddViaInline) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          val verilogOutput = new File(targetDir, "BlackBoxAdd.v")
          verilogOutput.exists() should be (true)
          verilogOutput.delete()
          Succeeded
        case _ =>
          Failed
      }
    }
    "Implementations can be contained in resource files" in {
      Driver.execute(Array("-X", "low", "--target-dir", targetDir), () => new UsesBlackBoxMinusViaResource) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          val verilogOutput = new File(targetDir, "BlackBoxTest.v")
          verilogOutput.exists() should be (true)
          verilogOutput.delete()
          Succeeded
        case _ =>
          Failed
      }
    }
  }
}
