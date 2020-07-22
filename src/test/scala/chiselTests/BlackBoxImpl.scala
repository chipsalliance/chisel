// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.{HasBlackBoxInline, HasBlackBoxResource, HasBlackBoxPath}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import firrtl.FirrtlExecutionSuccess
import org.scalacheck.Test.Failed
import org.scalatest.Succeeded
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers


class BlackBoxAdd(n : Int) extends HasBlackBoxInline {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

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
  addResource("/chisel3/BlackBoxTest.v")
}

class BlackBoxMinusPath extends HasBlackBoxPath {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })
  addPath(new File("src/test/resources/chisel3/BlackBoxTest.v").getCanonicalPath)
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

class UsesBlackBoxMinusViaPath extends Module {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val mod0 = Module(new BlackBoxMinusPath)

  mod0.io.in1 := io.in1
  mod0.io.in2 := io.in2
  io.out := mod0.io.out
}

class BlackBoxImplSpec extends AnyFreeSpec with Matchers {
  val targetDir = "test_run_dir"
  val stage = new ChiselStage
  "BlackBox can have verilator source implementation" - {
    "Implementations can be contained in-line" in {
      stage.execute(Array("-X", "verilog", "--target-dir", targetDir),
                    Seq(ChiselGeneratorAnnotation(() => new UsesBlackBoxAddViaInline)))
      val verilogOutput = new File(targetDir, "BlackBoxAdd.v")
      verilogOutput.exists() should be (true)
      verilogOutput.delete()
    }
    "Implementations can be contained in resource files" in {
      stage.execute(Array("-X", "low", "--target-dir", targetDir),
                    Seq(ChiselGeneratorAnnotation(() => new UsesBlackBoxMinusViaResource)))
      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput.exists() should be (true)
      verilogOutput.delete()
    }
    "Implementations can be contained in arbitrary files" in {
      stage.execute(Array("-X", "low", "--target-dir", targetDir),
                    Seq(ChiselGeneratorAnnotation(() => new UsesBlackBoxMinusViaPath)))
      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput.exists() should be (true)
      verilogOutput.delete()
      Succeeded
    }
  }
}
