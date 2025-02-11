// SPDX-License-Identifier: Apache-2.0

package chiselTests

import java.io.File
import chisel3._
import chisel3.util.{HasBlackBoxInline, HasBlackBoxPath, HasBlackBoxResource}
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.ChiselStage
import firrtl.options.TargetDirAnnotation
import firrtl.transforms.{BlackBoxNotFoundException, BlackBoxTargetDirAnno}
import org.scalacheck.Test.Failed
import org.scalatest.Succeeded
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class BlackBoxAdd(n: Int) extends HasBlackBoxInline {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  setInline(
    "BlackBoxAdd.v",
    s"""
       |module BlackBoxAdd(
       |    input  [15:0] in,
       |    output [15:0] out
       |);
       |  assign out = in + $n;
       |endmodule
    """.stripMargin
  )
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

class BlackBoxResourceNotFound extends HasBlackBoxResource {
  val io = IO(new Bundle {})
  addResource("/missing.resource")
}

class UsesMissingBlackBoxResource extends RawModule {
  val foo = Module(new BlackBoxResourceNotFound)
}

class BlackBoxImplSpec extends AnyFreeSpec with Matchers {
  val targetDir = "test_run_dir"
  val stage = new ChiselStage
  "BlackBox can have verilator source implementation" - {
    "Implementations can be contained in-line" in {
      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesBlackBoxAddViaInline),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "BlackBoxAdd.v")
      verilogOutput.exists() should be(true)
      verilogOutput.delete()
    }
    "Implementations can be contained in resource files" in {
      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesBlackBoxMinusViaResource),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput.exists() should be(true)
      verilogOutput.delete()
    }
    // TODO: This is temporarily disabled until firtool 1.30.0 is released.  This requires:
    //   - https://github.com/llvm/circt/commit/0285a98d96b8df898e02c5ed9528f869bff80dcf
    "Implementations can be contained in arbitrary files" ignore {
      val targetDir = "test_run_dir/blackbox-path"
      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesBlackBoxMinusViaPath),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput.exists() should be(true)
      verilogOutput.delete()
      Succeeded
    }
    "Resource files that do not exist produce Chisel errors" in {
      assertThrows[BlackBoxNotFoundException] {
        ChiselStage.emitCHIRRTL(new UsesMissingBlackBoxResource)
      }
    }
  }
}
