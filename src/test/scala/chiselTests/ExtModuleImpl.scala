// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util.{HasExtModuleInline, HasExtModulePath, HasExtModuleResource}
import circt.stage.ChiselStage
import firrtl.options.TargetDirAnnotation
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.transforms.{BlackBoxNotFoundException, BlackBoxTargetDirAnno}
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

//scalastyle:off magic.number

class ExtModuleAdd(n: Int) extends ExtModule with HasExtModuleInline {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  //scalastyle:off regex
  setInline(
    "ExtModuleAdd.v",
    s"""
       |module ExtModuleAdd(
       |    input  [15:0] in,
       |    output [15:0] out
       |);
       |  assign out = in + $n;
       |endmodule
    """.stripMargin
  )
}

class UsesExtModuleAddViaInline extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val blackBoxAdd = Module(new ExtModuleAdd(5))
  blackBoxAdd.io.in := io.in
  io.out := blackBoxAdd.io.out
}

class ExtModuleMinus extends ExtModule with HasExtModuleResource {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })
  addResource("/chisel3/BlackBoxTest.v")
}

class ExtModuleMinusPath extends ExtModule with HasExtModulePath {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })
  addPath(
    new File("src/test/resources/chisel3/BlackBoxTest.v").getCanonicalPath
  )
}

class UsesExtModuleMinusViaResource extends Module {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val mod0 = Module(new ExtModuleMinus)

  mod0.io.in1 := io.in1
  mod0.io.in2 := io.in2
  io.out := mod0.io.out
}

class UsesExtModuleMinusViaPath extends Module {
  val io = IO(new Bundle {
    val in1 = Input(UInt(16.W))
    val in2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val mod0 = Module(new ExtModuleMinusPath)

  mod0.io.in1 := io.in1
  mod0.io.in2 := io.in2
  io.out := mod0.io.out
}

class ExtModuleResourceNotFound extends HasExtModuleResource {
  val io = IO(new Bundle {})
  addResource("/missing.resource")
}

class UsesMissingExtModuleResource extends RawModule {
  val foo = Module(new ExtModuleResourceNotFound)
}

class ExtModuleImplSpec extends AnyFreeSpec with Matchers {
  "ExtModule can have verilator source implementation" - {

    "Implementations can be contained in-line" in {
      val targetDir = "test_run_dir/extmodule-inline"

      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesExtModuleAddViaInline),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "ExtModuleAdd.v")
      verilogOutput should exist
      verilogOutput.delete()
    }

    "Implementations can be contained in resource files" in {
      val targetDir = "test_run_dir/extmodule-resource"
      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesExtModuleMinusViaResource),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput should exist
      verilogOutput.delete()
    }

    // TODO: This is temporarily disabled until firtool 1.30.0 is released.  This requires:
    //   - https://github.com/llvm/circt/commit/0285a98d96b8df898e02c5ed9528f869bff80dcf
    "Implementations can be contained in arbitrary files" ignore {
      val targetDir = "test_run_dir/extmodule-path"
      val annotations = Seq(
        TargetDirAnnotation(targetDir),
        ChiselGeneratorAnnotation(() => new UsesExtModuleMinusViaPath),
        BlackBoxTargetDirAnno(".")
      )
      (new ChiselStage).execute(Array("--target", "systemverilog", "--split-verilog"), annotations)

      val verilogOutput = new File(targetDir, "BlackBoxTest.v")
      verilogOutput should exist
      verilogOutput.delete()
    }

    "Resource files that do not exist produce Chisel errors" in {
      assertThrows[BlackBoxNotFoundException] {
        ChiselStage.emitCHIRRTL(new UsesMissingExtModuleResource)
      }
    }
  }
}
