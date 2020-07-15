// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.Counter
import firrtl.passes.CheckInitialization.RefNotInitializedException
import firrtl.util.BackendCompilationUtilities
import org.scalatest._
import org.scalatest.matchers.should.Matchers

class InvalidateAPISpec extends ChiselPropSpec with Matchers with BackendCompilationUtilities with Utils {

  def myGenerateFirrtl(t: => Module): String = (new ChiselStage).emitChirrtl(t)
  def compileFirrtl(t: => Module): Unit = {
    val testDir = createTestDirectory(this.getClass.getSimpleName)

    (new ChiselStage).execute(Array[String]("-td", testDir.getAbsolutePath, "--compiler", "verilog"),
                              Seq(ChiselGeneratorAnnotation(() => t)))
  }
  class TrivialInterface extends Bundle {
    val in  = Input(Bool())
    val out = Output(Bool())
  }

  // scalastyle:off line.size.limit
  property("an output connected to DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithDontCare extends Module {
      val io = IO(new TrivialInterface)
      io.out := DontCare
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithDontCare)
    firrtlOutput should include("io.out is invalid")
  }

  property("an output without a DontCare should NOT emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should not include("is invalid")
  }

  property("an output without a DontCare should emit a Firrtl \"is invalid\" with NotStrict CompileOptions") {
    import chisel3.ExplicitCompileOptions.NotStrict
    class ModuleWithoutDontCare extends Module {
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("io is invalid")
  }

  property("a bundle with a DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {
      val io = IO(new TrivialInterface)
      io <> DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("io.out is invalid")
    firrtlOutput should include("io.in is invalid")
  }

  property("a Vec with a DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions and bulk connect") {
    import chisel3.ExplicitCompileOptions.Strict
    val nElements = 5
    class ModuleWithoutDontCare extends Module {
      val io = IO(new Bundle {
        val outs = Output(Vec(nElements, Bool()))
      })
      io.outs <> DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    for (i <- 0 until nElements)
     firrtlOutput should include(s"io.outs[$i] is invalid")
  }

  property("a Vec with a DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions and mono connect") {
    import chisel3.ExplicitCompileOptions.Strict
    val nElements = 5
    class ModuleWithoutDontCare extends Module {
      val io = IO(new Bundle {
        val ins = Input(Vec(nElements, Bool()))
      })
      io.ins := DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    for (i <- 0 until nElements)
      firrtlOutput should include(s"io.ins[$i] is invalid")
  }

  property("a DontCare cannot be a connection sink (LHS) for := ") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithDontCareSink extends Module {
      val io = IO(new TrivialInterface)
      DontCare := io.in
    }
    val exception = intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate(new ModuleWithDontCareSink)
      }
    }
    exception.getMessage should include("DontCare cannot be a connection sink (LHS)")
  }

  property("a DontCare cannot be a connection sink (LHS) for <>") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithDontCareSink extends Module {
      val io = IO(new TrivialInterface)
      DontCare <> io.in
    }
    val exception = intercept[BiConnectException] {
      extractCause[BiConnectException] {
        ChiselStage.elaborate(new ModuleWithDontCareSink)
      }
    }
    exception.getMessage should include("DontCare cannot be a connection sink (LHS)")
  }

  property("FIRRTL should complain about partial initialization with Strict CompileOptions and conditional connect") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithIncompleteAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      when (counter.inc()) {
        io.out := true.B
      }
    }
    val exception = intercept[RefNotInitializedException] {
      compileFirrtl(new ModuleWithIncompleteAssignment)
    }
    exception.getMessage should include("is not fully initialized")
 }

  property("FIRRTL should not complain about partial initialization with Strict CompileOptions and conditional connect after unconditional connect") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithUnconditionalAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      io.out := false.B
      when (counter.inc()) {
        io.out := true.B
      }
    }
    compileFirrtl(new ModuleWithUnconditionalAssignment)
  }

  property("FIRRTL should not complain about partial initialization with Strict CompileOptions and conditional connect with otherwise clause") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithConditionalAndOtherwiseAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      when (counter.inc()) {
        io.out := true.B
      } otherwise {
        io.out := false.B
      }
    }

    compileFirrtl(new ModuleWithConditionalAndOtherwiseAssignment)
  }

  property("an output without a DontCare should NOT emit a Firrtl \"is invalid\" with overriden NotStrict CompileOptions") {
    import chisel3.ExplicitCompileOptions.NotStrict
    class ModuleWithoutDontCare extends Module {
      override val compileOptions = chisel3.ExplicitCompileOptions.NotStrict.copy(explicitInvalidate = true)
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should not include("is invalid")
  }

  property("an output without a DontCare should NOT emit a Firrtl \"is invalid\" with overriden NotStrict CompileOptions module definition") {
    import chisel3.ExplicitCompileOptions.NotStrict
    abstract class ExplicitInvalidateModule extends Module()(chisel3.ExplicitCompileOptions.NotStrict.copy(explicitInvalidate = true))
    class ModuleWithoutDontCare extends ExplicitInvalidateModule {
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should not include("is invalid")
  }

  property("an output without a DontCare should emit a Firrtl \"is invalid\" with overriden Strict CompileOptions") {
    import chisel3.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {
      override val compileOptions = chisel3.ExplicitCompileOptions.Strict.copy(explicitInvalidate = false)
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("is invalid")
  }

  property("an output without a DontCare should emit a Firrtl \"is invalid\" with overriden Strict CompileOptions module definition") {
    import chisel3.ExplicitCompileOptions.Strict
    abstract class ImplicitInvalidateModule extends Module()(chisel3.ExplicitCompileOptions.NotStrict.copy(explicitInvalidate = false))
    class ModuleWithoutDontCare extends ImplicitInvalidateModule {
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("is invalid")
  }
  // scalastyle:on line.size.limit
}
