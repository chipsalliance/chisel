// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util.Counter
import firrtl.passes.CheckInitialization.RefNotInitializedException
import org.scalatest._

class InvalidateAPISpec extends ChiselPropSpec with Matchers {

  def myGenerateFirrtl(t: => Module): String = Driver.emit(() => t)
  def compileFirrtl(t: => Module): Unit = {
    Driver.execute(Array[String]("--compiler", "verilog"), () => t)
  }

  class HardIP extends Bundle {
    val in  = Input(Bool())
    val out = Output(Bool())

    override def cloneType: this.type = (new HardIP).asInstanceOf[this.type]
  }

  property("an output connected to DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithDontCare extends Module {

      val io = IO(new HardIP)
      io.out := DontCare
      io.out := io.in
    }

    val firrtlOutput = myGenerateFirrtl(new ModuleWithDontCare)
    firrtlOutput should include("io.out is invalid")
  }

  property("an output without a DontCare should NOT emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io.out := io.in
    }

    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should not include("is invalid")
  }

  property("an output without a DontCare should emit a Firrtl \"is invalid\" with NotStrict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.NotStrict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io.out := io.in
    }

    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("io is invalid")
  }

  property("a bundle with a DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io <> DontCare
    }

    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("io is invalid")
  }

  property("a DontCare cannot be a connection sink (LHS) for := ") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithDontCareSink extends Module {

      val io = IO(new HardIP)
      DontCare := io.in
    }

    val exception = intercept[ChiselException] {
      elaborate(new ModuleWithDontCareSink)
    }
    exception.getMessage should include("DontCare cannot be a connection sink (LHS)")
  }

  property("a DontCare cannot be a connection sink (LHS) for <>") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithDontCareSink extends Module {

      val io = IO(new HardIP)
      DontCare <> io.in
    }

    val exception = intercept[ChiselException] {
      elaborate(new ModuleWithDontCareSink)
    }
    exception.getMessage should include("DontCare cannot be a connection sink (LHS)")
  }

  property("FIRRTL should complain about partial initialization with Strict CompileOptions and conditional connect") {
    import chisel3.core.ExplicitCompileOptions.Strict
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
    import chisel3.core.ExplicitCompileOptions.Strict
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
    import chisel3.core.ExplicitCompileOptions.Strict
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
}
