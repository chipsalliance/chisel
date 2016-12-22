// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._

class ExplicitInvalidateSpec extends ChiselPropSpec with Matchers {

  def generateFirrtl(t: => Module): String = Driver.emit(() => t)

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

    val firrtlOutput = generateFirrtl(new ModuleWithDontCare)
    firrtlOutput should include("io.out is invalid")
  }

  property("an output without a DontCare should NOT emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io.out := io.in
    }

    val firrtlOutput = generateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should not include("is invalid")
  }

  property("an output without a DontCare should emit a Firrtl \"is invalid\" with NotStrict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.NotStrict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io.out := io.in
    }

    val firrtlOutput = generateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("io is invalid")
  }

  property("a bundle with a DontCare should emit a Firrtl \"is invalid\" with Strict CompileOptions") {
    import chisel3.core.ExplicitCompileOptions.Strict
    class ModuleWithoutDontCare extends Module {

      val io = IO(new HardIP)
      io <> DontCare
    }

    val firrtlOutput = generateFirrtl(new ModuleWithoutDontCare)
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
}
