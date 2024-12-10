// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.choice.{addGroup, Case, Group, ModuleChoice}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import _root_.circt.stage.ChiselStage

object Platform extends Group {
  object FPGA extends Case
  object ASIC extends Case
}

class TargetIO(width: Int) extends Bundle {
  val in = Flipped(UInt(width.W))
  val out = UInt(width.W)
}

class FPGATarget extends FixedIOExtModule[TargetIO](new TargetIO(8))

class ASICTarget extends FixedIOExtModule[TargetIO](new TargetIO(8))

class VerifTarget extends FixedIORawModule[TargetIO](new TargetIO(8))

class ModuleChoiceSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {
  it should "emit options and cases" in {
    class ModuleWithChoice extends Module {
      val out = IO(UInt(8.W))

      val inst =
        ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.ASIC -> new ASICTarget))

      inst.in := 42.U(8.W)
      out := inst.out
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new ModuleWithChoice, Array("--full-stacktrace"))

    info("CHIRRTL emission looks correct")
    matchesAndOmits(chirrtl)(
      "option Platform :",
      "FPGA",
      "ASIC",
      "instchoice inst of VerifTarget, Platform :",
      "FPGA => FPGATarget",
      "ASIC => ASICTarget"
    )()
  }

  it should "require that all cases are part of the same option" in {

    class MixedOptions extends Module {
      object Performance extends Group {
        object Fast extends Case
        object Small extends Case
      }

      val out = IO(UInt(8.W))

      val inst =
        ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Performance.Fast -> new ASICTarget))

      inst.in := 42.U(8.W)
      out := inst.out
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new MixedOptions) }.getMessage() should include(
      "cannot mix choices from different groups: Platform, Performance"
    )

  }

  it should "require that at least one alternative is present" in {

    class MixedOptions extends Module {
      val out = IO(UInt(8.W))

      val inst =
        ModuleChoice(new VerifTarget)(Seq())

      inst.in := 42.U(8.W)
      out := inst.out
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new MixedOptions) }.getMessage() should include(
      "at least one alternative must be specified"
    )

  }

  it should "require that all cases are distinct" in {

    class MixedOptions extends Module {
      val out = IO(UInt(8.W))

      val inst =
        ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.FPGA -> new ASICTarget))

      inst.in := 42.U(8.W)
      out := inst.out
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new MixedOptions) }.getMessage() should include(
      "duplicate case 'FPGA'"
    )

  }

  it should "require that all IO bundles are type equivalent" in {

    class MixedIO extends Module {
      val out = IO(UInt(8.W))

      class Target16 extends FixedIOExtModule[TargetIO](new TargetIO(16))

      val inst =
        ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new Target16))

      inst.in := 42.U(8.W)
      out := inst.out
    }

    intercept[ChiselException] { ChiselStage.emitCHIRRTL(new MixedIO, Array("--throw-on-first-error")) }
      .getMessage() should include(
      "choice module IO bundles are not type equivalent"
    )

  }

}
class AddGroupSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {
  it should "emit options for a registered group even if there are no consumers" in {
    class ModuleWithoutChoice extends Module {
      addGroup(Platform)
      val out = IO(UInt(8.W))
      val in = IO(UInt(8.W))
      out := in
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new ModuleWithoutChoice, Array("--full-stacktrace"))

    info("CHIRRTL emission looks correct")
    matchesAndOmits(chirrtl)(
      "option Platform :",
      "FPGA",
      "ASIC"
    )()
  }
}
