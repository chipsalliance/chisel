// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.choice.{Case, Group, ModuleChoice}
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

class FPGATarget extends FixedIORawModule[TargetIO](new TargetIO(8)) {
  io.out := io.in
}

class ASICTarget extends FixedIOExtModule[TargetIO](new TargetIO(8))

class VerifTarget extends FixedIORawModule[TargetIO](new TargetIO(8)) {
  io.out := io.in
}

class ModuleChoiceSpec extends ChiselFlatSpec with Utils with FileCheck {
  it should "emit options and cases" in {
    class ModuleWithChoice extends Module {
      val out = IO(UInt(8.W))

      val inst =
        ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.ASIC -> new ASICTarget))

      inst.in := 42.U(8.W)
      out := inst.out
    }

    generateFirrtlAndFileCheck(new ModuleWithChoice)(
      """|CHECK: option Platform :
         |CHECK-NEXT: FPGA
         |CHECK-NEXT: ASIC
         |CHECK: instchoice inst of VerifTarget, Platform :
         |CHECK-NEXT: FPGA => FPGATarget
         |CHECK-NEXT: ASIC => ASICTarget""".stripMargin
    )
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

  it should "allow a subset of options to be provided" in {

    class SparseOptions extends Module {
      val out = IO(Output(UInt(8.W)))
      val in = IO(Input(UInt(8.W)))

      val inst = ModuleChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget))

      inst.in := in
      out := inst.out
    }

    // Note, because of a quirk in how [[Case]]s are registered, only those referenced
    // in the Module here are going to be captured. This will be fixed in a forthcoming PR
    // that implements an [[addLayer]] like feature for [[Group]]s
    generateFirrtlAndFileCheck(new SparseOptions)(
      """|CHECK: option Platform :
         |CHECK-NEXT: FPGA
         |CHECK: instchoice inst of VerifTarget, Platform :
         |CHECK-NEXT: FPGA => FPGATarget""".stripMargin
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
