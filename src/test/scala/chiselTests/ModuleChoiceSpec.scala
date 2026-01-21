// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.choice.{Case, Group, ModuleChoice}
import chisel3.experimental.hierarchy.Definition
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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

class ModuleWithChoice[T <: Data](
  default: => FixedIOBaseModule[T]
)(alternateImpls: Seq[(Case, () => FixedIOBaseModule[T])])
    extends Module {
  val inst: T = ModuleChoice[T](default, alternateImpls)
  val io:   T = IO(chiselTypeOf(inst))
  io <> inst
}

class ModuleChoiceSpec extends AnyFlatSpec with Matchers with FileCheck {
  it should "emit options and cases" in {
    class ModuleWithValidChoices
        extends ModuleWithChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.ASIC -> new ASICTarget))

    ChiselStage
      .emitCHIRRTL(new ModuleWithValidChoices)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK-NEXT: ASIC
           |CHECK: instchoice inst of VerifTarget, Platform :
           |CHECK-NEXT: FPGA => FPGATarget
           |CHECK-NEXT: ASIC => ASICTarget""".stripMargin
      )
  }

  it should "emit options and cases for Modules including definitions" in {
    class ModuleWithValidChoices
        extends ModuleWithChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.ASIC -> new ASICTarget))
    class TopWithDefinition extends Module {
      val definitionWithChoice = Definition(new ModuleWithValidChoices)
    }

    ChiselStage
      .emitCHIRRTL(new TopWithDefinition)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK-NEXT: ASIC
           |CHECK: instchoice inst of VerifTarget, Platform :
           |CHECK-NEXT: FPGA => FPGATarget
           |CHECK-NEXT: ASIC => ASICTarget""".stripMargin
      )
  }

  it should "require that all cases are part of the same option" in {

    object Performance extends Group {
      object Fast extends Case
      object Small extends Case
    }

    class MixedOptions
        extends ModuleWithChoice(new VerifTarget)(
          Seq(Platform.FPGA -> new FPGATarget, Performance.Fast -> new ASICTarget)
        )

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new MixedOptions) }.getMessage() should include(
      "cannot mix choices from different groups: Platform, Performance"
    )

  }

  it should "require that at least one alternative is present" in {

    class NoAlternatives extends ModuleWithChoice(new VerifTarget)(Seq())

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new NoAlternatives) }.getMessage() should include(
      "at least one alternative must be specified"
    )

  }

  it should "allow a subset of options to be provided" in {

    class SubsetOptions extends ModuleWithChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget))

    // Note, because of a quirk in how [[Case]]s are registered, only those referenced
    // in the Module here are going to be captured. This will be fixed in a forthcoming PR
    // that implements an [[addLayer]] like feature for [[Group]]s
    ChiselStage
      .emitCHIRRTL(new SubsetOptions)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK: instchoice inst of VerifTarget, Platform :
           |CHECK-NEXT: FPGA => FPGATarget""".stripMargin
      )
  }

  it should "require that all cases are distinct" in {

    class MixedOptions
        extends ModuleWithChoice(new VerifTarget)(Seq(Platform.FPGA -> new FPGATarget, Platform.FPGA -> new ASICTarget))

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new MixedOptions) }.getMessage() should include(
      "duplicate case 'FPGA'"
    )

  }

  it should "require that all IO bundles are type equivalent" in {

    class Target16 extends FixedIOExtModule[TargetIO](new TargetIO(16))

    class MixedIO extends ModuleWithChoice(new VerifTarget)(Seq(Platform.FPGA -> new Target16))

    intercept[ChiselException] { ChiselStage.emitCHIRRTL(new MixedIO, Array("--throw-on-first-error")) }
      .getMessage() should include(
      "choice module IO bundles are not type equivalent"
    )

  }

}
