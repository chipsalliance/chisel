// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.choice.{Case, DynamicGroup, Group, ModuleChoice}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DynamicGroupSpec extends AnyFlatSpec with Matchers with FileCheck {

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

  it should "emit options and cases with DynamicGroup" in {
    val platform = new DynamicGroup("Platform", Seq("FPGA", "ASIC"))

    class ModuleWithDynamicChoice extends Module {
      val inst = ModuleChoice(new VerifTarget)(Seq(
        platform("FPGA") -> new FPGATarget,
        platform("ASIC") -> new ASICTarget
      ))
      val io = IO(inst.cloneType)
      io <> inst
    }

    ChiselStage
      .emitCHIRRTL(new ModuleWithDynamicChoice)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK-NEXT: ASIC
           |CHECK: instchoice inst of VerifTarget, Platform :
           |CHECK-NEXT: FPGA => FPGATarget
           |CHECK-NEXT: ASIC => ASICTarget""".stripMargin
      )
  }

  it should "allow same DynamicGroup name to be reused" in {
    class ModuleWithReusedGroup extends Module {
      val platform1 = new DynamicGroup("Platform", Seq("FPGA", "ASIC"))
      val platform2 = new DynamicGroup("Platform", Seq("FPGA", "ASIC")) // Should share the same group

      val inst1 = ModuleChoice(new VerifTarget)(Seq(platform1("FPGA") -> new FPGATarget))
      val inst2 = ModuleChoice(new VerifTarget)(Seq(platform2("ASIC") -> new ASICTarget))
      val io1 = IO(inst1.cloneType)
      val io2 = IO(inst2.cloneType)
      io1 <> inst1
      io2 <> inst2
    }

    ChiselStage
      .emitCHIRRTL(new ModuleWithReusedGroup)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK-NEXT: ASIC
           |CHECK-NOT: option Platform :
           """.stripMargin
      )

  }

  it should "reject DynamicGroup with same name but different cases" in {
    class ModuleWithMismatchedCases extends Module {
      val platform1 = new DynamicGroup("Platform", Seq("FPGA", "ASIC"))
      val platform2 = new DynamicGroup("Platform", Seq("FPGA", "GPU"))
    }

    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new ModuleWithMismatchedCases)
    }

    exception.getMessage should include("DynamicGroup 'Platform' already exists with different case names")
    exception.getMessage should include("FPGA")
    exception.getMessage should include("ASIC")
    exception.getMessage should include("GPU")
  }

  it should "reject DynamicGroup with same cases but different order" in {
    class ModuleWithDifferentOrder extends Module {
      val platform1 = new DynamicGroup("Platform", Seq("FPGA", "ASIC"))
      val platform2 = new DynamicGroup("Platform", Seq("ASIC", "FPGA"))
    }

    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new ModuleWithDifferentOrder)
    }

    exception.getMessage should include("DynamicGroup 'Platform' already exists with different case names")
    exception.getMessage should include("FPGA")
    exception.getMessage should include("ASIC")
  }
}

