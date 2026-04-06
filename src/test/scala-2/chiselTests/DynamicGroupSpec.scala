// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.choice.{Case, DynamicCase, DynamicGroup, Group, ModuleChoice}
import chisel3.experimental.SourceInfo
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DynamicGroupSpec extends AnyFlatSpec with Matchers with FileCheck {

  class PlatformType(customName: String)(implicit sourceInfo: SourceInfo) extends DynamicGroup(customName) {
    object FPGA extends DynamicCase
    object ASIC extends DynamicCase
  }

  class PlatformGpuType(customName: String)(implicit sourceInfo: SourceInfo) extends DynamicGroup(customName) {
    object FPGA extends DynamicCase
    object GPU extends DynamicCase
  }

  class ReorderedPlatformType(customName: String)(implicit sourceInfo: SourceInfo) extends DynamicGroup(customName) {
    object ASIC extends DynamicCase
    object FPGA extends DynamicCase
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

  it should "emit options and cases with DynamicGroup" in {
    class ModuleWithDynamicChoice extends Module {
      val platform = new PlatformType("Platform")

      val inst = ModuleChoice(new VerifTarget)(
        Seq(
          platform.FPGA -> new FPGATarget,
          platform.ASIC -> new ASICTarget
        )
      )
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

  it should "reject DynamicGroup with same name but different cases" in {
    class ModuleWithMismatchedCases extends Module {
      val platform1 = new PlatformType("Platform")
      val platform2 = new PlatformGpuType("Platform")

      val inst1 = ModuleChoice(new VerifTarget)(Seq(platform1.FPGA -> new FPGATarget))
      val inst2 = ModuleChoice(new VerifTarget)(Seq(platform2.GPU -> new FPGATarget))
      val io1 = IO(inst1.cloneType)
      val io2 = IO(inst2.cloneType)
      io1 <> inst1
      io2 <> inst2
    }

    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new ModuleWithMismatchedCases)
    }

    exception.getMessage should include("Group 'Platform' has inconsistent case definitions")
  }

  it should "reject DynamicGroup with same cases but different order" in {
    class ModuleWithDifferentOrder extends Module {
      val platform1 = new PlatformType("Platform")
      val platform2 = new ReorderedPlatformType("Platform")

      val inst1 = ModuleChoice(new VerifTarget)(Seq(platform1.FPGA -> new FPGATarget))
      val inst2 = ModuleChoice(new VerifTarget)(Seq(platform2.ASIC -> new ASICTarget))
      val io1 = IO(inst1.cloneType)
      val io2 = IO(inst2.cloneType)
      io1 <> inst1
      io2 <> inst2
    }

    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new ModuleWithDifferentOrder)
    }

    exception.getMessage should include("Group 'Platform' has inconsistent case definitions")
  }

  it should "allow same DynamicGroup name across different submodules" in {
    class SubModule1 extends Module {
      val platform = new PlatformType("Platform")
      val inst = ModuleChoice(new VerifTarget)(Seq(platform.FPGA -> new FPGATarget, platform.ASIC -> new ASICTarget))
      val io = IO(inst.cloneType)
      io <> inst
    }

    class SubModule2 extends Module {
      val platform = new PlatformType("Platform") // Same name, same cases - should work
      val inst = ModuleChoice(new VerifTarget)(Seq(platform.FPGA -> new FPGATarget, platform.ASIC -> new ASICTarget))
      val io = IO(inst.cloneType)
      io <> inst
    }

    class TopModule extends Module {
      val sub1 = Module(new SubModule1)
      val sub2 = Module(new SubModule2)
      val io1 = IO(sub1.io.cloneType)
      val io2 = IO(sub2.io.cloneType)
      io1 <> sub1.io
      io2 <> sub2.io
    }

    ChiselStage
      .emitCHIRRTL(new TopModule)
      .fileCheck()(
        """|CHECK: option Platform :
           |CHECK-NEXT: FPGA
           |CHECK-NEXT: ASIC
           |CHECK-NOT: option Platform :
           |CHECK: module SubModule1 :
           |CHECK: instchoice inst of {{VerifTarget[_0-9]*}}, Platform :
           |CHECK: module SubModule2 :
           |CHECK: instchoice inst of {{VerifTarget[_0-9]*}}, Platform :
           """.stripMargin
      )
  }

  // Define a static group for testing static vs dynamic conflict
  object StaticPlatform extends Group {
    object FPGA extends Case
    object ASIC extends Case
  }

  it should "reject conflict between static Group and DynamicGroup with same name" in {
    class ModuleWithStaticGroup extends Module {
      val inst1 = ModuleChoice(new VerifTarget)(Seq(StaticPlatform.FPGA -> new FPGATarget))
      val platform = new PlatformGpuType("StaticPlatform") // Same name as static group, different cases
      val inst2 = ModuleChoice(new VerifTarget)(Seq(platform.GPU -> new FPGATarget))
      val io1 = IO(inst1.cloneType)
      val io2 = IO(inst2.cloneType)
      io1 <> inst1
      io2 <> inst2
    }

    val exception = intercept[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new ModuleWithStaticGroup)
    }

    exception.getMessage should include("StaticPlatform")
    exception.getMessage should include("inconsistent case definitions")
  }

}
