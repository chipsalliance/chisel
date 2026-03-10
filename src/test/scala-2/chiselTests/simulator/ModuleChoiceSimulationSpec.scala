// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import chisel3._
import chisel3.choice.{Case, Group, ModuleChoice, DynamicGroup}
import chisel3.simulator.{InstanceChoiceControl, Settings}
import chisel3.simulator.InstanceChoiceControl.SpecializationTime
import chisel3.simulator.scalatest.ChiselSim
import chisel3.testing.scalatest.FileCheck
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

object Platform extends Group {
  object FPGA extends Case
  object ASIC extends Case
}

class TargetIO extends Bundle {
  val out = Output(UInt(8.W))
}

/** Test module with ModuleChoice that outputs different values based on selection */
class ModuleChoiceTestModule extends Module {
  val out1, out2 = IO(Output(UInt(8.W)))

  class Return0 extends FixedIORawModule[TargetIO](new TargetIO) {
    io.out := 0.U
  }
  class Return1 extends FixedIORawModule[TargetIO](new TargetIO) {
    io.out := 1.U
  }
  class Return2 extends FixedIORawModule[TargetIO](new TargetIO) {
    io.out := 2.U
  }

  val choiceOut1 = ModuleChoice(new Return0)(
    Seq(
      Platform.FPGA -> new Return1,
      Platform.ASIC -> new Return2
    )
  )

  out1 := choiceOut1.out

  // Use a dynamic group
  val group = new DynamicGroup("Opt", Seq("Fast"))
  val choiceOut2 = ModuleChoice(new Return0)(
    Seq(
      group("Fast") -> new Return1
    )
  )

  out2 := choiceOut2.out
}

/** Test ModuleChoice with FirtoolCompilationTime */
class ModuleChoiceFirtoolCompilationTimeSpec extends AnyFunSpec with ChiselSim with Matchers {

  describe("ModuleChoice at FirtoolCompilationTime") {
    it("should select single choice") {
      val settings = Settings
        .default[ModuleChoiceTestModule]
        .withInstanceChoices(
          InstanceChoiceControl(List((SpecializationTime.FirtoolCompilationTime, "Platform", "FPGA")))
        )

      simulate(new ModuleChoiceTestModule, settings = settings) { dut =>
        dut.out1.peek().litValue shouldBe 1
        dut.out2.peek().litValue shouldBe 0
      }
    }

    it("should select multiple choices") {
      val settings = Settings
        .default[ModuleChoiceTestModule]
        .withInstanceChoices(
          InstanceChoiceControl(
            List(
              (SpecializationTime.FirtoolCompilationTime, "Platform", "FPGA"),
              (SpecializationTime.FirtoolCompilationTime, "Opt", "Fast")
            )
          )
        )

      simulate(new ModuleChoiceTestModule, settings = settings) { dut =>
        dut.out1.peek().litValue shouldBe 1
        dut.out2.peek().litValue shouldBe 1
      }
    }

  }
}

/** Test ModuleChoice with VerilogElaborationTime */
class ModuleChoiceVerilogElaborationTimeSpec extends AnyFunSpec with ChiselSim with Matchers with FileCheck {

  describe("ModuleChoice at VerilogElaborationTime") {
    it("should select single choice") {
      val settings = Settings
        .default[ModuleChoiceTestModule]
        .withInstanceChoices(
          InstanceChoiceControl(List((SpecializationTime.VerilogElaborationTime, "Platform", "FPGA")))
        )

      simulate(new ModuleChoiceTestModule, settings = settings) { dut =>
        dut.out1.peek().litValue shouldBe 1
        dut.out2.peek().litValue shouldBe 0
      }
    }

    it("should select multiple choices") {
      val settings = Settings
        .default[ModuleChoiceTestModule]
        .withInstanceChoices(
          InstanceChoiceControl(
            List(
              (SpecializationTime.VerilogElaborationTime, "Platform", "FPGA"),
              (SpecializationTime.VerilogElaborationTime, "Opt", "Fast")
            )
          )
        )

      simulate(new ModuleChoiceTestModule, settings = settings) { dut =>
        dut.out1.peek().litValue shouldBe 1
        dut.out2.peek().litValue shouldBe 1
      }
    }

    it("should detect conflicting choices for same option") {
      val settings = Settings
        .default[ModuleChoiceTestModule]
        .withInstanceChoices(
          InstanceChoiceControl(
            List(
              (SpecializationTime.VerilogElaborationTime, "Platform", "FPGA"),
              (SpecializationTime.VerilogElaborationTime, "Platform", "ASIC")
            )
          )
        )

      info("Verilator errors if conflicting targets are set")
      intercept[Exception] {
        simulate(new ModuleChoiceTestModule, settings = settings)(_ => {})
      }.getMessage.fileCheck() {
        """|CHECK: _ERROR_{{.*}}_must_not_be_set
           |""".stripMargin
      }
    }
  }
}
