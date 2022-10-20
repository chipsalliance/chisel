// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3.stage.ChiselStage
import chisel3.testers.{BasicTester, TesterDriver}

// Avoid collisions with regular BlackBox tests by putting ExtModule blackboxes
// in their own scope.
package extmoduletests {

  import chisel3._
  import chisel3.experimental.ExtModule

  class BlackBoxInverter extends ExtModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))
  }

  class BlackBoxPassthrough extends ExtModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))
  }
}

object Chisel3BlackBoxes {

  import chisel3._
  import chisel3.experimental._

  class ExtModuleTester extends BasicTester {
    val blackBoxPos = Module(new extmoduletests.BlackBoxInverter)
    val blackBoxNeg = Module(new extmoduletests.BlackBoxInverter)

    blackBoxPos.in := 1.U
    blackBoxNeg.in := 0.U

    assert(blackBoxNeg.out === 1.U)
    assert(blackBoxPos.out === 0.U)
    stop()
  }

  /** Instantiate multiple BlackBoxes with similar interfaces but different
    * functionality. Used to detect failures in BlackBox naming and module
    * deduplication.
    */

  class MultiExtModuleTester extends BasicTester {
    val blackBoxInvPos = Module(new extmoduletests.BlackBoxInverter)
    val blackBoxInvNeg = Module(new extmoduletests.BlackBoxInverter)
    val blackBoxPassPos = Module(new extmoduletests.BlackBoxPassthrough)
    val blackBoxPassNeg = Module(new extmoduletests.BlackBoxPassthrough)

    blackBoxInvPos.in := 1.U
    blackBoxInvNeg.in := 0.U
    blackBoxPassPos.in := 1.U
    blackBoxPassNeg.in := 0.U

    assert(blackBoxInvNeg.out === 1.U)
    assert(blackBoxInvPos.out === 0.U)
    assert(blackBoxPassNeg.out === 0.U)
    assert(blackBoxPassPos.out === 1.U)
    stop()
  }

  class ExtModuleWithSuggestName extends ExtModule {
    val in = IO(Input(UInt(8.W)))
    in.suggestName("foo")
    val out = IO(Output(UInt(8.W)))
  }

  class ExtModuleWithSuggestNameTester extends Module {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    val inst = Module(new ExtModuleWithSuggestName)
    inst.in := in
    out := inst.out
  }

  class SimpleIOBundle extends Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  }

  class ExtModuleWithFlatIO extends ExtModule {
    val badIO = FlatIO(new SimpleIOBundle)
  }

  class ExtModuleWithFlatIOTester extends Module {
    val io = IO(new SimpleIOBundle)
    val inst = Module(new ExtModuleWithFlatIO)
    io <> inst.badIO
  }

  class ExtModuleChisel3Tester extends Module {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    val inst = Module(new ExtModule {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
    })
    inst.in := in
    out := inst.out
  }
}

object CompatibilityBlackBoxes {

  import Chisel._
  import Chisel.experimental._

  class ExtModuleChiselTester extends Module {
    val io = IO(new Bundle {
      val in = Input(UInt(8.W))
      val out = Output(UInt(8.W))
    })
    val inst = Module(new BlackBox {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
    })
    inst.io.in := io.in
    io.out := inst.io.out
  }
}

class ExtModuleSpec extends ChiselFlatSpec {

  import chisel3._
  import chisel3.experimental._

  "A ExtModule inverter" should "work" in {
    assertTesterPasses({ new Chisel3BlackBoxes.ExtModuleTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "Multiple ExtModules" should "work" in {
    assertTesterPasses({ new Chisel3BlackBoxes.MultiExtModuleTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "DataMirror.modulePorts" should "work with ExtModule" in {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {})
      val m = Module(new extmoduletests.BlackBoxPassthrough)
      assert(DataMirror.modulePorts(m) == Seq("in" -> m.in, "out" -> m.out))
    })
  }

  behavior.of("ExtModule")

  it should "work with .suggestName (aka it should not require reflection for naming)" in {
    val chirrtl = ChiselStage.emitChirrtl(new Chisel3BlackBoxes.ExtModuleWithSuggestNameTester)
    chirrtl should include("input foo : UInt<8>")
    chirrtl should include("inst.foo <= in")
  }

  it should "work with FlatIO" in {
    val chirrtl = ChiselStage.emitChirrtl(new Chisel3BlackBoxes.ExtModuleWithFlatIOTester)
    chirrtl should include("io.out <= inst.out")
    chirrtl should include("inst.in <= io.in")
    chirrtl shouldNot include("badIO")
  }

  it should "not have invalidated ports in a chisel3._ context" in {
    val chirrtl = ChiselStage.emitChirrtl(new Chisel3BlackBoxes.ExtModuleChisel3Tester)
    chirrtl shouldNot include("inst.in is invalid")
    chirrtl shouldNot include("inst.out is invalid")
  }
  it should "have invalidated ports in a Chisel._ context" in {
    val chirrtl = ChiselStage.emitChirrtl(new CompatibilityBlackBoxes.ExtModuleChiselTester)
    chirrtl should include("inst.in is invalid")
    chirrtl should include("inst.out is invalid")
  }
}
