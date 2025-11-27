// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.HasExtModuleResource
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Avoid collisions with regular BlackBox tests by putting ExtModule blackboxes
// in their own scope.
package extmoduletests {

  class BlackBoxInverter extends ExtModule with HasExtModuleResource {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))

    addResource("/chisel3/BlackBoxInverter.v")
  }

  class BlackBoxPassthrough extends ExtModule with HasExtModuleResource {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))

    addResource("/chisel3/BlackBoxPassthrough.v")
  }
}

class ExtModuleTester extends Module {
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

class MultiExtModuleTester extends Module {
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

class ExtModuleInvalidatedTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  val inst = Module(new ExtModule {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
  })
  inst.in := in
  out := inst.out
}

class ExtModuleSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "A ExtModule inverter" should "work" in {
    simulate(new ExtModuleTester)(RunUntilFinished(3))
  }
  "Multiple ExtModules" should "work" in {
    simulate(new MultiExtModuleTester)(RunUntilFinished(3))
  }
  "DataMirror.modulePorts" should "work with ExtModule" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val m = Module(new extmoduletests.BlackBoxPassthrough)
      assert(DataMirror.modulePorts(m) == Seq("in" -> m.in, "out" -> m.out))
    })
  }

  behavior.of("ExtModule")

  it should "work with .suggestName (aka it should not require reflection for naming)" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleWithSuggestNameTester)
    chirrtl should include("input foo : UInt<8>")
    chirrtl should include("connect inst.foo, in")
  }

  it should "work with FlatIO" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleWithFlatIOTester)
    chirrtl should include("connect io.out, inst.out")
    chirrtl should include("connect inst.in, io.in")
    chirrtl shouldNot include("badIO")
  }

  it should "not have invalidated ports in a chisel3._ context" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleInvalidatedTester)
    chirrtl shouldNot include("invalidater inst.in")
    chirrtl shouldNot include("invalidate inst.out")
  }
}
