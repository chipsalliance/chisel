// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptionBundle(val hasIn: Boolean) extends Bundle {
  val in = if (hasIn) {
    Some(Input(Bool()))
  } else {
    None
  }
  val out = Output(Bool())
}

class OptionBundleModule(val hasIn: Boolean) extends Module {
  val io = IO(new OptionBundle(hasIn))
  if (hasIn) {
    io.out := io.in.get
  } else {
    io.out := false.B
  }
}

class SomeOptionBundleTester(expected: Boolean) extends Module {
  val mod = Module(new OptionBundleModule(true))
  mod.io.in.get := expected.asBool
  assert(mod.io.out === expected.asBool)
  stop()
}

class NoneOptionBundleTester() extends Module {
  val mod = Module(new OptionBundleModule(false))
  assert(mod.io.out === false.B)
  stop()
}

class InvalidOptionBundleTester() extends Module {
  val mod = Module(new OptionBundleModule(false))
  mod.io.in.get := true.B
  assert(false.B)
  stop()
}

class OptionBundleSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "A Bundle with an Option field" should "work properly if the Option field is not None" in {
    simulate(new SomeOptionBundleTester(true))(RunUntilFinished(3))
    simulate(new SomeOptionBundleTester(false))(RunUntilFinished(3))
  }

  "A Bundle with an Option field" should "compile if the Option field is None" in {
    simulate(new NoneOptionBundleTester())(RunUntilFinished(3))
  }

  "A Bundle with an Option field" should "assert out accessing a None Option field" in {
    intercept[Exception] {
      ChiselStage.emitCHIRRTL { new InvalidOptionBundleTester() }
    }
  }
}
