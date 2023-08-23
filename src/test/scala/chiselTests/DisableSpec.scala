// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import _root_.circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import chisel3.experimental.prefix

class DisableSpec extends AnyFlatSpec with Matchers {

  behavior.of("Disable")

  it should "should be None by default in a RawModule" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      Module.disableOption should be(None)
    })
  }

  it should "throw an exception when using Module.disable in a RawModule" in {
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new RawModule {
        Module.disable
      })
    }
    e.getMessage should include("No implicit disable")
  }

  it should "default to hasBeenReset in a Module" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      override def desiredName = "Top"
      val doDisable = Module.disableOption
    })
    chirrtl should include("intmodule HasBeenResetIntrinsic :")
    chirrtl should include("input clock : Clock")
    chirrtl should include("input reset : Reset")
    chirrtl should include("output out : UInt<1>")
    chirrtl should include("intrinsic = circt_has_been_reset")
    chirrtl should include("module Top :")
    chirrtl should include("inst HasBeenResetIntrinsic of HasBeenResetIntrinsic")
    chirrtl should include("connect HasBeenResetIntrinsic.clock, clock")
    chirrtl should include("connect HasBeenResetIntrinsic.reset, reset")
    chirrtl should include("node doDisable = eq(HasBeenResetIntrinsic.out, UInt<1>(0h0))")
  }

  it should "be None when there is a clock but no reset" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val clk = IO(Input(Clock()))
      withClock(clk) {
        Module.disableOption should be(None)
      }
    })
  }

  it should "be None when there is a reset but no clock" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val rst = IO(Input(AsyncReset()))
      withReset(rst) {
        Module.disableOption should be(None)
      }
    })
  }

  it should "be defined when there is a clock and a reset" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val clk = IO(Input(Clock()))
      val rst = IO(Input(AsyncReset()))
      withClockAndReset(clk, rst) {
        assert(Module.disableOption.isDefined)
      }
    })
  }

  it should "be possible to set it to Never" in {
    ChiselStage.emitCHIRRTL(new Module {
      assert(Module.disableOption.isDefined)
      withDisable(Disable.Never) {
        Module.disableOption should be(None)
      }
      assert(Module.disableOption.isDefined)
    })
  }

  it should "setting should propagate across module boundaries" in {
    ChiselStage.emitCHIRRTL(new Module {
      assert(Module.disableOption.isDefined)
      withDisable(Disable.Never) {
        Module.disableOption should be(None)
        val inst = Module(new Module {
          Module.disableOption should be(None)
        })
      }
      assert(Module.disableOption.isDefined)
    })
  }

  it should "be setable back to BeforeReset" in {
    ChiselStage.emitCHIRRTL(new Module {
      assert(Module.disableOption.isDefined)
      withDisable(Disable.Never) {
        Module.disableOption should be(None)
        val inst = Module(new Module {
          Module.disableOption should be(None)
          withDisable(Disable.BeforeReset) {
            assert(Module.disableOption.isDefined)
          }
        })
      }
      assert(Module.disableOption.isDefined)
    })
  }

  it should "default the node name to disable" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      Module.disableOption // No name given
    })
    chirrtl should include("node disable = eq(HasBeenResetIntrinsic.out, UInt<1>(0h0))")
  }

  it should "be impacted by prefix" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      prefix("foo") {
        Module.disableOption // No name given
      }
    })
    chirrtl should include("node foo_disable = eq(HasBeenResetIntrinsic.out, UInt<1>(0h0))")
  }
}
