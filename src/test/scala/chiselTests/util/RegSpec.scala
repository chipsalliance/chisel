package chiselTests.util

import chisel3._
import chisel3.util.{RegEnable, ShiftRegister, ShiftRegisters}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RegEnableSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.RegEnable")

  it should "have source locators when passed nextVal, ena" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := RegEnable(in, true.B)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset.*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed next, init, enable" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := RegEnable(in, true.B, true.B)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }
}

class ShiftRegisterSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.ShiftRegister")

  it should "have source locators when passed in, n, en" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegister(in, 2, true.B)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed in, n" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegister(in, 2)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed in, n, resetData, en" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegister(in, 2, false.B, true.B)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed in, n, en, useDualPortSram, name" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegister.mem(in, 2, true.B, false, Some("sr"))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }
}

class ShiftRegistersSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.ShiftRegisters")

  it should "have source locators when passed in, n, en" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegisters(in, 2, true.B)(0)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed in, n" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegisters(in, 2)(0)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

  it should "have source locators when passed in, n, resetData, en" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := ShiftRegisters(in, 2, false.B, true.B)(0)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

}
