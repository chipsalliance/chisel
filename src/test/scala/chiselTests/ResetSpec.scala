// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ResetAgnosticModule extends RawModule {
  val clk = IO(Input(Clock()))
  val rst = IO(Input(Reset()))
  val out = IO(Output(UInt(8.W)))

  val reg = withClockAndReset(clk, rst)(RegInit(0.U(8.W)))
  reg := reg + 1.U
  out := reg
}

class AbstractResetDontCareModule extends RawModule {
  import chisel3.util.Valid
  val monoPort = IO(Output(Reset()))
  monoPort := DontCare
  val monoWire = Wire(Reset())
  monoWire := DontCare
  val monoAggPort = IO(Output(Valid(Reset())))
  monoAggPort := DontCare
  val monoAggWire = Wire(Valid(Reset()))
  monoAggWire := DontCare

  // Can't bulk connect to Wire so only ports here
  val bulkPort = IO(Output(Reset()))
  bulkPort <> DontCare
  val bulkAggPort = IO(Output(Valid(Reset())))
  bulkAggPort <> DontCare
}

class ResetSpec extends AnyFlatSpec with Matchers {

  behavior.of("Reset")

  it should "be able to be connected to DontCare" in {
    ChiselStage.emitCHIRRTL(new AbstractResetDontCareModule)
  }

  it should "be able to drive Bool" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      val w = Wire(Reset())
      w := in
      out := w
    })
  }

  it should "be able to drive AsyncReset" in {
    ChiselStage.emitSystemVerilog(new RawModule {
      val in = IO(Input(AsyncReset()))
      val out = IO(Output(AsyncReset()))
      val w = Wire(Reset())
      w := in
      out := w
    })
  }

  it should "allow writing modules that are reset agnostic" in {
    val sync = ChiselStage.emitSystemVerilog(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      val inst = Module(new ResetAgnosticModule)
      inst.clk := clock
      inst.rst := reset
      assert(inst.rst.isInstanceOf[chisel3.ResetType])
      io.out := inst.out
    })
    sync should include("always @(posedge clk)")

    val async = ChiselStage.emitSystemVerilog(new Module {
      val io = IO(new Bundle {
        val out = Output(UInt(8.W))
      })
      val inst = Module(new ResetAgnosticModule)
      inst.clk := clock
      inst.rst := reset.asTypeOf(AsyncReset())
      assert(inst.rst.isInstanceOf[chisel3.ResetType])
      io.out := inst.out
    })
    async should include("always @(posedge clk or posedge rst)")
  }

  it should "be able to be derived from Bool" in {
    val fir = ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Input(Bool()))
      val out = IO(Output(Reset()))
      out := in.asReset
    })
    fir should include("asReset(in)")
  }

  behavior.of("Users")

  they should "be able to force implicit reset to be synchronous" in {
    val fir = ChiselStage.emitCHIRRTL(new Module with RequireSyncReset {
      reset shouldBe a[Bool]
    })
    fir should include("input reset : UInt<1>")
  }

  they should "be able to force implicit reset to be asynchronous" in {
    val fir = ChiselStage.emitCHIRRTL(new Module with RequireAsyncReset {
      reset shouldBe an[AsyncReset]
    })
    fir should include("input reset : AsyncReset")
  }

  they should "be able to override the value of the implicit reset" in {
    val verilog = ChiselStage.emitSystemVerilog(new Module {
      val gate = IO(Input(Bool()))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      override protected val implicitReset = reset.asBool || gate
      val r = RegInit(0.U)
      out := r
      r := in
    })
    verilog should include("if (reset | gate)")
  }

  they should "be able to add an implicit clock to a RawModule" in {
    val verilog = ChiselStage.emitSystemVerilog(new RawModule with ImplicitReset {
      val foo = IO(Input(Bool()))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      override protected val implicitReset = !foo

      val clk = IO(Input(Clock()))
      val r = withClock(clk)(RegInit(0.U))
      out := r
      r := in
    })
    verilog should include("if (~foo)")
  }

  they should "be able to have parameterized top level reset type" in {
    class MyModule(hasAsyncNotSyncReset: Boolean) extends Module {
      override def resetType = if (hasAsyncNotSyncReset) Module.ResetType.Asynchronous else Module.ResetType.Synchronous
    }
    val firAsync = ChiselStage.emitCHIRRTL(new MyModule(true) {
      reset shouldBe an[AsyncReset]
    })
    firAsync should include("input reset : AsyncReset")

    val firSync = ChiselStage.emitCHIRRTL(new MyModule(false) {
      reset shouldBe a[Bool]
    })
    firSync should include("input reset : UInt<1>")
  }

  "Chisel" should "error if sync and async modules are nested" in {
    a[ChiselException] should be thrownBy {
      ChiselStage.emitCHIRRTL(new Module with RequireAsyncReset {
        val mod = Module(new Module with RequireSyncReset)
      })
    }
  }

  it should "give a decent error message if you try to use a reset before defining it" in {
    val e = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule with ImplicitReset {
          val r = Module.reset
          val foo = IO(Input(AsyncReset()))
          override protected def implicitReset = foo
        },
        args = Array("--throw-on-first-error")
      )
    )
    e.getMessage should include(
      "The implicit reset is null which means the code that sets its definition has not yet executed."
    )
  }
}
