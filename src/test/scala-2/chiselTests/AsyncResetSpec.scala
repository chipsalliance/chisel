// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, Queue}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AsyncResetTester extends Module {
  val (_, cDiv) = Counter(true.B, 4)
  // First rising edge when count === 3
  val slowClk = cDiv.asClock

  val (count, done) = Counter(true.B, 16)

  val asyncResetNext = RegInit(false.B)
  asyncResetNext := count === 4.U
  val asyncReset = asyncResetNext.asAsyncReset

  val reg = withClockAndReset(slowClk, asyncReset) {
    RegInit(123.U(8.W))
  }
  reg := 5.U // Normal connection

  when(count === 3.U) {
    assert(reg === 5.U)
  }
  when(count >= 5.U && count < 7.U) {
    assert(reg === 123.U)
  }.elsewhen(count >= 7.U) {
    assert(reg === 5.U)
  }

  when(done) {
    stop()
  }
}

class AsyncResetAggregateTester extends Module {
  class MyBundle extends Bundle {
    val x = UInt(8.W)
    val y = UInt(8.W)
  }
  val (_, cDiv) = Counter(true.B, 4)
  // First rising edge when count === 3
  val slowClk = cDiv.asClock

  val (count, done) = Counter(true.B, 16)

  val asyncResetNext = RegInit(false.B)
  asyncResetNext := count === 4.U
  val asyncReset = asyncResetNext.asAsyncReset

  val reg = withClockAndReset(slowClk, asyncReset) {
    val init = Wire(Vec(2, new MyBundle))
    init(0).x := 0.U
    init(0).y := 0.U
    init(1).x := 0.U
    init(1).y := 0.U
    RegInit(init)
  }
  reg(0).x := 5.U // Normal connections
  reg(0).y := 6.U
  reg(1).x := 7.U
  reg(1).y := 8.U

  when(count === 3.U) {
    assert(reg(0).x === 5.U)
    assert(reg(0).y === 6.U)
    assert(reg(1).x === 7.U)
    assert(reg(1).y === 8.U)
  }
  when(count >= 5.U && count < 7.U) {
    assert(reg(0).x === 0.U)
    assert(reg(0).y === 0.U)
    assert(reg(1).x === 0.U)
    assert(reg(1).y === 0.U)
  }.elsewhen(count >= 7.U) {
    assert(reg(0).x === 5.U)
    assert(reg(0).y === 6.U)
    assert(reg(1).x === 7.U)
    assert(reg(1).y === 8.U)
  }

  when(done) {
    stop()
  }
}

class AsyncResetQueueTester extends Module {
  val (_, cDiv) = Counter(true.B, 4)
  val slowClk = cDiv.asClock

  val (count, done) = Counter(true.B, 16)

  val asyncResetNext = RegNext(false.B, false.B)
  val asyncReset = asyncResetNext.asAsyncReset

  val queue = withClockAndReset(slowClk, asyncReset) {
    Module(new Queue(UInt(8.W), 4))
  }
  queue.io.enq.valid := true.B
  queue.io.enq.bits := count

  queue.io.deq.ready := false.B

  val doCheck = RegNext(false.B, false.B)
  when(queue.io.count === 3.U) {
    asyncResetNext := true.B
    doCheck := true.B
  }
  when(doCheck) {
    assert(queue.io.count === 0.U)
  }

  when(done) {
    stop()
  }
}

class AsyncResetDontCareModule extends RawModule {
  import chisel3.util.Valid
  val monoPort = IO(Output(AsyncReset()))
  monoPort := DontCare
  val monoWire = Wire(AsyncReset())
  monoWire := DontCare
  val monoAggPort = IO(Output(Valid(AsyncReset())))
  monoAggPort := DontCare
  val monoAggWire = Wire(Valid(AsyncReset()))
  monoAggWire := DontCare

  // Can't bulk connect to Wire so only ports here
  val bulkPort = IO(Output(AsyncReset()))
  bulkPort <> DontCare
  val bulkAggPort = IO(Output(Valid(AsyncReset())))
  bulkAggPort <> DontCare
}

class AsyncResetSpec extends AnyFlatSpec with Matchers with ChiselSim {

  behavior.of("AsyncReset")

  it should "be able to be connected to DontCare" in {
    ChiselStage.emitCHIRRTL(new AsyncResetDontCareModule)
  }

  it should "be allowed with literal reset values" in {
    ChiselStage.emitCHIRRTL(new Module {
      withReset(reset.asAsyncReset)(RegInit(123.U))
    })
  }

  it should "NOT be allowed with non-literal reset values" in {
    val e = intercept[RuntimeException] {
      ChiselStage.emitSystemVerilog(new Module {
        val x = WireInit(123.U + 456.U)
        withReset(reset.asAsyncReset)(RegInit(x))
      })
    }
    e.getMessage should include
    ("error: register \"REG\" has an async reset, but its reset value \"x\" is not driven with a constant value through wires, nodes, or connects")
  }

  it should "NOT be allowed to connect directly to a Bool" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new Module {
        val bool = Wire(Bool())
        val areset = reset.asAsyncReset
        bool := areset
      })
    }
  }

  it should "simulate correctly" in {
    simulate(new AsyncResetTester)(RunUntilFinished(17))
  }

  it should "simulate correctly with aggregates" in {
    simulate(new AsyncResetAggregateTester)(RunUntilFinished(17))
  }

  it should "allow casting to and from Bool" in {
    ChiselStage.emitCHIRRTL(new Module {
      val r: Reset = reset
      val a: AsyncReset = WireInit(r.asAsyncReset)
      val b: Bool = a.asBool
      val c: AsyncReset = b.asAsyncReset
    })
  }

  it should "allow changing the reset type of whole modules like Queue" in {
    simulate(new AsyncResetQueueTester)(RunUntilFinished(17))
  }

  it should "support SInt regs" in {
    simulate(new Module {
      // Also check that it traces through wires
      val initValue = Wire(SInt())
      val reg = withReset(reset.asAsyncReset)(RegNext(initValue, 27.S))
      initValue := -43.S
      val (count, done) = Counter(true.B, 4)
      when(count === 0.U) {
        chisel3.assert(reg === 27.S)
      }.otherwise {
        chisel3.assert(reg === -43.S)
      }
      when(done) { stop() }
    })(RunUntilFinished(5))
  }

  it should "allow literals cast to Bundles as reset values" in {
    class MyBundle extends Bundle {
      val x = UInt(16.W)
      val y = UInt(16.W)
    }
    simulate(new Module {
      val reg = withReset(reset.asAsyncReset) {
        RegNext(0xbad0cad0L.U.asTypeOf(new MyBundle), 0xdeadbeefL.U.asTypeOf(new MyBundle))
      }
      val (count, done) = Counter(true.B, 4)
      when(count === 0.U) {
        chisel3.assert(reg.asUInt === 0xdeadbeefL.U)
      }.otherwise {
        chisel3.assert(reg.asUInt === 0xbad0cad0L.U)
      }
      when(done) { stop() }
    })(RunUntilFinished(5))
  }
  it should "allow literals cast to Vecs as reset values" in {
    simulate(new Module {
      val reg = withReset(reset.asAsyncReset) {
        RegNext(0xbad0cad0L.U.asTypeOf(Vec(4, UInt(8.W))), 0xdeadbeefL.U.asTypeOf(Vec(4, UInt(8.W))))
      }
      val (count, done) = Counter(true.B, 4)
      when(count === 0.U) {
        chisel3.assert(reg.asUInt === 0xdeadbeefL.U)
      }.otherwise {
        chisel3.assert(reg.asUInt === 0xbad0cad0L.U)
      }
      when(done) { stop() }
    })(RunUntilFinished(5))
  }
}
