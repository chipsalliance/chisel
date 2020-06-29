// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.{Counter, Queue}
import chisel3.testers.BasicTester
import firrtl.checks.CheckResets.NonLiteralAsyncResetValueException

class AsyncResetTester extends BasicTester {
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

  when (count === 3.U) {
    assert(reg === 5.U)
  }
  when (count >= 5.U && count < 7.U) {
    assert(reg === 123.U)
  } .elsewhen (count >= 7.U) {
    assert(reg === 5.U)
  }

  when (done) {
    stop()
  }
}

class AsyncResetAggregateTester extends BasicTester {
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

  when (count === 3.U) {
    assert(reg(0).x === 5.U)
    assert(reg(0).y === 6.U)
    assert(reg(1).x === 7.U)
    assert(reg(1).y === 8.U)
  }
  when (count >= 5.U && count < 7.U) {
    assert(reg(0).x === 0.U)
    assert(reg(0).y === 0.U)
    assert(reg(1).x === 0.U)
    assert(reg(1).y === 0.U)
  } .elsewhen (count >= 7.U) {
    assert(reg(0).x === 5.U)
    assert(reg(0).y === 6.U)
    assert(reg(1).x === 7.U)
    assert(reg(1).y === 8.U)
  }

  when (done) {
    stop()
  }
}

class AsyncResetQueueTester extends BasicTester {
  val (_, cDiv) = Counter(true.B, 4)
  val slowClk = cDiv.asClock

  val (count, done) = Counter(true.B, 16)

  val asyncResetNext = RegNext(false.B, false.B)
  val asyncReset = asyncResetNext.asAsyncReset

  val queue = withClockAndReset (slowClk, asyncReset) {
    Module(new Queue(UInt(8.W), 4))
  }
  queue.io.enq.valid := true.B
  queue.io.enq.bits := count

  queue.io.deq.ready := false.B

  val doCheck = RegNext(false.B, false.B)
  when (queue.io.count === 3.U) {
    asyncResetNext := true.B
    doCheck := true.B
  }
  when (doCheck) {
    assert(queue.io.count === 0.U)
  }

  when (done) {
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

class AsyncResetSpec extends ChiselFlatSpec with Utils {

  behavior of "AsyncReset"

  it should "be able to be connected to DontCare" in {
    ChiselStage.elaborate(new AsyncResetDontCareModule)
  }

  it should "be allowed with literal reset values" in {
    ChiselStage.elaborate(new BasicTester {
      withReset(reset.asAsyncReset)(RegInit(123.U))
    })
  }

  it should "NOT be allowed with non-literal reset values" in {
    a [NonLiteralAsyncResetValueException] should be thrownBy extractCause[NonLiteralAsyncResetValueException] {
      compile(new BasicTester {
        val x = WireInit(123.U + 456.U)
        withReset(reset.asAsyncReset)(RegInit(x))
      })
    }
  }

  it should "NOT be allowed to connect directly to a Bool" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new BasicTester {
        val bool = Wire(Bool())
        val areset = reset.asAsyncReset
        bool := areset
      })
    }
  }

  it should "simulate correctly" in {
    assertTesterPasses(new AsyncResetTester)
  }

  it should "simulate correctly with aggregates" in {
    assertTesterPasses(new AsyncResetAggregateTester)
  }

  it should "allow casting to and from Bool" in {
    ChiselStage.elaborate(new BasicTester {
      val r: Reset = reset
      val a: AsyncReset = WireInit(r.asAsyncReset)
      val b: Bool = a.asBool
      val c: AsyncReset = b.asAsyncReset
    })
  }

  it should "allow changing the reset type of whole modules like Queue" in {
    assertTesterPasses(new AsyncResetQueueTester)
  }

  it should "support SInt regs" in {
    assertTesterPasses(new BasicTester {
      // Also check that it traces through wires
      val initValue = Wire(SInt())
      val reg = withReset(reset.asAsyncReset)(RegNext(initValue, 27.S))
      initValue := -43.S
      val (count, done) = Counter(true.B, 4)
      when (count === 0.U) {
        chisel3.assert(reg === 27.S)
      } .otherwise {
        chisel3.assert(reg === -43.S)
      }
      when (done) { stop() }
    })
  }

  it should "support Fixed regs" in {
    import chisel3.experimental.{withReset => _, _}
    assertTesterPasses(new BasicTester {
      val reg = withReset(reset.asAsyncReset)(RegNext(-6.0.F(2.BP), 3.F(2.BP)))
      val (count, done) = Counter(true.B, 4)
      when (count === 0.U) {
        chisel3.assert(reg === 3.F(2.BP))
      } .otherwise {
        chisel3.assert(reg === -6.0.F(2.BP))
      }
      when (done) { stop() }
    })
  }

  it should "support Interval regs" in {
    import chisel3.experimental.{withReset => _, _}
    assertTesterPasses(new BasicTester {
      val reg = withReset(reset.asAsyncReset) {
        val x = RegInit(Interval(range"[0,13]"), 13.I)
        x := 7.I
        x
      }
      val (count, done) = Counter(true.B, 4)
      when (count === 0.U) {
        chisel3.assert(reg === 13.I)
      } .otherwise {
        chisel3.assert(reg === 7.I)
      }
      when (done) { stop() }
    })
  }

  it should "allow literals cast to Bundles as reset values" in {
    class MyBundle extends Bundle {
      val x = UInt(16.W)
      val y = UInt(16.W)
    }
    assertTesterPasses(new BasicTester {
      val reg = withReset(reset.asAsyncReset) {
        RegNext(0xbad0cad0L.U.asTypeOf(new MyBundle), 0xdeadbeefL.U.asTypeOf(new MyBundle))
      }
      val (count, done) = Counter(true.B, 4)
      when (count === 0.U) {
        chisel3.assert(reg.asUInt === 0xdeadbeefL.U)
      } .otherwise {
        chisel3.assert(reg.asUInt === 0xbad0cad0L.U)
      }
      when (done) { stop() }
    })
  }
  it should "allow literals cast to Vecs as reset values" in {
    assertTesterPasses(new BasicTester {
      val reg = withReset(reset.asAsyncReset) {
        RegNext(0xbad0cad0L.U.asTypeOf(Vec(4, UInt(8.W))), 0xdeadbeefL.U.asTypeOf(Vec(4, UInt(8.W))))
      }
      val (count, done) = Counter(true.B, 4)
      when (count === 0.U) {
        chisel3.assert(reg.asUInt === 0xdeadbeefL.U)
      } .otherwise {
        chisel3.assert(reg.asUInt === 0xbad0cad0L.U)
      }
      when (done) { stop() }
    })
  }
}
