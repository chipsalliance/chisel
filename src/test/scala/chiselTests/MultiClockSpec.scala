// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Counter
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage

/** Multi-clock test of a Reg using a different clock via withClock */
class ClockDividerTest extends BasicTester {
  val cDiv = RegInit(true.B) // start with falling edge to simplify clock relationship assert
  cDiv := !cDiv
  val clock2 = cDiv.asClock

  val reg1 = RegInit(0.U(8.W))
  reg1 := reg1 + 1.U
  val reg2 = withClock(clock2) { RegInit(0.U(8.W)) }
  reg2 := reg2 + 1.U

  when(reg1 < 10.U) {
    assert(reg2 === reg1 / 2.U) // 1:2 clock relationship
  }

  when(reg1 === 10.U) {
    stop()
  }
}

class MultiClockSubModuleTest extends BasicTester {
  class SubModule extends Module {
    val io = IO(new Bundle {
      val out = Output(UInt())
    })
    val (cycle, _) = Counter(true.B, 10)
    io.out := cycle
  }

  val (cycle, done) = Counter(true.B, 10)
  val cDiv = RegInit(true.B) // start with falling edge to simplify clock relationship assert
  cDiv := !cDiv

  val otherClock = cDiv.asClock
  val otherReset = cycle < 3.U

  val inst = withClockAndReset(otherClock, otherReset) { Module(new SubModule) }

  when(done) {
    // The counter in inst should come out of reset later and increment at half speed
    assert(inst.io.out === 3.U)
    stop()
  }
}

/** Test withReset changing the reset of a Reg */
class WithResetTest extends BasicTester {
  val reset2 = WireDefault(false.B)
  val reg = withReset(reset2 || reset.asBool) { RegInit(0.U(8.W)) }
  reg := reg + 1.U

  val (cycle, done) = Counter(true.B, 10)
  when(cycle < 7.U) {
    assert(reg === cycle)
  }.elsewhen(cycle === 7.U) {
    reset2 := true.B
  }.elsewhen(cycle === 8.U) {
    assert(reg === 0.U)
  }
  when(done) { stop() }
}

/** Test Mem ports with different clocks */
class MultiClockMemTest extends BasicTester {
  val cDiv = RegInit(true.B)
  cDiv := !cDiv
  val clock2 = cDiv.asClock

  val mem = Mem(8, UInt(32.W))

  val (cycle, done) = Counter(true.B, 20)

  // Write port 1 walks through writing 123
  val waddr = RegInit(0.U(3.W))
  waddr := waddr + 1.U
  when(cycle < 8.U) {
    mem(waddr) := 123.U
  }

  val raddr = waddr - 1.U
  val rdata = mem(raddr)

  // Check each write from write port 1
  when(cycle > 0.U && cycle < 9.U) {
    assert(rdata === 123.U)
  }

  // Write port 2 walks through writing 456 on 2nd time through
  withClock(clock2) {
    when(cycle >= 8.U && cycle < 16.U) {
      mem(waddr) := 456.U // write 456 to different address
    }
  }

  // Check that every even address gets 456
  when(cycle > 8.U && cycle < 17.U) {
    when(raddr % 2.U === 0.U) {
      assert(rdata === 456.U)
    }.otherwise {
      assert(rdata === 123.U)
    }
  }

  when(done) { stop() }
}

class MultiClockSpec extends ChiselFlatSpec with Utils {

  "withClock" should "scope the clock of registers" in {
    assertTesterPasses(new ClockDividerTest)
  }

  it should "scope ports of memories" in {
    assertTesterPasses(new MultiClockMemTest, annotations = TesterDriver.verilatorOnly)
  }

  it should "return like a normal Scala block" in {
    ChiselStage.emitCHIRRTL(new BasicTester {
      assert(withClock(this.clock) { 5 } == 5)
    })
  }

  "Differing clocks at memory and port instantiation" should "warn" in {
    class modMemDifferingClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = withClock(myClock) { Mem(4, UInt(8.W)) }
      val port0 = mem(0.U)
    }
    val (logMemDifferingClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modMemDifferingClock))
    logMemDifferingClock should include("memory is different")

    class modSyncReadMemDifferingClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = withClock(myClock) { SyncReadMem(4, UInt(8.W)) }
      val port0 = mem(0.U)
    }
    val (logSyncReadMemDifferingClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemDifferingClock))
    logSyncReadMemDifferingClock should include("memory is different")
  }

  "Differing clocks at memory and write accessor instantiation" should "warn" in {
    class modMemWriteDifferingClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = withClock(myClock) { Mem(4, UInt(8.W)) }
      mem(1.U) := 1.U
    }
    val (logMemWriteDifferingClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modMemWriteDifferingClock))
    logMemWriteDifferingClock should include("memory is different")

    class modSyncReadMemWriteDifferingClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = withClock(myClock) { SyncReadMem(4, UInt(8.W)) }
      mem.write(1.U, 1.U)
    }
    val (logSyncReadMemWriteDifferingClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemWriteDifferingClock))
    logSyncReadMemWriteDifferingClock should include("memory is different")
  }

  "Differing clocks at memory and read accessor instantiation" should "warn" in {
    class modSyncReadMemReadDifferingClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = withClock(myClock) { SyncReadMem(4, UInt(8.W)) }
      val readVal = mem.read(0.U)
    }
    val (logSyncReadMemReadDifferingClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemReadDifferingClock))
    logSyncReadMemReadDifferingClock should include("memory is different")
  }

  "Passing clock parameter to memory port instantiation" should "not warn" in {
    class modMemPortClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = Mem(4, UInt(8.W))
      val port0 = mem(0.U, myClock)
    }
    val (logMemPortClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modMemPortClock))
    (logMemPortClock should not).include("memory is different")

    class modSyncReadMemPortClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = SyncReadMem(4, UInt(8.W))
      val port0 = mem(0.U, myClock)
    }
    val (logSyncReadMemPortClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemPortClock))
    (logSyncReadMemPortClock should not).include("memory is different")
  }

  "Passing clock parameter to memory write accessor" should "not warn" in {
    class modMemWriteClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = Mem(4, UInt(8.W))
      mem.write(0.U, 0.U, myClock)
    }
    val (logMemWriteClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modMemWriteClock))
    (logMemWriteClock should not).include("memory is different")

    class modSyncReadMemWriteClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = SyncReadMem(4, UInt(8.W))
      mem.write(0.U, 0.U, myClock)
    }
    val (logSyncReadMemWriteClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemWriteClock))
    (logSyncReadMemWriteClock should not).include("memory is different")
  }

  "Passing clock parameter to memory read accessor" should "not warn" in {
    class modMemReadClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = Mem(4, UInt(8.W))
      val readVal = mem.read(0.U, myClock)
    }
    val (logMemReadClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modMemReadClock))
    (logMemReadClock should not).include("memory is different")

    class modSyncReadMemReadClock extends Module {
      val myClock = IO(Input(Clock()))
      val mem = SyncReadMem(4, UInt(8.W))
      val readVal = mem.read(0.U, myClock)
    }
    val (logSyncReadMemReadClock, _) = grabLog(ChiselStage.emitCHIRRTL(new modSyncReadMemReadClock))
    (logSyncReadMemReadClock should not).include("memory is different")
  }

  "withReset" should "scope the reset of registers" in {
    assertTesterPasses(new WithResetTest)
  }

  it should "scope the clock and reset of Modules" in {
    assertTesterPasses(new MultiClockSubModuleTest)
  }

  it should "return like a normal Scala block" in {
    ChiselStage.emitCHIRRTL(new BasicTester {
      assert(withReset(this.reset) { 5 } == 5)
    })
  }
  it should "support literal Bools" in {
    assertTesterPasses(new BasicTester {
      val reg = withReset(true.B) {
        RegInit(6.U)
      }
      reg := reg - 1.U
      // The reg is always in reset so will never decrement
      chisel3.assert(reg === 6.U)
      val (_, done) = Counter(true.B, 4)
      when(done) { stop() }
    })
  }

  "withClockAndReset" should "return like a normal Scala block" in {
    ChiselStage.emitCHIRRTL(new BasicTester {
      assert(withClockAndReset(this.clock, this.reset) { 5 } == 5)
    })
  }

  it should "scope the clocks and resets of asserts" in {
    // Check that assert can fire
    assertTesterFails(new BasicTester {
      withClockAndReset(clock, reset) {
        chisel3.assert(0.U === 1.U)
      }
      val (_, done) = Counter(true.B, 2)
      when(done) { stop() }
    })
    // Check that reset will block
    assertTesterPasses(new BasicTester {
      withClockAndReset(clock, true.B) {
        chisel3.assert(0.U === 1.U)
      }
      val (_, done) = Counter(true.B, 2)
      when(done) { stop() }
    })
    // Check that no rising edge will block
    assertTesterPasses(new BasicTester {
      withClockAndReset(false.B.asClock, reset) {
        chisel3.assert(0.U === 1.U)
      }
      val (_, done) = Counter(true.B, 2)
      when(done) { stop() }
    })
  }

  it should "support setting Clock and Reset to None" in {
    val e1 = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        withClockAndReset(None, Some(this.reset)) {
          Reg(UInt(8.W))
        }
      })
    }
    e1.getMessage should include("No implicit clock.")

    val e2 = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        withClockAndReset(Some(this.clock), None) {
          RegInit(0.U(8.W))
        }
      })
    }
    e2.getMessage should include("No implicit reset.")
  }
}
