// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class WhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value === 0.U) {
    out := 1.U
  }.elsewhen(cnt.value === 1.U) {
    out := 2.U
  }.elsewhen(cnt.value === 2.U) {
    out := 3.U
  }.otherwise {
    out := 0.U
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class OverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value <= 0.U) {
    out := 1.U
  }.elsewhen(cnt.value <= 1.U) {
    out := 2.U
  }.elsewhen(cnt.value <= 2.U) {
    out := 3.U
  }.otherwise {
    out := 0.U
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class NoOtherwiseOverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value <= 0.U) {
    out := 1.U
  }.elsewhen(cnt.value <= 1.U) {
    out := 2.U
  }.elsewhen(cnt.value <= 2.U) {
    out := 3.U
  }.elsewhen(cnt.value <= 3.U) {
    out := 0.U
  }.otherwise {
    out := DontCare
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class SubmoduleWhenTester extends BasicTester {
  val (cycle, done) = Counter(true.B, 3)
  when(done) { stop() }
  val children =
    Seq(Module(new PassthroughModule), Module(new PassthroughModule), Module(new PassthroughRawModule))
  children.foreach { child =>
    when(cycle === 1.U) {
      child.io.in := "hdeadbeef".U
      assert(child.io.out === "hdeadbeef".U)
    }.otherwise {
      child.io.in := "h0badcad0".U
      assert(child.io.out === "h0badcad0".U)
    }
  }
}

class WhenCondTester extends BasicTester {
  val pred = Wire(Vec(4, Bool()))
  val (cycle, done) = Counter(true.B, 1 << pred.size)
  // Cycle through every predicate
  pred := cycle.asBools
  val Seq(a, b, c, d) = pred // Just for nicer accessors
  // When want the when predicates on connection to optimize away,
  //   it's not necessary but it makes the Verilog prettier
  val w1, w2, w3, w4, w5, w6, w7 = WireInit(Bool(), DontCare)
  when(a) {
    w1 := when.cond
    when(b) {
      w2 := when.cond
    }.elsewhen(c) {
      w3 := when.cond
    }.elsewhen(d) {
      w4 := when.cond
    }.otherwise {
      w5 := when.cond
    }
  }.otherwise {
    w6 := when.cond
  }
  w7 := when.cond

  assert(w1 === a)
  assert(w2 === (a && b))
  assert(w3 === (a && !b && c))
  assert(w4 === (a && !b && !c && d))
  assert(w5 === (a && !b && !c && !d))
  assert(w6 === !a)
  assert(w7)

  when(done) { stop() }
}

class WhenSpec extends ChiselFlatSpec with Utils {
  "When, elsewhen, and otherwise with orthogonal conditions" should "work" in {
    assertTesterPasses { new WhenTester }
  }
  "When, elsewhen, and otherwise with overlapped conditions" should "work" in {
    assertTesterPasses { new OverlappedWhenTester }
  }
  "When and elsewhen without otherwise with overlapped conditions" should "work" in {
    assertTesterPasses { new NoOtherwiseOverlappedWhenTester }
  }
  "Conditional connections to submodule ports" should "be handled properly" in {
    assertTesterPasses(new SubmoduleWhenTester)
  }
  "when.cond" should "give the current when condition" in {
    assertTesterPasses(new WhenCondTester)
  }

  "Returning in a when scope" should "give a reasonable error message" in {
    val e = the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {
          val foo = Input(UInt(8.W))
          val bar = Input(UInt(8.W))
          val cond = Input(Bool())
          val out = Output(UInt(8.W))
        })
        def func(): UInt = {
          when(io.cond) {
            // This is bad, do not do this!!!
            return io.foo
          }
          return io.bar
        }
        io.out := func()
      })
    }
    e.getMessage should include("Cannot exit from a when() block with a \"return\"")
  }
}
