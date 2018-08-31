// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class WhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.B) { cnt.inc() }

  val out = Wire(UInt(3.W))
  when(cnt.value === 0.U) {
    out := 1.U
  } .elsewhen (cnt.value === 1.U) {
    out := 2.U
  } .elsewhen (cnt.value === 2.U) {
    out := 3.U
  } .otherwise {
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
  } .elsewhen (cnt.value <= 1.U) {
    out := 2.U
  } .elsewhen (cnt.value <= 2.U) {
    out := 3.U
  } .otherwise {
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
  } .elsewhen (cnt.value <= 1.U) {
    out := 2.U
  } .elsewhen (cnt.value <= 2.U) {
    out := 3.U
  } .elsewhen (cnt.value <= 3.U) {
    out := 0.U
  } .otherwise {
    out := DontCare
  }

  assert(out === cnt.value + 1.U)

  when(cnt.value === 3.U) {
    stop()
  }
}

class SubmoduleWhenTester extends BasicTester {
  val (cycle, done) = Counter(true.B, 3)
  when (done) { stop() }
  val children = Seq(Module(new PassthroughModule),
                     Module(new PassthroughMultiIOModule),
                     Module(new PassthroughRawModule))
  children.foreach { child =>
    when (cycle === 1.U) {
      child.io.in := "hdeadbeef".U
      assert(child.io.out === "hdeadbeef".U)
    } .otherwise {
      child.io.in := "h0badcad0".U
      assert(child.io.out === "h0badcad0".U)
    }
  }
}

class WhenSpec extends ChiselFlatSpec {
  "When, elsewhen, and otherwise with orthogonal conditions" should "work" in {
    assertTesterPasses{ new WhenTester }
  }
  "When, elsewhen, and otherwise with overlapped conditions" should "work" in {
    assertTesterPasses{ new OverlappedWhenTester }
  }
  "When and elsewhen without otherwise with overlapped conditions" should "work" in {
    assertTesterPasses{ new NoOtherwiseOverlappedWhenTester }
  }
  "Conditional connections to submodule ports" should "be handled properly" in {
    assertTesterPasses(new SubmoduleWhenTester)
  }

  "Returning in a when scope" should "give a reasonable error message" in {
    val e = the [ChiselException] thrownBy {
      elaborate(new Module {
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
    e.getMessage should include ("Cannot exit from a when() block with a \"return\"")
  }
}
