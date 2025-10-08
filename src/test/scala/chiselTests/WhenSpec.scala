// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import chisel3.util.Counter
import chisel3.experimental.{SourceInfo, SourceLine}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WhenTester() extends Module {
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

class OverlappedWhenTester() extends Module {
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

class NoOtherwiseOverlappedWhenTester() extends Module {
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

class SubmoduleWhenTester extends Module {
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

class WhenCondTester extends Module {
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

class WhenSpec extends AnyFlatSpec with Matchers with ChiselSim with FileCheck {
  "When, elsewhen, and otherwise with orthogonal conditions" should "work" in {
    simulate(new WhenTester)(RunUntilFinished(5))
  }
  "When, elsewhen, and otherwise with overlapped conditions" should "work" in {
    simulate(new OverlappedWhenTester)(RunUntilFinished(5))
  }
  "When and elsewhen without otherwise with overlapped conditions" should "work" in {
    simulate(new NoOtherwiseOverlappedWhenTester)(RunUntilFinished(5))
  }
  "Conditional connections to submodule ports" should "be handled properly" in {
    simulate(new SubmoduleWhenTester)(RunUntilFinished(4))
  }
  "when.cond" should "give the current when condition" in {
    simulate(new WhenCondTester)(RunUntilFinished(math.pow(2, 4).toInt + 1))
  }

  "Returning in a when scope" should "give a reasonable error message" in {
    val e = intercept[ChiselException] {
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

  "Using a value that has escaped from a when scope in a connection" should "give a reasonable error message" in {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 12, 3)
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(
        new Module {
          override def desiredName = "Top"
          val foo, bar = IO(Output(UInt(8.W)))
          val a = IO(Input(Bool()))
          lazy val w = Wire(UInt(8.W))
          when(a) {
            foo := w
          }
          bar := w
        },
        args = Array("--throw-on-first-error")
      )
    }
    val msg =
      "'Top.foo_w: Wire[UInt<8>]' has escaped the scope of the block (@[Foo.scala:12:3]) in which it was constructed."
    e.getMessage should include(msg)
  }

  "Using a value that has escaped from a when scope in an operation" should "give a reasonable error message" in {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 12, 3)
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(
        new Module {
          override def desiredName = "Top"
          val foo, bar = IO(Output(UInt(8.W)))
          val a = IO(Input(Bool()))
          lazy val w = Wire(UInt(8.W))
          when(a) {
            foo := w
          }
          bar := w + 1.U
        },
        args = Array("--throw-on-first-error")
      )
    }
    val msg =
      "'Top.foo_w: Wire[UInt<8>]' has escaped the scope of the block (@[Foo.scala:12:3]) in which it was constructed."
    e.getMessage should include(msg)
  }

  "Whens with empty clauses" should "emit an indented skip" in {
    class Top extends Module {
      val cond = IO(Input(Bool()))
      val out = IO(Output(UInt(8.W)))
      when(cond) {}
      out := 1.U
    }
    // Strict check so we can match the exact location of the skip
    ChiselStage
      .emitCHIRRTL(new Top)
      .fileCheck("--strict-whitespace", "--match-full-lines")(
        """|     CHECK:    when cond : @{{.*}}
           |CHECK-NEXT:      skip
           |CHECK-NEXT:    connect out, UInt<1>(0h1) @{{.*}}
           |""".stripMargin
      )
  }

  "Whens with empty else clauses" should "not emit the else clause" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val cond = IO(Input(Bool()))
      val out = IO(Output(UInt(8.W)))
      when(cond) {
        out := 1.U
      }
    })
    chirrtl should include("when")
    chirrtl shouldNot include("else")
    chirrtl shouldNot include("skip")
  }

  "Sibling when blocks" should "emit error for visibility check" in {
    class Foo extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      var c: Bool = null
      when(true.B) {
        c = WireInit(Bool(), true.B)
        out := c
      }.otherwise {
        c := false.B
        out := c
      }
    }
    val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new Foo, args = Array("--throw-on-first-error"))
    e.getMessage should include("has escaped the scope of the block")
  }
}
