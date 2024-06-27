// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.testers.BasicTester
import chisel3.experimental.SourceLine
import _root_.circt.stage.ChiselStage
import chiselTests.ChiselRunners

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class LTLSpec extends AnyFlatSpec with Matchers with ChiselRunners {
  it should "allow booleans to be used as sequences" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      Sequence.delay(a, 42)
    })
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("intrinsic(circt_ltl_delay<delay = 42, length = 0> : UInt<1>, a)")
  }

  it should "allow booleans to be used as properties" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      Property.eventually(a)
    })
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("intrinsic(circt_ltl_eventually : UInt<1>, a)")
  }

  class DelaysMod extends RawModule {
    val a, b, c = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val s0: Sequence = a.delay(1)
    val s1: Sequence = b.delayRange(2, 4)
    val s2: Sequence = c.delayAtLeast(5)
    val s3: Sequence = a ### b
    val s4: Sequence = a ##* b
    val s5: Sequence = a ##+ b
  }
  it should "support sequence delay operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new DelaysMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include(f"node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc")
    chirrtl should include(f"node delay_1 = intrinsic(circt_ltl_delay<delay = 2, length = 2> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node delay_2 = intrinsic(circt_ltl_delay<delay = 5> : UInt<1>, c) $sourceLoc")
    chirrtl should include(f"node delay_3 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node concat = intrinsic(circt_ltl_concat : UInt<1>, a, delay_3) $sourceLoc")
    chirrtl should include(f"node delay_4 = intrinsic(circt_ltl_delay<delay = 0> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_4) $sourceLoc")
    chirrtl should include(f"node delay_5 = intrinsic(circt_ltl_delay<delay = 1> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_5) $sourceLoc")
  }
  it should "compile sequence delay operations" in {
    ChiselStage.emitSystemVerilog(new DelaysMod)
  }

  class ConcatMod extends RawModule {
    val a, b, c, d, e = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val s0: Sequence = a.concat(b)
    val s1: Sequence = Sequence.concat(c, d, e) // (c concat d) concat e
  }
  it should "support sequence concat operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ConcatMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include("input d : UInt<1>")
    chirrtl should include("input e : UInt<1>")
    chirrtl should include(f"intrinsic(circt_ltl_concat : UInt<1>, a, b) $sourceLoc")
    chirrtl should include(f"node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, c, d) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_ltl_concat : UInt<1>, concat_1, e) $sourceLoc")
  }
  it should "compile sequence concat operations" in {
    ChiselStage.emitSystemVerilog(new ConcatMod)
  }

  class RepeatMod extends RawModule {
    val a, b, c, d, e = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val s0: Sequence = a.repeat(1)
    val s1: Sequence = b.repeatRange(2, 4)
    val s2: Sequence = c.repeatAtLeast(5)
    val s3: Sequence = d.gotoRepeat(1, 3)
    val s4: Sequence = e.nonConsecutiveRepeat(1, 3)
  }
  it should "support sequence repeat operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RepeatMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include("input d : UInt<1>")
    chirrtl should include("input e : UInt<1>")
    chirrtl should include(f"node repeat = intrinsic(circt_ltl_repeat<base = 1, more = 0> : UInt<1>, a) $sourceLoc")
    chirrtl should include(f"node repeat_1 = intrinsic(circt_ltl_repeat<base = 2, more = 2> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node repeat_2 = intrinsic(circt_ltl_repeat<base = 5> : UInt<1>, c) $sourceLoc")
    chirrtl should include(
      f"node goto_repeat = intrinsic(circt_ltl_goto_repeat<base = 1, more = 2> : UInt<1>, d) $sourceLoc"
    )
    chirrtl should include(
      f"node non_consecutive_repeat = intrinsic(circt_ltl_non_consecutive_repeat<base = 1, more = 2> : UInt<1>, e) $sourceLoc"
    )
  }
  it should "compile sequence repeat operations" in {
    ChiselStage.emitSystemVerilog(new RepeatMod)
  }

  class AndOrClockMod extends RawModule {
    val a, b = IO(Input(Bool()))
    val clock = IO(Input(Clock()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val s0: Sequence = a.delay()
    val s1: Sequence = s0.and(b)
    val s2: Sequence = s0.or(b)
    val si: Sequence = s0.intersect(b)
    val sn: Sequence = Sequence.intersect(si, s1, s2)
    val s3: Sequence = s0.clock(clock)
    val p0: Property = a.eventually
    val p1: Property = p0.and(b)
    val p2: Property = p0.or(b)
    val pi: Property = p0.intersect(b)
    val pn: Property = Property.intersect(pi, p1, p2)
    val p3: Property = p0.clock(clock)
    val u1: Sequence = s0.until(b)
    val u2: Property = p0.until(b)
  }
  it should "support and, or, intersect, and clock operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new AndOrClockMod)
    val sourceLoc = "@[Foo.scala 1:2]"

    // Sequences
    chirrtl should include(f"node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc")
    chirrtl should include(f"node and = intrinsic(circt_ltl_and : UInt<1>, delay, b) $sourceLoc")
    chirrtl should include(f"node or = intrinsic(circt_ltl_or : UInt<1>, delay, b) $sourceLoc")
    chirrtl should include(f"node intersect = intrinsic(circt_ltl_intersect : UInt<1>, delay, b) $sourceLoc")
    chirrtl should include(f"node intersect_1 = intrinsic(circt_ltl_intersect : UInt<1>, intersect, and) $sourceLoc")
    chirrtl should include(f"node intersect_2 = intrinsic(circt_ltl_intersect : UInt<1>, intersect_1, or) $sourceLoc")
    chirrtl should include(f"node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, delay, clock) $sourceLoc")

    // Properties
    chirrtl should include(f"node eventually = intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc")
    chirrtl should include(f"node and_1 = intrinsic(circt_ltl_and : UInt<1>, eventually, b) $sourceLoc")
    chirrtl should include(f"node or_1 = intrinsic(circt_ltl_or : UInt<1>, eventually, b) $sourceLoc")
    chirrtl should include(f"node intersect_3 = intrinsic(circt_ltl_intersect : UInt<1>, eventually, b) $sourceLoc")
    chirrtl should include(
      f"node intersect_4 = intrinsic(circt_ltl_intersect : UInt<1>, intersect_3, and_1) $sourceLoc"
    )
    chirrtl should include(f"node intersect_5 = intrinsic(circt_ltl_intersect : UInt<1>, intersect_4, or_1) $sourceLoc")
    chirrtl should include(f"node clock_2 = intrinsic(circt_ltl_clock : UInt<1>, eventually, clock) $sourceLoc")

    // Until
    chirrtl should include(f"node until = intrinsic(circt_ltl_until : UInt<1>, delay, b) $sourceLoc")
    chirrtl should include(f"node until_1 = intrinsic(circt_ltl_until : UInt<1>, eventually, b) $sourceLoc")
  }
  it should "compile and, or, intersect, and clock operations" in {
    ChiselStage.emitSystemVerilog(new AndOrClockMod)
  }

  class NotMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val p0: Property = Property.not(a)
  }
  it should "support property not operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new NotMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include(f"intrinsic(circt_ltl_not : UInt<1>, a) $sourceLoc")
  }
  it should "compile property not operation" in {
    ChiselStage.emitSystemVerilog(new NotMod)
  }

  class PropImplicationMod extends RawModule {
    val a, b = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val p0: Property = Property.implication(a, b)
    val p1: Property = a |-> b
    val p2: Property = Property.implicationNonOverlapping(a, b)
    val p3: Property = a |=> b
  }
  it should "support property implication operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new PropImplicationMod)
    val sourceLoc = "@[Foo.scala 1:2]"

    // Overlapping
    chirrtl should include(f"intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc")

    // Non-overlapping (emitted as `a ## true |-> b`)
    chirrtl should include(
      f"node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc"
    )
    chirrtl should include(f"node concat = intrinsic(circt_ltl_concat : UInt<1>, a, delay) $sourceLoc")
    chirrtl should include(f"node implication_2 = intrinsic(circt_ltl_implication : UInt<1>, concat, b) $sourceLoc")
    chirrtl should include(
      f"node delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc"
    )
    chirrtl should include(f"node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_1) $sourceLoc")
    chirrtl should include(f"node implication_3 = intrinsic(circt_ltl_implication : UInt<1>, concat_1, b) $sourceLoc")
  }
  it should "compile property implication operation" in {
    ChiselStage.emitSystemVerilog(new PropImplicationMod)
  }

  class EventuallyMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val p0: Property = a.eventually
  }
  it should "support property eventually operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new EventuallyMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include(f"intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc")
  }
  it should "compile property eventually operation" in {
    ChiselStage.emitSystemVerilog(new EventuallyMod)
  }

  class DisableMod extends RawModule {
    val a, b = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    val p0: Property = a.disable(b.asDisable)
  }
  it should "support property disable operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new DisableMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include(f"intrinsic(circt_ltl_disable : UInt<1>, a, b) $sourceLoc")
  }
  it should "compile property disable operation" in {
    ChiselStage.emitSystemVerilog(new DisableMod)
  }

  class BasicVerifMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info = SourceLine("Foo.scala", 1, 2)
    AssertProperty(a)
    AssumeProperty(a)
    CoverProperty(a)
  }
  it should "support simple property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new BasicVerifMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include(f"intrinsic(circt_verif_assert, a) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_verif_assume, a) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_verif_cover, a) $sourceLoc")
  }
  it should "compile simple property asserts/assumes/covers" in {
    ChiselStage.emitSystemVerilog(new BasicVerifMod)
  }

  it should "use clock and disable by default for properties" in {

    val properties = Seq(
      AssertProperty -> ("VerifAssertIntrinsic", "assert"),
      AssumeProperty -> ("VerifAssumeIntrinsic", "assume"),
      CoverProperty -> ("VerifCoverIntrinsic", "cover")
    )

    for ((prop, (intrinsic, op)) <- properties) {
      val chirrtl = ChiselStage.emitCHIRRTL(new Module {
        val a = IO(Input(Bool()))
        implicit val info = SourceLine("Foo.scala", 1, 2)
        prop(a)
      })
      val sourceLoc = "@[Foo.scala 1:2]"
      chirrtl should include("node has_been_reset = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)")
      chirrtl should include("node disable = eq(has_been_reset, UInt<1>(0h0))")
      chirrtl should include(f"node disable_1 = intrinsic(circt_ltl_disable : UInt<1>, a, disable) $sourceLoc")
      chirrtl should include(f"node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, disable_1, clock) $sourceLoc")
      chirrtl should include(f"intrinsic(circt_verif_$op, clock_1) $sourceLoc")
    }
  }

  class LabeledVerifMod extends RawModule {
    val a = IO(Input(Bool()))
    AssertProperty(a, label = Some("foo0"))
    AssumeProperty(a, label = Some("foo1"))
    CoverProperty(a, label = Some("foo2"))
  }
  it should "support labeled property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new LabeledVerifMod)
    chirrtl should include("intrinsic(circt_verif_assert<label = \"foo0\">, a)")
    chirrtl should include("intrinsic(circt_verif_assume<label = \"foo1\">, a")
    chirrtl should include("intrinsic(circt_verif_cover<label = \"foo2\">, a)")
  }
  it should "compile labeled property asserts/assumes/covers" in {
    ChiselStage.emitSystemVerilog(new LabeledVerifMod)
  }

  it should "support assert shorthands with clock and disable" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      val c = IO(Input(Clock()))
      AssertProperty(a, clock = Some(c))
      AssertProperty(a, disable = Some(b.asDisable))
      AssertProperty(a, clock = Some(c), disable = Some(b.asDisable))
    })

    // with clock; emitted as `assert(clock(a, c))`
    chirrtl should include("node clock = intrinsic(circt_ltl_clock : UInt<1>, a, c)")
    chirrtl should include("intrinsic(circt_verif_assert, clock)")

    // with disable; emitted as `assert(disable(a, b))`
    chirrtl should include("node disable = intrinsic(circt_ltl_disable : UInt<1>, a, b)")
    chirrtl should include("intrinsic(circt_verif_assert, disable)")

    // with clock and disable; emitted as `assert(clock(disable(a, b), c))`
    chirrtl should include("node disable_1 = intrinsic(circt_ltl_disable : UInt<1>, a, b)")
    chirrtl should include("node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, disable_1, c)")
    chirrtl should include("intrinsic(circt_verif_assert, clock_1)")
  }

  class SequenceConvMod extends RawModule {
    val a, b = IO(Input(Bool()))
    AssertProperty(Sequence(a))
    AssertProperty(Sequence(a, b))
    AssertProperty(Sequence(Delay(), a))
    AssertProperty(Sequence(a, Delay(), b))
    AssertProperty(Sequence(a, Delay(2), b))
    AssertProperty(Sequence(a, Delay(42, 1337), b))
    AssertProperty(Sequence(a, Delay(9001, None), b))
  }
  it should "support Sequence(...) convenience constructor" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new SequenceConvMod)
    // a
    chirrtl should include("intrinsic(circt_verif_assert, a)")

    // a b
    chirrtl should include("node concat = intrinsic(circt_ltl_concat : UInt<1>, a, b)")
    chirrtl should include("intrinsic(circt_verif_assert, concat)")

    // Delay() a
    chirrtl should include("node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)")
    chirrtl should include("intrinsic(circt_verif_assert, delay)")

    // a Delay() b
    chirrtl should include("node delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b)")
    chirrtl should include("node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_1)")
    chirrtl should include("intrinsic(circt_verif_assert, concat_1)")

    // a Delay(2) b
    chirrtl should include("node delay_2 = intrinsic(circt_ltl_delay<delay = 2, length = 0> : UInt<1>, b)")
    chirrtl should include("node concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_2)")
    chirrtl should include("intrinsic(circt_verif_assert, concat_2)")

    // a Delay(42, 1337) b
    chirrtl should include("node delay_3 = intrinsic(circt_ltl_delay<delay = 42, length = 1295> : UInt<1>, b)")
    chirrtl should include("node concat_3 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_3)")
    chirrtl should include("intrinsic(circt_verif_assert, concat_3)")

    // a Delay(9001, None) b
    chirrtl should include("node delay_4 = intrinsic(circt_ltl_delay<delay = 9001> : UInt<1>, b)")
    chirrtl should include("node concat_4 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_4)")
    chirrtl should include("intrinsic(circt_verif_assert, concat_4)")
  }
  it should "compile Sequence(...) convenience constructor" in {
    ChiselStage.emitSystemVerilog(new SequenceConvMod)
  }

  it should "fail correctly in verilator simulation" in {
    assertTesterFails(new BasicTester {
      withClockAndReset(clock, reset) {
        AssertProperty(0.U === 1.U)
      }
    })
  }
}
