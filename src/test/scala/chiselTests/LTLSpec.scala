// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.experimental.{SourceInfo, SourceLine}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class LTLSpec extends AnyFlatSpec with Matchers with ChiselSim {
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
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val s0:            Sequence = a.delay(1)
    val s1:            Sequence = b.delayRange(2, 4)
    val s2:            Sequence = c.delayAtLeast(5)
    val s3:            Sequence = a ### b
    val s4:            Sequence = a ##* b
    val s5:            Sequence = a ##+ b
  }
  it should "support sequence delay operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new DelaysMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include(
      f"node ltl_delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_delay_1 = intrinsic(circt_ltl_delay<delay = 2, length = 2> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(f"node ltl_delay_2 = intrinsic(circt_ltl_delay<delay = 5> : UInt<1>, c) $sourceLoc")
    chirrtl should include(
      f"node ltl_delay_3 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(f"node ltl_concat = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_3) $sourceLoc")
    chirrtl should include(f"node ltl_delay_4 = intrinsic(circt_ltl_delay<delay = 0> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node ltl_concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_4) $sourceLoc")
    chirrtl should include(f"node ltl_delay_5 = intrinsic(circt_ltl_delay<delay = 1> : UInt<1>, b) $sourceLoc")
    chirrtl should include(f"node ltl_concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_5) $sourceLoc")
  }
  it should "compile sequence delay operations" in {
    ChiselStage.emitSystemVerilog(new DelaysMod)
  }

  class ConcatMod extends RawModule {
    val a, b, c, d, e = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val s0:            Sequence = a.concat(b)
    val s1:            Sequence = Sequence.concat(c, d, e) // (c concat d) concat e
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
    chirrtl should include(f"node ltl_concat_1 = intrinsic(circt_ltl_concat : UInt<1>, c, d) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_ltl_concat : UInt<1>, ltl_concat_1, e) $sourceLoc")
  }
  it should "compile sequence concat operations" in {
    ChiselStage.emitSystemVerilog(new ConcatMod)
  }

  class RepeatMod extends RawModule {
    val a, b, c, d, e = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val s0:            Sequence = a.repeat(1)
    val s1:            Sequence = b.repeatRange(2, 4)
    val s2:            Sequence = c.repeatAtLeast(5)
    val s3:            Sequence = d.gotoRepeat(1, 3)
    val s4:            Sequence = e.nonConsecutiveRepeat(1, 3)
  }
  it should "support sequence repeat operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RepeatMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include("input d : UInt<1>")
    chirrtl should include("input e : UInt<1>")
    chirrtl should include(f"node ltl_repeat = intrinsic(circt_ltl_repeat<base = 1, more = 0> : UInt<1>, a) $sourceLoc")
    chirrtl should include(
      f"node ltl_repeat_1 = intrinsic(circt_ltl_repeat<base = 2, more = 2> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(f"node ltl_repeat_2 = intrinsic(circt_ltl_repeat<base = 5> : UInt<1>, c) $sourceLoc")
    chirrtl should include(
      f"node ltl_goto_repeat = intrinsic(circt_ltl_goto_repeat<base = 1, more = 2> : UInt<1>, d) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_non_consecutive_repeat = intrinsic(circt_ltl_non_consecutive_repeat<base = 1, more = 2> : UInt<1>, e) $sourceLoc"
    )
  }
  it should "compile sequence repeat operations" in {
    ChiselStage.emitSystemVerilog(new RepeatMod)
  }

  class AndOrClockMod extends RawModule {
    val a, b = IO(Input(Bool()))
    val clock = IO(Input(Clock()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val s0:            Sequence = a.delay()
    val s1:            Sequence = s0.and(b)
    val s2:            Sequence = s0.or(b)
    val si:            Sequence = s0.intersect(b)
    val sn:            Sequence = Sequence.intersect(si, s1, s2)
    val s3:            Sequence = s0.clock(clock)
    val p0:            Property = a.eventually
    val p1:            Property = p0.and(b)
    val p2:            Property = p0.or(b)
    val pi:            Property = p0.intersect(b)
    val pn:            Property = Property.intersect(pi, p1, p2)
    val p3:            Property = p0.clock(clock)
    val u1:            Sequence = s0.until(b)
    val u2:            Property = p0.until(b)
  }
  it should "support and, or, intersect, and clock operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new AndOrClockMod)
    val sourceLoc = "@[Foo.scala 1:2]"

    // Sequences
    chirrtl should include(
      f"node ltl_delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc"
    )
    chirrtl should include(f"node ltl_and = intrinsic(circt_ltl_and : UInt<1>, ltl_delay, b) $sourceLoc")
    chirrtl should include(f"node ltl_or = intrinsic(circt_ltl_or : UInt<1>, ltl_delay, b) $sourceLoc")
    chirrtl should include(f"node ltl_intersect = intrinsic(circt_ltl_intersect : UInt<1>, ltl_delay, b) $sourceLoc")
    chirrtl should include(
      f"node ltl_intersect_1 = intrinsic(circt_ltl_intersect : UInt<1>, ltl_intersect, ltl_and) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_intersect_2 = intrinsic(circt_ltl_intersect : UInt<1>, ltl_intersect_1, ltl_or) $sourceLoc"
    )
    chirrtl should include(f"node ltl_clock = intrinsic(circt_ltl_clock : UInt<1>, ltl_delay, clock) $sourceLoc")

    // Properties
    chirrtl should include(f"node ltl_eventually = intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc")
    chirrtl should include(f"node ltl_and_1 = intrinsic(circt_ltl_and : UInt<1>, ltl_eventually, b) $sourceLoc")
    chirrtl should include(f"node ltl_or_1 = intrinsic(circt_ltl_or : UInt<1>, ltl_eventually, b) $sourceLoc")
    chirrtl should include(
      f"node ltl_intersect_3 = intrinsic(circt_ltl_intersect : UInt<1>, ltl_eventually, b) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_intersect_4 = intrinsic(circt_ltl_intersect : UInt<1>, ltl_intersect_3, ltl_and_1) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_intersect_5 = intrinsic(circt_ltl_intersect : UInt<1>, ltl_intersect_4, ltl_or_1) $sourceLoc"
    )
    chirrtl should include(f"node ltl_clock_1 = intrinsic(circt_ltl_clock : UInt<1>, ltl_eventually, clock) $sourceLoc")

    // Until
    chirrtl should include(f"node ltl_until = intrinsic(circt_ltl_until : UInt<1>, ltl_delay, b) $sourceLoc")
    chirrtl should include(f"node ltl_until_1 = intrinsic(circt_ltl_until : UInt<1>, ltl_eventually, b) $sourceLoc")
  }
  it should "compile and, or, intersect, and clock operations" in {
    ChiselStage.emitSystemVerilog(new AndOrClockMod)
  }

  class NotMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val p0:            Property = Property.not(a)
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
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val p0:            Property = Property.implication(a, b)
    val p1:            Property = a |-> b
    val p2:            Property = Property.implicationNonOverlapping(a, b)
    val p3:            Property = a |=> b
  }
  it should "support property implication operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new PropImplicationMod)
    val sourceLoc = "@[Foo.scala 1:2]"

    // Overlapping
    chirrtl should include(f"intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc")
    chirrtl should include(f"intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc")

    // Non-overlapping (emitted as `a ## true |-> b`)
    chirrtl should include(
      f"node ltl_delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc"
    )
    chirrtl should include(f"node ltl_concat = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay) $sourceLoc")
    chirrtl should include(
      f"node ltl_implication_2 = intrinsic(circt_ltl_implication : UInt<1>, ltl_concat, b) $sourceLoc"
    )
    chirrtl should include(
      f"node ltl_delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc"
    )
    chirrtl should include(f"node ltl_concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_1) $sourceLoc")
    chirrtl should include(
      f"node ltl_implication_3 = intrinsic(circt_ltl_implication : UInt<1>, ltl_concat_1, b) $sourceLoc"
    )
  }
  it should "compile property implication operation" in {
    ChiselStage.emitSystemVerilog(new PropImplicationMod)
  }

  class EventuallyMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val p0:            Property = a.eventually
  }
  it should "support property eventually operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new EventuallyMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include(f"intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc")
  }
  it should "compile property eventually operation" in {
    ChiselStage.emitSystemVerilog(new EventuallyMod)
  }

  class BasicVerifMod extends RawModule {
    val a = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    AssertProperty(a)
    AssumeProperty(a)
    CoverProperty(a)
    RequireProperty(a)
    EnsureProperty(a)
  }
  it should "support simple property asserts/assumes/covers and put them in layer blocks" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new BasicVerifMod)
    val sourceLoc = "@[Foo.scala 1:2]"
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Assert")
    chirrtl should include(f"intrinsic(circt_verif_assert, a) $sourceLoc")
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Assume")
    chirrtl should include(f"intrinsic(circt_verif_assume, a) $sourceLoc")
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Cover")
    chirrtl should include(f"intrinsic(circt_verif_cover, a) $sourceLoc")
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Assume")
    chirrtl should include(f"intrinsic(circt_verif_require, a) $sourceLoc")
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Assert")
    chirrtl should include(f"intrinsic(circt_verif_ensure, a) $sourceLoc")
  }
  it should "compile simple property checks" in {
    ChiselStage.emitSystemVerilog(new BasicVerifMod)
  }
  it should "not create layer blocks if already in a layer block" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      layer.block(chisel3.layers.Verification.Cover) {
        AssertProperty(a)
      }
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Foo)
    chirrtl should include("layerblock Verification")
    chirrtl should include("layerblock Cover")
    (chirrtl should not).include("layerblock Assert")
  }

  it should "use clock and disable by default for properties" in {

    val properties = Seq(
      AssertProperty -> ("VerifAssertIntrinsic", "assert"),
      AssumeProperty -> ("VerifAssumeIntrinsic", "assume"),
      CoverProperty -> ("VerifCoverIntrinsic", "cover"),
      RequireProperty -> ("VerifRequireIntrinsic", "require"),
      EnsureProperty -> ("VerifEnsureIntrinsic", "ensure")
    )

    for ((prop, (intrinsic, op)) <- properties) {
      val chirrtl = ChiselStage.emitCHIRRTL(new Module {
        val a = IO(Input(Bool()))
        implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
        prop(a)
      })
      val sourceLoc = "@[Foo.scala 1:2]"
      chirrtl should include("node has_been_reset = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)")
      chirrtl should include("node disable = eq(has_been_reset, UInt<1>(0h0))")
      chirrtl should include(f"node ltl_clock = intrinsic(circt_ltl_clock : UInt<1>, a, clock) $sourceLoc")
      chirrtl should include(f"node _T = eq(disable, UInt<1>(0h0)) $sourceLoc")
      chirrtl should include(f"intrinsic(circt_verif_$op, ltl_clock, _T) $sourceLoc")
    }
  }

  class LabeledVerifMod extends RawModule {
    val a = IO(Input(Bool()))
    AssertProperty(a, label = Some("foo0"))
    AssumeProperty(a, label = Some("foo1"))
    CoverProperty(a, label = Some("foo2"))
    RequireProperty(a, label = Some("foo3"))
    EnsureProperty(a, label = Some("foo4"))
  }
  it should "support labeled property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new LabeledVerifMod)
    chirrtl should include("intrinsic(circt_verif_assert<label = \"foo0\">, a)")
    chirrtl should include("intrinsic(circt_verif_assume<label = \"foo1\">, a")
    chirrtl should include("intrinsic(circt_verif_cover<label = \"foo2\">, a)")
    chirrtl should include("intrinsic(circt_verif_require<label = \"foo3\">, a)")
    chirrtl should include("intrinsic(circt_verif_ensure<label = \"foo4\">, a)")
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
    chirrtl should include("node ltl_clock = intrinsic(circt_ltl_clock : UInt<1>, a, c)")
    chirrtl should include("intrinsic(circt_verif_assert, ltl_clock)")

    // with disable; emitted as `assert(a, disable)`
    chirrtl should include("node _T = eq(b, UInt<1>(0h0))")
    chirrtl should include("intrinsic(circt_verif_assert, a, _T)")

    // with clock and disable; emitted as `assert(clock(disable(a, b), c))`
    chirrtl should include("node ltl_clock_1 = intrinsic(circt_ltl_clock : UInt<1>, a, c)")
    chirrtl should include("node _T_1 = eq(b, UInt<1>(0h0))")
    chirrtl should include("intrinsic(circt_verif_assert, ltl_clock_1, _T_1)")
  }

  class SequenceConvMod extends RawModule {
    val a, b = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
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
    val sourceLoc = "@[Foo.scala 1:2]"

    // a
    chirrtl should include(s"intrinsic(circt_verif_assert, a) $sourceLoc")

    // a b
    chirrtl should include(s"node ltl_concat = intrinsic(circt_ltl_concat : UInt<1>, a, b) $sourceLoc")
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_concat) $sourceLoc")

    // Delay() a
    chirrtl should include(
      s"node ltl_delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc"
    )
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_delay) $sourceLoc")

    // a Delay() b
    chirrtl should include(
      s"node ltl_delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(s"node ltl_concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_1) $sourceLoc")
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_concat_1) $sourceLoc")

    // a Delay(2) b
    chirrtl should include(
      s"node ltl_delay_2 = intrinsic(circt_ltl_delay<delay = 2, length = 0> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(s"node ltl_concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_2) $sourceLoc")
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_concat_2) $sourceLoc")

    // a Delay(42, 1337) b
    chirrtl should include(
      s"node ltl_delay_3 = intrinsic(circt_ltl_delay<delay = 42, length = 1295> : UInt<1>, b) $sourceLoc"
    )
    chirrtl should include(s"node ltl_concat_3 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_3) $sourceLoc")
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_concat_3) $sourceLoc")

    // a Delay(9001, None) sb
    chirrtl should include(s"node ltl_delay_4 = intrinsic(circt_ltl_delay<delay = 9001> : UInt<1>, b) $sourceLoc")
    chirrtl should include(s"node ltl_concat_4 = intrinsic(circt_ltl_concat : UInt<1>, a, ltl_delay_4) $sourceLoc")
    chirrtl should include(s"intrinsic(circt_verif_assert, ltl_concat_4) $sourceLoc")
  }
  it should "compile Sequence(...) convenience constructor" in {
    ChiselStage.emitSystemVerilog(new SequenceConvMod)
  }

  it should "fail correctly in verilator simulation" in {
    intercept[chisel3.simulator.Exceptions.AssertionFailed] {
      simulate(new Module {
        withClockAndReset(clock, reset) {
          AssertProperty(0.U === 1.U)
        }
      })(RunUntilFinished(3))
    }
  }

  class LayerBlockMod extends RawModule {
    val a, b = IO(Input(Bool()))
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    AssertProperty(Sequence(Delay(), a))
    AssumeProperty(a |-> b)
  }

  it should "wrap all intrinsics in layerblocks" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new LayerBlockMod)
    val sourceLoc = "@[Foo.scala 1:2]"

    val assertBlockLoc = chirrtl.indexOf(s"layerblock Assert : $sourceLoc")
    val delayIntrinsicLoc = chirrtl.indexOf(
      s"intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)"
    )
    val assumeblockLoc = chirrtl.indexOf(s"layerblock Assume : $sourceLoc")
    val implicationIntrinsicLoc = chirrtl.indexOf(
      s"intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc"
    )

    assert(assertBlockLoc < delayIntrinsicLoc)
    assert(assumeblockLoc < implicationIntrinsicLoc)
  }
}
