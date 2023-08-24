// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.testers.BasicTester
import _root_.circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class LTLSpec extends AnyFlatSpec with Matchers {
  it should "allow booleans to be used as sequences" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      Sequence.delay(a, 42)
    })
    chirrtl should include("intmodule LTLDelayIntrinsic_42_0 :")
    chirrtl should include("input in : UInt<1>")
    chirrtl should include("output out : UInt<1>")
    chirrtl should include("intrinsic = circt_ltl_delay")
    chirrtl should include("parameter delay = 42")
    chirrtl should include("parameter length = 0")

    chirrtl should include("inst ltl_delay of LTLDelayIntrinsic_42_0")
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("connect ltl_delay.in, a")
  }

  it should "allow booleans to be used as properties" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      Property.eventually(a)
    })
    chirrtl should include("intmodule LTLEventuallyIntrinsic :")
    chirrtl should include("input in : UInt<1>")
    chirrtl should include("output out : UInt<1>")
    chirrtl should include("intrinsic = circt_ltl_eventually")

    chirrtl should include("inst ltl_eventually of LTLEventuallyIntrinsic")
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("connect ltl_eventually.in, a")
  }

  it should "support sequence delay operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b, c = IO(Input(Bool()))
      val s0: Sequence = a.delay(1)
      val s1: Sequence = b.delayRange(2, 4)
      val s2: Sequence = c.delayAtLeast(5)
      val s3: Sequence = a ### b
      val s4: Sequence = a ##* b
      val s5: Sequence = a ##+ b
    })
    chirrtl should include("inst ltl_delay of LTLDelayIntrinsic_1_0")
    chirrtl should include("inst ltl_delay_1 of LTLDelayIntrinsic_2_2")
    chirrtl should include("inst ltl_delay_2 of LTLDelayIntrinsic_5")
    chirrtl should include("inst ltl_delay_3 of LTLDelayIntrinsic_1_0")
    chirrtl should include("inst ltl_delay_4 of LTLDelayIntrinsic_0")
    chirrtl should include("inst ltl_delay_5 of LTLDelayIntrinsic_1")
    chirrtl should include("connect ltl_delay.in, a")
    chirrtl should include("connect ltl_delay_1.in, b")
    chirrtl should include("connect ltl_delay_2.in, c")
  }

  it should "support sequence concat operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b, c, d, e = IO(Input(Bool()))
      val s0: Sequence = a.concat(b)
      val s1: Sequence = Sequence.concat(c, d, e) // (c concat d) concat e
    })
    chirrtl should include("inst ltl_concat of LTLConcatIntrinsic")
    chirrtl should include("connect ltl_concat.lhs, a")
    chirrtl should include("connect ltl_concat.rhs, b")
    chirrtl should include("inst ltl_concat_1 of LTLConcatIntrinsic")
    chirrtl should include("connect ltl_concat_1.lhs, c")
    chirrtl should include("connect ltl_concat_1.rhs, d")
    chirrtl should include("inst ltl_concat_2 of LTLConcatIntrinsic")
    chirrtl should include("connect ltl_concat_2.lhs, ltl_concat_1.out")
    chirrtl should include("connect ltl_concat_2.rhs, e")
  }

  it should "support and, or, and clock operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      val clock = IO(Input(Clock()))
      val s0: Sequence = a.delay()
      val s1: Sequence = s0.and(b)
      val s2: Sequence = s0.or(b)
      val s3: Sequence = s0.clock(clock)
      val p0: Property = a.eventually
      val p1: Property = p0.and(b)
      val p2: Property = p0.or(b)
      val p3: Property = p0.clock(clock)
    })

    // Sequences
    chirrtl should include("inst ltl_and of LTLAndIntrinsic")
    chirrtl should include("connect ltl_and.lhs, ltl_delay.out")
    chirrtl should include("connect ltl_and.rhs, b")

    chirrtl should include("inst ltl_or of LTLOrIntrinsic")
    chirrtl should include("connect ltl_or.lhs, ltl_delay.out")
    chirrtl should include("connect ltl_or.rhs, b")

    chirrtl should include("inst ltl_clock of LTLClockIntrinsic")
    chirrtl should include("connect ltl_clock.in, ltl_delay.out")
    chirrtl should include("connect ltl_clock.clock, clock")

    // Properties
    chirrtl should include("inst ltl_and_1 of LTLAndIntrinsic")
    chirrtl should include("connect ltl_and_1.lhs, ltl_eventually.out")
    chirrtl should include("connect ltl_and_1.rhs, b")

    chirrtl should include("inst ltl_or_1 of LTLOrIntrinsic")
    chirrtl should include("connect ltl_or_1.lhs, ltl_eventually.out")
    chirrtl should include("connect ltl_or_1.rhs, b")

    chirrtl should include("inst ltl_clock_1 of LTLClockIntrinsic")
    chirrtl should include("connect ltl_clock_1.in, ltl_eventually.out")
    chirrtl should include("connect ltl_clock_1.clock, clock")
  }

  it should "support property not operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      val p0: Property = Property.not(a)
    })
    chirrtl should include("inst ltl_not of LTLNotIntrinsic")
    chirrtl should include("connect ltl_not.in, a")
  }

  it should "support property implication operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      val p0: Property = Property.implication(a, b)
      val p1: Property = a |-> b
      val p2: Property = Property.implicationNonOverlapping(a, b)
      val p3: Property = a |=> b
    })

    // Overlapping
    chirrtl should include("inst ltl_implication of LTLImplicationIntrinsic")
    chirrtl should include("connect ltl_implication.lhs, a")
    chirrtl should include("connect ltl_implication.rhs, b")

    chirrtl should include("inst ltl_implication_1 of LTLImplicationIntrinsic")
    chirrtl should include("connect ltl_implication_1.lhs, a")
    chirrtl should include("connect ltl_implication_1.rhs, b")

    // Non-overlapping (emitted as `a ## true |-> b`)
    chirrtl should include("inst ltl_delay of LTLDelayIntrinsic_1_0")
    chirrtl should include("connect ltl_delay.in, UInt<1>(0h1)")
    chirrtl should include("inst ltl_concat of LTLConcatIntrinsic")
    chirrtl should include("connect ltl_concat.lhs, a")
    chirrtl should include("connect ltl_concat.rhs, ltl_delay.out")
    chirrtl should include("inst ltl_implication_2 of LTLImplicationIntrinsic")
    chirrtl should include("connect ltl_implication_2.lhs, ltl_concat.out")
    chirrtl should include("connect ltl_implication_2.rhs, b")

    chirrtl should include("inst ltl_delay_1 of LTLDelayIntrinsic_1_0")
    chirrtl should include("connect ltl_delay_1.in, UInt<1>(0h1)")
    chirrtl should include("inst ltl_concat_1 of LTLConcatIntrinsic")
    chirrtl should include("connect ltl_concat_1.lhs, a")
    chirrtl should include("connect ltl_concat_1.rhs, ltl_delay_1.out")
    chirrtl should include("inst ltl_implication_3 of LTLImplicationIntrinsic")
    chirrtl should include("connect ltl_implication_3.lhs, ltl_concat_1.out")
    chirrtl should include("connect ltl_implication_3.rhs, b")
  }

  it should "support property eventually operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      val p0: Property = a.eventually
    })
    chirrtl should include("inst ltl_eventually of LTLEventuallyIntrinsic")
    chirrtl should include("connect ltl_eventually.in, a")
  }

  it should "support property disable operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      val p0: Property = a.disable(b.asDisable)
    })
    chirrtl should include("node _T = bits(b, 0, 0)")
    chirrtl should include("inst ltl_disable of LTLDisableIntrinsic")
    chirrtl should include("connect ltl_disable.in, a")
    chirrtl should include("connect ltl_disable.condition, _T")
  }

  it should "support simple property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      AssertProperty(a)
      AssumeProperty(a)
      CoverProperty(a)
    })
    (chirrtl should not).include("parameter label")
    chirrtl should include("inst verif of VerifAssertIntrinsic")
    chirrtl should include("inst verif_1 of VerifAssumeIntrinsic")
    chirrtl should include("inst verif_2 of VerifCoverIntrinsic")
    chirrtl should include("connect verif.property, a")
    chirrtl should include("connect verif_1.property, a")
    chirrtl should include("connect verif_2.property, a")
  }

  it should "use clock and disable by default for properties" in {

    val properties = Seq(
      AssertProperty -> "VerifAssertIntrinsic",
      AssumeProperty -> "VerifAssumeIntrinsic",
      CoverProperty -> "VerifCoverIntrinsic"
    )

    for ((prop, intrinsic) <- properties) {
      val chirrtl = ChiselStage.emitCHIRRTL(new Module {
        val a = IO(Input(Bool()))
        prop(a)
      })
      chirrtl should include("inst HasBeenResetIntrinsic of HasBeenResetIntrinsic")
      chirrtl should include("node disable = eq(HasBeenResetIntrinsic.out, UInt<1>(0h0))")
      chirrtl should include("inst ltl_disable of LTLDisableIntrinsic")
      chirrtl should include("connect ltl_disable.in, a")
      chirrtl should include("connect ltl_disable.condition, disable")
      chirrtl should include("inst ltl_clock of LTLClockIntrinsic")
      chirrtl should include("connect ltl_clock.in, ltl_disable.out")
      chirrtl should include("connect ltl_clock.clock, clock")
      chirrtl should include(s"inst verif of $intrinsic")
      chirrtl should include("connect verif.property, ltl_clock.out")
    }
  }

  it should "support labeled property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      AssertProperty(a, label = Some("foo0"))
      AssumeProperty(a, label = Some("foo1"))
      CoverProperty(a, label = Some("foo2"))
    })
    chirrtl should include("parameter label = \"foo0\"")
    chirrtl should include("parameter label = \"foo1\"")
    chirrtl should include("parameter label = \"foo2\"")
    chirrtl should include("inst verif of VerifAssertIntrinsic_foo0")
    chirrtl should include("inst verif_1 of VerifAssumeIntrinsic_foo1")
    chirrtl should include("inst verif_2 of VerifCoverIntrinsic_foo2")
    chirrtl should include("connect verif.property, a")
    chirrtl should include("connect verif_1.property, a")
    chirrtl should include("connect verif_2.property, a")
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
    chirrtl should include("inst ltl_clock of LTLClockIntrinsic")
    chirrtl should include("connect ltl_clock.in, a")
    chirrtl should include("connect ltl_clock.clock, c")
    chirrtl should include("inst verif of VerifAssertIntrinsic")
    chirrtl should include("connect verif.property, ltl_clock.out")

    // with disable; emitted as `assert(disable(a, b))`
    chirrtl should include("node x2 = bits(b, 0, 0)")
    chirrtl should include("inst ltl_disable of LTLDisableIntrinsic")
    chirrtl should include("connect ltl_disable.in, a")
    chirrtl should include("connect ltl_disable.condition, x2")
    chirrtl should include("inst verif_1 of VerifAssertIntrinsic")
    chirrtl should include("connect verif_1.property, ltl_disable.out")

    // with clock and disable; emitted as `assert(clock(disable(a, b), c))`
    chirrtl should include("node _T = bits(b, 0, 0)")
    chirrtl should include("inst ltl_disable_1 of LTLDisableIntrinsic")
    chirrtl should include("connect ltl_disable_1.in, a")
    chirrtl should include("connect ltl_disable_1.condition, _T")
    chirrtl should include("inst ltl_clock_1 of LTLClockIntrinsic")
    chirrtl should include("connect ltl_clock_1.in, ltl_disable_1.out")
    chirrtl should include("connect ltl_clock_1.clock, c")
    chirrtl should include("inst verif_2 of VerifAssertIntrinsic")
    chirrtl should include("connect verif_2.property, ltl_clock_1.out")
  }

  it should "support Sequence(...) convenience constructor" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      AssertProperty(Sequence(a))
      AssertProperty(Sequence(a, b))
      AssertProperty(Sequence(Delay(), a))
      AssertProperty(Sequence(a, Delay(), b))
      AssertProperty(Sequence(a, Delay(2), b))
      AssertProperty(Sequence(a, Delay(42, 1337), b))
      AssertProperty(Sequence(a, Delay(9001, None), b))
    })
    // a
    chirrtl should include("connect verif.property, a")

    // a b
    chirrtl should include("connect ltl_concat.lhs, a")
    chirrtl should include("connect ltl_concat.rhs, b")
    chirrtl should include("connect verif_1.property, ltl_concat.out")

    // Delay() a
    chirrtl should include("inst ltl_delay of LTLDelayIntrinsic_1_0")
    chirrtl should include("connect ltl_delay.in, a")
    chirrtl should include("connect verif_2.property, ltl_delay.out")

    // a Delay() b
    chirrtl should include("inst ltl_delay_1 of LTLDelayIntrinsic_1_0")
    chirrtl should include("connect ltl_delay_1.in, b")
    chirrtl should include("connect ltl_concat_1.lhs, a")
    chirrtl should include("connect ltl_concat_1.rhs, ltl_delay_1")
    chirrtl should include("connect verif_3.property, ltl_concat_1.out")

    // a Delay(2) b
    chirrtl should include("inst ltl_delay_2 of LTLDelayIntrinsic_2_0")
    chirrtl should include("connect ltl_delay_2.in, b")
    chirrtl should include("connect ltl_concat_2.lhs, a")
    chirrtl should include("connect ltl_concat_2.rhs, ltl_delay_2")
    chirrtl should include("connect verif_4.property, ltl_concat_2.out")

    // a Delay(42, 1337) b
    chirrtl should include("inst ltl_delay_3 of LTLDelayIntrinsic_42_1295")
    chirrtl should include("connect ltl_delay_3.in, b")
    chirrtl should include("connect ltl_concat_3.lhs, a")
    chirrtl should include("connect ltl_concat_3.rhs, ltl_delay_3")
    chirrtl should include("connect verif_5.property, ltl_concat_3.out")

    // a Delay(9001, None) b
    chirrtl should include("inst ltl_delay_4 of LTLDelayIntrinsic_9001")
    chirrtl should include("connect ltl_delay_4.in, b")
    chirrtl should include("connect ltl_concat_4.lhs, a")
    chirrtl should include("connect ltl_concat_4.rhs, ltl_delay_4")
    chirrtl should include("connect verif_6.property, ltl_concat_4.out")
  }
}
