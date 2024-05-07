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
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include("node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)")
    chirrtl should include("node delay_1 = intrinsic(circt_ltl_delay<delay = 2, length = 2> : UInt<1>, b)")
    chirrtl should include("node delay_2 = intrinsic(circt_ltl_delay<delay = 5> : UInt<1>, c)")
    chirrtl should include("node delay_3 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b)")
    chirrtl should include("node concat = intrinsic(circt_ltl_concat : UInt<1>, a, delay_3")
    chirrtl should include("node delay_4 = intrinsic(circt_ltl_delay<delay = 0> : UInt<1>, b)")
    chirrtl should include("node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_4")
    chirrtl should include("node delay_5 = intrinsic(circt_ltl_delay<delay = 1> : UInt<1>, b)")
    chirrtl should include("node concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_5")
  }

  it should "support sequence concat operations" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b, c, d, e = IO(Input(Bool()))
      val s0: Sequence = a.concat(b)
      val s1: Sequence = Sequence.concat(c, d, e) // (c concat d) concat e
    })
    chirrtl should include("input a : UInt<1>")
    chirrtl should include("input b : UInt<1>")
    chirrtl should include("input c : UInt<1>")
    chirrtl should include("input d : UInt<1>")
    chirrtl should include("input e : UInt<1>")
    chirrtl should include("intrinsic(circt_ltl_concat : UInt<1>, a, b")
    chirrtl should include("node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, c, d")
    chirrtl should include("intrinsic(circt_ltl_concat : UInt<1>, concat_1, e")
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
    chirrtl should include("node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)")
    chirrtl should include("node and = intrinsic(circt_ltl_and : UInt<1>, delay, b)")
    chirrtl should include("node or = intrinsic(circt_ltl_or : UInt<1>, delay, b)")
    chirrtl should include("node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, delay, clock)")

    // Properties
    chirrtl should include("node eventually = intrinsic(circt_ltl_eventually : UInt<1>, a)")
    chirrtl should include("node and_1 = intrinsic(circt_ltl_and : UInt<1>, eventually, b)")
    chirrtl should include("node or_1 = intrinsic(circt_ltl_or : UInt<1>, eventually, b)")
    chirrtl should include("node clock_2 = intrinsic(circt_ltl_clock : UInt<1>, eventually, clock)")
  }

  it should "support property not operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      val p0: Property = Property.not(a)
    })
    chirrtl should include("intrinsic(circt_ltl_not : UInt<1>, a)")
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
    chirrtl should include("intrinsic(circt_ltl_implication : UInt<1>, a, b)")
    chirrtl should include("intrinsic(circt_ltl_implication : UInt<1>, a, b)")

    // Non-overlapping (emitted as `a ## true |-> b`)
    chirrtl should include("node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1))")
    chirrtl should include("node concat = intrinsic(circt_ltl_concat : UInt<1>, a, delay)")
    chirrtl should include("node implication_2 = intrinsic(circt_ltl_implication : UInt<1>, concat, b)")
    chirrtl should include("node delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1))")
    chirrtl should include("node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_1)")
    chirrtl should include("node implication_3 = intrinsic(circt_ltl_implication : UInt<1>, concat_1, b)")
  }

  it should "support property eventually operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      val p0: Property = a.eventually
    })
    chirrtl should include("intrinsic(circt_ltl_eventually : UInt<1>, a)")
  }

  it should "support property disable operation" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a, b = IO(Input(Bool()))
      val p0: Property = a.disable(b.asDisable)
    })
    chirrtl should include("intrinsic(circt_ltl_disable : UInt<1>, a, b)")
  }

  it should "support simple property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      AssertProperty(a)
      AssumeProperty(a)
      CoverProperty(a)
    })
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, a)")
    chirrtl should include("intrinsic(circt_verif_assume : UInt<1>, a)")
    chirrtl should include("intrinsic(circt_verif_cover : UInt<1>, a)")
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
        prop(a)
      })
      chirrtl should include("node has_been_reset = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)")
      chirrtl should include("node disable = eq(has_been_reset, UInt<1>(0h0))")
      chirrtl should include("node disable_1 = intrinsic(circt_ltl_disable : UInt<1>, a, disable)")
      chirrtl should include("node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, disable_1, clock)")
      chirrtl should include(f"node $op = intrinsic(circt_verif_$op : UInt<1>, clock_1)")
    }
  }

  it should "support labeled property asserts/assumes/covers" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(Bool()))
      AssertProperty(a, label = Some("foo0"))
      AssumeProperty(a, label = Some("foo1"))
      CoverProperty(a, label = Some("foo2"))
    })
    chirrtl should include("intrinsic(circt_verif_assert<label = \"foo0\"> : UInt<1>, a)")
    chirrtl should include("intrinsic(circt_verif_assume<label = \"foo1\"> : UInt<1>, a")
    chirrtl should include("intrinsic(circt_verif_cover<label = \"foo2\"> : UInt<1>, a)")
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
    chirrtl should include("node assert = intrinsic(circt_verif_assert : UInt<1>, clock)")

    // with disable; emitted as `assert(disable(a, b))`
    chirrtl should include("node disable = intrinsic(circt_ltl_disable : UInt<1>, a, b)")
    chirrtl should include("node assert_1 = intrinsic(circt_verif_assert : UInt<1>, disable)")

    // with clock and disable; emitted as `assert(clock(disable(a, b), c))`
    chirrtl should include("node disable_1 = intrinsic(circt_ltl_disable : UInt<1>, a, b)")
    chirrtl should include("node clock_1 = intrinsic(circt_ltl_clock : UInt<1>, disable_1, c)")
    chirrtl should include("node assert_2 = intrinsic(circt_verif_assert : UInt<1>, clock_1)")
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
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, a)")

    // a b
    chirrtl should include("node concat = intrinsic(circt_ltl_concat : UInt<1>, a, b)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, concat)")

    // Delay() a
    chirrtl should include("node delay = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, delay)")

    // a Delay() b
    chirrtl should include("node delay_1 = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b)")
    chirrtl should include("node concat_1 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_1)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, concat_1)")

    // a Delay(2) b
    chirrtl should include("node delay_2 = intrinsic(circt_ltl_delay<delay = 2, length = 0> : UInt<1>, b)")
    chirrtl should include("node concat_2 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_2)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, concat_2)")

    // a Delay(42, 1337) b
    chirrtl should include("node delay_3 = intrinsic(circt_ltl_delay<delay = 42, length = 1295> : UInt<1>, b)")
    chirrtl should include("node concat_3 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_3)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, concat_3)")

    // a Delay(9001, None) b
    chirrtl should include("node delay_4 = intrinsic(circt_ltl_delay<delay = 9001> : UInt<1>, b)")
    chirrtl should include("node concat_4 = intrinsic(circt_ltl_concat : UInt<1>, a, delay_4)")
    chirrtl should include("intrinsic(circt_verif_assert : UInt<1>, concat_4)")
  }
}
