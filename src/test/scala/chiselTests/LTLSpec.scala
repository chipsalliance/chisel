// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import chisel3.experimental.{SourceInfo, SourceLine}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class LTLSpec extends AnyFlatSpec with Matchers with ChiselSim with FileCheck {
  it should "allow booleans to be used as sequences" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Bool()))
        Sequence.delay(a, 42)
      })
      .fileCheck()(
        """|CHECK: input a : UInt<1>
           |CHECK: intrinsic(circt_ltl_delay<delay = 42, length = 0> : UInt<1>, a)
           |""".stripMargin
      )
  }

  it should "allow booleans to be used as properties" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a = IO(Input(Bool()))
        Property.eventually(a)
      })
      .fileCheck()(
        """|CHECK: input a : UInt<1>
           |CHECK: intrinsic(circt_ltl_eventually : UInt<1>, a)
           |""".stripMargin
      )
  }

  class DelaysMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b, c = IO(Input(Bool()))
    val s0: Sequence = a.delay(1)
    val s1: Sequence = b.delayRange(2, 4)
    val s2: Sequence = c.delayAtLeast(5)
    val s3: Sequence = a ### b
    val s4: Sequence = a ##* b
    val s5: Sequence = a ##+ b
  }
  it should "support sequence delay operations" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new DelaysMod)
      .fileCheck()(
        s"""|CHECK: input a : UInt<1>
            |CHECK: input b : UInt<1>
            |CHECK: input c : UInt<1>
            |CHECK: intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc
            |CHECK: intrinsic(circt_ltl_delay<delay = 2, length = 2> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_delay<delay = 5> : UInt<1>, c) $sourceLoc
            |CHECK: node [[D3:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_concat : UInt<1>, a, [[D3]]) $sourceLoc
            |CHECK: node [[D4:.*]] = intrinsic(circt_ltl_delay<delay = 0> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_concat : UInt<1>, a, [[D4]]) $sourceLoc
            |CHECK: node [[D5:.*]] = intrinsic(circt_ltl_delay<delay = 1> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_concat : UInt<1>, a, [[D5]]) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile sequence delay operations" in {
    ChiselStage.emitSystemVerilog(new DelaysMod)
  }

  class PastMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b = IO(Input(Bool()))
    val clock = IO(Input(Clock()))
    val s0: Sequence = a.past()
    val s1: Sequence = a.past(3)
    val s2: Sequence = Sequence.past(b, 2)
    val s3: Sequence = a.past(clock)
    val s4: Sequence = a.past(2, clock)
  }
  it should "support sequence past operations" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new PastMod)
      .fileCheck()(
        s"""|CHECK: input a : UInt<1>
            |CHECK: input b : UInt<1>
            |CHECK: input clock : Clock
            |CHECK: intrinsic(circt_ltl_past<delay = 1> : UInt<1>, a) $sourceLoc
            |CHECK: intrinsic(circt_ltl_past<delay = 3> : UInt<1>, a) $sourceLoc
            |CHECK: intrinsic(circt_ltl_past<delay = 2> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_past<delay = 1> : UInt<1>, a, clock) $sourceLoc
            |CHECK: intrinsic(circt_ltl_past<delay = 2> : UInt<1>, a, clock) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile sequence past operations" in {
    ChiselStage.emitSystemVerilog(new PastMod)
  }

  class ConcatMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b, c, d, e = IO(Input(Bool()))
    val s0: Sequence = a.concat(b)
    val s1: Sequence = Sequence.concat(c, d, e) // (c concat d) concat e
  }
  it should "support sequence concat operations" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new ConcatMod)
      .fileCheck()(
        s"""|CHECK: input a : UInt<1>
            |CHECK: input b : UInt<1>
            |CHECK: input c : UInt<1>
            |CHECK: input d : UInt<1>
            |CHECK: input e : UInt<1>
            |CHECK: intrinsic(circt_ltl_concat : UInt<1>, a, b) $sourceLoc
            |CHECK: node [[CD:.*]] = intrinsic(circt_ltl_concat : UInt<1>, c, d) $sourceLoc
            |CHECK: intrinsic(circt_ltl_concat : UInt<1>, [[CD]], e) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile sequence concat operations" in {
    ChiselStage.emitSystemVerilog(new ConcatMod)
  }

  class RepeatMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b, c, d, e = IO(Input(Bool()))
    val s0: Sequence = a.repeat(1)
    val s1: Sequence = b.repeatRange(2, 4)
    val s2: Sequence = c.repeatAtLeast(5)
    val s3: Sequence = d.gotoRepeat(1, 3)
    val s4: Sequence = e.nonConsecutiveRepeat(1, 3)
  }
  it should "support sequence repeat operations" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new RepeatMod)
      .fileCheck()(
        s"""|CHECK: input a : UInt<1>
            |CHECK: input b : UInt<1>
            |CHECK: input c : UInt<1>
            |CHECK: input d : UInt<1>
            |CHECK: input e : UInt<1>
            |CHECK: intrinsic(circt_ltl_repeat<base = 1, more = 0> : UInt<1>, a) $sourceLoc
            |CHECK: intrinsic(circt_ltl_repeat<base = 2, more = 2> : UInt<1>, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_repeat<base = 5> : UInt<1>, c) $sourceLoc
            |CHECK: intrinsic(circt_ltl_goto_repeat<base = 1, more = 2> : UInt<1>, d) $sourceLoc
            |CHECK: intrinsic(circt_ltl_non_consecutive_repeat<base = 1, more = 2> : UInt<1>, e) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile sequence repeat operations" in {
    ChiselStage.emitSystemVerilog(new RepeatMod)
  }

  class AndOrClockMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b = IO(Input(Bool()))
    val clock = IO(Input(Clock()))
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
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new AndOrClockMod)
      .fileCheck()(
        s"""|CHECK: node [[DEL:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc
            |CHECK: node [[SAND:.*]] = intrinsic(circt_ltl_and : UInt<1>, [[DEL]], b) $sourceLoc
            |CHECK: node [[SOR:.*]] = intrinsic(circt_ltl_or : UInt<1>, [[DEL]], b) $sourceLoc
            |CHECK: node [[SI:.*]] = intrinsic(circt_ltl_intersect : UInt<1>, [[DEL]], b) $sourceLoc
            |CHECK: node [[SIA:.*]] = intrinsic(circt_ltl_intersect : UInt<1>, [[SI]], [[SAND]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_intersect : UInt<1>, [[SIA]], [[SOR]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_clock : UInt<1>, [[DEL]], clock) $sourceLoc
            |
            |CHECK: node [[EV:.*]] = intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc
            |CHECK: node [[PAND:.*]] = intrinsic(circt_ltl_and : UInt<1>, [[EV]], b) $sourceLoc
            |CHECK: node [[POR:.*]] = intrinsic(circt_ltl_or : UInt<1>, [[EV]], b) $sourceLoc
            |CHECK: node [[PI:.*]] = intrinsic(circt_ltl_intersect : UInt<1>, [[EV]], b) $sourceLoc
            |CHECK: node [[PIA:.*]] = intrinsic(circt_ltl_intersect : UInt<1>, [[PI]], [[PAND]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_intersect : UInt<1>, [[PIA]], [[POR]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_clock : UInt<1>, [[EV]], clock) $sourceLoc
            |
            |CHECK: intrinsic(circt_ltl_until : UInt<1>, [[DEL]], b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_until : UInt<1>, [[EV]], b) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile and, or, intersect, and clock operations" in {
    ChiselStage.emitSystemVerilog(new AndOrClockMod)
  }

  class NotMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a = IO(Input(Bool()))
    val p0: Property = Property.not(a)
  }
  it should "support property not operation" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new NotMod)
      .fileCheck()(
        s"""|CHECK: intrinsic(circt_ltl_not : UInt<1>, a) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile property not operation" in {
    ChiselStage.emitSystemVerilog(new NotMod)
  }

  class PropImplicationMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b = IO(Input(Bool()))
    val p0: Property = Property.implication(a, b)
    val p1: Property = a |-> b
    val p2: Property = Property.implicationNonOverlapping(a, b)
    val p3: Property = a |=> b
  }
  it should "support property implication operation" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new PropImplicationMod)
      .fileCheck()(
        s"""|CHECK: intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc
            |CHECK: intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc
            |
            |CHECK: node [[D0:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc
            |CHECK: node [[C0:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D0]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_implication : UInt<1>, [[C0]], b) $sourceLoc
            |CHECK: node [[D1:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, UInt<1>(0h1)) $sourceLoc
            |CHECK: node [[C1:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D1]]) $sourceLoc
            |CHECK: intrinsic(circt_ltl_implication : UInt<1>, [[C1]], b) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile property implication operation" in {
    ChiselStage.emitSystemVerilog(new PropImplicationMod)
  }

  class EventuallyMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a = IO(Input(Bool()))
    val p0: Property = a.eventually
  }
  it should "support property eventually operation" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new EventuallyMod)
      .fileCheck()(
        s"""|CHECK: intrinsic(circt_ltl_eventually : UInt<1>, a) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile property eventually operation" in {
    ChiselStage.emitSystemVerilog(new EventuallyMod)
  }

  class BasicVerifMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a = IO(Input(Bool()))
    AssertProperty(a)
    AssumeProperty(a)
    CoverProperty(a)
    RequireProperty(a)
    EnsureProperty(a)
  }
  it should "support simple property asserts/assumes/covers and put them in layer blocks" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new BasicVerifMod)
      .fileCheck()(
        s"""|CHECK: layerblock Verification
            |CHECK: layerblock Assert
            |CHECK: intrinsic(circt_verif_assert, a) $sourceLoc
            |CHECK: layerblock Verification
            |CHECK: layerblock Assume
            |CHECK: intrinsic(circt_verif_assume, a) $sourceLoc
            |CHECK: layerblock Verification
            |CHECK: layerblock Cover
            |CHECK: intrinsic(circt_verif_cover, a) $sourceLoc
            |CHECK: layerblock Verification
            |CHECK: layerblock Assume
            |CHECK: intrinsic(circt_verif_require, a) $sourceLoc
            |CHECK: layerblock Verification
            |CHECK: layerblock Assert
            |CHECK: intrinsic(circt_verif_ensure, a) $sourceLoc
            |""".stripMargin
      )
  }
  it should "compile simple property checks" in {
    ChiselStage.emitSystemVerilog(new BasicVerifMod)
  }
  it should "not create layer blocks if already in a layer block" in {
    class Foo extends RawModule {
      implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
      val a = IO(Input(Bool()))
      layer.block(chisel3.layers.Verification.Cover) {
        AssertProperty(a)
      }
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK:     layerblock Verification
           |CHECK:     layerblock Cover
           |CHECK-NOT: layerblock Assert
           |CHECK:     intrinsic(circt_verif_assert, a)
           |""".stripMargin
      )
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
      val sourceLoc = "@[Foo.scala 1:2]"
      ChiselStage
        .emitCHIRRTL(new Module {
          implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
          val a = IO(Input(Bool()))
          prop(a)
        })
        .fileCheck()(
          s"""|CHECK: node [[HBR:.*]] = intrinsic(circt_has_been_reset : UInt<1>, clock, reset)
              |CHECK: node [[DIS:.*]] = eq([[HBR]], UInt<1>(0h0))
              |CHECK: node [[CLK:.*]] = intrinsic(circt_ltl_clock : UInt<1>, a, clock) $sourceLoc
              |CHECK: node [[EN:.*]] = eq([[DIS]], UInt<1>(0h0)) $sourceLoc
              |CHECK: intrinsic(circt_verif_$op, [[CLK]], [[EN]]) $sourceLoc
              |""".stripMargin
        )
    }
  }

  class LabeledVerifMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a = IO(Input(Bool()))
    AssertProperty(a, label = Some("foo0"))
    AssumeProperty(a, label = Some("foo1"))
    CoverProperty(a, label = Some("foo2"))
    RequireProperty(a, label = Some("foo3"))
    EnsureProperty(a, label = Some("foo4"))
  }
  it should "support labeled property asserts/assumes/covers" in {
    ChiselStage
      .emitCHIRRTL(new LabeledVerifMod)
      .fileCheck()(
        """|CHECK: intrinsic(circt_verif_assert<label = "foo0">, a)
           |CHECK: intrinsic(circt_verif_assume<label = "foo1">, a)
           |CHECK: intrinsic(circt_verif_cover<label = "foo2">, a)
           |CHECK: intrinsic(circt_verif_require<label = "foo3">, a)
           |CHECK: intrinsic(circt_verif_ensure<label = "foo4">, a)
           |""".stripMargin
      )
  }
  it should "compile labeled property asserts/assumes/covers" in {
    ChiselStage.emitSystemVerilog(new LabeledVerifMod)
  }

  it should "support assert shorthands with clock and disable" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val a, b = IO(Input(Bool()))
        val c = IO(Input(Clock()))
        AssertProperty(a, clock = Some(c))
        AssertProperty(a, disable = Some(b.asDisable))
        AssertProperty(a, clock = Some(c), disable = Some(b.asDisable))
      })
      .fileCheck()(
        // with clock; emitted as `assert(clock(a, c))`
        """|CHECK: node [[C0:.*]] = intrinsic(circt_ltl_clock : UInt<1>, a, c)
           |CHECK: intrinsic(circt_verif_assert, [[C0]])
           |
           |CHECK: node [[T0:.*]] = eq(b, UInt<1>(0h0))
           |CHECK: intrinsic(circt_verif_assert, a, [[T0]])
           |
           |CHECK: node [[C1:.*]] = intrinsic(circt_ltl_clock : UInt<1>, a, c)
           |CHECK: node [[T1:.*]] = eq(b, UInt<1>(0h0))
           |CHECK: intrinsic(circt_verif_assert, [[C1]], [[T1]])
           |""".stripMargin
      )
  }

  class SequenceConvMod extends RawModule {
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
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
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new SequenceConvMod)
      .fileCheck()(
        s"""|CHECK: intrinsic(circt_verif_assert, a) $sourceLoc
            |
            |CHECK: node [[CAB:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, b) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[CAB]]) $sourceLoc
            |
            |CHECK: node [[D0:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[D0]]) $sourceLoc
            |
            |CHECK: node [[D1:.*]] = intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, b) $sourceLoc
            |CHECK: node [[C1:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D1]]) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[C1]]) $sourceLoc
            |
            |CHECK: node [[D2:.*]] = intrinsic(circt_ltl_delay<delay = 2, length = 0> : UInt<1>, b) $sourceLoc
            |CHECK: node [[C2:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D2]]) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[C2]]) $sourceLoc
            |
            |CHECK: node [[D3:.*]] = intrinsic(circt_ltl_delay<delay = 42, length = 1295> : UInt<1>, b) $sourceLoc
            |CHECK: node [[C3:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D3]]) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[C3]]) $sourceLoc
            |
            |CHECK: node [[D4:.*]] = intrinsic(circt_ltl_delay<delay = 9001> : UInt<1>, b) $sourceLoc
            |CHECK: node [[C4:.*]] = intrinsic(circt_ltl_concat : UInt<1>, a, [[D4]]) $sourceLoc
            |CHECK: intrinsic(circt_verif_assert, [[C4]]) $sourceLoc
            |""".stripMargin
      )
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
    implicit val info: SourceInfo = SourceLine("Foo.scala", 1, 2)
    val a, b = IO(Input(Bool()))
    AssertProperty(Sequence(Delay(), a))
    AssumeProperty(a |-> b)
  }

  it should "wrap all intrinsics in layerblocks" in {
    val sourceLoc = "@[Foo.scala 1:2]"
    ChiselStage
      .emitCHIRRTL(new LayerBlockMod)
      .fileCheck()(
        s"""|CHECK: layerblock Assert : $sourceLoc
            |CHECK:   intrinsic(circt_ltl_delay<delay = 1, length = 0> : UInt<1>, a)
            |CHECK: layerblock Assume : $sourceLoc
            |CHECK:   intrinsic(circt_ltl_implication : UInt<1>, a, b) $sourceLoc
            |""".stripMargin
      )
  }
}
