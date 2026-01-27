// SPDX-License-Identifier: Apache-2.0

package chiselTests.reflect

import chisel3._
import chisel3.experimental.Analog
import chisel3.reflect.DataMirror
import circt.stage.ChiselStage
import org.scalactic.source.Position
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object CheckTypeEquivalenceSpec {
  private def test[A <: Data, B <: Data](gen1: A, gen2: B, expectSame: Boolean)(implicit pos: Position): Unit = {
    class TestModule extends RawModule {
      val foo = IO(Flipped(gen1))
      val bar = IO(gen2)
      if (expectSame) {
        assert(DataMirror.checkTypeEquivalence(gen1, gen2), s"Unbound types $gen1 and $gen2 should be the same")
        assert(DataMirror.checkTypeEquivalence(foo, bar), s"Bound values $foo and $bar should be the same")
      } else {
        assert(!DataMirror.checkTypeEquivalence(gen1, gen2), s"Unbound types $gen1 and $gen2 should NOT be the same")
        assert(!DataMirror.checkTypeEquivalence(foo, bar), s"Bound values $foo and $bar NOT should be the same")
      }
    }
    ChiselStage.emitCHIRRTL(new TestModule)
  }
  private def testSame[A <: Data, B <: Data](gen1: A, gen2: B)(implicit pos: Position): Unit = test(gen1, gen2, true)
  private def testDiff[A <: Data, B <: Data](gen1: A, gen2: B)(implicit pos: Position): Unit = test(gen1, gen2, false)

  private def elementTypes =
    Seq(UInt(8.W), UInt(), SInt(8.W), SInt(), Bool(), Clock(), Reset(), AsyncReset(), Analog(8.W))
  private def elementTypeCombinations = elementTypes.combinations(2).map { pair => (pair(0), pair(1)) }

  // Use of 2 Data here is arbitrary, they aren't being compared
  private class BoxBundle[A <: Data, B <: Data](gen1: A, gen2: B) extends Bundle {
    val foo = gen1.cloneType
    val bar = gen2.cloneType
  }
}

class CheckTypeEquivalenceSpec extends AnyFlatSpec with Matchers {
  import CheckTypeEquivalenceSpec._

  behavior.of("DataMirror.checkTypeEquivalence")

  it should "support equivalence of Element types" in {
    for (tpe <- elementTypes) {
      testSame(tpe, tpe.cloneType)
    }
  }

  it should "show non-equivalence of Element types" in {
    for ((gen1, gen2) <- elementTypeCombinations) {
      testDiff(gen1, gen2)
    }
  }

  it should "support equivalence of Vecs" in {
    // Shouldn't need to check many different sizes
    for (size <- 0 to 2) {
      for (eltTpe <- elementTypes) {
        testSame(Vec(size, eltTpe), Vec(size, eltTpe))
      }
    }
  }

  it should "support non-equivalence of Vecs" in {
    // Shouldn't need to check many different sizes
    for (size <- 0 to 2) {
      for (eltTpe <- elementTypes) {
        testDiff(Vec(size, eltTpe), Vec(size + 1, eltTpe))
      }
    }
  }

  it should "support equivalence of Bundles" in {
    for ((gen1, gen2) <- elementTypeCombinations) {
      testSame(new BoxBundle(gen1, gen2), new BoxBundle(gen1, gen2))
    }
  }

  it should "support non-equivalence of Bundles" in {
    for ((gen1, gen2) <- elementTypeCombinations) {
      // Note 2nd argument has arguments backwards
      testDiff(new BoxBundle(gen1, gen2), new BoxBundle(gen2, gen1))
    }
  }

  // Test for https://github.com/chipsalliance/chisel/issues/3922
  it should "check all fields of a Bundle (not just first or last)" in {
    class StartAndEndSameBundle(extraField: Boolean) extends Bundle {
      val foo = UInt(8.W)
      val mid = if (extraField) Some(UInt(8.W)) else None
      val bar = UInt(8.W)
    }
    // Sanity checks
    testSame(new StartAndEndSameBundle(true), new StartAndEndSameBundle(true))
    testSame(new StartAndEndSameBundle(false), new StartAndEndSameBundle(false))
    // Real check
    testDiff(new StartAndEndSameBundle(true), new StartAndEndSameBundle(false))
  }
}
