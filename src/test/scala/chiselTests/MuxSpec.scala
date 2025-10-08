// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{log2Ceil, MuxLookup}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec

class MuxTester extends Module {
  assert(Mux(0.B, 1.U, 2.U) === 2.U)
  assert(Mux(1.B, 1.U, 2.U) === 1.U)
  val dontCareMux1 = Wire(UInt())
  dontCareMux1 := Mux(0.B, DontCare, 4.U) // note: Mux output of type Element
  assert(dontCareMux1 === 4.U)

  val dontCareMux2 = Wire(UInt())
  dontCareMux2 := Mux(1.B, 3.U, DontCare) // note: Mux output of type Element
  assert(dontCareMux2 === 3.U)

  Mux(0.B, 3.U, DontCare) // just to make sure nothing crashes, any result is valid
  stop()
}

class MuxBetween[A <: Data, B <: Data](genA: A, genB: B) extends RawModule {
  val in0 = IO(Input(genA))
  val in1 = IO(Input(genB))
  val sel = IO(Input(Bool()))
  val result = Mux(sel, in0, in1)
}

class MuxSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "Mux" should "pass basic checks" in {
    simulate(new MuxTester)(RunUntilFinished(3))
  }

  it should "give reasonable error messages for mismatched user-defined types" in {
    class MyBundle(w: Int) extends Bundle {
      val foo = UInt(w.W)
      val bar = UInt(8.W)
    }
    val e = the[Exception] thrownBy {
      ChiselStage.emitCHIRRTL(new MuxBetween(new MyBundle(8), new MyBundle(16)))
    }
    e.getMessage should include(
      "can't create Mux with non-equivalent types _.foo: Left (MuxBetween.in1.foo: IO[UInt<16>]) and Right (MuxBetween.in0.foo: IO[UInt<8>]) have different widths."
    )
  }

}

class MuxLookupEnumTester extends Module {
  object TestEnum extends ChiselEnum {
    val a = Value
    val b = Value(7.U)
    val c = Value
  }
  val mapping = TestEnum.all.zipWithIndex.map { case (e, i) =>
    e -> i.U
  }
  assert(MuxLookup(TestEnum.a, 3.U)(mapping) === 0.U)
  assert(MuxLookup(TestEnum.b, 3.U)(mapping) === 1.U)
  assert(MuxLookup(TestEnum.c, 3.U)(mapping) === 2.U)

  val incompleteMapping = Seq(TestEnum.a -> 0.U, TestEnum.c -> 2.U)
  assert(MuxLookup(TestEnum.a, 3.U)(incompleteMapping) === 0.U)
  assert(MuxLookup(TestEnum.b, 3.U)(incompleteMapping) === 3.U)
  assert(MuxLookup(TestEnum.c, 3.U)(incompleteMapping) === 2.U)

  stop()
}

class MuxLookupEnumSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "MuxLookup with enum selector" should "pass basic checks" in {
    simulate(new MuxLookupEnumTester)(RunUntilFinished(3))
  }
}

class MuxLookupWrapper(keyWidth: Int, default: Int, mapping: () => Seq[(UInt, UInt)]) extends RawModule {
  val outputWidth = log2Ceil(default).max(keyWidth) // make room for default value
  val key = IO(Input(UInt(keyWidth.W)))
  val output = IO(Output(UInt(outputWidth.W)))
  output := MuxLookup(key, default.U)(mapping())
}

class MuxLookupExhaustiveSpec extends AnyPropSpec with Matchers {
  val keyWidth = 2
  val default = 9 // must be less than 10 to avoid hex/decimal mismatches
  val firrtlLit = s"""UInt<4>(0h$default)"""

  // Assumes there are no literals with 'UInt<4>(0h09)' in the output FIRRTL
  // Assumes no binary recoding in output

  val incomplete = () => Seq(0.U -> 1.U, 1.U -> 2.U, 2.U -> 3.U)
  property("The default value should not be optimized away for an incomplete MuxLookup") {
    ChiselStage.emitCHIRRTL(new MuxLookupWrapper(keyWidth, default, incomplete)) should include(firrtlLit)
  }

  val exhaustive = () => (3.U -> 0.U) +: incomplete()
  property("The default value should be optimized away for an exhaustive MuxLookup") {
    (ChiselStage.emitCHIRRTL(new MuxLookupWrapper(keyWidth, default, exhaustive)) should not).include(firrtlLit)
  }

  val overlap = () => (4096.U -> 0.U) +: incomplete()
  property("The default value should not be optimized away for a MuxLookup with 2^{keyWidth} non-distinct mappings") {
    ChiselStage.emitCHIRRTL(new MuxLookupWrapper(keyWidth, default, overlap)) should include(firrtlLit)
  }

  val nonLiteral = () => { val foo = Wire(UInt()); (foo -> 1.U) +: incomplete() }
  property("The default value should not be optimized away for a MuxLookup with a non-literal") {
    ChiselStage.emitCHIRRTL(new MuxLookupWrapper(keyWidth, default, nonLiteral)) should include(firrtlLit)
  }

  val nonLiteralStillFull = () => { val foo = Wire(UInt()); (foo -> 1.U) +: exhaustive() }
  property("The default value should be optimized away for a MuxLookup with a non-literal that is still full") {
    (ChiselStage.emitCHIRRTL(new MuxLookupWrapper(keyWidth, default, nonLiteralStillFull)) should not)
      .include(firrtlLit)
  }

}
