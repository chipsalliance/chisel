// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.{log2Ceil, MuxLookup}
import circt.stage.ChiselStage

class MuxTester extends BasicTester {
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

class MuxSpec extends ChiselFlatSpec {
  "Mux" should "pass basic checks" in {
    assertTesterPasses { new MuxTester }
  }
}

class MuxLookupEnumTester extends BasicTester {
  object TestEnum extends ChiselEnum {
    val a = Value
    val b = Value(7.U)
    val c = Value
  }
  val mapping = TestEnum.all.zipWithIndex.map {
    case (e, i) =>
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

class MuxLookupEnumSpec extends ChiselFlatSpec {
  "MuxLookup with enum selector" should "pass basic checks" in {
    assertTesterPasses { new MuxLookupEnumTester }
  }
}

class MuxLookupWrapper(keyWidth: Int, default: Int, mapping: () => Seq[(UInt, UInt)]) extends RawModule {
  val outputWidth = log2Ceil(default).max(keyWidth) // make room for default value
  val key = IO(Input(UInt(keyWidth.W)))
  val output = IO(Output(UInt(outputWidth.W)))
  output := MuxLookup(key, default.U)(mapping())
}

class MuxLookupExhaustiveSpec extends ChiselPropSpec {
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
