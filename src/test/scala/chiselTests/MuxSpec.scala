// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util.{MuxLookup, log2Ceil}
import chisel3.testers.BasicTester

class MuxTester extends BasicTester {
  assert(Mux(0.B, 1.U, 2.U) === 2.U)
  assert(Mux(1.B, 1.U, 2.U) === 1.U)
  val dontCareMux1 = Wire(UInt())
  dontCareMux1 := Mux(0.B, DontCare, 4.U)  // note: Mux output of type Element
  assert(dontCareMux1 === 4.U)

  val dontCareMux2 = Wire(UInt())
  dontCareMux2 := Mux(1.B, 3.U, DontCare)  // note: Mux output of type Element
  assert(dontCareMux2 === 3.U)

  Mux(0.B, 3.U, DontCare)  // just to make sure nothing crashes, any result is valid
  stop()
}

class MuxSpec extends ChiselFlatSpec {
  "Mux" should "pass basic checks" in {
    assertTesterPasses { new MuxTester }
  }
}

class MuxLookupWrapper(keyWidth: Int, default: Int, mapping: () => Seq[(UInt, UInt)]) extends RawModule {
  val outputWidth = log2Ceil(default).max(keyWidth) // make room for default value
  val key = IO(Input(UInt(keyWidth.W)))
  val output = IO(Output(UInt(outputWidth.W)))
  output := MuxLookup(key, default.U, mapping())
}

class MuxLookupExhaustiveSpec extends ChiselPropSpec {
  val keyWidth = 2
  val default = 9 // must be less than 10 to avoid hex/decimal mismatches

  // Assumes there are no temps with '9' in the name -> fails conservatively
  // Assumes no binary recoding in output

  val incomplete = { () => Seq(0.U -> 1.U, 1.U -> 2.U, 2.U -> 3.U) }
  property("The default value should not be optimized away for an incomplete MuxLookup") {
    val c = Driver.emit { () => new MuxLookupWrapper(keyWidth, default, incomplete) }
    c.contains(default.toString) should be (true) // not optimized away
  }

  val exhaustive = { () => Seq(0.U -> 1.U, 1.U -> 2.U, 2.U -> 3.U, 3.U -> 0.U) }
  property("The default value should be optimized away for an exhaustive MuxLookup") {
    val c = Driver.emit { () => new MuxLookupWrapper(keyWidth, default, exhaustive) }
    c.contains(default.toString) should be (false) // optimized away
  }

  val overlap = { () => Seq(0.U -> 1.U, 1.U -> 2.U, 2.U -> 3.U, 4096.U -> 0.U) }
  property("The default value should be optimized away for a MuxLookup with 2^{keyWidth} non-distinct mappings") {
    val c = Driver.emit { () => new MuxLookupWrapper(keyWidth, default, overlap) }
    c.contains(default.toString) should be (true) // not optimized away
  }

}
