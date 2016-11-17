// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
//import chisel3.core.ExplicitCompileOptions.Strict

class ValueTester(w: Int, values: List[Int]) extends BasicTester {
  val v = Vec(values.map(_.asUInt(w.W))) // TODO: does this need a Wire? Why no error?
  for ((a,b) <- v.zip(values)) {
    assert(a === b.asUInt)
  }
  stop()
}

class TabulateTester(n: Int) extends BasicTester {
  val v = Vec(Range(0, n).map(i => UInt(i * 2)))
  val x = Vec(Array.tabulate(n){ i => UInt(i * 2) })
  val u = Vec.tabulate(n)(i => UInt(i*2))

  assert(v.asUInt() === x.asUInt())
  assert(v.asUInt() === u.asUInt())
  assert(x.asUInt() === u.asUInt())

  stop()
}

class ShiftRegisterTester(n: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, n*2)
  val shifter = Reg(Vec(n, UInt.width(log2Up(n))))
  (shifter, shifter drop 1).zipped.foreach(_ := _)
  shifter(n-1) := cnt
  when (cnt >= n.asUInt) {
    val expected = cnt - n.asUInt
    assert(shifter(0) === expected)
  }
  when (wrap) {
    stop()
  }
}

class VecSpec extends ChiselPropSpec {
  property("Vecs should be assignable") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new ValueTester(w, v) }
    }
  }

  property("Vecs should tabulate correctly") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new TabulateTester(n) } }
  }

  property("Regs of vecs should be usable as shift registers") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new ShiftRegisterTester(n) } }
  }
}
