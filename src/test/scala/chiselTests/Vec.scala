// See LICENSE for license details.

package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class ValueTester(w: Int, values: List[Int]) extends BasicTester {
  val v = Vec(values.map(UInt(_, width = w))) // TODO: does this need a Wire? Why no error?
  for ((a,b) <- v.zip(values)) {
    assert(a === UInt(b))
  }
  stop()
}

class TabulateTester(n: Int) extends BasicTester {
  val v = Vec(Range(0, n).map(i => UInt(i * 2)))
  val x = Vec(Array.tabulate(n){ i => UInt(i * 2) })
  val u = Vec.tabulate(n)(i => UInt(i*2))

  assert(v.toBits === x.toBits)
  assert(v.toBits === u.toBits)
  assert(x.toBits === u.toBits)

  stop()
}

class ShiftRegisterTester(n: Int) extends BasicTester {
  val (cnt, wrap) = Counter(Bool(true), n*2)
  val shifter = Reg(Vec(UInt(width = log2Up(n)), n))
  (shifter, shifter drop 1).zipped.foreach(_ := _)
  shifter(n-1) := cnt
  when (cnt >= UInt(n)) {
    val expected = cnt - UInt(n)
    assert(shifter(0) === expected)
  }
  when (wrap) {
    stop()
  }
}

class VecSpec extends ChiselPropSpec {
  property("Vecs should be assignable") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assert(execute{ new ValueTester(w, v) })
    }
  }

  property("Vecs should tabulate correctly") {
    forAll(smallPosInts) { (n: Int) => assert(execute{ new TabulateTester(n) }) }
  }

  property("Regs of vecs should be usable as shift registers") {
    forAll(smallPosInts) { (n: Int) => assert(execute{ new ShiftRegisterTester(n) }) }
  }
}
