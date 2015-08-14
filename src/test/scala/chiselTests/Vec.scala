package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class VecSpec extends ChiselPropSpec {

  class ValueTester(w: Int, values: List[Int]) extends BasicTester {
    io.done := Bool(true)
    val v = Vec(values.map(UInt(_, width = w))) // TODO: does this need a Wire? Why no error?
    io.error := v.zip(values).map { case(a,b) => 
      a != UInt(b)
    }.foldLeft(UInt(0))(_##_)
  }

  property("Vecs should be assignable") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assert(execute{ new ValueTester(w, v) })
    }
  }

  class TabulateTester(n: Int) extends BasicTester {
    io.done := Bool(true)
    val v = Vec(Range(0, n).map(i => UInt(i * 2)))
    val x = Vec(Array.tabulate(n){ i => UInt(i * 2) })
    val u = Vec.tabulate(n)(i => UInt(i*2))
    when(v.toBits != x.toBits) { io.error := UInt(1) }
    when(v.toBits != u.toBits) { io.error := UInt(2) }
    when(x.toBits != u.toBits) { io.error := UInt(3) }
  }

  property("Vecs should tabulate correctly") {
    forAll(smallPosInts) { (n: Int) => assert(execute{ new TabulateTester(n) }) }
  }

  class ShiftRegisterTester(n: Int) extends BasicTester {
    val (cnt, wrap) = Counter(Bool(true), n*2)
    when(wrap) { io.done := Bool(true) }

    val shifter = Vec(Reg(UInt(width = log2Up(n))), n)
    (shifter, shifter drop 1).zipped.foreach(_ := _)
    shifter(n-1) := cnt
    val expected = cnt - UInt(n)
    when(cnt >= UInt(n) && expected != shifter(0)) { io.done := Bool(true); io.error := expected }
  }

  property("Vecs of regs should be usable as shift registers") {
    forAll(smallPosInts) { (n: Int) => assert(execute{ new ShiftRegisterTester(n) }) }
  }
}
