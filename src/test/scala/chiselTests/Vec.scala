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

  assert(v.asUInt() === x.asUInt())
  assert(v.asUInt() === u.asUInt())
  assert(x.asUInt() === u.asUInt())

  stop()
}

class ShiftRegisterTester(n: Int) extends BasicTester {
  val (cnt, wrap) = Counter(Bool(true), n*2)
  val shifter = Reg(Vec(n, UInt(width = log2Up(n))))
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

class FunBundle extends Bundle {
  val stuff = UInt(width = 10)
}

class ZeroModule extends Module {
  val io = new Bundle {
    val mem = UInt(width = 10)
    val interrupts = Vec(2, Bool()).asInput
    val mmio_axi = Vec(0, new FunBundle)
    val mmio_ahb = Vec(0, new FunBundle).flip
  }
  
  io.mmio_axi <> io.mmio_ahb
  
  io.mem := UInt(0)
  when (io.interrupts(0)) { io.mem := UInt(1) }
  when (io.interrupts(1)) { io.mem := UInt(2) }
}

class ZeroTester extends BasicTester {
  val foo = Module(new ZeroModule)
  foo.io.interrupts := Vec.tabulate(2) { _ => Bool(true) }
  assert (foo.io.mem === UInt(2))
  stop()
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
  
  property("Dual empty Vectors") {
    assertTesterPasses{ new ZeroTester }
  }
}
