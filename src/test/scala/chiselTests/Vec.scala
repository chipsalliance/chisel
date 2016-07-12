// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class RegTesterMod( vecSize : Int ) extends Module {
  val io = new Bundle {
    val in = Vec( vecSize, UInt( INPUT ) )
    val out = Vec( vecSize, UInt( OUTPUT ) )
  }
  val vecReg = RegInit( Vec( vecSize, UInt( 0 ) ) )
  vecReg := io.in
  io.out := vecReg
}

class IOTesterMod( vecSize : Int ) extends Module {
  val io = new Bundle {
    val in = Vec( vecSize, UInt( INPUT ) )
    val out = Vec( vecSize, UInt( OUTPUT ) )
  }
  io.out := io.in
}

class RegTester(w: Int, values: List[Int] ) extends BasicTester {
  val v = Vec(values.map(UInt(_, width = w)))
  val dut = Module( new RegTesterMod( values.length ) )
  val doneReg = RegInit( Bool(false) )
  dut.io.in := v
  when ( doneReg ) {
    for ((a,b) <- dut.io.out.zip(values))
      assert(a === UInt(b))
    stop()
  } .otherwise {
    doneReg := Bool(true)
    for ( a <- dut.io.out)
      assert(a === UInt(0))
  }
}

class IOTester(w: Int, values: List[Int] ) extends BasicTester {
  val v = Vec(values.map(UInt(_, width = w))) // TODO: does this need a Wire? Why no error?
  val dut = Module( new IOTesterMod( values.length ) )
  dut.io.in := v
  for ((a,b) <- dut.io.out.zip(values)) {
    assert(a === UInt(b))
  }
  stop()
}

class IOTesterModFill( vecSize : Int ) extends Module {
  val io = new Bundle {
    val in = Vec.fill( vecSize ) { UInt( INPUT ) }
    val out = Vec.fill( vecSize ) { UInt( OUTPUT ) }
  }
  io.out := io.in
}

class IOTesterFill(w: Int, values: List[Int] ) extends BasicTester {
  val v = Vec(values.map(UInt(_, width = w))) // TODO: does this need a Wire? Why no error?
  val dut = Module( new IOTesterModFill( values.length ) )
  dut.io.in := v
  for ((a,b) <- dut.io.out.zip(values)) {
    assert(a === UInt(b))
  }
  stop()
}

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

  property("Vecs should be passed through vec IO") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new IOTester(w, v) }
    }
  }

  property("Vecs should be passed through vec IO with fill") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new IOTesterFill(w, v) }
    }
  }

  property("A Reg of a Vec should operate correctly") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new RegTester(w, v) }
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
