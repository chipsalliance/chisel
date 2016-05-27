// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel._
import chisel.testers.BasicTester
import chisel.util._

class Complex[T <: Data](val re: T, val im: T) extends Bundle {
  override def cloneType: this.type =
    new Complex(re.cloneType, im.cloneType).asInstanceOf[this.type]
}

class ComplexAssign(w: Int) extends Module {
  val io = new Bundle {
    val e   = new Bool(INPUT)
    val in  = new Complex(UInt(width = w), UInt(width = w)).asInput
    val out = new Complex(UInt(width = w), UInt(width = w)).asOutput
  }
  when (io.e) {
    val tmp = Wire(new Complex(UInt(width = w), UInt(width = w)))
    tmp := io.in
    io.out.re := tmp.re
    io.out.im := tmp.im
  } .otherwise {
    io.out.re := UInt(0)
    io.out.im := UInt(0)
  }
}

class ComplexAssignTester(enList: List[Boolean], re: Int, im: Int) extends BasicTester {
  val (cnt, wrap) = Counter(Bool(true), enList.size)
  val dut = Module(new ComplexAssign(32))
  dut.io.in.re := UInt(re)
  dut.io.in.im := UInt(im)
  dut.io.e := Vec(enList.map(Bool(_)))(cnt)
  val re_correct = dut.io.out.re === Mux(dut.io.e, dut.io.in.re, UInt(0))
  val im_correct = dut.io.out.im === Mux(dut.io.e, dut.io.in.im, UInt(0))
  assert(re_correct && im_correct)
  when(wrap) {
    stop()
  }
}

class ComplexAssignSpec extends ChiselPropSpec {
  property("All complex assignments should return the correct result") {
    forAll(enSequence(2), safeUInts, safeUInts) { (en: List[Boolean], re: Int, im: Int) =>
      assertTesterPasses{ new ComplexAssignTester(en, re, im) }
    }
  }
}
