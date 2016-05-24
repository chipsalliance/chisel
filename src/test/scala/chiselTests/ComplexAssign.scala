// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._

import Chisel.testers.BasicTester

class Complex[T <: Data](val re: T, val im: T) extends Bundle {
  override def cloneType: this.type =
    new Complex(re.newType, im.newType).asInstanceOf[this.type]
}

class ComplexAssign(w: Int) extends Module {
  val io = IO(new Bundle {
    val e   = Input(Bool())
    val in  = Input(new Complex(UInt(width = w), UInt(width = w)))
    val out = Output(new Complex(UInt(width = w), UInt(width = w)))
  })
  when (io.e) {
    val tmp = Wire(new Complex(UInt(width = w), UInt(width = w)))
    tmp := io.in
    io.out.re := tmp.re
    io.out.im := tmp.im
  } .otherwise {
    io.out.re := 0.asUInt
    io.out.im := 0.asUInt
  }
}

class ComplexAssignTester(enList: List[Boolean], re: Int, im: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.asBool, enList.size)
  val dut = Module(new ComplexAssign(32))
  dut.io.in.re := re.asUInt
  dut.io.in.im := im.asUInt
  dut.io.e := Vec(enList.map(_.asBool))(cnt)
  val re_correct = dut.io.out.re === Mux(dut.io.e, dut.io.in.re, 0.asUInt)
  val im_correct = dut.io.out.im === Mux(dut.io.e, dut.io.in.im, 0.asUInt)
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
