// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class Complex[T <: Data](val re: T, val im: T) extends Bundle

class ComplexAssign(w: Int) extends Module {
  val io = IO(new Bundle {
    val e = Input(Bool())
    val in = Input(new Complex(UInt(w.W), UInt(w.W)))
    val out = Output(new Complex(UInt(w.W), UInt(w.W)))
  })
  when(io.e) {
    val tmp = Wire(new Complex(UInt(w.W), UInt(w.W)))
    tmp := io.in
    io.out.re := tmp.re
    io.out.im := tmp.im
  }.otherwise {
    io.out.re := 0.U
    io.out.im := 0.U
  }
}

class ComplexAssignTester(enList: List[Boolean], re: Int, im: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, enList.size)
  val dut = Module(new ComplexAssign(32))
  dut.io.in.re := re.asUInt
  dut.io.in.im := im.asUInt
  dut.io.e := VecInit(enList.map(_.asBool))(cnt)
  val re_correct = dut.io.out.re === Mux(dut.io.e, dut.io.in.re, 0.U)
  val im_correct = dut.io.out.im === Mux(dut.io.e, dut.io.in.im, 0.U)
  assert(re_correct && im_correct)
  when(wrap) {
    stop()
  }
}

class ComplexAssignSpec extends ChiselPropSpec {
  property("All complex assignments should return the correct result") {
    forAll(enSequence(2), safeUInts, safeUInts) { (en: List[Boolean], re: Int, im: Int) =>
      assertTesterPasses { new ComplexAssignTester(en, re, im) }
    }
  }
}
