// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class Tbl(w: Int, n: Int) extends Module {
  val io = IO(new Bundle {
    val wi  = Input(UInt(log2Ceil(n).W))
    val ri  = Input(UInt(log2Ceil(n).W))
    val we  = Input(Bool())
    val  d  = Input(UInt(w.W))
    val  o  = Output(UInt(w.W))
  })
  val m = Mem(n, UInt(w.W))
  io.o := m(io.ri)
  when (io.we) {
    m(io.wi) := io.d
    when(io.ri === io.wi) {
      io.o := io.d
    }
  }
}

class TblTester(w: Int, n: Int, idxs: List[Int], values: List[Int]) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, idxs.size)
  val dut = Module(new Tbl(w, n))
  val vvalues = VecInit(values.map(_.asUInt))
  val vidxs = VecInit(idxs.map(_.asUInt))
  val prev_idx = vidxs(cnt - 1.U)
  val prev_value = vvalues(cnt - 1.U)
  dut.io.wi := vidxs(cnt)
  dut.io.ri := prev_idx
  dut.io.we := true.B //TODO enSequence
  dut.io.d := vvalues(cnt)
  when (cnt > 0.U) {
    when (prev_idx === vidxs(cnt)) {
      assert(dut.io.o === vvalues(cnt))
    } .otherwise {
      assert(dut.io.o === prev_value)
    }
  }
  when(wrap) {
    stop()
  }
}

class TblSpec extends ChiselPropSpec {
  property("All table reads should return the previous write") {
    forAll(safeUIntPairN(8)) { case(w: Int, pairs: List[(Int, Int)]) =>
      // Provide an appropriate whenever clause.
      // ScalaTest will try and shrink the values on error to determine the smallest values that cause the error.
      whenever(w > 0 && pairs.length > 0) {
        val (idxs, values) = pairs.unzip
        assertTesterPasses{ new TblTester(w, 1 << w, idxs, values) }
      }
    }
  }
}
