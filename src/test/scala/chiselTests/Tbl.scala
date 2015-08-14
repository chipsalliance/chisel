package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class Tbl(w: Int, n: Int) extends Module {
  val io = new Bundle {
    val wi  = UInt(INPUT, log2Ceil(w))
    val ri  = UInt(INPUT, log2Ceil(w))
    val we  = Bool(INPUT)
    val  d  = UInt(INPUT, w)
    val  o  = UInt(OUTPUT, w)
  }
  val m = Mem(UInt(width = w), n)
  io.o := m(io.ri)
  when (io.we) { 
    m(io.wi) := io.d
    when(io.ri === io.wi) { io.o := io.d }
  }
}

class TblSpec extends ChiselPropSpec {

  class TblTester(w: Int, n: Int, idxs: List[Int], values: List[Int]) extends BasicTester {
    val (cnt, wrap) = Counter(Bool(true), idxs.size)
    val dut = Module(new Tbl(w, n))
    val vvalues = Vec(values.map(UInt(_)))
    val vidxs = Vec(idxs.map(UInt(_)))
    val prev_idx = vidxs(cnt - UInt(1))
    val prev_value = vvalues(cnt - UInt(1))
    dut.io.wi := vidxs(cnt)
    dut.io.ri := prev_idx
    dut.io.we := Bool(true) //TODO enSequence
    dut.io.d := vvalues(cnt)
    when(cnt > UInt(0) && dut.io.o != prev_value) { io.done := Bool(true); io.error := prev_idx }
    when(wrap) { io.done := Bool(true) }
  }

  property("All table reads should return the previous write") {
    forAll(safeUIntPairN(8)) { case(w: Int, pairs: List[(Int, Int)]) =>
      val (idxs, values) = pairs.unzip
      assert(execute{ new TblTester(w, 1 << w, idxs, values) }) 
    }
  }
}
