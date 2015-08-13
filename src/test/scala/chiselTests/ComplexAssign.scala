package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

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

class ComplexAssignSpec extends ChiselPropSpec {

  class ComplexAssignTester(enList: List[Boolean], re: Int, im: Int) extends BasicTester {
    val (cnt, wrap) = Counter(Bool(true), enList.size)
    val dut = Module(new ComplexAssign(32))
    dut.io.in.re := UInt(re)
    dut.io.in.im := UInt(im)
    dut.io.e := Vec(enList.map(Bool(_)))(cnt)
    val re_correct = dut.io.out.re === Mux(dut.io.e, dut.io.in.re, UInt(0))
    val im_correct = dut.io.out.im === Mux(dut.io.e, dut.io.in.im, UInt(0))
    when(!re_correct || !im_correct) {
      io.done := Bool(true); io.error := cnt 
    } .elsewhen(wrap) { io.done := Bool(true) }
  }
     
  property("All complex assignments should return the correct result") {
    forAll(enSequence(16), safeUInts, safeUInts) { (en: List[Boolean], re: Int, im: Int) =>
      assert(execute{ new ComplexAssignTester(en, re, im) }) 
    }
  }
}
