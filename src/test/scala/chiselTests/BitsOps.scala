package Chisel
import Chisel.testers._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.prop.GeneratorDrivenPropertyChecks._

class BitwiseOps(w: Int) extends Module {
  val io = new Bundle {
    val a = Bits(INPUT, w)
    val b = Bits(INPUT, w)
    val not = Bits(OUTPUT, w)
    val and = Bits(OUTPUT, w)
    val or  = Bits(OUTPUT, w)
    val xor = Bits(OUTPUT, w)
  }
  io.not := ~io.a
  io.and := io.a & io.b
  io.or := io.a | io.b
  io.xor := io.a ^ io.b
}

class BitwiseOpsSpec extends ChiselSpec {

  class BitwiseOpsTester(w: Int, a: Int, b: Int) extends BasicTester {
    val mask = (1 << w)-1;
    val dut = Module(new BitwiseOps(w))
    io.done := Bool(true)
    dut.io.a := UInt(a) 
    dut.io.b := UInt(b)
    when(dut.io.not != UInt(mask & ~a)) { io.error := UInt(1) }
    when(dut.io.and != UInt(mask & (a & b))) { io.error := UInt(2) }
    when(dut.io.or  != UInt(mask & (a | b))) { io.error := UInt(3) }
    when(dut.io.xor != UInt(mask & (a ^ b))) { io.error := UInt(4) }
  }

  "BitwiseOps" should "return the correct result" in {
    forAll(safeUInts, safeUInts) { (a: Int, b: Int) =>
      assert(TesterDriver.execute{ new BitwiseOpsTester(32, a, b) }) 
    }
  }
}
