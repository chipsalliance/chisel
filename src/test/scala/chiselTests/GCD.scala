package Chisel
import Chisel.testers._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.prop.TableDrivenPropertyChecks._

class GCD extends Module {
  val io = new Bundle {
    val a  = Bits(INPUT,  16)
    val b  = Bits(INPUT,  16)
    val e  = Bool(INPUT)
    val z  = Bits(OUTPUT, 16)
    val v  = Bool(OUTPUT)
  }
  val x = Reg(Bits(width = 16))
  val y = Reg(Bits(width = 16))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === Bits(0)
}

class GCDSpec extends ChiselSpec {

  class GCDTester(a: Int, b: Int, z: Int) extends BasicTester {
    val dut = Module(new GCD)
    val first = Reg(init=Bool(true))
    dut.io.a := UInt(a)
    dut.io.b := UInt(b)
    dut.io.e := first
    when(first) { first := Bool(false) }
    when(dut.io.v) {
      io.done := Bool(true)
      io.error := (dut.io.z != UInt(z)).toUInt
    }
  }
  
  val gcds = Table(
    ("a", "b", "z"),  // First tuple defines column names
    ( 64,  48,  16),  // Subsequent tuples define the data
    ( 12,   9,   3),
    ( 48,  64,  12))

  "GCD" should "return the correct result" in {
    forAll (gcds) { (a: Int, b: Int, z: Int) => 
      assert(TesterDriver.execute{ new GCDTester(a, b, z) })
    }
  }
}
