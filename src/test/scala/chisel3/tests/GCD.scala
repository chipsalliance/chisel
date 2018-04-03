// See LICENSE for license details.

package chisel3.tests

import chisel3._
import chisel3.tester._
import org.scalatest._
import org.scalatest.prop._

class GCD extends Module {
  val io = IO(new Bundle {
    val a  = Input(UInt(32.W))
    val b  = Input(UInt(32.W))
    val e  = Input(Bool())
    val z  = Output(UInt(32.W))
    val v  = Output(Bool())
  })
  val x = Reg(UInt(32.W))
  val y = Reg(UInt(32.W))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === 0.U
}

/*
class GCDTester(a: Int, b: Int, z: Int) extends BasicTester {
  val dut = Module(new GCD)
  val first = RegInit(true.B)
  dut.io.a := a.U
  dut.io.b := b.U
  dut.io.e := first
  when(first) { first := false.B }
  when(!first && dut.io.v) {
    assert(dut.io.z === z.U)
    stop()
  }
}
*/

class GCDSpec extends PropSpec with ChiselScalatestTester with PropertyChecks {

  //TODO: use generators and this function to make z's
  def gcd(a: Int, b: Int): Int = if(b == 0) a else gcd(b, a%b)

  val gcds = Table(
    ("a", "b", "z"),  // First tuple defines column names
    ( 64,  48,  16),  // Subsequent tuples define the data
    ( 12,   9,   3),
    ( 48,  64,  16))

/*
  property("GCD should elaborate") {
    elaborate { new GCD }
  }
*/
  private val backendNames = Array[String] ("firrtl")
  for (backendName <- backendNames) {
    val options = new TesterOptionsManager {
      testerOptions = TesterOptions(backendName = backendName)
    }


    property("GCDTester should return the correct result (with $backendName)") {
      forAll (gcds) { (a: Int, b: Int, z: Int) =>
        test(new Module{
  val io = IO(new Bundle {
    val a  = Input(UInt(32.W))
    val b  = Input(UInt(32.W))
    val e  = Input(Bool())
    val z  = Output(UInt(32.W))
    val v  = Output(Bool())
  })
  val x = Reg(UInt(32.W))
  val y = Reg(UInt(32.W))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === 0.U
	}, options) { dut =>
          val first = RegInit(true.B)
          dut.io.a := a.U
          dut.io.b := b.U
          dut.io.e := first
          when(first) { first := false.B }
          when(!first && dut.io.v) {
            dut.io.z.expect(z.U)
          }.otherwise {
            dut.clock.step()
	  }
        }
      }
    }
  }
}
