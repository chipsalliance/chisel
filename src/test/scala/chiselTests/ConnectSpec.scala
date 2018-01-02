// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{Analog, FixedPoint}
import chisel3.testers.BasicTester

abstract class CrossCheck extends Bundle {
  val in: Data
  val out: Data
}

class CrossConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Input(inType)
    val out = Output(outType)
  })
  io.out := io.in
}

class PipeInternalWires extends Module {
  import chisel3.util.Pipe
  val io = IO(new Bundle {
    val a = Input(Bool())
    val b = Input(UInt(32.W))
  })
  val pipe = Module(new Pipe(UInt(32.W), 32))
  pipe.io.enq.valid <> io.a
  pipe.io.enq.bits <> io.b
}

class CrossConnectTester(inType: Data, outType: Data) extends BasicTester {
  val dut = Module(new CrossConnects(inType, outType))
  dut.io := DontCare
  stop()
}

class ConnectSpec extends ChiselPropSpec {
  property("SInt := SInt should succeed") {
    assertTesterPasses{ new CrossConnectTester(SInt(16.W), SInt(16.W)) }
  }
  property("SInt := UInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(UInt(16.W), SInt(16.W)) }
  }
  property("SInt := FixedPoint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
  }
  property("UInt := UInt should succeed") {
    assertTesterPasses{ new CrossConnectTester(UInt(16.W), UInt(16.W)) }
  }
  property("UInt := SInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), UInt(16.W)) }
  }
  property("UInt := FixedPoint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
  }

  property("Clock := Clock should succeed") {
    assertTesterPasses{ new CrossConnectTester(Clock(), Clock()) }
  }
  property("Clock := UInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(Clock(), UInt(16.W)) }
  }

  property("FixedPoint := FixedPoint should succeed") {
    assertTesterPasses{ new CrossConnectTester(FixedPoint(16.W, 8.BP), FixedPoint(16.W, 8.BP)) }
  }
  property("FixedPoint := SInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), FixedPoint(16.W, 8.BP)) }
  }
  property("FixedPoint := UInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(UInt(16.W), FixedPoint(16.W, 8.BP)) }
  }

  property("Analog := Analog should fail") {
    intercept[ChiselException]{ new CrossConnectTester(Analog(16.W), Analog(16.W)) }
  }
  property("Analog := FixedPoint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(Analog(16.W), FixedPoint(16.W, 8.BP)) }
  }
  property("FixedPoint := Analog should fail") {
    intercept[ChiselException]{ new CrossConnectTester(FixedPoint(16.W, 8.BP), Analog(16.W)) }
  }
  property("Analog := UInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(Analog(16.W), UInt(16.W)) }
  }
  property("Analog := SInt should fail") {
    intercept[ChiselException]{ new CrossConnectTester(Analog(16.W), SInt(16.W)) }
  }
  property("UInt := Analog should fail") {
    intercept[ChiselException]{ new CrossConnectTester(UInt(16.W), Analog(16.W)) }
  }
  property("SInt := Analog should fail") {
    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), Analog(16.W)) }
  }
  property("Pipe internal connections should succeed") {
    elaborate( new PipeInternalWires)
  }
}
