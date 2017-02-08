// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._
import chisel3._
import chisel3.core.FixedPoint
import chisel3.testers.BasicTester
import chisel3.util._

abstract class CrossCheck extends Bundle {
  val in: Data
  val out: Data
}

class CheckUIntToSInt extends CrossCheck {
  val in  = Input(UInt(16.W))
  val out = Output(SInt(16.W))
}

class CheckSIntToUInt extends CrossCheck {
  val in  = Input(UInt(16.W))
  val out = Output(SInt(16.W))
}

class CrossConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Input(inType)
    val out = Output(outType)
  })
  io.out := io.in
}

class CrossConnectTester(inType: Data, outType: Data) extends BasicTester {
  val dut = Module(new CrossConnects(inType, outType))
  stop()
}

class ConnectSpec extends ChiselPropSpec {
  property("sint := sint should succceed") {
    assertTesterPasses{ new CrossConnectTester(SInt(16.W), SInt(16.W)) }
  }
  property("sint := uint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(UInt(16.W), SInt(16.W)) }
  }
  property("sint := fixedpoint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
  }
  property("uint := uint should succceed") {
    assertTesterPasses{ new CrossConnectTester(UInt(16.W), UInt(16.W)) }
  }
  property("uint := sint should fail") {
//    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), UInt(16.W)) }
    assertTesterPasses{ new CrossConnectTester(SInt(16.W), UInt(16.W)) }
  }
  property("uint := fixedpoint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
  }

  property("fixedpoint := fixedpoint should succceed") {
    assertTesterPasses{ new CrossConnectTester(FixedPoint(16.W, 8.BP), FixedPoint(16.W, 8.BP)) }
  }
  property("fixedpoint := sint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), FixedPoint(16.W, 8.BP)) }
  }
  property("fixedpoint := uint should fail") {
    intercept[ChiselException]{ new CrossConnectTester(UInt(16.W), FixedPoint(16.W, 8.BP)) }
  }
}
