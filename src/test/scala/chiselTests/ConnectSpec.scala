// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{FixedPoint, Analog}
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
    intercept[ChiselException]{ new CrossConnectTester(SInt(16.W), UInt(16.W)) }
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

class FigureCrossCheckSpec extends ChiselPropSpec {
  val typesToCheck = List(
    ("uint",  UInt(16.W)),
    ("sint",  SInt(16.W)),
    ("fp",    FixedPoint(16.W, 8.BP)),
    ("ana",   Analog(16.W)),
    ("bool",  Bool())
  )

  for {
    (k1, f1) <- typesToCheck
    (k2, f2) <- typesToCheck
  } {
    property(s"checking connect for $k1 := $k2") {
      try {
        assertTesterPasses(new CrossConnectTester(f1, f2))
        println(s"Can connect $k1 := $k2")
      }
      catch {
        case c: ChiselException =>
          println(s"Got a chisel exception for $k1 := $k2")
        case c: Throwable =>
          println(s"Got a ${c.getClass.getName} for $k1 := $k2")
      }
    }
  }
}
