// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.Analog
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

abstract class CrossCheck extends Bundle {
  val in:  Data
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

class ConnectSpec extends ChiselPropSpec with Utils {
  property("SInt := SInt should succeed") {
    assertTesterPasses { new CrossConnectTester(SInt(16.W), SInt(16.W)) }
  }
  property("SInt := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(UInt(16.W), SInt(16.W)) }
      }
    }
  }
  property("UInt := UInt should succeed") {
    assertTesterPasses { new CrossConnectTester(UInt(16.W), UInt(16.W)) }
  }
  property("UInt := SInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(SInt(16.W), UInt(16.W)) }
      }
    }
  }
  property("Clock := Clock should succeed") {
    assertTesterPasses { new CrossConnectTester(Clock(), Clock()) }
  }
  property("Clock := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(Clock(), UInt(16.W)) }
      }
    }
  }

  property("Analog := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(Analog(16.W), Analog(16.W)) }
      }
    }
  }
  property("Analog := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(Analog(16.W), UInt(16.W)) }
      }
    }
  }
  property("Analog := SInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(Analog(16.W), SInt(16.W)) }
      }
    }
  }
  property("UInt := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(UInt(16.W), Analog(16.W)) }
      }
    }
  }
  property("SInt := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new CrossConnectTester(SInt(16.W), Analog(16.W)) }
      }
    }
  }
  property("Pipe internal connections should succeed") {
    ChiselStage.emitCHIRRTL(new PipeInternalWires)
  }

  property("Connect error messages should have meaningful information") {
    class InnerExample extends Module {
      val myReg = RegInit(0.U(8.W))
    }

    class OuterAssignExample extends Module {
      val inner = Module(new InnerExample())
      inner.myReg := false.B // ERROR
    }

    val assignError = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL { new OuterAssignExample } }
    val expectedAssignError = """.*@: myReg in InnerExample cannot be written from module OuterAssignExample."""
    (assignError.getMessage should fullyMatch).regex(expectedAssignError)

    class OuterReadExample extends Module {
      val myReg = RegInit(0.U(8.W))
      val inner = Module(new InnerExample())
      myReg := inner.myReg // ERROR
    }

    val readError = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL { new OuterReadExample } }
    val expectedReadError = """.*@: myReg in InnerExample cannot be read from module OuterReadExample."""
    (readError.getMessage should fullyMatch).regex(expectedReadError)

    val typeMismatchError = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL {
        new RawModule {
          val myUInt = Wire(UInt(4.W))
          val mySInt = Wire(SInt(4.W))
          myUInt := mySInt
        }
      }
    }
    val expectedTypeMismatchError = """.*@: Sink \(UInt<4>\) and Source \(SInt<4>\) have different types."""
    (typeMismatchError.getMessage should fullyMatch).regex(expectedTypeMismatchError)
  }
}
