// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{Analog, FixedPoint}
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester

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

class CrossStrictConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
    val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
  })
  io.out :<>= io.in
}

class NotCommutativeCrossStrictConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
    val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
  })
  io.in :<>= io.out
}

class NotWritableCrossStrictConnects(inType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
  })
  val tmp = Wire(Flipped(inType))
  io.in :<>= tmp
}

class CrossStrictConnectsWithWires(inType: Data, outType: Data, nTmps: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
    val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
  })
  require(nTmps > 0)
  val wiresIn = Seq.fill(nTmps)(Wire(inType))
  val wiresOut = Seq.fill(nTmps)(Wire(outType))
  (Seq(io.out) ++ wiresOut ++ wiresIn).zip(wiresOut ++ wiresIn :+ io.in).foreach {
    case (l, r) =>
      l :<>= r
  }
}

class ConnectSpec extends ChiselPropSpec with Utils {
  property("SInt := SInt should succeed") {
    assertTesterPasses { new CrossConnectTester(SInt(16.W), SInt(16.W)) }
  }
  property("SInt := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(UInt(16.W), SInt(16.W)) }
      }
    }
  }
  property("SInt := FixedPoint should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
      }
    }
  }
  property("UInt := UInt should succeed") {
    assertTesterPasses { new CrossConnectTester(UInt(16.W), UInt(16.W)) }
  }
  property("UInt := SInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(SInt(16.W), UInt(16.W)) }
      }
    }
  }
  property("UInt := FixedPoint should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(FixedPoint(16.W, 8.BP), UInt(16.W)) }
      }
    }
  }

  property("Clock := Clock should succeed") {
    assertTesterPasses { new CrossConnectTester(Clock(), Clock()) }
  }
  property("Clock := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(Clock(), UInt(16.W)) }
      }
    }
  }

  property("FixedPoint := FixedPoint should succeed") {
    assertTesterPasses { new CrossConnectTester(FixedPoint(16.W, 8.BP), FixedPoint(16.W, 8.BP)) }
  }
  property("FixedPoint := SInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(SInt(16.W), FixedPoint(16.W, 8.BP)) }
      }
    }
  }
  property("FixedPoint := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(UInt(16.W), FixedPoint(16.W, 8.BP)) }
      }
    }
  }

  property("Analog := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(Analog(16.W), Analog(16.W)) }
      }
    }
  }
  property("Analog := FixedPoint should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(Analog(16.W), FixedPoint(16.W, 8.BP)) }
      }
    }
  }
  property("FixedPoint := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(FixedPoint(16.W, 8.BP), Analog(16.W)) }
      }
    }
  }
  property("Analog := UInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(Analog(16.W), UInt(16.W)) }
      }
    }
  }
  property("Analog := SInt should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(Analog(16.W), SInt(16.W)) }
      }
    }
  }
  property("UInt := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(UInt(16.W), Analog(16.W)) }
      }
    }
  }
  property("SInt := Analog should fail") {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.elaborate { new CrossConnectTester(SInt(16.W), Analog(16.W)) }
      }
    }
  }
  property("Pipe internal connections should succeed") {
    ChiselStage.elaborate(new PipeInternalWires)
  }

  property("Connect error messages should have meaningful information") {
    class InnerExample extends Module {
      val myReg = RegInit(0.U(8.W))
    }

    class OuterAssignExample extends Module {
      val inner = Module(new InnerExample())
      inner.myReg := false.B // ERROR
    }

    val assignError = the[ChiselException] thrownBy { ChiselStage.elaborate { new OuterAssignExample } }
    val expectedAssignError = """.*@: myReg in InnerExample cannot be written from module OuterAssignExample."""
    (assignError.getMessage should fullyMatch).regex(expectedAssignError)

    class OuterReadExample extends Module {
      val myReg = RegInit(0.U(8.W))
      val inner = Module(new InnerExample())
      myReg := inner.myReg // ERROR
    }

    val readError = the[ChiselException] thrownBy { ChiselStage.elaborate { new OuterReadExample } }
    val expectedReadError = """.*@: myReg in InnerExample cannot be read from module OuterReadExample."""
    (readError.getMessage should fullyMatch).regex(expectedReadError)

    val typeMismatchError = the[ChiselException] thrownBy {
      ChiselStage.elaborate {
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

  // (D)irectional Bulk Connect tests
  property("(D.a) SInt :<>= SInt should succeed") {
    ChiselStage.elaborate { new CrossStrictConnects(SInt(16.W), SInt(16.W)) }
  }
  property("(D.b) UInt :<>= UInt should succeed") {
    ChiselStage.elaborate { new CrossStrictConnects(UInt(16.W), UInt(16.W)) }
  }
  property("(D.c) SInt :<>= UInt should fail") {
    intercept[ChiselException] { ChiselStage.elaborate { new CrossStrictConnects(UInt(16.W), SInt(16.W)) } }
  }
  property("(D.d) Decoupled :<>= Decoupled should succeed") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl { new CrossStrictConnects(new Decoupled, new Decoupled) }
    assert(out.contains("io.out <= io.in"))
  }
  property("(D.d) Aggregates with same-named fields should succeed") {
    class Foo extends Bundle {
      val foo = Bool()
      val bar = Flipped(Bool())
    }
    class FooLike extends Bundle {
      val foo = Bool()
      val bar = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl { new CrossStrictConnects(new Foo, new FooLike) }
    assert(out.contains("io.out <= io.in"))
  }
  property("(D.d) Decoupled[Foo] :<>= Decoupled[Foo-Like] should succeed") {
    class Foo extends Bundle {
      val foo = Bool()
      val bar = Flipped(Bool())
    }
    class FooLike extends Bundle {
      val foo = Bool()
      val bar = Flipped(Bool())
    }
    class Decoupled[T <: Data](gen: => T) extends Bundle {
      val bits = gen
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl {
      new CrossStrictConnects(new Decoupled(new Foo()), new Decoupled(new FooLike()))
    }
    assert(out.contains("io.out <= io.in"))
  }
  property("(D.e) different relative flips, but same absolute flippage is an error") {
    class X(yflip: Boolean, zflip: Boolean) extends Bundle {
      val y = if (yflip) Flipped(new Y(zflip)) else new Y(zflip)
    }
    class Y(flip: Boolean) extends Bundle {
      val z = if (flip) Flipped(Bool()) else Bool()
    }
    intercept[ChiselException] {
      ChiselStage.emitVerilog { new CrossStrictConnects(new X(true, false), new X(false, true)) }
    }
  }
  property("(D.f) :<>= is not commutative.") {
    intercept[ChiselException] {
      ChiselStage.elaborate { new NotCommutativeCrossStrictConnects(UInt(16.W), UInt(16.W)) }
    }
  }
  property("(D.g) UInt :<>= UInt should succeed with intermediate Wires") {
    ChiselStage.elaborate { new CrossStrictConnectsWithWires(UInt(16.W), UInt(16.W), 1) }
  }
  property("(D.h) Decoupled :<>= Decoupled should succeed with intermediate Wires") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl { new CrossStrictConnectsWithWires(new Decoupled, new Decoupled, 2) }
    assert(out.contains("io.out <= wiresOut_0"))
    assert(out.contains("wiresOut_1 <= wiresIn_0"))
    assert(out.contains("wiresIn_1 <= io.in"))
  }
  property("(D.i) Aggregates :<>= with missing fields should not succeed, no matter the direction.") {
    class Foo extends Bundle {
      val foo = Bool()
    }
    class FooBar extends Bundle {
      val foo = Bool()
      val bar = Bool()
    }
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossStrictConnects(new Foo(), new FooBar()) }
    }
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossStrictConnects(new FooBar(), new Foo()) }
    }
  }
  property("(D.j) Cannot :<>= to something that is not writable.") {

    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new NotWritableCrossStrictConnects(UInt(16.W)) }
    }
  }
  property("(D.k) Can :<>= to Vecs of the same length") {
    val out = (new ChiselStage).emitChirrtl { new CrossStrictConnects(Vec(3, UInt(16.W)), Vec(3, UInt(16.W))) }
    assert(out.contains("io.out <= io.in"))
  }
  property("(D.l)  :<>= between Vecs of different same length should not succeed, no matter the direction") {
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossStrictConnects(Vec(2, UInt(16.W)), Vec(3, UInt(16.W))) }
    }
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossStrictConnects(Vec(3, UInt(16.W)), Vec(2, UInt(16.W))) }
    }

  }

}
