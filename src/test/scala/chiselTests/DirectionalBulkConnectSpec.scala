// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{Analog, FixedPoint}
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester

class CrossDirectionalBulkConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
    val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
  })
  io.out :<>= io.in
}

class NotCommutativeCrossDirectionalBulkConnects(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
    val out = Flipped(Flipped(outType)) // no clonetype, no Aligned (yet)
  })
  io.in :<>= io.out
}

class NotWritableCrossDirectionalBulkConnects(inType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(inType)
  })
  val tmp = Wire(Flipped(inType))
  io.in :<>= tmp
}

class CrossDirectionalBulkConnectsWithWires(inType: Data, outType: Data, nTmps: Int) extends Module {
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

class DirectionalBulkConnectSpec extends ChiselPropSpec with Utils {

  // (D)irectional Bulk Connect tests
  property("(D.a) SInt :<>= SInt should succeed") {
    ChiselStage.elaborate { new CrossDirectionalBulkConnects(SInt(16.W), SInt(16.W)) }
  }
  property("(D.b) UInt :<>= UInt should succeed") {
    ChiselStage.elaborate { new CrossDirectionalBulkConnects(UInt(16.W), UInt(16.W)) }
  }
  property("(D.c) SInt :<>= UInt should fail") {
    intercept[ChiselException] { ChiselStage.elaborate { new CrossDirectionalBulkConnects(UInt(16.W), SInt(16.W)) } }
  }
  property("(D.d) Decoupled :<>= Decoupled should succeed") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Decoupled, new Decoupled) }
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
    val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Foo, new FooLike) }
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
      new CrossDirectionalBulkConnects(new Decoupled(new Foo()), new Decoupled(new FooLike()))
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
      ChiselStage.emitVerilog { new CrossDirectionalBulkConnects(new X(true, false), new X(false, true)) }
    }
  }
  property("(D.f) :<>= is not commutative.") {
    intercept[ChiselException] {
      ChiselStage.elaborate { new NotCommutativeCrossDirectionalBulkConnects(UInt(16.W), UInt(16.W)) }
    }
  }
  property("(D.g) UInt :<>= UInt should succeed with intermediate Wires") {
    ChiselStage.elaborate { new CrossDirectionalBulkConnectsWithWires(UInt(16.W), UInt(16.W), 1) }
  }
  property("(D.h) Decoupled :<>= Decoupled should succeed with intermediate Wires") {
    class Decoupled extends Bundle {
      val bits = UInt(3.W)
      val valid = Bool()
      val ready = Flipped(Bool())
    }
    val out = ChiselStage.emitChirrtl { new CrossDirectionalBulkConnectsWithWires(new Decoupled, new Decoupled, 2) }
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
      ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new Foo(), new FooBar()) }
    }
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(new FooBar(), new Foo()) }
    }
  }
  property("(D.j) Cannot :<>= to something that is not writable.") {

    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new NotWritableCrossDirectionalBulkConnects(UInt(16.W)) }
    }
  }
  property("(D.k) Can :<>= to Vecs of the same length") {
    val out = (new ChiselStage).emitChirrtl { new CrossDirectionalBulkConnects(Vec(3, UInt(16.W)), Vec(3, UInt(16.W))) }
    assert(out.contains("io.out <= io.in"))
  }
  property("(D.l) :<>= between Vecs of different length should not succeed, no matter the direction") {
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(Vec(2, UInt(16.W)), Vec(3, UInt(16.W))) }
    }
    intercept[ChiselException] {
      ChiselStage.emitChirrtl { new CrossDirectionalBulkConnects(Vec(3, UInt(16.W)), Vec(2, UInt(16.W))) }
    }

  }
  property(
    "(D.m) :<>= is NOT equivalent to Chisel.:= in that  `A Module with missing bundle fields when compiled with the Chisel compatibility package` *should* `throw an exception` "
  ) {
    // This is copied from CompatibilitySpec but the := is replaced with :<>=
    class SmallBundle extends Bundle {
      val f1 = UInt(4.W)
      val f2 = UInt(5.W)
    }
    class BigBundle extends SmallBundle {
      val f3 = UInt(6.W)
    }

    class ConnectFieldMismatchModule extends Module {
      val io = IO(new Bundle {
        val in = Input((new SmallBundle))
        val out = Output((new BigBundle))
      })
      io.out :<>= io.in
    }
    intercept[ChiselException] {
      ChiselStage.elaborate { new ConnectFieldMismatchModule() }
    }
  }

  property("(D.n) :<>= works with DataView to connect a bundle that is a subtype") {
    import chisel3.experimental.dataview._

    class SmallBundle extends Bundle {
      val f1 = UInt(4.W)
      val f2 = UInt(5.W)
    }
    class BigBundle extends SmallBundle {
      val f3 = UInt(6.W)
    }

    class ConnectSupertype extends Module {
      val io = IO(new Bundle {
        val in = Input((new SmallBundle))
        val out = Output((new BigBundle))
      })
      io.out := DontCare
      io.out.viewAsSupertype(Output(new SmallBundle)) :<>= io.in
    }
    val out = (new ChiselStage).emitChirrtl(gen = new ConnectSupertype(), args = Array("--full-stacktrace"))
    assert(out.contains("io.out.f1 <= io.in.f1"))
    assert(out.contains("io.out.f2 <= io.in.f2"))
    assert(!out.contains("io.out.f3 <= io.in.f3"))
    assert(!out.contains("io.out <= io.in"))
    assert(!out.contains("io.out <- io.in"))
  }
  property("(D.o) :<>= works with DataView to connect a two Bundles with a common trait") {
    import chisel3.experimental.dataview._

    class SmallBundle extends Bundle {
      val common = Input(UInt(4.W))
    }
    class BigA extends SmallBundle {
      val a = Output(UInt(6.W))
    }
    class BigB extends SmallBundle {
      val b = Input(UInt(6.W))
    }

    class ConnectCommonTrait extends Module {
      val io = IO(new Bundle {
        val in = (new BigA)
        val out = Flipped((new BigB))
      })
      io.in := DontCare
      io.out := DontCare
      io.out.viewAsSupertype(Flipped(new SmallBundle)) :<>= io.in.viewAsSupertype(new SmallBundle)
    }
    val out = ChiselStage.emitChirrtl { new ConnectCommonTrait() }
    assert(!out.contains("io.out <= io.in"))
    assert(!out.contains("io.out <- io.in"))
    assert(out.contains("io.out.common <= io.in.common"))
    assert(!out.contains("io.out.b <= io.in.b"))
    assert(!out.contains("io.in.a  <= io.out.a"))
  }

}
