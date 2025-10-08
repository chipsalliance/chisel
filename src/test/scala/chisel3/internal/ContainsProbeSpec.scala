package chisel3.internal

import chisel3._
import chisel3.probe._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ContainsProbeSpec extends AnyFunSpec with Matchers {
  it("1. should return false for elements that dont") {
    class TestMod extends RawModule {
      val a = IO(Input(Bool()))
      val b = Bool()

      require(!containsProbe(a))
      require(!containsProbe(b))
    }
    ChiselStage.emitCHIRRTL(new TestMod)
  }

  it("2. should return true for elements that do") {
    class TestMod extends RawModule {

      val a = IO(Probe(Bool()))
      val b = Probe(Bool())

      require(containsProbe(a))
      require(containsProbe(b))
    }
    ChiselStage.emitCHIRRTL(new TestMod)

  }

  it("3. should return false for aggregates") {
    class TestMod extends RawModule {

      val a = IO(DecoupledIO(Bool()))
      val b = DecoupledIO(Bool())

      require(!containsProbe(a))
      require(!containsProbe(b))
    }
    ChiselStage.emitCHIRRTL(new TestMod)
  }

  it("4. return true for aggregates that are probes") {
    class TestMod extends RawModule {

      val a = IO(Probe(DecoupledIO(Bool())))
      val b = Probe(DecoupledIO(Bool()))

      require(containsProbe(a))
      require(containsProbe(b))

      val aCopy = IO(chiselTypeOf(a))
      val bCopy = IO(b)
      require(containsProbe(aCopy))
      require(containsProbe(bCopy))
    }
    ChiselStage.emitCHIRRTL(new TestMod)
  }
  it("5. return true for aggregates that contain probes") {
    class TestMod extends RawModule {

      val a = IO(DecoupledIO(Probe(Bool())))
      val b = DecoupledIO(Probe(Bool()))
      val c = IO(Vec(3, Probe(Bool())))
      val d = Vec(3, Probe(Bool()))

      require(containsProbe(a))
      require(containsProbe(b))
      require(containsProbe(c))
      require(containsProbe(d))

      val aCopy = IO(chiselTypeOf(a))
      val bCopy = IO(b)
      val cCopy = IO(chiselTypeOf(c))
      val dCopy = IO(d)
      require(containsProbe(aCopy))
      require(containsProbe(bCopy))
      require(containsProbe(cCopy))
      require(containsProbe(dCopy))
    }
    ChiselStage.emitCHIRRTL(new TestMod)
  }
}
