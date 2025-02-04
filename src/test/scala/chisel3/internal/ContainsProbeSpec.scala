package chisel3.internal

import chisel3._
import chisel3.probe._
import chisel3.util.DecoupledIO
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage
import chiselTests.ChiselFunSpec

class ContainsProbeSpec extends ChiselFunSpec {
  it("1. should return false for elements that dont"){
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

  it("3. should return false for aggregates"){
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
     }
    ChiselStage.emitCHIRRTL(new TestMod)
  }
  it("5. return true for aggregates that contain probes"){
    class TestMod extends RawModule {

      val a = IO(DecoupledIO(Probe(Bool())))
      val b = DecoupledIO(Probe(Bool()))
      val c = IO(Vec(3, Probe(Bool())))
      val d = Vec(3, Probe(Bool()))

      require(containsProbe(a))
      require(containsProbe(b))
      require(containsProbe(c))
     }
    ChiselStage.emitCHIRRTL(new TestMod)
  }
}