// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import chisel3.experimental.{Analog, attach, BaseModule, RawModule}

// The RawModule to test
class AnalogPortRawModule(val vec_ports: Int = 4) extends RawModule {
  val s_analog     = IO(Analog(32.W))
  val s_in         = IO(Input(UInt(32.W)))
  val s_out        = IO(Output(UInt(32.W)))
  val v_analog     = IO(Vec(vec_ports, Analog(32.W)))
  val v_in         = IO(Input(Vec(vec_ports, UInt(32.W))))
  val v_out        = IO(Output(Vec(vec_ports, UInt(32.W))))
  
  val sig_wr_mod = Module(new AnalogWriterBlackBox)
  attach(sig_wr_mod.io.bus, s_analog)
  sig_wr_mod.io.in := s_in
  
  val sig_rd_mod = Module(new AnalogReaderBlackBox)
  attach(sig_rd_mod.io.bus, s_analog)
  s_out := sig_rd_mod.io.out
  
  (v_analog zip v_in).foreach{ case( ana, in ) => 
    val vec_wr_mod = Module(new AnalogWriterBlackBox)
    attach(vec_wr_mod.io.bus, ana)
    vec_wr_mod.io.in := in
  }
  
  (v_analog zip v_out).foreach{ case( ana, out ) => 
    val vec_rd_mod = Module(new AnalogReaderBlackBox)
    attach(vec_rd_mod.io.bus, ana)
    out := vec_rd_mod.io.out
  }
}

// The Chisel tester (Just borrowed the AnalogTester)
abstract class AnalogPortRawModuleTester(val vec_ports: Int = 4) extends BasicTester {
  final val BusValue = "hdeadbeef".U

  final val (cycle, done) = Counter(true.B, 2)
  when (done) { stop() }

  final val raw_module = Module(new AnalogPortRawModule(vec_ports))
  raw_module.s_in := BusValue
  raw_module.v_in.foreach{ case x => x := BusValue }

  final def check: Unit = {
    assert(raw_module.s_out === BusValue)
    raw_module.v_out.foreach{ case x => assert( x === BusValue) }
  }
}

class AnalogPortRawModuleSpec extends ChiselFlatSpec {
  behavior of "AnalogPortRawModule"
  
  it should "work with 1 vector bulk connected" in {
    assertTesterPasses(new AnalogPortRawModuleTester(1) {
      check
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }
  
  it should "work with 2 vector bulk connected" in {
    assertTesterPasses(new AnalogPortRawModuleTester(2) {
      check
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }
  
  it should "work with 3 vector bulk connected" in {
    assertTesterPasses(new AnalogPortRawModuleTester(3) {
      check
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }
}
