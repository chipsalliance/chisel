// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import chisel3.experimental.{Analog, attach, BaseModule}

// IO for Modules that just connect bus to out
class AnalogReaderIO extends Bundle {
  val bus = Analog(32.W)
  val out = Output(UInt(32.W))
}
// IO for Modules that drive bus from in (there should be only 1)
class AnalogWriterIO extends Bundle {
  val bus = Analog(32.W)
  val in = Input(UInt(32.W))
}

trait AnalogReader {
  self: BaseModule =>
  final val io = self.IO(new AnalogReaderIO)
}

class AnalogReaderBlackBox extends BlackBox with AnalogReader

class AnalogReaderWrapper extends Module with AnalogReader {
  val mod = Module(new AnalogReaderBlackBox)
  io <> mod.io
}
class AnalogWriterBlackBox extends BlackBox {
  val io = IO(new AnalogWriterIO)
}
// Connects two Analog ports
class AnalogConnector extends Module {
  val io = IO(new Bundle {
    val bus1 = Analog(32.W)
    val bus2 = Analog(32.W)
  })
  io.bus1 <> io.bus2
}

// Parent class for tests connecing up AnalogReaders and AnalogWriters
abstract class AnalogTester extends BasicTester {
  final val BusValue = "hdeadbeef".U

  final val (cycle, done) = Counter(true.B, 2)
  when (done) { stop() }

  final val writer = Module(new AnalogWriterBlackBox)
  writer.io.in := BusValue

  final def check(reader: BaseModule with AnalogReader): Unit =
    assert(reader.io.out === BusValue)
}

class AnalogSpec extends ChiselFlatSpec {
  behavior of "Analog"

  it should "NOT be bindable to registers" in {
    a [ChiselException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {})
        val reg = Reg(Analog(32.W))
      }}
    }
  }

  it should "NOT be bindable to a direction" in {
    a [ChiselException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {
          val a = Input(Analog(32.W))
        })
      }}
    }
    a [ChiselException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {
          val a = Output(Analog(32.W))
        })
      }}
    }
  }

  it should "be flippable" in {
    elaborate { new Module {
      val io = IO(new Bundle {
        val a = Flipped(Analog(32.W))
      })
    }}
  }

  // There is no binding on the type of a memory
  // Should this be an error?
  ignore should "NOT be a legal type for Mem" in {
    a [ChiselException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {})
        val mem = Mem(16, Analog(32.W))
      }}
    }
  }

  it should "NOT be bindable to Mem ports" in {
    a [ChiselException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {})
        val mem = Mem(16, Analog(32.W))
        val port = mem(5.U)
      }}
    }
  }

  // TODO This should probably be caught in Chisel
  // Also note this relies on executing Firrtl from Chisel directly
  it should "NOT be connectable to UInts" in {
    a [Exception] should be thrownBy {
      runTester { new BasicTester {
        val uint = Wire(init = 0.U(32.W))
        val sint = Wire(Analog(32.W))
        sint := uint
      }}
    }
  }

  it should "work with 2 blackboxes bulk connected" in {
    assertTesterPasses(new AnalogTester {
      val mod = Module(new AnalogReaderBlackBox)
      mod.io.bus <> writer.io.bus
      check(mod)
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  it should "error if any bulk connected more than once" in {
    a [ChiselException] should be thrownBy {
      elaborate(new Module {
        val io = IO(new Bundle {})
        val wires = List.fill(3)(Wire(Analog(32.W)))
        wires(0) <> wires(1)
        wires(0) <> wires(2)
      })
    }
  }

  it should "work with 3 blackboxes attached" in {
    assertTesterPasses(new AnalogTester {
      val mods = Seq.fill(2)(Module(new AnalogReaderBlackBox))
      attach(writer.io.bus, mods(0).io.bus, mods(1).io.bus)
      mods.foreach(check(_))
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  it should "work with 3 blackboxes separately attached via a wire" in {
    assertTesterPasses(new AnalogTester {
      val mods = Seq.fill(2)(Module(new AnalogReaderBlackBox))
      val busWire = Wire(Analog(32.W))
      attach(busWire, writer.io.bus)
      attach(busWire, mods(0).io.bus)
      attach(mods(1).io.bus, busWire)
      mods.foreach(check(_))
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  // This does not currently work in Verilator unless Firrtl does constant prop and dead code
  // elimination on these wires
  ignore should "work with intermediate wires attached to each other" in {
    assertTesterPasses(new AnalogTester {
      val mod = Module(new AnalogReaderBlackBox)
      val busWire = Seq.fill(2)(Wire(Analog(32.W)))
      attach(busWire(0), writer.io.bus)
      attach(busWire(1), mod.io.bus)
      attach(busWire(0), busWire(1))
      check(mod)
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  it should "work with blackboxes at different levels of the module hierarchy" in {
    assertTesterPasses(new AnalogTester {
      val mods = Seq(Module(new AnalogReaderBlackBox), Module(new AnalogReaderWrapper))
      val busWire = Wire(writer.io.bus.cloneType)
      attach(writer.io.bus, mods(0).io.bus, mods(1).io.bus)
      mods.foreach(check(_))
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  // This does not currently work in Verilator, but does work in VCS
  ignore should "support two analog ports in the same module" in {
    assertTesterPasses(new AnalogTester {
      val reader = Module(new AnalogReaderBlackBox)
      val connector = Module(new AnalogConnector)
      connector.io.bus1 <> writer.io.bus
      reader.io.bus <> connector.io.bus2
      check(reader)
    }, Seq("/chisel3/AnalogBlackBox.v"))
  }

  it should "NOT support conditional connection of analog types" in {
    a [ChiselException] should be thrownBy {
      assertTesterPasses(new AnalogTester {
        val mod = Module(new AnalogReaderBlackBox)
        when (cycle > 3.U) {
          mod.io.bus <> writer.io.bus
        }
        check(mod)
      }, Seq("/chisel3/AnalogBlackBox.v"))
    }
  }
}

