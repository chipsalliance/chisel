// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util._
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.experimental.{attach, Analog, BaseModule}

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
  def out: UInt
  def bus: Analog
}

class AnalogReaderBlackBox extends BlackBox with AnalogReader {
  val io = IO(new AnalogReaderIO)
  def out = io.out
  def bus = io.bus
}

class AnalogReaderWrapper extends Module with AnalogReader {
  val io = IO(new AnalogReaderIO)
  def out = io.out
  def bus = io.bus
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

class VecAnalogReaderWrapper extends RawModule with AnalogReader {
  val vecbus = IO(Vec(1, Analog(32.W)))
  val out = IO(Output(UInt(32.W)))
  val mod = Module(new AnalogReaderBlackBox)
  def bus = vecbus(0)
  mod.io.bus <> bus
  out := mod.io.out
}

class VecBundleAnalogReaderWrapper extends RawModule with AnalogReader {
  val vecBunBus = IO(
    Vec(
      1,
      new Bundle {
        val analog = Analog(32.W)
      }
    )
  )
  def bus = vecBunBus(0).analog
  val out = IO(Output(UInt(32.W)))
  val mod = Module(new AnalogReaderBlackBox)
  mod.io.bus <> bus
  out := mod.io.out
}

// Parent class for tests connecing up AnalogReaders and AnalogWriters
abstract class AnalogTester extends BasicTester {
  final val BusValue = "hdeadbeef".U

  final val (cycle, done) = Counter(true.B, 2)
  when(done) { stop() }

  final val writer = Module(new AnalogWriterBlackBox)
  writer.io.in := BusValue

  final def check(reader: BaseModule with AnalogReader): Unit =
    assert(reader.out === BusValue)
}

class AnalogSpec extends ChiselFlatSpec with Utils {
  behavior.of("Analog")

  it should "NOT be bindable to registers" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {})
          val reg = Reg(Analog(32.W))
        }
      }
    }
  }

  it should "NOT be bindable to a direction" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {
            val a = Input(Analog(32.W))
          })
        }
      }
    }
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {
            val a = Output(Analog(32.W))
          })
        }
      }
    }
  }

  it should "be flippable" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {
          val a = Flipped(Analog(32.W))
        })
      }
    }
  }

  // There is no binding on the type of a memory
  // Should this be an error?
  ignore should "NOT be a legal type for Mem" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {})
          val mem = Mem(16, Analog(32.W))
        }
      }
    }
  }

  it should "NOT be bindable to Mem ports" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {})
          val mem = Mem(16, Analog(32.W))
          val port = mem(5.U)
        }
      }
    }
  }

  // TODO This should probably be caught in Chisel
  // Also note this relies on executing Firrtl from Chisel directly
  it should "NOT be connectable to UInts" in {
    a[Exception] should be thrownBy {
      runTester {
        new BasicTester {
          val uint = WireDefault(0.U(32.W))
          val sint = Wire(Analog(32.W))
          sint := uint
        }
      }
    }
  }

  it should "work with 2 blackboxes bulk connected" in {
    assertTesterPasses(
      new AnalogTester {
        val mod = Module(new AnalogReaderBlackBox)
        mod.io.bus <> writer.io.bus
        check(mod)
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  it should "error if any bulk connected more than once" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val wires = List.fill(3)(Wire(Analog(32.W)))
        wires(0) <> wires(1)
        wires(0) <> wires(2)
      })
    }
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val wires = List.fill(2)(Wire(Analog(32.W)))
        wires(0) <> DontCare
        wires(0) <> wires(1)
      })
    }
  }

  it should "allow DontCare connection" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val a = Analog(1.W)
      })
      io.a := DontCare
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val a = Analog(1.W)
      })
      io.a <> DontCare
    })
  }

  it should "work in bidirectional Aggregate wires" in {
    class MyBundle extends Bundle {
      val x = Input(UInt(8.W))
      val y = Analog(8.W)
    }
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val a = new MyBundle
      })
      val w = Wire(new MyBundle)
      w <> io.a
    })
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val a = Vec(1, new MyBundle)
      })
      val w = Wire(Vec(1, new MyBundle))
      w <> io.a
    })
  }

  it should "work with 3 blackboxes attached" in {
    assertTesterPasses(
      new AnalogTester {
        val mods = Seq.fill(2)(Module(new AnalogReaderBlackBox))
        attach(writer.io.bus, mods(0).io.bus, mods(1).io.bus)
        mods.foreach(check(_))
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  it should "work with 3 blackboxes separately attached via a wire" in {
    assertTesterPasses(
      new AnalogTester {
        val mods = Seq.fill(2)(Module(new AnalogReaderBlackBox))
        val busWire = Wire(Analog(32.W))
        attach(busWire, writer.io.bus)
        attach(busWire, mods(0).io.bus)
        attach(mods(1).io.bus, busWire)
        mods.foreach(check(_))
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  // This does not currently work in Verilator unless Firrtl does constant prop and dead code
  // elimination on these wires
  ignore should "work with intermediate wires attached to each other" in {
    assertTesterPasses(
      new AnalogTester {
        val mod = Module(new AnalogReaderBlackBox)
        val busWire = Seq.fill(2)(Wire(Analog(32.W)))
        attach(busWire(0), writer.io.bus)
        attach(busWire(1), mod.io.bus)
        attach(busWire(0), busWire(1))
        check(mod)
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  it should "work with blackboxes at different levels of the module hierarchy" in {
    assertTesterPasses(
      new AnalogTester {
        val mods = Seq(Module(new AnalogReaderBlackBox), Module(new AnalogReaderWrapper))
        val busWire = Wire(writer.io.bus.cloneType)
        attach(writer.io.bus, mods(0).bus, mods(1).bus)
        mods.foreach(check(_))
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  // This does not currently work in Verilator, but does work in VCS
  ignore should "support two analog ports in the same module" in {
    assertTesterPasses(
      new AnalogTester {
        val reader = Module(new AnalogReaderBlackBox)
        val connector = Module(new AnalogConnector)
        connector.io.bus1 <> writer.io.bus
        reader.io.bus <> connector.io.bus2
        check(reader)
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  it should "NOT support conditional connection of analog types" in {
    a[ChiselException] should be thrownBy {
      assertTesterPasses(
        new AnalogTester {
          val mod = Module(new AnalogReaderBlackBox)
          when(cycle > 3.U) {
            mod.io.bus <> writer.io.bus
          }
          check(mod)
        },
        Seq("/chisel3/AnalogBlackBox.v")
      )
    }
  }

  it should "work with Vecs of Analog" in {
    assertTesterPasses(
      new AnalogTester {
        val mod = Module(new VecAnalogReaderWrapper)
        mod.bus <> writer.io.bus
        check(mod)
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }

  it should "work with Vecs of Bundles of Analog" in {
    assertTesterPasses(
      new AnalogTester {
        val mod = Module(new VecBundleAnalogReaderWrapper)
        mod.bus <> writer.io.bus
        check(mod)
      },
      Seq("/chisel3/AnalogBlackBox.v"),
      TesterDriver.verilatorOnly
    )
  }
}
