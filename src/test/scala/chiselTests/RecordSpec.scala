// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.OpaqueType
import chisel3.reflect.DataMirror
import chisel3.testers.BasicTester
import chisel3.util.{Counter, Queue}
import circt.stage.ChiselStage

import scala.collection.immutable.{ListMap, SeqMap}

object RecordSpec {
  class MyBundle extends Bundle {
    val foo = UInt(32.W)
    val bar = UInt(32.W)
  }
  // Useful for constructing types from CustomBundle
  // This is a def because each call to this needs to return a new instance
  def fooBarType: CustomBundle = new CustomBundle("foo" -> UInt(32.W), "bar" -> UInt(32.W))

  class MyModule(output: => Record, input: => Record) extends Module {
    val io = IO(new Bundle {
      val in = Input(input)
      val out = Output(output)
    })
    io.out <> io.in
  }

  class ConnectionTestModule(output: => Record, input: => Record) extends Module {
    val io = IO(new Bundle {
      val inMono = Input(input)
      val outMono = Output(output)
      val inBi = Input(input)
      val outBi = Output(output)
    })
    io.outMono := io.inMono
    io.outBi <> io.inBi
  }

  class RecordSerializationTest extends BasicTester {
    val recordType = new CustomBundle("fizz" -> UInt(16.W), "buzz" -> UInt(16.W))
    val record = Wire(recordType)
    // Note that "buzz" was added later than "fizz" and is therefore higher order
    record("fizz") := "hdead".U
    record("buzz") := "hbeef".U
    // To UInt
    val uint = record.asUInt
    assert(uint.getWidth == 32) // elaboration time
    assert(uint === "hbeefdead".U)
    // Back to Record
    val record2 = uint.asTypeOf(recordType)
    assert("hdead".U === record2("fizz").asUInt)
    assert("hbeef".U === record2("buzz").asUInt)
    stop()
  }

  class RecordQueueTester extends BasicTester {
    val queue = Module(new Queue(fooBarType, 4))
    queue.io <> DontCare
    queue.io.enq.valid := false.B
    val (cycle, done) = Counter(true.B, 4)

    when(cycle === 0.U) {
      queue.io.enq.bits("foo") := 1234.U
      queue.io.enq.bits("bar") := 5678.U
      queue.io.enq.valid := true.B
    }
    when(cycle === 1.U) {
      queue.io.deq.ready := true.B
      assert(queue.io.deq.valid === true.B)
      assert(queue.io.deq.bits("foo").asUInt === 1234.U)
      assert(queue.io.deq.bits("bar").asUInt === 5678.U)
    }
    when(done) {
      stop()
    }
  }

  class AliasedRecord extends Module {
    val field = UInt(32.W)
    val io = IO(new CustomBundle("in" -> Input(field), "out" -> Output(field)))
  }

  class RecordIOModule extends Module {
    val io = IO(new CustomBundle("in" -> Input(UInt(32.W)), "out" -> Output(UInt(32.W))))
    io("out") := io("in")
  }

  class RecordIOTester extends BasicTester {
    val mod = Module(new RecordIOModule)
    mod.io("in") := 1234.U
    assert(mod.io("out").asUInt === 1234.U)
    stop()
  }

  class RecordDigitTester extends BasicTester {
    val wire = Wire(new CustomBundle("0" -> UInt(32.W)))
    wire("0") := 123.U
    assert(wire("0").asUInt === 123.U)
    stop()
  }

  class RecordTypeTester extends BasicTester {
    val wire0 = Wire(new CustomBundle("0" -> UInt(32.W)))
    val wire1 = Reg(new CustomBundle("0" -> UInt(32.W)))
    val wire2 = Wire(new CustomBundle("1" -> UInt(32.W)))
    require(DataMirror.checkTypeEquivalence(wire0, wire1))
    require(!DataMirror.checkTypeEquivalence(wire1, wire2))
  }
}

class RecordSpec extends ChiselFlatSpec with Utils {
  import RecordSpec._

  behavior.of("Records")

  they should "bulk connect similarly to Bundles" in {
    ChiselStage.emitCHIRRTL { new MyModule(fooBarType, fooBarType) }
  }

  they should "bulk connect to Bundles" in {
    ChiselStage.emitCHIRRTL { new MyModule(new MyBundle, fooBarType) }
  }

  they should "emit FIRRTL bulk connects when possible" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      gen = new ConnectionTestModule(fooBarType, fooBarType)
    )
    chirrtl should include("connect io.outMono, io.inMono @")
    chirrtl should include("connect io.outBi, io.inBi @")
  }

  they should "not allow aliased fields" in {
    class AliasedFieldRecord extends Record {
      val foo = UInt(8.W)
      val elements = SeqMap("foo" -> foo, "bar" -> foo)
    }

    val e = intercept[AliasedAggregateFieldException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new AliasedFieldRecord)
        }
      }
    }
    e.getMessage should include("contains aliased fields named (bar,foo)")
  }

  they should "follow UInt serialization/deserialization API" in {
    assertTesterPasses { new RecordSerializationTest }
  }

  they should "work as the type of a Queue" in {
    assertTesterPasses { new RecordQueueTester }
  }

  they should "work as the type of a Module's io" in {
    assertTesterPasses { new RecordIOTester }
  }

  they should "support digits as names of fields" in {
    assertTesterPasses { new RecordDigitTester }
  }

  "Bulk connect on Record" should "check that the fields match" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL { new MyModule(fooBarType, new CustomBundle("bar" -> UInt(32.W))) }
    }).getMessage should include("Right Record missing field")

    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL { new MyModule(new CustomBundle("bar" -> UInt(32.W)), fooBarType) }
    }).getMessage should include("Left Record missing field")
  }

  "CustomBundle" should "work like built-in aggregates" in {
    ChiselStage.emitCHIRRTL(new Module {
      val gen = new CustomBundle("foo" -> UInt(32.W))
      val io = IO(Output(gen))
      val wire = Wire(gen)
      io := wire
    })
  }

  "CustomBundle" should "check the types" in {
    ChiselStage.emitCHIRRTL { new RecordTypeTester }
  }

  "Record with unstable elements" should "error" in {
    class MyRecord extends Record {
      def elements = SeqMap("a" -> UInt(8.W))
    }
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(Input(new MyRecord))
      })
    }
    e.getMessage should include("does not return the same objects when calling .elements multiple times")
  }

  "Bundle types which couldn't be cloned by the plugin" should "throw an error" in {
    class CustomBundleBroken(elts: (String, Data)*) extends Record {
      val elements = ListMap(elts.map {
        case (field, elt) =>
          field -> elt
      }: _*)
      def apply(elt: String): Data = elements(elt)
    }
    val err = the[ChiselException] thrownBy {
      val recordType = new CustomBundleBroken("fizz" -> UInt(16.W), "buzz" -> UInt(16.W))
      val record = Wire(recordType)
      val uint = record.asUInt
      val record2 = uint.asTypeOf(recordType)
    }

    err.getMessage should include("bundle plugin was unable to clone")
  }

  "Attempting to create a Record with bound nested elements" should "error" in {
    class InnerNestedRecord[T <: Data](gen: T) extends Record {
      val elements = SeqMap("a" -> gen)
    }
    class NestedRecord[T <: Data](gen: T) extends Record {
      val inner1 = new InnerNestedRecord(gen)
      val inner2 = new InnerNestedRecord(UInt(4.W))
      val elements = SeqMap("a" -> inner1, "b" -> inner2)
    }
    class MyRecord[T <: Data](gen: T) extends Record {
      val nested = new NestedRecord(gen)
      val elements = SeqMap("a" -> nested)
    }

    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        val myReg = RegInit(0.U(8.W))
        val io = IO(Input(new MyRecord(myReg)))
      })
    }
    e.getMessage should include("must be a Chisel type, not hardware")
  }
}
