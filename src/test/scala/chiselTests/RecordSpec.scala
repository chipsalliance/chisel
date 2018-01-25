// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.{Counter, Queue}
import chisel3.experimental.{DataMirror, requireIsChiselType}
import scala.collection.immutable.ListMap

// An example of how Record might be extended
// In this case, CustomBundle is a Record constructed from a Tuple of (String, Data)
//   it is a possible implementation of a programmatic "Bundle"
//   (and can by connected to MyBundle below)
final class CustomBundle(elts: (String, Data)*) extends Record {
  val elements = ListMap(elts map { case (field, elt) =>
    requireIsChiselType(elt)
    field -> elt
  }: _*)
  def apply(elt: String): Data = elements(elt)
  override def cloneType = {
    val cloned = elts.map { case (n, d) => n -> DataMirror.internal.chiselTypeClone(d) }
    (new CustomBundle(cloned: _*)).asInstanceOf[this.type]
  }
}

trait RecordSpecUtils {
  class MyBundle extends Bundle {
    val foo = UInt(32.W)
    val bar = UInt(32.W)
    override def cloneType = (new MyBundle).asInstanceOf[this.type]
  }
  // Useful for constructing types from CustomBundle
  // This is a def because each call to this needs to return a new instance
  def fooBarType = new CustomBundle("foo" -> UInt(32.W), "bar" -> UInt(32.W))

  class MyModule(output: => Record, input: => Record) extends Module {
    val io = IO(new Bundle {
      val in = Input(input)
      val out = Output(output)
    })
    io.out <> io.in
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

    when (cycle === 0.U) {
      queue.io.enq.bits("foo") := 1234.U
      queue.io.enq.bits("bar") := 5678.U
      queue.io.enq.valid := true.B
    }
    when (cycle === 1.U) {
      queue.io.deq.ready := true.B
      assert(queue.io.deq.valid === true.B)
      assert(queue.io.deq.bits("foo").asUInt === 1234.U)
      assert(queue.io.deq.bits("bar").asUInt === 5678.U)
    }
    when (done) {
      stop()
    }
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
}

class RecordSpec extends ChiselFlatSpec with RecordSpecUtils {
  behavior of "Records"

  they should "bulk connect similarly to Bundles" in {
    elaborate { new MyModule(fooBarType, fooBarType) }
  }

  they should "bulk connect to Bundles" in {
    elaborate { new MyModule(new MyBundle, fooBarType) }
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
    (the [ChiselException] thrownBy {
      elaborate { new MyModule(fooBarType, new CustomBundle("bar" -> UInt(32.W))) }
    }).getMessage should include ("Right Record missing field")

    (the [ChiselException] thrownBy {
      elaborate { new MyModule(new CustomBundle("bar" -> UInt(32.W)), fooBarType) }
    }).getMessage should include ("Left Record missing field")
  }

  "CustomBundle" should "work like built-in aggregates" in {
    elaborate(new Module {
      val gen = new CustomBundle("foo" -> UInt(32.W))
      val io = IO(Output(gen))
      val wire = Wire(gen)
      io := wire
    })
  }
}
