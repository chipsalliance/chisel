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

  class SingleElementRecord extends Record with OpaqueType {
    private val underlying = UInt(8.W)
    val elements = SeqMap("" -> underlying)

    def +(that: SingleElementRecord): SingleElementRecord = {
      val _w = Wire(new SingleElementRecord)
      _w.underlying := this.underlying + that.underlying
      _w
    }
  }

  class SingleElementRecordModule extends Module {
    val in1 = IO(Input(new SingleElementRecord))
    val in2 = IO(Input(new SingleElementRecord))
    val out = IO(Output(new SingleElementRecord))

    val r = new SingleElementRecord

    out := in1 + in2
  }

  class InnerRecord extends Record with OpaqueType {
    val k = new InnerInnerRecord
    val elements = SeqMap("" -> k)
  }

  class InnerInnerRecord extends Record with OpaqueType {
    val k = new SingleElementRecord
    val elements = SeqMap("" -> k)
  }

  class NestedRecordModule extends Module {
    val in = IO(Input(new InnerRecord))
    val out = IO(Output(new InnerRecord))
    val inst = Module(new InnerModule)
    inst.io.foo := in
    out := inst.io.bar
  }

  class InnerModule extends Module {
    val io = IO(new Bundle {
      val foo = Input(new InnerRecord)
      val bar = Output(new InnerRecord)
    })

    // DO NOT do this; just for testing element connections
    io.bar.elements.head._2 := io.foo.elements.head._2
  }

  class NamedSingleElementRecord extends Record with OpaqueType {
    private val underlying = UInt(8.W)
    val elements = SeqMap("unused" -> underlying)
  }

  class NamedSingleElementModule extends Module {
    val in = IO(Input(new NamedSingleElementRecord))
    val out = IO(Output(new NamedSingleElementRecord))
    out := in
  }

  class ErroneousOverride extends Record with OpaqueType {
    private val underlyingA = UInt(8.W)
    private val underlyingB = UInt(8.W)
    val elements = SeqMap("x" -> underlyingA, "y" -> underlyingB)

    override def opaqueType = true
  }

  class ErroneousOverrideModule extends Module {
    val in = IO(Input(new ErroneousOverride))
    val out = IO(Output(new ErroneousOverride))
    out := in
  }

  class NotActuallyOpaqueType extends Record with OpaqueType {
    private val underlyingA = UInt(8.W)
    private val underlyingB = UInt(8.W)
    val elements = SeqMap("x" -> underlyingA, "y" -> underlyingB)

    override def opaqueType = false
  }

  class NotActuallyOpaqueTypeModule extends Module {
    val in = IO(Input(new NotActuallyOpaqueType))
    val out = IO(Output(new NotActuallyOpaqueType))
    out := in
  }

  // Illustrate how to dyanmically decide between OpaqueType or not
  sealed trait MaybeBoxed[T <: Data] extends Record {
    def underlying: T
    def boxed:      Boolean
  }
  object MaybeBoxed {
    def apply[T <: Data](gen: T, boxed: Boolean): MaybeBoxed[T] = {
      if (boxed) new Boxed(gen) else new Unboxed(gen)
    }
  }
  class Boxed[T <: Data](gen: T) extends MaybeBoxed[T] {
    def boxed = true
    lazy val elements = SeqMap("underlying" -> gen)
    def underlying = elements.head._2
  }
  class Unboxed[T <: Data](gen: T) extends MaybeBoxed[T] with OpaqueType {
    def boxed = false
    lazy val elements = SeqMap("" -> gen)
    def underlying = elements.head._2
  }
}

class RecordSpec extends ChiselFlatSpec with Utils {
  import RecordSpec._

  behavior.of("Records")

  they should "bulk connect similarly to Bundles" in {
    ChiselStage.elaborate { new MyModule(fooBarType, fooBarType) }
  }

  they should "bulk connect to Bundles" in {
    ChiselStage.elaborate { new MyModule(new MyBundle, fooBarType) }
  }

  they should "emit FIRRTL bulk connects when possible" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      gen = new ConnectionTestModule(fooBarType, fooBarType)
    )
    chirrtl should include("io.outMono <= io.inMono @")
    chirrtl should include("io.outBi <= io.inBi @")
  }

  they should "not allow aliased fields" in {
    class AliasedFieldRecord extends Record {
      val foo = UInt(8.W)
      val elements = SeqMap("foo" -> foo, "bar" -> foo)
    }

    val e = intercept[AliasedAggregateFieldException] {
      ChiselStage.elaborate {
        new Module {
          val io = IO(new AliasedFieldRecord)
        }
      }
    }
    e.getMessage should include("contains aliased fields named (bar,foo)")
  }

  they should "support OpaqueType for maps with single unnamed elements" in {
    val singleElementChirrtl = ChiselStage.emitCHIRRTL { new SingleElementRecordModule }
    singleElementChirrtl should include("input in1 : UInt<8>")
    singleElementChirrtl should include("input in2 : UInt<8>")
    singleElementChirrtl should include("add(in1, in2)")
  }

  they should "work correctly for toTarget in nested OpaqueType Records" in {
    var mod: NestedRecordModule = null
    ChiselStage.elaborate { mod = new NestedRecordModule; mod }
    val testStrings = Seq(
      mod.inst.io.foo.toTarget.serialize,
      mod.inst.io.foo.k.toTarget.serialize,
      mod.inst.io.foo.k.k.toTarget.serialize,
      mod.inst.io.foo.elements.head._2.toTarget.serialize,
      mod.inst.io.foo.k.elements.head._2.toTarget.serialize,
      mod.inst.io.foo.k.k.elements.head._2.toTarget.serialize
    )
    testStrings.foreach(x => assert(x == "~NestedRecordModule|InnerModule>io.foo"))
  }

  they should "work correctly with DataMirror in nested OpaqueType Records" in {
    var mod: NestedRecordModule = null
    ChiselStage.elaborate { mod = new NestedRecordModule; mod }
    val ports = DataMirror.fullModulePorts(mod.inst)
    val expectedPorts = Seq(
      ("clock", mod.inst.clock),
      ("reset", mod.inst.reset),
      ("io", mod.inst.io),
      ("io_bar", mod.inst.io.bar),
      ("io_bar", mod.inst.io.bar.k),
      ("io_bar", mod.inst.io.bar.k.k),
      ("io_bar", mod.inst.io.bar.k.k.elements.head._2),
      ("io_foo", mod.inst.io.foo),
      ("io_foo", mod.inst.io.foo.k),
      ("io_foo", mod.inst.io.foo.k.k),
      ("io_foo", mod.inst.io.foo.k.k.elements.head._2)
    )
    ports shouldBe expectedPorts
  }

  they should "work correctly when connecting nested OpaqueType elements" in {
    val nestedRecordChirrtl = ChiselStage.emitCHIRRTL { new NestedRecordModule }
    nestedRecordChirrtl should include("input in : UInt<8>")
    nestedRecordChirrtl should include("output out : UInt<8>")
    nestedRecordChirrtl should include("inst.io.foo <= in")
    nestedRecordChirrtl should include("out <= inst.io.bar")
    nestedRecordChirrtl should include("output io : { flip foo : UInt<8>, bar : UInt<8>}")
    nestedRecordChirrtl should include("io.bar <= io.foo")
  }

  they should "throw an error when map contains a named element and OpaqueType is mixed in" in {
    (the[Exception] thrownBy extractCause[Exception] {
      ChiselStage.elaborate { new NamedSingleElementModule }
    }).getMessage should include("Opaque types must have exactly one element with an empty name")
  }

  they should "throw an error when map contains more than one element and OpaqueType is mixed in" in {
    (the[Exception] thrownBy extractCause[Exception] {
      ChiselStage.elaborate { new ErroneousOverrideModule }
    }).getMessage should include("Opaque types must have exactly one element with an empty name")
  }

  they should "work correctly when an OpaqueType overrides the def as false" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new NotActuallyOpaqueTypeModule)
    chirrtl should include("input in : { y : UInt<8>, x : UInt<8>}")
    chirrtl should include("output out : { y : UInt<8>, x : UInt<8>}")
    chirrtl should include("out <= in")
  }

  they should "support conditional OpaqueTypes via traits and factory methods" in {
    class MyModule extends Module {
      val in0 = IO(Input(MaybeBoxed(UInt(8.W), true)))
      val out0 = IO(Output(MaybeBoxed(UInt(8.W), true)))
      val in1 = IO(Input(MaybeBoxed(UInt(8.W), false)))
      val out1 = IO(Output(MaybeBoxed(UInt(8.W), false)))
      out0 := in0
      out1 := in1
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("input in0 : { underlying : UInt<8>}")
    chirrtl should include("input in1 : UInt<8>")
  }

  they should "work with .toTarget" in {
    var m: SingleElementRecordModule = null
    ChiselStage.elaborate { m = new SingleElementRecordModule; m }
    val q = m.in1.toTarget.toString
    assert(q == "~SingleElementRecordModule|SingleElementRecordModule>in1")
  }

  they should "NOT work with .toTarget on non-data OpaqueType Record" in {
    var m: SingleElementRecordModule = null
    ChiselStage.elaborate { m = new SingleElementRecordModule; m }
    a[ChiselException] shouldBe thrownBy { m.r.toTarget }
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
      ChiselStage.elaborate { new MyModule(fooBarType, new CustomBundle("bar" -> UInt(32.W))) }
    }).getMessage should include("Right Record missing field")

    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new MyModule(new CustomBundle("bar" -> UInt(32.W)), fooBarType) }
    }).getMessage should include("Left Record missing field")
  }

  "CustomBundle" should "work like built-in aggregates" in {
    ChiselStage.elaborate(new Module {
      val gen = new CustomBundle("foo" -> UInt(32.W))
      val io = IO(Output(gen))
      val wire = Wire(gen)
      io := wire
    })
  }

  "CustomBundle" should "check the types" in {
    ChiselStage.elaborate { new RecordTypeTester }
  }

  "Record with unstable elements" should "error" in {
    class MyRecord extends Record {
      def elements = SeqMap("a" -> UInt(8.W))
    }
    val e = the[ChiselException] thrownBy {
      ChiselStage.elaborate(new Module {
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

  "Attempting to create a Record with bound elements" should "error" in {
    class MyRecord[T <: Data](gen: T) extends Record {
      val elements = SeqMap("a" -> gen)
    }
    val e = the[ChiselException] thrownBy {
      ChiselStage.elaborate(new Module {
        val myReg = RegInit(0.U(8.W))
        val io = IO(Input(new MyRecord(myReg)))
      })
    }
    e.getMessage should include("record is already a bound hardware")
  }
}
