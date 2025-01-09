// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.OpaqueType
import chisel3.reflect.DataMirror
import circt.stage.ChiselStage

import scala.collection.immutable.SeqMap

object OpaqueTypeSpec {

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

  class MaybeNoAsUInt(noAsUInt: Boolean) extends Record with OpaqueType {
    lazy val elements = SeqMap("" -> UInt(8.W))
    override protected def errorOnAsUInt = noAsUInt
  }
}

class OpaqueTypeSpec extends ChiselFlatSpec with Utils {
  import OpaqueTypeSpec._

  behavior.of("OpaqueTypes")

  they should "support OpaqueType for maps with single unnamed elements" in {
    val singleElementChirrtl = ChiselStage.emitCHIRRTL { new SingleElementRecordModule }
    singleElementChirrtl should include("input in1 : UInt<8>")
    singleElementChirrtl should include("input in2 : UInt<8>")
    singleElementChirrtl should include("add(in1, in2)")
  }

  they should "work correctly for toTarget in nested OpaqueType Records" in {
    var mod: NestedRecordModule = null
    ChiselStage.emitCHIRRTL { mod = new NestedRecordModule; mod }
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
    ChiselStage.emitCHIRRTL { mod = new NestedRecordModule; mod }
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
    nestedRecordChirrtl should include("connect inst.io.foo, in")
    nestedRecordChirrtl should include("connect out, inst.io.bar")
    nestedRecordChirrtl should include("output io : { flip foo : UInt<8>, bar : UInt<8>}")
    nestedRecordChirrtl should include("connect io.bar, io.foo")
  }

  they should "throw an error when map contains a named element and OpaqueType is mixed in" in {
    (the[Exception] thrownBy extractCause[Exception] {
      ChiselStage.emitCHIRRTL { new NamedSingleElementModule }
    }).getMessage should include("Opaque types must have exactly one element with an empty name")
  }

  they should "throw an error when map contains more than one element and OpaqueType is mixed in" in {
    (the[Exception] thrownBy extractCause[Exception] {
      ChiselStage.emitCHIRRTL { new ErroneousOverrideModule }
    }).getMessage should include("Opaque types must have exactly one element with an empty name")
  }

  they should "work correctly when an OpaqueType overrides the def as false" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new NotActuallyOpaqueTypeModule)
    chirrtl should include("input in : { y : UInt<8>, x : UInt<8>}")
    chirrtl should include("output out : { y : UInt<8>, x : UInt<8>}")
    chirrtl should include("connect out, in")
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
    ChiselStage.emitCHIRRTL { m = new SingleElementRecordModule; m }
    val q = m.in1.toTarget.toString
    assert(q == "~SingleElementRecordModule|SingleElementRecordModule>in1")
  }

  they should "NOT work with .toTarget on non-data OpaqueType Record" in {
    var m: SingleElementRecordModule = null
    ChiselStage.emitCHIRRTL { m = new SingleElementRecordModule; m }
    a[ChiselException] shouldBe thrownBy { m.r.toTarget }
  }

  they should "support making .asUInt illegal" in {
    class AsUIntTester(gen: Data) extends RawModule {
      val in = IO(Input(gen))
      val out = IO(Output(UInt()))
      out :#= in.asUInt
    }
    // First check that it works when it should
    val chirrtl = ChiselStage.emitCHIRRTL(new AsUIntTester(new MaybeNoAsUInt(false)))
    chirrtl should include("connect out, in")

    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new AsUIntTester(new MaybeNoAsUInt(true)), Array("--throw-on-first-error"))
    }
    e.getMessage should include("MaybeNoAsUInt does not support .asUInt.")
  }

  they should "support give a decent error for .asUInt nested in an Aggregate" in {
    class AsUIntTester(gen: Data) extends RawModule {
      val in = IO(Input(Valid(gen)))
      val out = IO(Output(UInt()))
      out :#= in.asUInt
    }
    // First check that it works when it should
    val chirrtl = ChiselStage.emitCHIRRTL(new AsUIntTester(new MaybeNoAsUInt(false)))
    chirrtl should include("cat(in.valid, in.bits)")

    val e1 = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new AsUIntTester(new MaybeNoAsUInt(true)), Array("--throw-on-first-error"))
    }
    e1.getMessage should include("Field '_.bits' of type MaybeNoAsUInt does not support .asUInt.")
  }
}
