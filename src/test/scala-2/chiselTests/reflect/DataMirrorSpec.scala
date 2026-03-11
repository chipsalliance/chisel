// SPDX-License-Identifier: Apache-2.0

package chiselTests.reflect

import chisel3._
import chisel3.experimental.dataview._
import chisel3.experimental.hierarchy._
import chisel3.probe.Probe
import chisel3.properties.Property
import chisel3.reflect.DataMirror
import chisel3.util.Decoupled
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object DataMirrorSpec {
  import org.scalatest.matchers.should.Matchers._
  class GrandChild(parent: RawModule) extends Module {
    val internal = WireInit(false.B)
    DataMirror.getParent(this) should be(Some(parent))
    DataMirror.isVisible(internal) should be(true)
    DataMirror.isVisible(internal.viewAs[Bool]) should be(true)
  }
  @instantiable
  class Child(parent: RawModule) extends Module {
    val inst = Module(new GrandChild(this))
    @public val io = IO(Input(Bool()))
    val internal = WireInit(false.B)
    lazy val underWhen = WireInit(false.B)
    when(true.B) {
      underWhen := true.B // trigger the lazy val
      DataMirror.isVisible(underWhen) should be(true)
      DataMirror.isVisible((internal, underWhen).viewAs) should be(true)
    }
    val mixedView = (io, underWhen).viewAs
    DataMirror.getParent(inst) should be(Some(this))
    DataMirror.getParent(this) should be(Some(parent))
    DataMirror.isVisible(io) should be(true)
    DataMirror.isVisible(io.viewAs[Bool]) should be(true)
    DataMirror.isVisible(internal) should be(true)
    DataMirror.isVisible(internal.viewAs[Bool]) should be(true)
    DataMirror.isVisible(inst.internal) should be(false)
    DataMirror.isVisible(inst.internal.viewAs[Bool]) should be(false)
    DataMirror.isVisible(underWhen) should be(false)
    DataMirror.isVisible(underWhen.viewAs) should be(false)
    DataMirror.isVisible(mixedView) should be(false)
    DataMirror.isVisible(mixedView._1) should be(true)
  }
  @instantiable
  class Parent extends Module {
    @public val io = IO(Input(Bool()))
    @public val inst = Module(new Child(this))
    @public val internal = WireInit(io)
    @public val tuple = (io, internal).viewAs
    inst.io := internal
    DataMirror.getParent(inst) should be(Some(this))
    DataMirror.getParent(this) should be(None)
    DataMirror.isVisible(inst.io) should be(true)
    DataMirror.isVisible(inst.io.viewAs[Bool]) should be(true)
    DataMirror.isVisible(inst.internal) should be(false)
    DataMirror.isVisible(inst.internal.viewAs[Bool]) should be(false)
    DataMirror.isVisible(inst.inst.internal) should be(false)
    DataMirror.isVisible(inst.inst.internal.viewAs[Bool]) should be(false)
    DataMirror.isVisible(inst.mixedView) should be(false)
    DataMirror.isVisible(inst.mixedView._1) should be(true)
    DataMirror.isVisible(tuple) should be(true)
  }
}

class DataMirrorSpec extends AnyFlatSpec with Matchers {
  import DataMirrorSpec._

  behavior.of("DataMirror")

  def assertBinding(x: Data, io: Boolean, wire: Boolean, reg: Boolean) = {
    DataMirror.isIO(x) should be(io)
    DataMirror.isWire(x) should be(wire)
    DataMirror.isReg(x) should be(reg)
  }

  def assertIO(x: Data) = assertBinding(x, true, false, false)

  def assertWire(x: Data) = assertBinding(x, false, true, false)

  def assertReg(x: Data) = assertBinding(x, false, false, true)

  def assertNone(x: Data) = assertBinding(x, false, false, false)

  it should "validate bindings" in {
    class MyModule extends Module {
      val typ = UInt(4.W)
      val vectyp = Vec(8, UInt(4.W))
      val io = IO(new Bundle {
        val in = Input(UInt(4.W))
        val vec = Input(vectyp)
        val out = Output(UInt(4.W))
      })
      val vec = Wire(vectyp)
      val regvec = Reg(vectyp)
      val wire = Wire(UInt(4.W))
      val reg = RegNext(wire)

      assertIO(io)
      assertIO(io.in)
      assertIO(io.out)
      assertIO(io.vec(1))
      assertIO(io.vec)
      assertWire(vec)
      assertWire(vec(0))
      assertWire(wire)
      assertReg(reg)
      assertReg(regvec)
      assertReg(regvec(2))
      assertNone(typ)
      assertNone(vectyp)
    }
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  it should "support getParent and isVisible for normal modules" in {
    ChiselStage.emitCHIRRTL(new Parent)
  }

  it should "support getParent and isVisible for normal modules even when used in a D/I context" in {
    class Top extends Module {
      val defn = Definition(new Parent)
      val inst = Instance(defn)
      DataMirror.getParent(this) should be(None)
      DataMirror.isVisible(inst.io) should be(true)
      DataMirror.isVisible(inst.io.viewAs) should be(true)
      DataMirror.isVisible(inst.inst.io) should be(false)
      DataMirror.isVisible(inst.inst.io.viewAs) should be(false)
      DataMirror.isVisible(inst.internal) should be(false)
      DataMirror.isVisible(inst.internal.viewAs) should be(false)
      DataMirror.isVisible(inst.tuple) should be(false)
      DataMirror.isVisible(inst.tuple._1) should be(true)
    }
    ChiselStage.emitCHIRRTL(new Top)
  }

  it should "support getting name guesses even though they may change" in {
    import DataMirror.queryNameGuess
    class MyModule extends Module {
      val io = {
        val port = IO(new Bundle {
          val foo = Output(UInt(8.W))
        })
        queryNameGuess(port) should be("io_port")
        queryNameGuess(port.foo) should be("io_port.foo")
        port
      }
      queryNameGuess(io) should be("io")
      queryNameGuess(io.foo) should be("io.foo")
      io.suggestName("potato")
      queryNameGuess(io) should be("potato")
      queryNameGuess(io.foo) should be("potato.foo")
    }
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  it should "not support name guesses for non-hardware" in {
    an[ExpectedHardwareException] should be thrownBy DataMirror.queryNameGuess(UInt(8.W))
  }

  it should "support querying if a Data is a Property" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val notProperty = IO(Input(Bool()))
      val property = IO(Input(Property[Int]()))

      DataMirror.isProperty(notProperty) shouldBe false
      DataMirror.isProperty(property) shouldBe true
    })
  }

  "chiselTypeClone" should "preserve Scala type information" in {
    class MyModule extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(DataMirror.internal.chiselTypeClone(in)))
      // The connection checks the types
      out :#= in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
  }

  "isFullyAligned" should "work" in {
    class InputOutputTest extends Bundle {
      val incoming = Input(Decoupled(UInt(8.W)))
      val outgoing = Output(Decoupled(UInt(8.W)))
      val mixed = Decoupled(UInt(8.W))
    }
    // Top-level negative test.
    assert(!DataMirror.isFullyAligned(new InputOutputTest()))

    // Various positive tests, coerced.
    assert(DataMirror.isFullyAligned(new InputOutputTest().incoming))
    assert(DataMirror.isFullyAligned(new InputOutputTest().outgoing))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest())))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest()).outgoing))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest())))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest()).incoming))
    assert(DataMirror.isFullyAligned(Output(new InputOutputTest()).incoming.ready))

    // Negative test mixed, check positive when coerced.
    assert(!DataMirror.isFullyAligned(new InputOutputTest().mixed))
    assert(DataMirror.isFullyAligned(Input(new InputOutputTest().mixed)))

    // Check DecoupledIO directly, as well as coerced.
    assert(!DataMirror.isFullyAligned(Decoupled(UInt(8.W))))
    assert(DataMirror.isFullyAligned(Input(Decoupled(UInt(8.W)))))

    // Positive test, simple vector + flipped vector.
    assert(DataMirror.isFullyAligned(Vec(2, UInt(1.W))))
    assert(DataMirror.isFullyAligned(Flipped(Vec(2, UInt(1.W)))))

    // Positive test, zero-length vector of non-aligned elements.
    assert(DataMirror.isFullyAligned(Vec(0, Decoupled(UInt(8.W)))))

    // Negative test: vector of flipped (?).
    assert(!DataMirror.isFullyAligned(Vec(2, Flipped(UInt(1.W)))))

    // Check empty bundle (no members).
    assert(DataMirror.isFullyAligned(new Bundle {}))
    assert(DataMirror.isFullyAligned(Flipped(new Bundle {})))

    // Check ground type.
    assert(DataMirror.isFullyAligned(UInt(8.W)))
  }

  "getLayerColor" should "return a layer color if one exists" in {
    object A extends layer.Layer(layer.LayerConfig.Extract())
    class Foo extends Bundle {
      val a = Bool()
      val b = Probe(Bool())
      val c = Probe(Bool(), A)
    }

    val foo = new Foo

    info("a non-probe returns None")
    DataMirror.getLayerColor(foo.a) should be(None)

    info("an uncolored probe returns None")
    DataMirror.getLayerColor(foo.b) should be(None)

    info("a probe colored with A returns Some(A)")
    DataMirror.getLayerColor(foo.c) should be(Some(A))
  }

  "currentModulePorts" should "return an in-progress module's IOs" in {
    class Foo extends RawModule {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      val wire = Wire(Bool())
      val child = Module(new RawModule {})

      val ports0 = DataMirror.internal.currentModulePorts(this)

      val other = IO(Input(Bool()))

      val ports1 = DataMirror.internal.currentModulePorts(this)
    }

    ChiselStage.emitCHIRRTL(new RawModule {
      val foo = Module(new Foo)
      foo.ports0 should be(Seq(foo.in, foo.out))
      foo.ports1 should be(Seq(foo.in, foo.out, foo.other))
    })
  }

  "currentInstancePorts" should "return an Instance's IOs" in {
    @instantiable
    class Foo extends RawModule {
      @public val in = IO(Input(Bool()))
      @public val out = IO(Output(Bool()))
    }

    ChiselStage.emitCHIRRTL(new RawModule {
      val io = IO(Output(Bool()))

      // Check instantiate API.
      val fooInst = Instantiate(new Foo)

      val fooInstPorts = DataMirror.internal.currentInstancePorts(fooInst)

      fooInstPorts should be(Seq(fooInst.in, fooInst.out))

      // Check toInstance API.
      val fooBoxed = Module(new Foo).toInstance

      val fooBoxedPorts = DataMirror.internal.currentInstancePorts(fooBoxed)

      fooBoxedPorts should be(Seq(fooBoxed.in, fooBoxed.out))

      // Check toInstance API on an in-progress Module.
      val thisInst = this.toInstance

      val thisInstPorts = DataMirror.internal.currentInstancePorts(thisInst)

      thisInstPorts should be(Seq(io))
    })
  }

  "modulePorts and fullModulePorts" should "return an Instance of a module's IOs" in {
    @instantiable
    class Bar extends Module {
      @public val io = IO(new Bundle {
        val vec = Vec(2, Bool())
        val x = UInt(4.W)
      })
    }

    class Foo extends Module {
      val definition = Definition(new Bar)
      val instA = Instance(definition)
      val portsA = DataMirror.modulePorts(instA)

      val instB = (Module(new Bar)).toInstance
      val portsB = DataMirror.fullModulePorts(instB)
    }

    ChiselStage.emitCHIRRTL(new Module {
      val foo = Module(new Foo)
      foo.portsA.map(_._1) should be(Seq("clock", "reset", "io"))
      foo.portsB.map(_._1) should be(
        Seq(
          "clock",
          "reset",
          "io",
          "io_x",
          "io_vec",
          "io_vec_0",
          "io_vec_1"
        )
      )
    })
  }
}
