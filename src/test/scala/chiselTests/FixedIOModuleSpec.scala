// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import scala.collection.immutable.ListMap
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.experimental.SourceInfo
import chisel3.probe.Probe

class FixedIOModuleSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  "FixedIOModule" should "create a module with flattened IO" in {

    class Foo(width: Int) extends FixedIORawModule[UInt](UInt(width.W)) {
      io :<>= DontCare
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo(8)))("output io : UInt<8>")()
  }

  "PolymoprhicIOModule error behavior" should "disallow further IO creation" in {
    class Foo extends FixedIORawModule[Bool](Bool()) {
      val a = IO(Bool())
    }
    val exception = intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new Foo, Array("--throw-on-first-error"))
    }
    exception.getMessage should include("This module cannot have IOs instantiated after disallowing IOs")
  }

  "FixedIOBlackBox" should "create a module with flattend IO" in {

    class Bar extends FixedIOExtModule(UInt(1.W))

    class BazIO extends Bundle {
      val a = UInt(2.W)
    }

    class Baz extends FixedIOExtModule(new BazIO)

    class Foo extends RawModule {
      val bar = Module(new Bar)
      val baz = Module(new Baz)
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "output io : UInt<1>",
      "output a : UInt<2>"
    )()
  }

  "User defined RawModules" should "be able to lock down their ios" in {

    class Bar extends RawModule {
      val in = IO(Input(UInt(1.W)))
      val out = IO(Output(UInt(1.W)))

      endIOCreation()

      val other = IO(Input(UInt(1.W)))
    }

    val exception = intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new Bar, Array("--throw-on-first-error"))
    }
    exception.getMessage should include("This module cannot have IOs instantiated after disallowing IOs")
  }

  "User defined RawModules" should "be able to lock down their ios in a scope" in {

    class Bar(illegalIO: Boolean) extends RawModule {
      val in = IO(Input(UInt(1.W)))
      val out = IO(Output(UInt(1.W)))

      withoutIO {
        if (illegalIO) {
          val other = IO(Input(UInt(1.W)))
        }
      }
      val end = IO(Input(UInt(1.W)))
    }

    val exception = intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new Bar(true), Array("--throw-on-first-error"))
    }
    exception.getMessage should include("This module cannot have IOs instantiated after disallowing IOs")
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Bar(false)))(
      "input in : UInt<1>",
      "output out : UInt<1>",
      "input end : UInt<1>"
    )()
  }

  "User defined FixedIORaw/ExtModules" should "be able to have Probes in their IOs" in {

    class Agg extends Bundle {
      val foo = Bool()
      val bar = Bool()
    }

    class FixedIO extends Bundle {
      val elem = Probe(Bool())
      // val agg = Probe(new Agg())
    }

    class ExampleRaw extends FixedIORawModule[FixedIO](new FixedIO()) {

      val elemWire = Wire(Bool())
      elemWire := false.B
      probe.define(io.elem, probe.ProbeValue(elemWire))

      //val aggWire = Wire(new Agg())
      //aggWire := DontCare
      // probe.define(io.agg, probe.ProbeValue(aggWire))
    }

    class ExampleExt extends FixedIOExtModule[FixedIO](new FixedIO())

    class Parent extends Module {
      val childRaw = Module(new ExampleRaw())
      val childExt = Module(new ExampleExt())
      val outElemRaw = IO(Bool())
      val probeElemWireRaw = Wire(Probe(Bool()))
      outElemRaw := probe.read(probeElemWireRaw)
      probeElemWireRaw :<>= childRaw.io.elem
      val probeElemWireExt = Wire(Probe(Bool()))
      val outElemExt = IO(Bool())
      outElemExt := probe.read(probeElemWireExt)
      probeElemWireExt :<>= childExt.io.elem

      /* Doesn't work yet
  val outAggRaw = IO(new Agg())
  val probeAggWireRaw = Wire(Probe(new Agg()))
  outAggRaw := probe.read(probeAggWireRaw)
  probeAggWireRaw :<>= childRaw.io.agg
  val probeAggWireExt = Wire(Probe(new Agg()))
  val outAggExt = IO(new Agg())
  outAggExt := probe.read(probeAggWireExt)
  probeAggWireExt :<>= childExt.io.agg
       */
    }

    print(circt.stage.ChiselStage.emitCHIRRTL(new Parent))

  }
}
