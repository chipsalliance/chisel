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

    /* Unused -- we can't yet have a Probed Aggregate in a FixedIO___Module
    class Agg extends Bundle {
      val foo = Bool()
      val bar = Bool()
    }
     */

    /*Unused -- we can't yet have an Aggregate of Probes in a FixedIO__Module
    class Nested extends Bundle {
      val foo = Probe(Bool())
      val bar = Probe(Bool())
    }
     */

    class FixedIO extends Bundle {
      val elem = Probe(Bool())
      // Doesn't work yet
      // val agg = Probe(new Agg())
      // val nested = new Nested()
    }

    class ExampleRaw extends FixedIORawModule[FixedIO](new FixedIO()) {

      val elemWire = Wire(Bool())
      elemWire := false.B
      probe.define(io.elem, probe.ProbeValue(elemWire))

      /* Doesn't work yet
      val aggWire = Wire(new Agg())
      aggWire := DontCare
      probe.define(io.agg, probe.ProbeValue(aggWire))
       */

      /* Doesn't work yet
      val nestedWire = Wire(new Nested())
      val nestedFoo = WireInit(false.B)
      val nestedBar = WireInit(false.B)
      probe.define(nestedWire.foo, probe.ProbeValue(nestedFoo))
      probe.define(nestedWire.bar, probe.ProbeValue(nestedBar))
      io.nested :<>= nestedWire
       */
    }

    class ExampleExt extends FixedIOExtModule[FixedIO](new FixedIO())

    class Parent extends Module {
      val childRaw = Module(new ExampleRaw())
      val childExt = Module(new ExampleExt())

      // Check Probe(Element)
      val outElemRaw = IO(Bool())
      val probeElemWireRaw = Wire(Probe(Bool()))
      outElemRaw := probe.read(probeElemWireRaw)
      probeElemWireRaw :<>= childRaw.io.elem

      val probeElemWireExt = Wire(Probe(Bool()))
      val outElemExt = IO(Bool())
      outElemExt := probe.read(probeElemWireExt)
      probeElemWireExt :<>= childExt.io.elem

      // Check Probe(Aggregate)
      /** Doesn't work yet
        *      val outAggRaw = IO(new Agg())
        *      val probeAggWireRaw = Wire(Probe(new Agg()))
        *      outAggRaw := probe.read(probeAggWireRaw)
        *      probeAggWireRaw :<>= childRaw.io.agg
        *
        *      val probeAggWireExt = Wire(Probe(new Agg()))
        *      val outAggExt = IO(new Agg())
        *      outAggExt := probe.read(probeAggWireExt)
        *      probeAggWireExt :<>= childExt.io.agg
        */

      // Check Aggregate(Probes)
      /* Doesn't work yet
      val probeNestedWireRaw = Wire(new Nested())
      val outNestedRawFoo = IO(Bool())
      val outNestedRawBar = IO(Bool())
      outNestedRawFoo := probe.read(probeNestedWireRaw.foo)
      outNestedRawBar := probe.read(probeNestedWireRaw.bar)
      probeNestedWireRaw :<>= childRaw.io.nested

      val probeNestedWireExt = Wire(new Nested())
      val outNestedExtFoo = IO(Bool())
      val outNestedExtBar = IO(Bool())
      outNestedExtFoo := probe.read(probeNestedWireExt.foo)
      outNestedExtBar := probe.read(probeNestedWireExt.bar)
      probeNestedWireExt :<>= childExt.io.nested
       */

    }

    println(ChiselStage.emitCHIRRTL(new Parent, Array("--full-stacktrace")))
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Parent))(
      "output elem : Probe<UInt<1>>",
      "output elem : Probe<UInt<1>>",
      "define probeElemWireRaw = childRaw.elem",
      "define probeElemWireExt = childExt.elem"
    )()

  }
}
