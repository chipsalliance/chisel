// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.experimental.hierarchy.{instantiable, Definition, Instance, Instantiate}
import chisel3.probe.Probe
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.collection.immutable.ListMap

class FixedIOModuleSpec extends AnyFlatSpec with Matchers with FileCheck {

  "FixedIOModule" should "create a module with flattened IO" in {

    class Foo(width: Int) extends FixedIORawModule[UInt](UInt(width.W)) {
      io :<>= DontCare
    }

    ChiselStage.emitCHIRRTL(new Foo(8)) should include("output io : UInt<8>")
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

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: extmodule Bar :
           |CHECK:         output io : UInt<1>
           |CHECK-LABEL: extmodule Baz :
           |CHECK:         output a : UInt<2>
           |CHECK-LABEL: public module Foo :
           |""".stripMargin
      )
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

    ChiselStage
      .emitCHIRRTL(new Bar(false))
      .fileCheck()(
        """|CHECK-LABEL: public module Bar :
           |CHECK:         input in : UInt<1>
           |CHECK:         output out : UInt<1>
           |CHECK:         input end : UInt<1>
           |""".stripMargin
      )
  }

  class Agg extends Bundle {
    val foo = Bool()
    val bar = Bool()
  }

  class AggWithProbes extends Bundle {
    val foo = Probe(Bool())
    val bar = Probe(Bool())
  }

  class MixedBundle(probeAgg: Boolean, aggProbe: Boolean) extends Bundle {
    val elem = Probe(Bool())
    val agg = Option.when(probeAgg)(Probe(new Agg()))
    val nested = Option.when(aggProbe)(new AggWithProbes())
  }

  class ExampleRaw(probeAgg: Boolean, aggProbe: Boolean)
      extends FixedIORawModule[MixedBundle](new MixedBundle(probeAgg, aggProbe)) {

    val elemWire = Wire(Bool())
    elemWire := false.B
    probe.define(io.elem, probe.ProbeValue(elemWire))

    if (probeAgg) {
      val aggWire = Wire(new Agg())
      aggWire := DontCare
      probe.define(io.agg.get, probe.ProbeValue(aggWire))
    }

    if (aggProbe) {
      val nestedWire = Wire(new AggWithProbes())
      val nestedFoo = WireInit(false.B)
      val nestedBar = WireInit(false.B)
      probe.define(nestedWire.foo, probe.ProbeValue(nestedFoo))
      probe.define(nestedWire.bar, probe.ProbeValue(nestedBar))
      io.nested.get :<>= nestedWire
    }
  }

  class ExampleExt(probeAgg: Boolean, aggProbe: Boolean)
      extends FixedIOExtModule[MixedBundle](new MixedBundle(probeAgg, aggProbe))

  class Parent(probeAgg: Boolean, aggProbe: Boolean) extends Module {
    val childRaw = Module(new ExampleRaw(probeAgg, aggProbe))
    val childExt = Module(new ExampleExt(probeAgg, aggProbe))

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
    if (probeAgg) {
      val outAggRaw = IO(new Agg())
      val probeAggWireRaw = Wire(Probe(new Agg()))
      outAggRaw := probe.read(probeAggWireRaw)
      probeAggWireRaw :<>= childRaw.io.agg.get
      val probeAggWireExt = Wire(Probe(new Agg()))
      val outAggExt = IO(new Agg())
      outAggExt := probe.read(probeAggWireExt)
      probeAggWireExt :<>= childExt.io.agg.get
    }

    // Check Aggregate(Probe)
    if (aggProbe) {
      val probeNestedWireRaw = Wire(new AggWithProbes())
      val outNestedRawFoo = IO(Bool())
      val outNestedRawBar = IO(Bool())
      outNestedRawFoo := probe.read(probeNestedWireRaw.foo)
      outNestedRawBar := probe.read(probeNestedWireRaw.bar)
      probeNestedWireRaw :<>= childRaw.io.nested.get

      val probeNestedWireExt = Wire(new AggWithProbes())
      val outNestedExtFoo = IO(Bool())
      val outNestedExtBar = IO(Bool())
      outNestedExtFoo := probe.read(probeNestedWireExt.foo)
      outNestedExtBar := probe.read(probeNestedWireExt.bar)
      probeNestedWireExt :<>= childExt.io.nested.get
    }

  }

  "FixedIORaw/ExtModules" should "be able to have a Record with Probes of Elements in their IOs" in {
    ChiselStage
      .emitCHIRRTL(new Parent(false, false))
      .fileCheck()(
        """|CHECK-LABEL: module ExampleRaw :
           |CHECK:         output elem : Probe<UInt<1>>
           |CHECK-LABEL: extmodule ExampleExt
           |CHECK:         output elem : Probe<UInt<1>>
           |CHECK-LABEL: public module Parent :
           |CHECK:         define probeElemWireRaw = childRaw.elem
           |CHECK:         define probeElemWireExt = childExt.elem
           |""".stripMargin
      )
  }

  "FixedIORaw/ExtModules" should "be able to have a Record with Probes of Aggregates in their IOs" in {
    ChiselStage
      .emitCHIRRTL(new Parent(true, false))
      .fileCheck()(
        """|CHECK-LABEL: module ExampleRaw :
           |CHECK:         output agg : Probe<{ foo : UInt<1>, bar : UInt<1>}>
           |CHECK-LABEL: extmodule ExampleExt
           |CHECK:         output agg : Probe<{ foo : UInt<1>, bar : UInt<1>}>
           |CHECK-LABEL: public module Parent :
           |CHECK:         define probeAggWireRaw = childRaw.agg
           |CHECK:         define probeAggWireExt = childExt.agg
           |""".stripMargin
      )
  }

  "FixedIORaw/ExtModules" should "be able to have a Record with Aggregates with Probes in their IOs" in {
    ChiselStage
      .emitCHIRRTL(new Parent(false, true))
      .fileCheck()(
        """|CHECK-LABEL: module ExampleRaw :
           |CHECK:         output nested : { foo : Probe<UInt<1>>, bar : Probe<UInt<1>>}
           |CHECK-LABEL: extmodule ExampleExt
           |CHECK:         output nested : { foo : Probe<UInt<1>>, bar : Probe<UInt<1>>}
           |CHECK-LABEL: public module Parent :
           |CHECK:         define probeNestedWireRaw.bar = childRaw.nested.bar
           |CHECK:         define probeNestedWireRaw.foo = childRaw.nested.foo
           |CHECK:         define probeNestedWireExt.bar = childExt.nested.bar
           |CHECK:         define probeNestedWireExt.foo = childExt.nested.foo
           |""".stripMargin
      )
  }
  "FixedIOExtModules" should "be able to have a Probe(Element) as its FixedIO" in {
    class ProbeElemExt extends FixedIOExtModule(Probe(Bool()))
    class Parent extends RawModule {
      val child = Module(new ProbeElemExt)
      val ioElem = IO(Output(Bool()))
      val wireElem = Wire(Probe(Bool()))
      wireElem :<>= child.io
      ioElem := probe.read(wireElem)
    }
    ChiselStage
      .emitCHIRRTL(new Parent)
      .fileCheck()(
        """|CHECK-LABEL: extmodule ProbeElemExt :
           |CHECK:         output io : Probe<UInt<1>>
           |CHECK-LABEL: public module Parent :
           |CHECK:         define wireElem = child.io
           |""".stripMargin
      )
  }
  "FixedIOExtModules" should "be able to have a Probe(Aggregate) as its FixedIO" in {
    class ProbeAggExt extends FixedIOExtModule(Probe(new Agg()))
    class Parent extends RawModule {
      val child = Module(new ProbeAggExt)
      val ioAgg = IO(Output(new Agg()))
      val wireAgg = Wire(Probe(new Agg()))
      wireAgg :<>= child.io
      ioAgg := probe.read(wireAgg)
    }
    ChiselStage
      .emitCHIRRTL(new Parent)
      .fileCheck()(
        """|CHECK-LABEL: extmodule ProbeAggExt :
           |CHECK:         output io : Probe<{ foo : UInt<1>, bar : UInt<1>}>
           |CHECK-LABEL: public module Parent :
           |CHECK:         define wireAgg = child.io
           |""".stripMargin
      )
  }

  "FixedIOExtModules" should "be able to have an Aggregate with Probes as its FixedIO" in {
    class ProbeNestedExt extends FixedIOExtModule(new AggWithProbes())
    class Parent extends RawModule {
      val child = Module(new ProbeNestedExt)
      val ioBar = IO(Output(new Bool()))
      val wireNested = Wire(new AggWithProbes())
      wireNested :<>= child.io
      ioBar := probe.read(wireNested.bar)
    }
    ChiselStage
      .emitCHIRRTL(new Parent)
      .fileCheck()(
        """|CHECK-LABEL: extmodule ProbeNestedExt :
           |CHECK:         output foo : Probe<UInt<1>>
           |CHECK:         output bar : Probe<UInt<1>>
           |CHECK-LABEL: public module Parent :
           |CHECK:         define wireNested.bar = child.bar
           |CHECK:         define wireNested.foo = child.foo
           |""".stripMargin
      )
  }

  "FixedIOModule" should "work with D/I API" in {

    class Foo(width: Int) extends FixedIORawModule[UInt](UInt(width.W)) {
      io :<>= DontCare
    }

    class Bar extends RawModule {
      val fooDef = Definition(new Foo(8))
      val foo1 = Instance(fooDef)
      val foo2 = Instance(fooDef)
      foo1.io :<>= DontCare
      foo2.io :<>= DontCare
    }

    ChiselStage.emitCHIRRTL(new Bar) should include("module Foo :")
  }

  "FixedIOModule" should "create a module with flattened IO with clock and reset" in {
    class Foo(width: Int) extends FixedIOModule[UInt](UInt(width.W)) {
      override def resetType = Module.ResetType.Synchronous
      io :<>= DontCare
    }

    ChiselStage
      .emitCHIRRTL(new Foo(8))
      .fileCheck()(
        """| CHECK: module Foo :
           | CHECK:   input clock : Clock
           | CHECK:   input reset : UInt<1>
           | CHECK:   output io : UInt<8>
           |""".stripMargin
      )
  }
}
