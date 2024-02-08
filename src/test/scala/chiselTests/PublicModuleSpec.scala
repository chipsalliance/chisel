// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe.{Probe, ProbeValue, define}
import circt.stage.ChiselStage

class PublicModuleSpec extends ChiselFlatSpec with MatchesAndOmits {

  class Baz extends RawModule

  class Bar extends RawModule with Public {
    val a = IO(Output(Probe(Bool())))
    val baz = Module(new Baz)

    val b = WireInit(Bool(), true.B)
    dontTouch(b)

    define(a, ProbeValue(b))
  }

  class Foo extends RawModule {
    val a = IO(Output(Probe(Bool())))
    val bar = Module(new Bar)
    define(a, bar.a)
  }

  "The main module" should "be marked public" in {

    println(ChiselStage.emitCHIRRTL(new Foo))
    println(ChiselStage.emitSystemVerilog(new Foo))

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "module Baz",
      "public module Bar",
      "module Foo"
    )(
      "public module Baz",
      "public module Foo"
    )

  }

}
