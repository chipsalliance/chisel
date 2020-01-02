// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.incremental.{Cache, ExportCache, ItemTag, Stash}
import chisel3.stage.phases.Elaborate
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselStage, PrintFullStackTraceAnnotation}
import firrtl.options.{Phase, Stage}
import org.scalatest.{FlatSpec, Matchers}


class ChiselStageSpec extends FlatSpec with Matchers {
  case class Simple() extends RawModule with Cacheable {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    out := in
  }

  class Bar() extends RawModule {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    val simpleHandle = InstanceHandle("simple", ItemTag(classOf[Simple], Nil), None)
    simpleHandle{ simple =>
      simple.in := in
      out := simple.out
    }
  }

  behavior of classOf[Stash].toString

  class Fixture { val stage: Stage = new ChiselStage }

  it should "export a cache after elaboration" in new Fixture {
    val caches = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Simple), ExportCache("simple", None, false))
    ).collect{
      case c: Cache => c
    }

    //info("original annotations removed")
    val x = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Bar()), ExportCache("bar", None, false), PrintFullStackTraceAnnotation ) ++ caches
    ).collect{
      case c: ChiselCircuitAnnotation =>
        c.circuit.name should be ("Bar")
        c
    }
    println(x)
  }

}

