// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.incremental.{Cache, ExportCache, ItemTag, Stash}
import chisel3.stage.phases.Elaborate
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselStage, NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}
import firrtl.options.{Phase, Stage}
import firrtl.stage.FirrtlCircuitAnnotation
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
    val simple = InstanceHandle(ItemTag[Simple](Nil), None)
    simple(_.in) := in
    out := simple(_.out)
  }

  behavior of classOf[Stash].toString

  class Fixture { val stage: Stage = new ChiselStage }

  it should "export a cache after elaboration" in new Fixture {
    val simpleResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Simple), ExportCache("simple", None, false))
    )
    simpleResult.collect {
      case f: FirrtlCircuitAnnotation => println(f.circuit.serialize)
    }
    val caches = simpleResult.collect{
      case c: Cache => c
    }

    //info("original annotations removed")
    val barResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Bar()),
        ExportCache("bar", None, false),
        NoRunFirrtlCompilerAnnotation,
        PrintFullStackTraceAnnotation
      ) ++ caches
    )
    barResult.collect{
      case c: FirrtlCircuitAnnotation =>
        c.circuit.main should be ("Bar")
        println(c.circuit.serialize)
    }
  }

}

