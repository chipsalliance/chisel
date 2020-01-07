// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.incremental.{Cache, ExportCache, ItemTag, Stash}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}
import firrtl.options.{Phase, Stage}
import firrtl.stage.FirrtlCircuitAnnotation
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

case class Simple()(implicit val ttag: ClassTag[Simple]) extends RawModule with Cacheable[Simple] {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  out := in
  val secretWord = "secretz"
}

trait SimpleInterface {
  val in: UInt
  val out: UInt
  val secretWord: String
}

class Bar extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  val simple = InstanceHandle(ItemTag[Simple](Nil), None)
  simple(_.in) := in
  out := simple(_.out)
}

// add materialize
// try macro writing
// add multi-elaboration stage
// fix packages

class ChiselStageSpec extends FlatSpec with Matchers {

  class Fixture { val stage: Stage = new ChiselStage }

  "Stash" should "export a cache after elaboration, then import" in new Fixture {
    // First elaborate Simple module
    val simpleResult = stage.run( Seq(ChiselGeneratorAnnotation(() => new Simple), ExportCache("simple", None, false)) )
    // Print FIRRTL
    simpleResult.collect { case f: FirrtlCircuitAnnotation => println(f.circuit.serialize) }
    // Collect elaborated Simple (inside a Cache)
    val caches = simpleResult.collect{
      case c: Cache => c
    }

    // Secondly, elaborate Bar with caches
    val barResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Bar()),
        ExportCache("bar", None, false),
        NoRunFirrtlCompilerAnnotation,
        PrintFullStackTraceAnnotation
      ) ++ caches
    )
    // Print FIRRTL
    barResult.collect{
      case c: FirrtlCircuitAnnotation =>
        c.circuit.main should be ("Bar")
        c.circuit.modules.map(_.name) should contain ("Simple")
        println(c.circuit.serialize)
    }
  }

  "Stash" should "enable serialization of cache" in new Fixture {
    val directory = "test_run_dir/ChiselStageSpec/caches/simple"
    val packge = "simple"

    // First elaborate Simple module
    //val simpleResult = stage.run( Seq(ChiselGeneratorAnnotation(() => new Simple), ExportCache(packge, Some(directory), false)) )
    //simpleResult.collect { case f: FirrtlCircuitAnnotation => println(f.circuit.serialize) }

    // Secondly, elaborate Bar with caches
    val barResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Bar()),
        Cache.load(directory, packge),
        NoRunFirrtlCompilerAnnotation,
        PrintFullStackTraceAnnotation
      )
    )

    // Print FIRRTL
    barResult.collect{
      case c: FirrtlCircuitAnnotation =>
        c.circuit.main should be ("Bar")
        c.circuit.modules.map(_.name) should contain ("Simple")
        println(c.circuit.serialize)
    }

  }
}

