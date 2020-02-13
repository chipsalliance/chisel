// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.aop.injecting.InjectingAspect
import chisel3.incremental.{Cache, ExportCache, ItemTag, Stash}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}
import firrtl.annotations.DeletedAnnotation
import firrtl.{EmittedCircuitAnnotation, EmittedFirrtlCircuitAnnotation, LowFirrtlCompiler}
import firrtl.options.{Phase, Stage}
import firrtl.stage.{CompilerAnnotation, FirrtlCircuitAnnotation}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

case class Simple()(implicit val ttag: ClassTag[Simple]) extends RawModule with Cacheable[Simple] {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  out := in
  val secretWord = "secretz"
}

class Bar extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  val simple = InstanceHandle(ItemTag[Simple](Nil), None)
  val simple2 = InstanceHandle(ItemTag[Simple](Nil), None)
  simple(_.in) := in
  out := simple(_.out)
  println(simple(_.secretWord))
}

class Simple2 extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  out := in
  val secretWord = "secretz"
}

case class Foo(n: Int)(implicit val ttag: ClassTag[Foo]) extends RawModule with Cacheable[Foo] {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  val finalDriver = (0 until n).foldLeft(in) { (drivingPort, i) =>
    val simple = Module(new Simple2())
    simple.in := drivingPort
    simple.out
  }
  out := finalDriver
}

class FooParent extends RawModule {
  val in:  UInt = IO(Input(UInt(3.W)))
  val out: UInt = IO(Output(UInt(3.W)))
  val foo = InstanceHandle(ItemTag[Foo](Seq(5)), None)
  foo(_.in) := in
  out := foo(_.out)
}

class CountAndDriver()(implicit x: ClassTag[CountAndDriver]) extends Module {//with Cacheable[CountAndDriver] {
  //override def productArity: Int = 0
  //override def productElement(n: Int): Any = Nil

  val ttag = x

  val io = IO(new Bundle {
    val default_value = Input(Bool())
    val count = Input(UInt(16.W))
    val driven_value = Output(Bool())
  })

  io.driven_value := true.B
  val counter = RegInit(0.U(16.W))
  when (~io.default_value)
  {
    when (counter < io.count) { counter := counter + 1.U }
  }
  when (io.count <= counter) { io.driven_value := io.default_value }
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
    val simpleResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Simple),
        ExportCache(packge, Some(directory), false)
      )
    )
    simpleResult.collect { case f: FirrtlCircuitAnnotation => println(f.circuit.serialize) }

    val barStitch = InjectingAspect(
      { b: Bar => Seq(b) },
      { b: Bar =>
        printf("Value! %d", b.out)
      }
    )

    val findSecrets = InjectingAspect(
      { b: Bar => Seq(b) },
      { b: Bar =>
        printf(b.simple(_.secretWord))
      }
    )

    // Secondly, elaborate Bar with caches
    val barResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new Bar()),
        Cache.load(directory, packge),
        PrintFullStackTraceAnnotation,
        CompilerAnnotation(new LowFirrtlCompiler),
        barStitch,
        findSecrets
      )
    )

    // Print FIRRTL
    barResult.collect {
      case DeletedAnnotation("firrtl.stage.phases.WriteEmitted", c: EmittedFirrtlCircuitAnnotation) =>
        println(c.value.value)
        c.value.name should be ("Bar")
    }

  }

  "Stash" should "serialize module with children" in new Fixture {
    // First elaborate Simple module
    val directory = "test_run_dir/ChiselStageSpec/caches/foo"
    val packge = "foo"

    val fooResult = stage.run( Seq(ChiselGeneratorAnnotation(() => new Foo(5)), ExportCache(packge, Some(directory), false)) )
    // Print FIRRTL
    fooResult.collect { case f: FirrtlCircuitAnnotation => println(f.circuit.serialize) }
    // Collect elaborated Simple (inside a Cache)
    val caches = fooResult.collect{
      case c: Cache => c
    }

    // Secondly, elaborate fooParent with caches
    val fooParentResult = stage.run(
      Seq(ChiselGeneratorAnnotation(() => new FooParent()),
        NoRunFirrtlCompilerAnnotation,
        PrintFullStackTraceAnnotation,
        Cache.load(directory, packge)
      )
    )
    // Print FIRRTL
    fooParentResult.collect{
      case c: FirrtlCircuitAnnotation =>
        c.circuit.main should be ("FooParent")
        c.circuit.modules.map(_.name) should contain ("Foo")
        println(c.circuit.serialize)
    }
  }

  "Stash" should "serialize problematic modules" in new Fixture {
    val directory = "test_run_dir/ChiselStageSpec/caches/problematic"
    val packge = "problematic"

    val fooResult = stage.run( Seq(ChiselGeneratorAnnotation(() => new CountAndDriver), ExportCache(packge, Some(directory), false)) )
    // Print FIRRTL
    fooResult.collect { case f: FirrtlCircuitAnnotation => println(f.circuit.serialize) }
    // Collect elaborated Simple (inside a Cache)
    Cache.load(directory, packge)
  }
}
