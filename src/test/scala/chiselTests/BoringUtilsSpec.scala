// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util.Counter
import chisel3.testers.BasicTester
import chisel3.experimental.{BaseModule, ChiselAnnotation, RunFirrtlTransform}
import chisel3.util.experimental.BoringUtils

import firrtl.{CircuitForm, CircuitState, ChirrtlForm, DependencyAPIMigration, Transform}
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Dependency
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}
import firrtl.passes.wiring.{WiringException, WiringTransform}
import firrtl.stage.Forms

abstract class ShouldntAssertTester(cyclesToWait: BigInt = 4) extends BasicTester {
  val dut: BaseModule
  val (_, done) = Counter(true.B, 2)
  when (done) { stop() }
}

class StripNoDedupAnnotation extends Transform with DependencyAPIMigration {
  override def prerequisites = Forms.ChirrtlForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Dependency[WiringTransform] +: Forms.ChirrtlEmitters
  override def invalidates(a: Transform) = false
  def execute(state: CircuitState): CircuitState = {
    state.copy(annotations = state.annotations.filter{ case _: NoDedupAnnotation => false; case _ => true })
  }
}

class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners {

  class BoringInverter extends Module {
    val io = IO(new Bundle{})
    val a = Wire(UInt(1.W))
    val notA = Wire(UInt(1.W))
    val b = Wire(UInt(1.W))
    a := 0.U
    notA := ~a
    b := a
    chisel3.assert(b === 1.U)
    BoringUtils.addSource(notA, "x")
    BoringUtils.addSink(b, "x")
  }

  behavior of "BoringUtils.{addSink, addSource}"

  it should "connect two wires within a module" in {
    runTester(new ShouldntAssertTester { val dut = Module(new BoringInverter) } ) should be (true)
  }

  trait WireX { this: BaseModule =>
    val x = Wire(UInt(4.W))
    chisel3.experimental.annotate(new ChiselAnnotation {
      def toFirrtl: Annotation = DontTouchAnnotation(x.toNamed)
    })
  }

  class Source extends RawModule with WireX {
    val in = IO(Input(UInt()))
    x := in
  }

  class Sink extends RawModule with WireX {
    val out = IO(Output(UInt()))
    x := 0.U // Default value. Output is zero unless we bore...
    out := x
  }

  class Top(val width: Int) extends MultiIOModule {
    /* From the perspective of deduplication, all sources are identical and all sinks are identical. */
    val sources = Seq.fill(3)(Module(new Source))
    val sinks = Seq.fill(6)(Module(new Sink))

    /* Sources are differentiated by their input connections only. */
    sources.zip(Seq(0, 1, 2)).map{ case (a, b) => a.in := b.U }

    /* Sinks are differentiated by their post-boring outputs. */
    sinks.zip(Seq(0, 1, 1, 2, 2, 2)).map{ case (a, b) => chisel3.assert(a.out === b.U) }
  }

  /** This is testing a complicated wiring pattern and exercising
    * the necessity of disabling deduplication for sources and sinks.
    * Without disabling deduplication, this test will fail.
    */
  class TopTester extends ShouldntAssertTester {
    val dut = Module(new Top(4))
    BoringUtils.bore(dut.sources(1).x, Seq(dut.sinks(1).x, dut.sinks(2).x))
    BoringUtils.bore(dut.sources(2).x, Seq(dut.sinks(3).x, dut.sinks(4).x, dut.sinks(5).x))
  }

  trait FailViaDedup { this: TopTester =>
    case object FooAnnotation extends NoTargetAnnotation
    chisel3.experimental.annotate(
      new ChiselAnnotation with RunFirrtlTransform {
        def toFirrtl: Annotation = FooAnnotation
        def transformClass: Class[_ <: Transform] = classOf[StripNoDedupAnnotation] } )
  }

  behavior of "BoringUtils.bore"

  it should "connect across modules using BoringUtils.bore" in {
    runTester(new TopTester) should be (true)
  }

  it should "throw an exception if NoDedupAnnotations are removed" in {
    intercept[WiringException] { runTester(new TopTester with FailViaDedup) }
      .getMessage should startWith ("Unable to determine source mapping for sink")
  }

  class InternalBore extends RawModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))
    out := false.B
    BoringUtils.bore(in, Seq(out))
  }

  class InternalBoreTester extends ShouldntAssertTester {
    val dut = Module(new InternalBore)
    dut.in := true.B
    chisel3.assert(dut.out === true.B)
  }

  it should "work for an internal (same module) BoringUtils.bore" in {
    runTester(new InternalBoreTester) should be (true)
  }

}
