// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3._
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation}
import chisel3.stage.phases.Elaborate

class ElaborateSpec extends FlatSpec with Matchers {

  class Foo extends Module {
    override def desiredName: String = "Foo"
    val io = IO(
      new Bundle {
        val in = Input(Bool())
        val out = Output(Bool())
      })

    io.out := ~io.in
  }

  class Bar extends Foo {
    override def desiredName: String = "Bar"
  }

  behavior of Elaborate.name

  it should "expand ChiselGeneratorAnnotations into ChiselCircuitAnnotations and delete originals" in {
    val annotations = Seq( ChiselGeneratorAnnotation(() => new Foo),
                           ChiselGeneratorAnnotation(() => new Bar) )
    val out = Elaborate.transform(annotations)

    info("original annotations removed")
    out.collect{ case a: ChiselGeneratorAnnotation => a } should be (empty)

    info("circuits created with the expected names")
    out.collect{ case a: ChiselCircuitAnnotation => a.circuit.name } should be (Seq("Foo", "Bar"))
  }

}
