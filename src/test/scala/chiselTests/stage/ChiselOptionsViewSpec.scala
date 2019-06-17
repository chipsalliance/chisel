// See LICENSE for license details.

package chiselTests.stage

import org.scalatest.{FlatSpec, Matchers}

import firrtl.options.Viewer.view

import chisel3.stage._
import chisel3.internal.firrtl.Circuit

class ChiselOptionsViewSpec extends FlatSpec with Matchers {

  behavior of ChiselOptionsView.getClass.getName

  it should "construct a view from an AnnotationSeq" in {
    val bar = Circuit("bar", Seq.empty, Seq.empty)
    val annotations = Seq(
      NoRunFirrtlCompilerAnnotation,
      PrintFullStackTraceAnnotation,
      ChiselOutputFileAnnotation("foo"),
      ChiselCircuitAnnotation(bar)
    )
    val out = view[ChiselOptions](annotations)

    info("runFirrtlCompiler was set to false")
    out.runFirrtlCompiler should be (false)

    info("printFullStackTrace was set to true")
    out.printFullStackTrace should be (true)

    info("outputFile was set to 'foo'")
    out.outputFile should be (Some("foo"))

    info("chiselCircuit was set to circuit 'bar'")
    out.chiselCircuit should be (Some(bar))

  }

}
