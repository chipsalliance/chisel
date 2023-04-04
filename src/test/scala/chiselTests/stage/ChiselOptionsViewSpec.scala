// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage

import firrtl.options.Viewer.view
import firrtl.RenameMap

import chisel3.stage._
import chisel3.internal.firrtl.Circuit
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ChiselOptionsViewSpec extends AnyFlatSpec with Matchers {

  behavior.of(ChiselOptionsView.getClass.getName)

  it should "construct a view from an AnnotationSeq" in {
    val bar = Circuit("bar", Seq.empty, Seq.empty, RenameMap())
    val annotations = Seq(
      PrintFullStackTraceAnnotation,
      ChiselOutputFileAnnotation("foo"),
      ChiselCircuitAnnotation(bar)
    )
    val out = view[ChiselOptions](annotations)

    info("printFullStackTrace was set to true")
    out.printFullStackTrace should be(true)

    info("outputFile was set to 'foo'")
    out.outputFile should be(Some("foo"))

    info("chiselCircuit was set to circuit 'bar'")
    out.chiselCircuit should be(Some(bar))

  }

}
