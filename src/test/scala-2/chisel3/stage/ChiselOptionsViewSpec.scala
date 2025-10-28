// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.options.Viewer.view
import firrtl.RenameMap

import chisel3.ElaboratedCircuit
import chisel3.stage._
import chisel3.internal.firrtl.ir.Circuit
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ChiselOptionsViewSpec extends AnyFlatSpec with Matchers {

  behavior.of(ChiselOptionsView.getClass.getName)

  it should "construct a view from an AnnotationSeq" in {
    val bar = Circuit("bar", Seq.empty, Seq.empty, RenameMap("bar"), Seq.empty, Seq.empty, Seq.empty, false, Seq.empty)
    val circuit = ElaboratedCircuit(bar, Seq.empty)
    val annotations = Seq(
      PrintFullStackTraceAnnotation,
      ChiselOutputFileAnnotation("foo"),
      ChiselCircuitAnnotation(circuit),
      SuppressSourceInfoAnnotation
    )
    val out = view[ChiselOptions](annotations)

    info("printFullStackTrace was set to true")
    out.printFullStackTrace should be(true)

    info("outputFile was set to 'foo'")
    out.outputFile should be(Some("foo"))

    info("elaboratedCircuit was set to circuit 'circuit'")
    out.elaboratedCircuit should be(Some(circuit))

    info("suppressSourceInfo was set to true")
    out.suppressSourceInfo should be(true)
  }

}
