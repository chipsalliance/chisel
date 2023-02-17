// SPDX-License-Identifier: Apache-2.0

package firrtlTests.stage

import firrtl.stage._

import firrtl.{ir, Parser}
import firrtl.options.Viewer.view
import firrtl.stage.{FirrtlOptions, FirrtlOptionsView}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FirrtlOptionsViewSpec extends AnyFlatSpec with Matchers {

  behavior.of(FirrtlOptionsView.getClass.getName)

  def circuitIR(main: String): ir.Circuit = ir.Circuit(
    ir.NoInfo,
    Seq(
      ir.Module(
        ir.NoInfo,
        main,
        Seq.empty,
        ir.Block(
          ir.DefNode(
            ir.NoInfo,
            "x",
            ir.UIntLiteral(0, ir.IntWidth(1))
          )
        )
      )
    ),
    main
  )

  val grault: ir.Circuit = circuitIR("grault")

  val annotations = Seq(
    /* FirrtlOptions */
    OutputFileAnnotation("bar"),
    InfoModeAnnotation("use"),
    FirrtlCircuitAnnotation(grault)
  )

  it should "construct a view from an AnnotationSeq" in {
    val out = view[FirrtlOptions](annotations)

    out.outputFileName should be(Some("bar"))
    out.infoModeName should be("use")
    out.firrtlCircuit should be(Some(grault))
  }

  /* This test only exists to catch changes to existing behavior. This test does not indicate that this is the correct
   * behavior, only that modifications to existing code will not change behavior that people may expect.
   */
  it should "overwrite or append to earlier annotation information with later annotation information" in {
    val grault_ = circuitIR("thud_")

    val overwrites = Seq(
      OutputFileAnnotation("bar_"),
      InfoModeAnnotation("gen"),
      FirrtlCircuitAnnotation(grault_)
    )

    val out = view[FirrtlOptions](annotations ++ overwrites)

    out.outputFileName should be(Some("bar_"))
    out.infoModeName should be("gen")
    out.firrtlCircuit should be(Some(grault_))
  }

}
