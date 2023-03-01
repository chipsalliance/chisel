// SPDX-License-Identifier: Apache-2.0

package circtTests.stage.phases

import firrtl.ir
import firrtl.options.Phase
import firrtl.stage.{FirrtlCircuitAnnotation, OutputFileAnnotation}
import circt.stage.phases.AddImplicitOutputFile
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AddImplicitOutputFileSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new AddImplicitOutputFile }

  val circuit = ir.Circuit(
    ir.NoInfo,
    Seq(
      ir.Module(
        ir.NoInfo,
        "foo",
        Seq.empty,
        ir.Block(
          ir.DefNode(
            ir.NoInfo,
            "a",
            ir.UIntLiteral(0, ir.IntWidth(1))
          )
        )
      )
    ),
    "foo"
  )

  behavior.of(classOf[AddImplicitOutputFile].toString)

  it should "default to an output file named 'a'" in new Fixture {
    phase.transform(Seq.empty).toSeq should be(Seq(OutputFileAnnotation("a")))
  }

  it should "set the output file based on a FirrtlCircuitAnnotation's main" in new Fixture {
    val in = Seq(FirrtlCircuitAnnotation(circuit))
    val out = OutputFileAnnotation(circuit.main) +: in
    phase.transform(in).toSeq should be(out)
  }

  it should "do nothing if an OutputFileAnnotation or EmitAllModulesAnnotation already exists" in new Fixture {

    info("OutputFileAnnotation works")
    val outputFile = Seq(OutputFileAnnotation("Bar"), FirrtlCircuitAnnotation(circuit))
    phase.transform(outputFile).toSeq should be(outputFile)

  }

}
