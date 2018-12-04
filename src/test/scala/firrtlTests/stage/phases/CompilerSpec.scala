// See LICENSE for license details.

package firrtlTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import firrtl.{Compiler => _, _}
import firrtl.options.{Phase, PhasePrerequisiteException}
import firrtl.stage.{CompilerAnnotation, FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}
import firrtl.stage.phases.Compiler

class CompilerSpec extends FlatSpec with Matchers {

  class Fixture { val phase: Phase = new Compiler }

  behavior of classOf[Compiler].toString

  it should "do nothing for an empty AnnotationSeq" in new Fixture {
    phase.transform(Seq.empty).toSeq should be (empty)
  }

  /** A circuit with a parameterized main (top name) that is different at High, Mid, and Low FIRRTL forms. */
  private def chirrtl(main: String): String =
    s"""|circuit $main:
        |  module $main:
        |    output foo: {bar: UInt}
        |
        |    foo.bar <= UInt<4>("h0")
        |""".stripMargin

  it should "compile a circuit to Low FIRRTL when using the Verilog compiler" in new Fixture {
    val compiler = new VerilogCompiler

    val circuitIn = Parser.parse(chirrtl("top"))
    val circuitOut = compiler.compile(CircuitState(circuitIn, ChirrtlForm), Seq.empty).circuit

    val input = Seq(
      FirrtlCircuitAnnotation(circuitIn),
      CompilerAnnotation(compiler) )

    val expected = Seq(FirrtlCircuitAnnotation(circuitOut))

    phase.transform(input).toSeq should be (expected)
  }

  it should "compile multiple FirrtlCircuitAnnotations" in new Fixture {
    val (nc, hfc, mfc, lfc, vc, svc) = (
      new NoneCompiler,
      new HighFirrtlCompiler,
      new MiddleFirrtlCompiler,
      new LowFirrtlCompiler,
      new VerilogCompiler,
      new SystemVerilogCompiler )
    val (ce, hfe, mfe, lfe, ve, sve) = (
      new ChirrtlEmitter,
      new HighFirrtlEmitter,
      new MiddleFirrtlEmitter,
      new LowFirrtlEmitter,
      new VerilogEmitter,
      new SystemVerilogEmitter )

    val a = Seq(
      /* Default Compiler is HighFirrtlCompiler */
      CompilerAnnotation(hfc),

      /* First compiler group, use NoneCompiler */
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("a"))),
      CompilerAnnotation(nc),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),

      /* Second compiler group, use default HighFirrtlCompiler */
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("b"))),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),
      RunFirrtlTransformAnnotation(hfe),
      EmitCircuitAnnotation(hfe.getClass),

      /* Third compiler group, use MiddleFirrtlCompiler */
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("c"))),
      CompilerAnnotation(mfc),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),
      RunFirrtlTransformAnnotation(hfe),
      EmitCircuitAnnotation(hfe.getClass),
      RunFirrtlTransformAnnotation(mfe),
      EmitCircuitAnnotation(mfe.getClass),

      /* Fourth compiler group, use LowFirrtlCompiler*/
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("d"))),
      CompilerAnnotation(lfc),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),
      RunFirrtlTransformAnnotation(hfe),
      EmitCircuitAnnotation(hfe.getClass),
      RunFirrtlTransformAnnotation(mfe),
      EmitCircuitAnnotation(mfe.getClass),
      RunFirrtlTransformAnnotation(lfe),
      EmitCircuitAnnotation(lfe.getClass),

      /* Fifth compiler group, use VerilogCompiler */
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("e"))),
      CompilerAnnotation(vc),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),
      RunFirrtlTransformAnnotation(hfe),
      EmitCircuitAnnotation(hfe.getClass),
      RunFirrtlTransformAnnotation(mfe),
      EmitCircuitAnnotation(mfe.getClass),
      RunFirrtlTransformAnnotation(lfe),
      EmitCircuitAnnotation(lfe.getClass),
      RunFirrtlTransformAnnotation(ve),
      EmitCircuitAnnotation(ve.getClass),

      /* Sixth compiler group, use SystemVerilogCompiler */
      FirrtlCircuitAnnotation(Parser.parse(chirrtl("f"))),
      CompilerAnnotation(svc),
      RunFirrtlTransformAnnotation(ce),
      EmitCircuitAnnotation(ce.getClass),
      RunFirrtlTransformAnnotation(hfe),
      EmitCircuitAnnotation(hfe.getClass),
      RunFirrtlTransformAnnotation(mfe),
      EmitCircuitAnnotation(mfe.getClass),
      RunFirrtlTransformAnnotation(lfe),
      EmitCircuitAnnotation(lfe.getClass),
      RunFirrtlTransformAnnotation(sve),
      EmitCircuitAnnotation(sve.getClass)
    )

    val output = phase.transform(a)

    info("with the same number of output FirrtlCircuitAnnotations")
    output
      .collect{ case a: FirrtlCircuitAnnotation => a }
      .size should be (6)

    info("and all expected EmittedAnnotations should be generated")
    output
      .collect{ case a: EmittedAnnotation[_] => a }
      .size should be (20)
  }

}
