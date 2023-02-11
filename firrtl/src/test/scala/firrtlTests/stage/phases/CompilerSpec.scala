// SPDX-License-Identifier: Apache-2.0

package firrtlTests.stage.phases

import scala.collection.mutable

import firrtl.{Compiler => _, _}
import firrtl.options.Phase
import firrtl.stage.{CompilerAnnotation, FirrtlCircuitAnnotation, Forms, RunFirrtlTransformAnnotation}
import firrtl.stage.phases.Compiler
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CompilerSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new Compiler }

  behavior.of(classOf[Compiler].toString)

  it should "do nothing for an empty AnnotationSeq" in new Fixture {
    phase.transform(Seq.empty).toSeq should be(empty)
  }

  /** A circuit with a parameterized main (top name) that is different at High, Mid, and Low FIRRTL forms. */
  private def chirrtl(main: String): String =
    s"""|circuit $main:
        |  module $main:
        |    output foo: {bar: UInt}
        |    foo.bar <= UInt<4>("h0")
        |""".stripMargin

  it should "compile multiple FirrtlCircuitAnnotations" in new Fixture {
    val (nc, hfc, mfc, lfc) = (
      new NoneCompiler,
      new HighFirrtlCompiler,
      new MiddleFirrtlCompiler,
      new LowFirrtlCompiler
    )
    val (ce, hfe, mfe, lfe) = (
      new ChirrtlEmitter,
      new HighFirrtlEmitter,
      new MiddleFirrtlEmitter,
      new LowFirrtlEmitter
    )

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
      EmitCircuitAnnotation(lfe.getClass)
    )

    val output = phase.transform(a)

    info("with the same number of output FirrtlCircuitAnnotations")
    output.collect { case a: FirrtlCircuitAnnotation => a }.size should be(6)

    info("and all expected EmittedAnnotations should be generated")
    output.collect { case a: EmittedAnnotation[_] => a }.size should be(20)
  }

  it should "run transforms in sequential order" in new Fixture {
    import CompilerSpec.{FirstTransform, SecondTransform}

    val circuitIn = Parser.parse(chirrtl("top"))
    val annotations =
      Seq(
        FirrtlCircuitAnnotation(circuitIn),
        RunFirrtlTransformAnnotation(new FirstTransform),
        RunFirrtlTransformAnnotation(new SecondTransform)
      )
    phase.transform(annotations)

    CompilerSpec.globalState.toSeq should be(Seq(classOf[FirstTransform], classOf[SecondTransform]))
  }

}

object CompilerSpec {

  private[CompilerSpec] val globalState: mutable.Queue[Class[_ <: Transform]] =
    mutable.Queue.empty[Class[_ <: Transform]]

  class LoggingTransform extends Transform {
    override def inputForm = UnknownForm
    override def outputForm = UnknownForm
    override def prerequisites = Forms.HighForm
    override def invalidates(a: Transform) = false
    def execute(c: CircuitState): CircuitState = {
      globalState += this.getClass
      c
    }
  }

  class FirstTransform extends LoggingTransform
  class SecondTransform extends LoggingTransform

}
