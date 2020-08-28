// See LICENSE for license details.

package firrtl.stage.phases.tests

import firrtl.{HighFirrtlCompiler, HighFirrtlEmitter, LowFirrtlCompiler}
import firrtl.options.{Dependency, OptionsException}
import firrtl.stage.{CompilerAnnotation, RunFirrtlTransformAnnotation}
import firrtl.stage.phases.ConvertCompilerAnnotations

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ConvertCompilerAnnotationsSpec extends AnyFlatSpec with Matchers {

  class Fixture {
    val phase = new ConvertCompilerAnnotations
  }

  behavior.of(classOf[ConvertCompilerAnnotations].getName)

  it should "convert a deprecated CompilerAnnotation to a RunFirrtlTransformAnnotation" in new Fixture {
    val annotations = Seq(CompilerAnnotation(new HighFirrtlCompiler))
    phase
      .transform(annotations)
      .map {
        case RunFirrtlTransformAnnotation(a) => Dependency.fromTransform(a)
      }
      .toSeq should be(Seq(Dependency[HighFirrtlEmitter]))
  }

  it should "throw an exception if multiple CompilerAnnotations are specified" in new Fixture {
    val annotations = Seq(
      CompilerAnnotation(new HighFirrtlCompiler),
      CompilerAnnotation(new LowFirrtlCompiler)
    )
    intercept[OptionsException] {
      phase.transform(annotations)
    }.getMessage should include("Zero or one CompilerAnnotation may be specified")
  }
}
