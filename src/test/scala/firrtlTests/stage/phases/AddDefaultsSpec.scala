// See LICENSE for license details.

package firrtlTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import firrtl.NoneCompiler
import firrtl.annotations.Annotation
import firrtl.stage.phases.AddDefaults
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.stage.{CompilerAnnotation, InfoModeAnnotation}
import firrtl.options.{Phase, TargetDirAnnotation}

class AddDefaultsSpec extends FlatSpec with Matchers {

  class Fixture { val phase: Phase = new AddDefaults }

  behavior of classOf[AddDefaults].toString

  it should "add expected default annotations and nothing else" in new Fixture {
    val expected = Seq(
      (a: Annotation) => a match { case BlackBoxTargetDirAnno(b) => b == TargetDirAnnotation().directory },
      (a: Annotation) => a match { case CompilerAnnotation(b) => b.getClass == CompilerAnnotation().compiler.getClass },
      (a: Annotation) => a match { case InfoModeAnnotation(b) => b == InfoModeAnnotation().modeName } )

    phase.transform(Seq.empty).zip(expected).map { case (x, f) => f(x) should be (true) }
  }

  it should "not overwrite existing annotations" in new Fixture {
    val input = Seq(
      BlackBoxTargetDirAnno("foo"),
      CompilerAnnotation(new NoneCompiler()),
      InfoModeAnnotation("ignore"))

    phase.transform(input).toSeq should be (input)
  }
}
