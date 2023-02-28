// SPDX-License-Identifier: Apache-2.0

package firrtl.testutils

import org.scalatest.flatspec.AnyFlatSpec
import firrtl.ir.Circuit
import firrtl.options.{Dependency, IdentityLike}
import firrtl.passes.{PassExceptions, RemoveEmpty}
import firrtl.stage.Forms
import firrtl._
import firrtl.annotations._
import logger._
import org.scalatest.flatspec.AnyFlatSpec

// An example methodology for testing Firrtl Passes
// Spec class should extend this class
abstract class SimpleTransformSpec extends AnyFlatSpec with FirrtlMatchers with Compiler with LazyLogging {
  // Utility function
  def squash(c: Circuit): Circuit = RemoveEmpty.run(c)

  // Executes the test. Call in tests.
  // annotations cannot have default value because scalatest trait Suite has a default value
  def execute(input: String, check: String, annotations: Seq[Annotation]): CircuitState = {
    val finalState = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annotations))
    val actual = RemoveEmpty.run(parse(finalState.getEmittedCircuit.value)).serialize
    val expected = parse(check).serialize
    logger.debug(actual)
    logger.debug(expected)
    (actual) should be(expected)
    finalState
  }

  def executeWithAnnos(
    input:            String,
    check:            String,
    annotations:      Seq[Annotation],
    checkAnnotations: Seq[Annotation]
  ): CircuitState = {
    val finalState = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annotations))
    val actual = RemoveEmpty.run(parse(finalState.getEmittedCircuit.value)).serialize
    val expected = parse(check).serialize
    logger.debug(actual)
    logger.debug(expected)
    (actual) should be(expected)

    annotations.foreach { anno =>
      logger.debug(anno.serialize)
    }

    finalState.annotations.toSeq.foreach { anno =>
      logger.debug(anno.serialize)
    }
    checkAnnotations.foreach { check =>
      (finalState.annotations.toSeq) should contain(check)
    }
    finalState
  }
  // Executes the test, should throw an error
  // No default to be consistent with execute
  def failingexecute(input: String, annotations: Seq[Annotation]): Exception = {
    intercept[PassExceptions] {
      compile(CircuitState(parse(input), ChirrtlForm, annotations), Seq.empty)
    }
  }
}

@deprecated(
  "Use a TransformManager including 'ReRunResolveAndCheck' as a target. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class CustomResolveAndCheck(form: CircuitForm) extends SeqTransform {
  def inputForm = form
  def outputForm = form
  def transforms: Seq[Transform] = Seq[Transform](new ResolveAndCheck)
}

/** Transform that re-runs resolve and check transforms as late as possible, but before any emitters. */
object ReRunResolveAndCheck extends Transform with DependencyAPIMigration with IdentityLike[CircuitState] {

  override val optionalPrerequisites = Forms.LowFormOptimized
  override val optionalPrerequisiteOf = Forms.ChirrtlEmitters

  override def invalidates(a: Transform) = {
    val resolveAndCheck = Forms.Resolved.toSet -- Forms.WorkingIR
    resolveAndCheck.contains(Dependency.fromTransform(a))
  }

  override def execute(a: CircuitState) = transform(a)

}

trait LowTransformSpec extends SimpleTransformSpec {
  def emitter = new LowFirrtlEmitter
  def transform: Transform
  def transforms: Seq[Transform] = transform +: ReRunResolveAndCheck +: Forms.LowForm.map(_.getObject)
}

trait MiddleTransformSpec extends SimpleTransformSpec {
  def emitter = new MiddleFirrtlEmitter
  def transform: Transform
  def transforms: Seq[Transform] = transform +: ReRunResolveAndCheck +: Forms.MidForm.map(_.getObject)
}

trait HighTransformSpec extends SimpleTransformSpec {
  def emitter = new HighFirrtlEmitter
  def transform: Transform
  def transforms = transform +: ReRunResolveAndCheck +: Forms.HighForm.map(_.getObject)
}
