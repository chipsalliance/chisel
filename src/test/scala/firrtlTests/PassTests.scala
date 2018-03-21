// See LICENSE for license details.

package firrtlTests

import java.io.{StringWriter,Writer}
import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.junit.JUnitRunner
import firrtl.ir.Circuit
import firrtl.Parser.UseInfo
import firrtl.passes.{Pass, PassExceptions, RemoveEmpty}
import firrtl._
import firrtl.annotations._
import logger._

// An example methodology for testing Firrtl Passes
// Spec class should extend this class
abstract class SimpleTransformSpec extends FlatSpec with FirrtlMatchers with Compiler with LazyLogging {
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
      (actual) should be (expected)
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

class CustomResolveAndCheck(form: CircuitForm) extends SeqTransform {
  def inputForm = form
  def outputForm = form
  def transforms: Seq[Transform] = Seq[Transform](new ResolveAndCheck)
}

trait LowTransformSpec extends SimpleTransformSpec {
   def emitter = new LowFirrtlEmitter
   def transform: Transform
   def transforms: Seq[Transform] = Seq(
      new ChirrtlToHighFirrtl(),
      new IRToWorkingIR(),
      new ResolveAndCheck(),
      new HighFirrtlToMiddleFirrtl(),
      new MiddleFirrtlToLowFirrtl(),
      new CustomResolveAndCheck(LowForm),
      transform
   )
}

trait MiddleTransformSpec extends SimpleTransformSpec {
   def emitter = new MiddleFirrtlEmitter
   def transform: Transform
   def transforms: Seq[Transform] = Seq(
      new ChirrtlToHighFirrtl(),
      new IRToWorkingIR(),
      new ResolveAndCheck(),
      new HighFirrtlToMiddleFirrtl(),
      new CustomResolveAndCheck(MidForm),
      transform
   )
}

trait HighTransformSpec extends SimpleTransformSpec {
   def emitter = new HighFirrtlEmitter
   def transform: Transform
   def transforms = Seq(
      new ChirrtlToHighFirrtl(),
      new IRToWorkingIR(),
      new CustomResolveAndCheck(HighForm),
      transform
   )
}
