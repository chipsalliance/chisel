// See LICENSE for license details.

package firrtlTests

import java.io.{StringWriter,Writer}
import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.junit.JUnitRunner
import firrtl.ir.Circuit
import firrtl.Parser.UseInfo
import firrtl.passes.{Pass, PassExceptions, RemoveEmpty}
import firrtl._
import logger._

// An example methodology for testing Firrtl Passes
// Spec class should extend this class
abstract class SimpleTransformSpec extends FlatSpec with Matchers with Compiler with LazyLogging {
   // Utility function
   def parse(s: String): Circuit = Parser.parse(s.split("\n").toIterator, infoMode = UseInfo)
   def squash(c: Circuit): Circuit = RemoveEmpty.run(c)

   // Executes the test. Call in tests.
   def execute(annotations: AnnotationMap, input: String, check: String): Unit = {
      val finalState = compileAndEmit(CircuitState(parse(input), ChirrtlForm, Some(annotations)))
      val actual = RemoveEmpty.run(parse(finalState.getEmittedCircuit.value)).serialize
      val expected = parse(check).serialize
      logger.debug(actual)
      logger.debug(expected)
      (actual) should be (expected)
   }
   // Executes the test, should throw an error
   def failingexecute(annotations: AnnotationMap, input: String): Exception = {
      intercept[PassExceptions] {
         compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), Seq.empty)
      }
   }
}

class CustomResolveAndCheck(form: CircuitForm) extends PassBasedTransform {
  private val wrappedTransform = new ResolveAndCheck
  def inputForm = form
  def outputForm = form
  def passSeq = wrappedTransform.passSeq
}

trait LowTransformSpec extends SimpleTransformSpec {
   def emitter = new LowFirrtlEmitter
   def transform: Transform
   def transforms = Seq(
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
   def transforms = Seq(
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
      new ResolveAndCheck(),
      transform
   )
}
