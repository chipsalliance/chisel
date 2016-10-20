package firrtlTests

import com.typesafe.scalalogging.LazyLogging
import java.io.{StringWriter,Writer}
import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.junit.JUnitRunner
import firrtl.ir.Circuit
import firrtl.Parser.IgnoreInfo
import firrtl.passes.{Pass, PassExceptions, RemoveEmpty}
import firrtl.{
   Transform,
   PassBasedTransform,
   CircuitState,
   CircuitForm,
   ChirrtlForm,
   HighForm,
   MidForm,
   LowForm,
   SimpleRun,
   ChirrtlToHighFirrtl,
   IRToWorkingIR,
   ResolveAndCheck,
   HighFirrtlToMiddleFirrtl,
   MiddleFirrtlToLowFirrtl,
   FirrtlEmitter,
   Compiler,
   Parser
}
import firrtl.Annotations.AnnotationMap


// An example methodology for testing Firrtl Passes
// Spec class should extend this class
abstract class SimpleTransformSpec extends FlatSpec with Matchers with Compiler with LazyLogging {
   def emitter = new FirrtlEmitter

   // Utility function
   def parse(s: String): Circuit = Parser.parse(s.split("\n").toIterator, infoMode = IgnoreInfo)

   // Executes the test. Call in tests.
   def execute(writer: Writer, annotations: AnnotationMap, input: String, check: String) = {
      compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
      val actual = RemoveEmpty.run(parse(writer.toString)).serialize
      val expected = parse(check).serialize
      logger.debug(actual)
      logger.debug(expected)
      (actual) should be (expected)
   }
   // Executes the test, should throw an error
   def failingexecute(writer: Writer, annotations: AnnotationMap, input: String): Exception = {
      intercept[PassExceptions] {
         compile(CircuitState(parse(input), ChirrtlForm, Some(annotations)), writer)
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
   def transform: Transform
   def transforms = Seq(
      new ChirrtlToHighFirrtl(),
      new IRToWorkingIR(),
      new ResolveAndCheck(),
      transform
   )
}
