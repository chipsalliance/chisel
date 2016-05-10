package firrtlTests

import java.io.StringWriter

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.{
   Parser,
   Named,
   ModuleName,
   ComponentName,
   CircuitAnnotation,
   StringAnnotation,
   BrittleCircuitAnnotation,
   UnknownCAKind,
   Compiler,
   CompilerResult,
   Annotation,
   RenameMap,
   VerilogCompiler
}

/**
 * An example methodology for testing Firrtl annotations.
 */
abstract class AnnotationSpec extends FlatSpec {
   def parse (s: String): Circuit = Parser.parse(s.split("\n").toIterator)
   def compiler: Compiler
   def input: String
   def getResult (annotation: CircuitAnnotation): CompilerResult = {
      val writer = new StringWriter()
      compiler.compile(parse(input), Seq(annotation), writer)
   }
}


/**
 * An example test for testing module annotations
 */
class BrittleModuleAnnotationSpec extends AnnotationSpec with Matchers {
   val compiler = new VerilogCompiler()
   val input =
"""
circuit Top :
  module Top :
    input a : UInt<1>[2]
    node x = a
"""
   val message = "This is Top"
   val map: Map[Named, Annotation] = Map(ModuleName("Top") -> StringAnnotation(message))
   val annotation = BrittleCircuitAnnotation(UnknownCAKind, map)
   "The annotation" should "get passed through the compiler" in {
      (getResult(annotation).annotations.head == annotation) should be (true)
   }
}
