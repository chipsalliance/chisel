// See LICENSE for license details.

package firrtlTests

import java.io._

import com.typesafe.scalalogging.LazyLogging
import scala.sys.process._
import org.scalatest._
import org.scalatest.prop._
import scala.io.Source

import firrtl._
import firrtl.ir._
import firrtl.Parser.UseInfo
import firrtl.annotations._
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}
import firrtl.util.BackendCompilationUtilities

trait FirrtlRunners extends BackendCompilationUtilities {

  val cppHarnessResourceName: String = "/firrtl/testTop.cpp"

  /** Compiles input Firrtl to Verilog */
  def compileToVerilog(input: String, annotations: AnnotationMap = AnnotationMap(Seq.empty)): String = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val compiler = new VerilogCompiler
    val res = compiler.compileAndEmit(CircuitState(circuit, HighForm, Some(annotations)))
    res.getEmittedCircuit.value
  }
  /** Compile a Firrtl file
    *
    * @param prefix is the name of the Firrtl file without path or file extension
    * @param srcDir directory where all Resources for this test are located
    * @param annotations Optional Firrtl annotations
    */
  def compileFirrtlTest(
      prefix: String,
      srcDir: String,
      customTransforms: Seq[Transform] = Seq.empty,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)): File = {
    val testDir = createTestDirectory(prefix)
    copyResourceToFile(s"${srcDir}/${prefix}.fir", new File(testDir, s"${prefix}.fir"))

    val optionsManager = new ExecutionOptionsManager(prefix) with HasFirrtlOptions {
      commonOptions = CommonOptions(topName = prefix, targetDirName = testDir.getPath)
      firrtlOptions = FirrtlExecutionOptions(
                        infoModeName = "ignore",
                        customTransforms = customTransforms,
                        annotations = annotations.annotations.toList)
    }
    firrtl.Driver.execute(optionsManager)

    testDir
  }
  /** Execute a Firrtl Test
    *
    * @param prefix is the name of the Firrtl file without path or file extension
    * @param srcDir directory where all Resources for this test are located
    * @param verilogPrefixes names of option Verilog resources without path or file extension
    * @param annotations Optional Firrtl annotations
    */
  def runFirrtlTest(
      prefix: String,
      srcDir: String,
      verilogPrefixes: Seq[String] = Seq.empty,
      customTransforms: Seq[Transform] = Seq.empty,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)) = {
    val testDir = compileFirrtlTest(prefix, srcDir, customTransforms, annotations)
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    // Note file copying side effect
    val verilogFiles = verilogPrefixes map { vprefix =>
      val file = new File(testDir, s"$vprefix.v")
      copyResourceToFile(s"$srcDir/$vprefix.v", file)
      file
    }

    verilogToCpp(prefix, testDir, verilogFiles, harness).!
    cppToExe(prefix, testDir).!
    assert(executeExpectingSuccess(prefix, testDir))
  }
}

trait FirrtlMatchers extends Matchers {
  def dontTouch(path: String): Annotation = {
    val parts = path.split('.')
    require(parts.size >= 2, "Must specify both module and component!")
    val name = ComponentName(parts.tail.mkString("."), ModuleName(parts.head, CircuitName("Top")))
    DontTouchAnnotation(name)
  }
  def dontDedup(mod: String): Annotation = {
    require(mod.split('.').size == 1, "Can only specify a Module, not a component or instance")
    NoDedupAnnotation(ModuleName(mod, CircuitName("Top")))
  }
  // Replace all whitespace with a single space and remove leading and
  //   trailing whitespace
  // Note this is intended for single-line strings, no newlines
  def normalized(s: String): String = {
    require(!s.contains("\n"))
    s.replaceAll("\\s+", " ").trim
  }
  def parse(str: String) = Parser.parse(str.split("\n").toIterator, UseInfo)
  /** Helper for executing tests
    * compiler will be run on input then emitted result will each be split into
    * lines and normalized.
    */
  def executeTest(
      input: String,
      expected: Seq[String],
      compiler: Compiler,
      annotations: Seq[Annotation] = Seq.empty) = {
    val annoMap = AnnotationMap(annotations)
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, Some(annoMap)))
    val lines = finalState.getEmittedCircuit.value split "\n" map normalized
    for (e <- expected) {
      lines should contain (e)
    }
  }
}

object FirrtlCheckers extends FirrtlMatchers {
  import matchers._
  implicit class TestingFunctionsOnCircuitState(val state: CircuitState) extends AnyVal {
    def search(pf: PartialFunction[Any, Boolean]): Boolean = state.circuit.search(pf)
  }
  implicit class TestingFunctionsOnCircuit(val circuit: Circuit) extends AnyVal {
    def search(pf: PartialFunction[Any, Boolean]): Boolean = {
      val f = pf.lift
      def rec(node: Any): Boolean = {
        f(node) match {
          // If the partial function is defined on this node, return its result
          case Some(res) => res
          // Otherwise keep digging
          case None =>
            require(node.isInstanceOf[Product] || !node.isInstanceOf[FirrtlNode],
                    "Error! Unexpected FirrtlNode that does not implement Product!")
            val iter = node match {
              case p: Product => p.productIterator
              case i: Iterable[Any] => i.iterator
              case _ => Iterator.empty
            }
            iter.foldLeft(false) {
              case (res, elt) => if (res) res else rec(elt)
            }
        }
      }
      rec(circuit)
    }
  }

  /** Checks that the emitted circuit has the expected line, both will be normalized */
  def containLine(expectedLine: String) = new CircuitStateStringMatcher(expectedLine)

  class CircuitStateStringMatcher(expectedLine: String) extends Matcher[CircuitState] {
    override def apply(state: CircuitState): MatchResult = {
      val emitted = state.getEmittedCircuit.value
      MatchResult(
        emitted.split("\n").map(normalized).contains(normalized(expectedLine)),
        emitted + "\n did not contain \"" + expectedLine + "\"",
        s"${state.circuit.main} contained $expectedLine"
      )
    }
  }

  def containTree(pf: PartialFunction[Any, Boolean]) = new CircuitStatePFMatcher(pf)

  class CircuitStatePFMatcher(pf: PartialFunction[Any, Boolean]) extends Matcher[CircuitState] {
    override def apply(state: CircuitState): MatchResult = {
      MatchResult(
        state.search(pf),
        state.circuit.serialize + s"\n did not contain $pf",
        s"${state.circuit.main} contained $pf"
      )
    }
  }
}

abstract class FirrtlPropSpec extends PropSpec with PropertyChecks with FirrtlRunners with LazyLogging

abstract class FirrtlFlatSpec extends FlatSpec with FirrtlRunners with FirrtlMatchers with LazyLogging

// Who tests the testers?
class TestFirrtlFlatSpec extends FirrtlFlatSpec {
  import FirrtlCheckers._

  val c = parse("""
    |circuit Test:
    |  module Test :
    |    input in : UInt<8>
    |    output out : UInt<8>
    |    out <= in
    |""".stripMargin)
  val state = CircuitState(c, ChirrtlForm)
  val compiled = (new LowFirrtlCompiler).compileAndEmit(state, List.empty)

  // While useful, ScalaTest helpers should be used over search
  behavior of "Search"

  it should "be supported on Circuit" in {
    assert(c search {
      case Connect(_, Reference("out",_), Reference("in",_)) => true
    })
  }
  it should "be supported on CircuitStates" in {
    assert(state search {
      case Connect(_, Reference("out",_), Reference("in",_)) => true
    })
  }
  it should "be supported on the results of compilers" in {
    assert(compiled search {
      case Connect(_, WRef("out",_,_,_), WRef("in",_,_,_)) => true
    })
  }

  // Use these!!!
  behavior of "ScalaTest helpers"

  they should "work for lines of emitted text" in {
    compiled should containLine (s"input in : UInt<8>")
    compiled should containLine (s"output out : UInt<8>")
    compiled should containLine (s"out <= in")
  }

  they should "work for partial functions matching on subtrees" in {
    val UInt8 = UIntType(IntWidth(8)) // BigInt unapply is weird
    compiled should containTree { case Port(_, "in", Input, UInt8) => true }
    compiled should containTree { case Port(_, "out", Output, UInt8) => true }
    compiled should containTree { case Connect(_, WRef("out",_,_,_), WRef("in",_,_,_)) => true }
  }
}

/** Super class for execution driven Firrtl tests */
abstract class ExecutionTest(name: String, dir: String, vFiles: Seq[String] = Seq.empty) extends FirrtlPropSpec {
  property(s"$name should execute correctly") {
    runFirrtlTest(name, dir, vFiles)
  }
}
/** Super class for compilation driven Firrtl tests */
abstract class CompilationTest(name: String, dir: String) extends FirrtlPropSpec {
  property(s"$name should compile correctly") {
    compileFirrtlTest(name, dir)
  }
}


