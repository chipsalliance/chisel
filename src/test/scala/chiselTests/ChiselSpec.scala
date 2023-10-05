// SPDX-License-Identifier: Apache-2.0

package chiselTests

import _root_.logger.Logger
import chisel3._
import chisel3.aop.Aspect
import chisel3.stage.{ChiselGeneratorAnnotation, PrintFullStackTraceAnnotation}
import chisel3.testers._
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import chisel3.simulator._
import svsim._
import firrtl.annotations.Annotation
import firrtl.ir.Circuit
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.util.BackendCompilationUtilities
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import org.scalacheck._
import org.scalatest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import java.io.{ByteArrayOutputStream, PrintStream}
import java.security.Permission
import scala.reflect.ClassTag
import java.text.SimpleDateFormat
import java.util.Calendar

/** Common utility functions for Chisel unit tests. */
trait ChiselRunners extends Assertions {
  private val verilatorBackend = verilator.Backend.initializeFromProcessEnvironment()
  private val timeStampFormat = new SimpleDateFormat("yyyyMMddHHmmss")
  def runTester(
    t:                    => BasicTester,
    additionalVResources: Seq[String] = Seq(),
    annotations:          AnnotationSeq = Seq()
  ): Boolean = {
    val workspacePath = Seq(
      "test_run_dir",
      getClass.getSimpleName,
      /// This is taken from the legacy `TesterDriver` class. It isn't ideal and we hope to improve this eventually.
      timeStampFormat.format(Calendar.getInstance().getTime())
    ).mkString("/")
    val workspace = new Workspace(workspacePath)
    workspace.reset()
    val elaboratedModule = workspace.elaborateGeneratedModule({ () => t })
    additionalVResources.foreach(workspace.addPrimarySourceFromResource(getClass(), _))
    workspace.generateAdditionalSources()
    val simulation = workspace.compile(verilatorBackend)(
      "verilator", {
        import CommonCompilationSettings._
        CommonCompilationSettings(
          availableParallelism = AvailableParallelism.UpTo(Runtime.getRuntime().availableProcessors()),
          optimizationStyle = OptimizationStyle.OptimizeForCompilationSpeed,
          verilogPreprocessorDefines = Seq(
            VerilogPreprocessorDefine("ASSERT_VERBOSE_COND", s"!${Workspace.testbenchModuleName}.reset"),
            VerilogPreprocessorDefine("PRINTF_COND", s"!${Workspace.testbenchModuleName}.reset"),
            VerilogPreprocessorDefine("STOP_COND", s"!${Workspace.testbenchModuleName}.reset")
          )
        )
      },
      verilator.Backend
        .CompilationSettings(disabledWarnings = Seq("WIDTH", "STMTDLY"), disableFatalExitOnWarnings = true),
      customSimulationWorkingDirectory = None,
      verbose = false
    )
    try {
      simulation
        .runElaboratedModule(elaboratedModule) { module =>
          val dut = module.wrapped
          val clock = module.port(dut.clock)
          val reset = module.port(dut.reset)
          reset.set(1)
          clock.tick(
            timestepsPerPhase = 1,
            maxCycles = 10,
            inPhaseValue = 1,
            outOfPhaseValue = 0,
            sentinel = None
          )
          reset.set(0)

          val elapsedCycles = clock.tick(
            timestepsPerPhase = 1,
            maxCycles = 10000,
            inPhaseValue = 1,
            outOfPhaseValue = 0,
            sentinel = None
          )
        }
      true
    } catch {
      // We eventually want to have a more structured way of detecting assertions, but this works for now.
      case svsim.Simulation.UnexpectedEndOfMessages =>
        val filename = s"$workspacePath/workdir-verilator/simulation-log.txt"
        for (line <- scala.io.Source.fromFile(filename).getLines()) {
          if (line.contains("Verilog $finish")) {
            // We don't immediately exit on $finish, so we need to ignore assertions that happen after a call to $finish
            return true
          }
          if (line.contains("Assertion failed")) {
            return false
          }
        }
        return true
    }
  }
  def assertTesterPasses(
    t:                    => BasicTester,
    additionalVResources: Seq[String] = Seq(),
    annotations:          AnnotationSeq = Seq()
  ): Unit = {
    assert(runTester(t, additionalVResources, annotations))
  }
  def assertTesterFails(
    t:                    => BasicTester,
    additionalVResources: Seq[String] = Seq(),
    annotations:          Seq[chisel3.aop.Aspect[_]] = Seq()
  ): Unit = {
    assert(!runTester(t, additionalVResources, annotations))
  }

  def assertKnownWidth(expected: Int)(gen: => Data): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(testPoint.getWidth === expected)
      // Sanity check that firrtl doesn't change the width
      testPoint := 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      dontTouch(testPoint)
    }
    val verilog = ChiselStage.emitSystemVerilog(new TestModule, Array.empty, Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("testPoint"))
      case 1 =>
        assert(verilog.contains(s"testPoint"))
        assert(!verilog.contains(s"0] testPoint"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] testPoint"))
    }
  }

  def assertInferredWidth(expected: Int)(gen: => Data): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(!testPoint.isWidthKnown, s"Asserting that width should be inferred yet width is known to Chisel!")
      testPoint := 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      dontTouch(testPoint)
    }
    val verilog = ChiselStage.emitSystemVerilog(new TestModule, Array.empty, Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("testPoint"))
      case 1 =>
        assert(verilog.contains(s"testPoint"))
        assert(!verilog.contains(s"0] testPoint"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] testPoint"))
    }
  }

  /** Compiles a Chisel Module to Verilog
    * NOTE: This uses the "test_run_dir" as the default directory for generated code.
    * @param t the generator for the module
    * @return the Verilog code as a string.
    */
  def compile(t: => RawModule): String = {
    (new ChiselStage)
      .execute(
        Array("--target-dir", BackendCompilationUtilities.createTestDirectory(this.getClass.getSimpleName).toString),
        Seq(ChiselGeneratorAnnotation(() => t), CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog))
      )
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a.value
      }
      .getOrElse(fail("No Verilog circuit was emitted by the FIRRTL compiler!"))
  }

  def elaborateAndGetModule[A <: RawModule](t: => A): A = {
    var res: Any = null
    ChiselStage.emitCHIRRTL {
      res = t
      res.asInstanceOf[A]
    }
    res.asInstanceOf[A]
  }

  /** Compiles a Chisel Module to FIRRTL
    * NOTE: This uses the "test_run_dir" as the default directory for generated code.
    * @param t the generator for the module
    * @return The FIRRTL Circuit and Annotations _before_ FIRRTL compilation
    */
  def getFirrtlAndAnnos(t: => RawModule, providedAnnotations: Seq[Annotation] = Nil): (Circuit, Seq[Annotation]) = {
    TestUtils.getChirrtlAndAnnotations(t, providedAnnotations)
  }
}

/** Spec base class for BDD-style testers. */
abstract class ChiselFlatSpec extends AnyFlatSpec with ChiselRunners with Matchers

/** Spec base class for BDD-style testers. */
abstract class ChiselFreeSpec extends AnyFreeSpec with ChiselRunners with Matchers

/** Spec base class for BDD-style testers. */
abstract class ChiselFunSpec extends AnyFunSpec with ChiselRunners with Matchers

/** Spec base class for property-based testers. */
abstract class ChiselPropSpec extends AnyPropSpec with ChiselRunners with ScalaCheckPropertyChecks with Matchers {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 8, minSize = 1, sizeRange = 3)

  // Generator for small positive integers.
  val smallPosInts = Gen.choose(1, 4)

  // Generator for positive (ascending or descending) ranges.
  def posRange: Gen[Range] = for {
    dir <- Gen.oneOf(true, false)
    step <- Gen.choose(1, 3)
    m <- Gen.choose(1, 10)
    n <- Gen.choose(1, 10)
  } yield {
    if (dir) {
      Range(m, (m + n) * step, step)
    } else {
      Range((m + n) * step, m, -step)
    }
  }

  // Generator for widths considered "safe".
  val safeUIntWidth = Gen.choose(1, 30)

  // Generators for integers that fit within "safe" widths.
  val safeUInts = Gen.choose(0, (1 << 30))

  // Generators for vector sizes.
  val vecSizes = Gen.choose(0, 4)

  // Generator for string representing an arbitrary integer.
  val binaryString = for (i <- Arbitrary.arbitrary[Int]) yield "b" + i.toBinaryString

  // Generator for a sequence of Booleans of size n.
  def enSequence(n: Int): Gen[List[Boolean]] = Gen.containerOfN[List, Boolean](n, Gen.oneOf(true, false))

  // Generator which gives a width w and a list (of size n) of numbers up to w bits.
  def safeUIntN(n: Int): Gen[(Int, List[Int])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i)

  // Generator which gives a width w and a numbers up to w bits.
  val safeUInt = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i)

  // Generator which gives a width w and a list (of size n) of a pair of numbers up to w bits.
  def safeUIntPairN(n: Int): Gen[(Int, List[(Int, Int)])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
    j <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i.zip(j))

  // Generator which gives a width w and a pair of numbers up to w bits.
  val safeUIntPair = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
    j <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i, j)
}

trait Utils {

  /** Run some Scala thunk and return STDOUT and STDERR as strings.
    * @param thunk some Scala code
    * @return a tuple containing STDOUT, STDERR, and what the thunk returns
    */
  def grabStdOutErr[T](thunk: => T): (String, String, T) = {
    val stdout, stderr = new ByteArrayOutputStream()
    val ret = scala.Console.withOut(stdout) { scala.Console.withErr(stderr) { thunk } }
    (stdout.toString, stderr.toString, ret)
  }

  /** Run some Scala thunk and return all logged messages as Strings
    * @param thunk some Scala code
    * @return a tuple containing LOGGED, and what the thunk returns
    */
  def grabLog[T](thunk: => T): (String, T) = {
    val baos = new ByteArrayOutputStream()
    val stream = new PrintStream(baos, true, "utf-8")
    val ret = Logger.makeScope(Nil) {
      Logger.setOutput(stream)
      thunk
    }
    (baos.toString, ret)
  }

  /** Encodes a System.exit exit code
    * @param status the exit code
    */
  private case class ExitException(status: Int) extends SecurityException(s"Found a sys.exit with code $status")

  /** A tester which runs generator and uses an aspect to check the returned object
    * @param gen function to generate a Chisel module
    * @param f a function to check the Chisel module
    * @tparam T the Chisel module class
    */
  def aspectTest[T <: RawModule](gen: () => T)(f: T => Unit)(implicit scalaMajorVersion: Int): Unit = {
    // Runs chisel stage
    def run[T <: RawModule](gen: () => T, annotations: AnnotationSeq): AnnotationSeq = {
      new ChiselStage().run(
        Seq(
          ChiselGeneratorAnnotation(gen),
          CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL),
          PrintFullStackTraceAnnotation
        ) ++ annotations
      )
    }
    // Creates a wrapping aspect to contain checking function
    case object BuiltAspect extends Aspect[T] {
      override def toAnnotation(top: T): AnnotationSeq = { f(top); Nil }
    }
    val currentMajorVersion = scala.util.Properties.versionNumberString.split('.')(1).toInt
    if (currentMajorVersion >= scalaMajorVersion) {
      run(gen, Seq(BuiltAspect))
    }
  }

  /** Run some code and rethrow an exception with a specific type if an exception of that type occurs anywhere in the
    * stack trace.
    *
    * This is useful for "extracting" one exception that may be wrapped by other exceptions.
    *
    * Example usage:
    * {{{
    * a [ChiselException] should be thrownBy extractCause[ChiselException] { /* ... */ }
    * }}}
    *
    * @param thunk some code to run
    * @tparam A the type of the exception to extract
    * @return nothing
    */
  def extractCause[A <: Throwable: ClassTag](thunk: => Any): Unit = {
    def unrollCauses(a: Throwable): Seq[Throwable] = a match {
      case null => Seq.empty
      case _    => a +: unrollCauses(a.getCause)
    }

    val exceptions: Seq[_ <: Throwable] =
      try {
        thunk
        Seq.empty
      } catch {
        case a: Throwable => unrollCauses(a)
      }

    exceptions.collectFirst { case a: A => a } match {
      case Some(a) => throw a
      case None =>
        exceptions match {
          case Nil    => ()
          case h :: t => throw h
        }
    }

  }
}

/** Contains helpful function to assert both statements to match, and statements to omit */
trait MatchesAndOmits {
  private def matches(lines: List[String], matchh: String): Option[String] = lines.filter(_.contains(matchh)).lastOption
  private def omits(line:    String, omit:         String): Option[(String, String)] =
    if (line.contains(omit)) Some((omit, line)) else None
  private def omits(lines: List[String], omit: String): Seq[(String, String)] = lines.flatMap { omits(_, omit) }
  def matchesAndOmits(output: String)(matchList: String*)(omitList: String*): Unit = {
    val lines = output.split("\n").toList
    val unmatched = matchList.flatMap { m =>
      if (matches(lines, m).nonEmpty) None else Some(m)
    }.map(x => s"  > $x was unmatched")
    val unomitted = omitList.flatMap { o => omits(lines, o) }.map {
      case (o, l) => s"  > $o was not omitted in ($l)"
    }
    val results = unmatched ++ unomitted
    assert(results.isEmpty, results.mkString("\n"))
  }
}
