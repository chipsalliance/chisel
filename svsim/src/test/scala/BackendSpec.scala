// SPDX-License-Identifier: Apache-2.0

package svsimTests

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import svsim._
import java.io.{BufferedReader, FileReader}
import java.nio.file.Path
import scala.util.matching.Regex
import svsimTests.Resources.TestWorkspace

class VCSSpec extends BackendSpec {

  override val finishRe = "^\\$finish called from file.*$".r

  import vcs.Backend.CompilationSettings._
  val backend = vcs.Backend.initializeFromProcessEnvironment()
  val compilationSettings = vcs.Backend.CompilationSettings(
    traceSettings = TraceSettings(
      enableVcd = true
    ),
    licenceExpireWarningTimeout = Some(0),
    archOverride = Some("linux")
  )
  backend match {
    case Some(backend) => test("vcs", backend)(compilationSettings)
    case None          => ignore("Svsim backend 'vcs'") {}
  }
}

/**
  * A Backend trivially wrapping the Verilator backend to demonstrate custom out-of-package backends.
  */
case class CustomVerilatorBackend(actualBackend: verilator.Backend) extends Backend {
  type CompilationSettings = verilator.Backend.CompilationSettings
  def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): Backend.Parameters = {
    actualBackend.generateParameters(
      outputBinaryName,
      topModuleName,
      additionalHeaderPaths,
      commonSettings,
      backendSpecificSettings
    )
  }

  override def escapeDefine(string: String): String = string

  override val assertionFailed = "^.*Assertion failed in.*".r
}

class VerilatorSpec extends BackendSpec {

  override val finishRe = "^.*: Verilog \\$finish".r

  import verilator.Backend.CompilationSettings._
  val backend = CustomVerilatorBackend(verilator.Backend.initializeFromProcessEnvironment())
  val compilationSettings = verilator.Backend.CompilationSettings(
    traceStyle = Some(TraceStyle.Vcd(traceUnderscore = false))
  )
  test("verilator", backend)(compilationSettings)
}

trait BackendSpec extends AnyFunSpec with Matchers {

  /** A regulare expression that matches a line in a backend-specific simulation
    * log indicating that a Verilog `$finish` took place.
    */
  def finishRe: Regex

  def test[Backend <: svsim.Backend](
    name:    String,
    backend: Backend
  )(compilationSettings: backend.CompilationSettings) = {
    describe(s"Svsim backend '$name'") {
      val workspace = new svsim.Workspace(path = s"test_run_dir/${getClass().getSimpleName()}")
      var simulation: Simulation = null

      it("fails to compile a testbench without generated sources") {
        import Resources._
        workspace.reset()
        workspace.elaborateGCD()
        assertThrows[Exception] {
          simulation = workspace.compile(
            backend
          )(
            workingDirectoryTag = name,
            commonSettings = CommonCompilationSettings(),
            backendSpecificSettings = compilationSettings,
            customSimulationWorkingDirectory = None,
            verbose = false
          )
        }
      }

      it("compiles an example testbench") {
        import Resources._
        workspace.reset()
        workspace.elaborateGCD()
        workspace.generateAdditionalSources()
        simulation = workspace.compile(
          backend
        )(
          workingDirectoryTag = name,
          commonSettings = CommonCompilationSettings(),
          backendSpecificSettings = compilationSettings,
          customSimulationWorkingDirectory = None,
          verbose = false
        )
      }

      it("fails with a trailing exception") {
        final case class TrailingException() extends Throwable
        assertThrows[TrailingException] {
          simulation.run(
            verbose = false,
            executionScriptLimit = None
          ) { controller =>
            val clock = controller.port("clock")
            clock.check { _ =>
              throw TrailingException()
            }
          }
        }
      }

      it("simulates correctly") {
        simulation.run(
          verbose = false,
          executionScriptLimit = None
        ) { controller =>
          val clock = controller.port("clock")
          val a = controller.port("a")
          val b = controller.port("b")
          val loadValues = controller.port("loadValues")
          val isValid = controller.port("isValid")
          val result = controller.port("result")

          controller.setTraceEnabled(true)

          val aVal = -0x0240000000000000L
          val bVal = +0x0180000000000000L
          a.set(aVal)
          a.check(isSigned = true) { value =>
            assert(value.asBigInt === aVal)
          }
          a.check(isSigned = false) { value =>
            assert(value.asBigInt === 0x7dc0000000000000L)
          }
          b.set(bVal)
          b.check(isSigned = true) { value =>
            assert(value.asBigInt === bVal)
          }
          loadValues.set(1)

          // Maunally tick the clock
          clock.set(0)
          controller.run(1)
          clock.set(1)
          controller.run(1)

          loadValues.set(0)
          clock.tick(
            inPhaseValue = 0,
            outOfPhaseValue = 1,
            timestepsPerPhase = 1,
            maxCycles = 10,
            sentinel = Some(isValid, 1)
          )

          var isValidChecked: Boolean = false
          isValid.check { value =>
            isValidChecked = true
            assert(value.asBigInt === 1)
          }
          assert(isValidChecked === false)
          var isResultChecked: Boolean = false
          result.check { value =>
            isResultChecked = true
            assert(value.asBigInt === 0x00c0000000000000L)
          }
          assert(isResultChecked === false)

          controller.completeInFlightCommands()
          assert(isValidChecked === true)
          assert(isResultChecked === true)

          val log = controller.readLog()
          assert(log.contains("Calculating GCD of 7dc0000000000000 and 0180000000000000"))
          assert(log.contains("Calculated GCD to be 00c0000000000000"))
        }

        val traceReader = new BufferedReader(new FileReader(s"${simulation.workingDirectoryPath}/trace.vcd"))
        traceReader.lines().count() must be > 1L
      }

      // Check sendBits()
      it("communicates data to the Scala side correctly (#4593)") {
        import Resources._
        workspace.reset()
        workspace.elaborateSIntTest()
        workspace.generateAdditionalSources()
        simulation = workspace.compile(
          backend
        )(
          workingDirectoryTag = name,
          commonSettings = CommonCompilationSettings(),
          backendSpecificSettings = compilationSettings,
          customSimulationWorkingDirectory = None,
          verbose = false
        )
        simulation.run(
          verbose = false,
          executionScriptLimit = None
        ) { controller =>
          val bitWidths: Seq[Int] = List(8, 31, 32, 33)
          val outConstPorts = bitWidths.map(b => controller.port(s"out_const_${b}"))

          controller.setTraceEnabled(true)

          for (idxBitWidth <- 0 until bitWidths.length) {
            val bitWidth = bitWidths(idxBitWidth)
            val outConst = outConstPorts(idxBitWidth)
            val outConstVal = BigInt(-1) << (bitWidth - 1)
            var isOutConstChecked: Boolean = false
            outConst.check(isSigned = true) { value =>
              isOutConstChecked = true
              assert(value.asBigInt === outConstVal)
            }
            assert(isOutConstChecked === false)
            controller.completeInFlightCommands()
            assert(isOutConstChecked === true)
          }
        }
      }

      // Check both scanHexBits() and sendBits()
      it("communicates data from and to the Scala side correctly") {
        simulation.run(
          verbose = false,
          executionScriptLimit = None
        ) { controller =>
          val bitWidths: Seq[Int] = List(8, 31, 32, 33)
          val inPorts = bitWidths.map(b => controller.port(s"in_${b}"))
          val outPorts = bitWidths.map(b => controller.port(s"out_${b}"))

          controller.setTraceEnabled(true)

          // Some values near bounds
          def boundValues(bitWidth: Int): Seq[BigInt] = {
            val minVal = BigInt(-1) << (bitWidth - 1)
            val maxVal = (BigInt(1) << (bitWidth - 1)) - 1
            val deltaRange = maxVal.min(BigInt(257))
            val valueNearZero = for { v <- -deltaRange to deltaRange } yield v
            val valueNearMax = for { delta <- BigInt(0) to deltaRange } yield maxVal - delta
            val valueNearMin = for { delta <- BigInt(0) to deltaRange } yield minVal + delta
            valueNearMin ++ valueNearZero ++ valueNearMax
          }

          for (idxBitWidth <- 0 until bitWidths.length) {
            val bitWidth = bitWidths(idxBitWidth)
            val in = inPorts(idxBitWidth)
            val out = outPorts(idxBitWidth)

            val inValues = boundValues(bitWidth)
            val outValues = inValues
            for ((inVal, outVal) <- inValues.zip(outValues)) {
              in.set(inVal)
              in.check(isSigned = true) { value =>
                assert(value.asBigInt === inVal)
              }
              var isOutChecked: Boolean = false
              out.check(isSigned = true) { value =>
                isOutChecked = true
                assert(value.asBigInt === inVal)
              }
              assert(isOutChecked === false)

              controller.completeInFlightCommands()
              assert(isOutChecked === true)
            }
          }
        }
      }

      it("handles initial statements correctly (#3962)") {
        workspace.reset()
        workspace.elaborateInitialTest()
        workspace.generateAdditionalSources()
        simulation = workspace.compile(
          backend
        )(
          workingDirectoryTag = name,
          commonSettings = CommonCompilationSettings(),
          backendSpecificSettings = compilationSettings,
          customSimulationWorkingDirectory = None,
          verbose = false
        )
        simulation.run(
          verbose = false,
          executionScriptLimit = None
        ) { controller =>
          controller.port("b").check(isSigned = false) { value =>
            assert(value.asBigInt === 1)
          }
        }
      }

      it("ends the simulation on '$finish' (#4700)") {
        workspace.reset()
        workspace.elaborateFinishTest()
        workspace.generateAdditionalSources()
        simulation = workspace.compile(
          backend
        )(
          workingDirectoryTag = name,
          commonSettings = CommonCompilationSettings(),
          backendSpecificSettings = compilationSettings,
          customSimulationWorkingDirectory = None,
          verbose = false
        )
        simulation.run(
          verbose = false,
          executionScriptLimit = None
        ) { controller =>
          val clock = controller.port("clock")
          clock.tick(
            inPhaseValue = 0,
            outOfPhaseValue = 1,
            timestepsPerPhase = 1,
            maxCycles = 8,
            sentinel = None
          )
        }

        new BufferedReader(new FileReader(s"${simulation.workingDirectoryPath}/simulation-log.txt")).lines
          .filter(finishRe.matches(_))
          .toArray
          .size must be(1)
      }
    }
  }
}
