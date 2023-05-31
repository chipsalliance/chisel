// SPDX-License-Identifier: Apache-2.0

package svsimTests

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import svsim._
import java.io.{BufferedReader, FileReader}

class VCSSpec extends BackendSpec {
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
}

class VerilatorSpec extends BackendSpec {
  import verilator.Backend.CompilationSettings._
  val backend = CustomVerilatorBackend(verilator.Backend.initializeFromProcessEnvironment())
  val compilationSettings = verilator.Backend.CompilationSettings(
    traceStyle = Some(TraceStyle.Vcd(traceUnderscore = false))
  )
  test("verilator", backend)(compilationSettings)
}

trait BackendSpec extends AnyFunSpec with Matchers {
  def test[Backend <: svsim.Backend](
    name:                String,
    backend:             Backend
  )(compilationSettings: backend.CompilationSettings
  ) = {
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
    }
  }
}
