package chisel3
package simulator

import svsim._

/**
  * Wrapper for unifying compilation and runtime options for different backends
  *
  * @param backend                      the selected [[Backend]]
  * @param verilogPreprocessorDefines   SystemVerilog defines as a map of `MACRO`->definition
  * @param optimizationStyle
  * @param availableParallelism
  * @param defaultTimescale
  * @param libraryExtensions
  * @param libraryPaths
  * @param disabledWarnings
  * @param disableFatalExitOnWarnings
  * @param traceStyle
  * @param outputSplit
  * @param outputSplitCFuncs
  * @param enableAllAssertions
  * @param customSimulationWorkingDirectory
  * @param verboseCompile
  * @param verboseRun
  * @param workspaceRoot
  * @param workingDirectoryPrefix
  * @param executionScriptLimit
  * @param executionScriptEnabled
  * @param conservativeCommandResolution
  * @param resetWorkspace                 completely delete workspace folders and their content before running the simulation
  * @param randomlyInitializeRegisters
  * @param chiselArgs
  * @param firtoolArgs
  */
case class ChiselSimSettings[B <: Backend](
  backend:                          B,
  traceStyle:                       TraceStyle = TraceStyle.NoTrace,
  verilogPreprocessorDefines:       Seq[(String, Any)] = Seq(),
  optimizationStyle:                CommonCompilationSettings.OptimizationStyle = CommonCompilationSettings.OptimizationStyle.Default,
  defaultTimescale:                 Option[CommonCompilationSettings.Timescale] = None,
  libraryPaths:                     Option[Seq[String]] = None,
  libraryExtensions:                Option[Seq[String]] = None,
  disabledWarnings:                 Seq[String] = Seq(/*"WIDTH", */ "STMTDLY"),
  disableFatalExitOnWarnings:       Boolean = true,
  outputSplit:                      Option[Int] = Some(20_000),
  outputSplitCFuncs:                Option[Int] = Some(20_000),
  enableAllAssertions:              Boolean = true,
  customSimulationWorkingDirectory: Option[String] = None,
  verboseCompile:                   Boolean = false,
  verboseRun:                       Boolean = false,
  testRunDir:                       Option[String] = Some("test_run_dir"),
  workingDirectoryPrefix:           String = "workdir",
  executionScriptLimit:             Option[Int] = None, // Some(0),
  executionScriptEnabled:           Boolean = false,
  conservativeCommandResolution:    Boolean = false,
  resetWorkspace:                   Boolean = false,
  randomlyInitializeRegisters:      Boolean = true,
  availableParallelism: CommonCompilationSettings.AvailableParallelism =
    CommonCompilationSettings.AvailableParallelism.Default,
  chiselArgs:  Seq[String] = Seq("--throw-on-first-error", "--full-stacktrace"),
  firtoolArgs: Seq[String] = Seq("-preserve-values=all", "-preserve-aggregate=vec")
  ///
) {

  private def preprocessorDefines: Seq[CommonCompilationSettings.VerilogPreprocessorDefine] =
    (verilogPreprocessorDefines ++ Seq(
      "ASSERT_VERBOSE_COND" -> s"!${Workspace.testbenchModuleName}.reset",
      "PRINTF_COND" -> s"!${Workspace.testbenchModuleName}.reset",
      "STOP_COND" -> s"!${Workspace.testbenchModuleName}.reset"
    ) ++ Option
      .when(randomlyInitializeRegisters)("RANDOMIZE_REG_INIT" -> 1) ++ Seq[(String, Any)](
    )).map {
      case (k, None) => CommonCompilationSettings.VerilogPreprocessorDefine(k)
      case (k, v)    => CommonCompilationSettings.VerilogPreprocessorDefine(k, v.toString)
    }.distinct

  def commonSettings: CommonCompilationSettings = CommonCompilationSettings(
    verilogPreprocessorDefines = preprocessorDefines,
    optimizationStyle = optimizationStyle,
    availableParallelism = availableParallelism,
    defaultTimescale = defaultTimescale,
    libraryExtensions = libraryExtensions,
    libraryPaths = libraryPaths
  )

  def backendSettings: backend.CompilationSettings = {
    backend match {
      case _: verilator.Backend =>
        verilator.Backend
          .CompilationSettings(
            traceStyle = traceStyle match {
              case TraceStyle.Vcd(filename, traceUnderscore) =>
                Some(verilator.Backend.CompilationSettings.TraceStyle.Vcd(traceUnderscore, filename))
              case TraceStyle.Fst(filename, traceUnderscore) =>
                Some(verilator.Backend.CompilationSettings.TraceStyle.Fst(traceUnderscore, filename))
              case TraceStyle.NoTrace =>
                None
              case _ =>
                throw new Exception(s"Trace style ${traceStyle} is not supported by the Verilator backend")
            },
            disabledWarnings = disabledWarnings,
            disableFatalExitOnWarnings = disableFatalExitOnWarnings,
            outputSplit = outputSplit,
            outputSplitCFuncs = outputSplitCFuncs,
            enableAllAssertions = enableAllAssertions
          )
          .asInstanceOf[backend.CompilationSettings]
      case _: vcs.Backend =>
        val (enableVcd, enableVpd) = traceStyle match {
          case TraceStyle.NoTrace   => (false, false)
          case TraceStyle.Vcd(_, _) => (true, false)
          case TraceStyle.Vpd(_, _) => (false, true)
          case _                    => throw new Exception(s"Trace style ${traceStyle} is not supported by the VCS backend")
        }
        vcs.Backend
          .CompilationSettings(
            xProp = None,
            randomlyInitializeRegisters = randomlyInitializeRegisters,
            traceSettings = vcs.Backend.CompilationSettings.TraceSettings(
              enableVcd = enableVcd,
              enableVpd = enableVpd
            ),
            simulationSettings = vcs.Backend.SimulationSettings(
              customWorkingDirectory = customSimulationWorkingDirectory,
              assertionSettings = None
            ),
            licenceExpireWarningTimeout = None,
            archOverride = None,
            waitForLicenseIfUnavailable = true
          )
          .asInstanceOf[backend.CompilationSettings]

      case _ => throw new Exception("Unknown backend")
    }
  }

  lazy val backendName = backend match {
    case _: verilator.Backend => "verilator"
    case _: vcs.Backend       => "vcs"
    case _ => "unknown"
  }

  def withTrace(traceStyle: TraceStyle): ChiselSimSettings[B] = this.copy(traceStyle = traceStyle)
}

object ChiselSimSettings {
  def defaultVerilatorSettings: ChiselSimSettings[verilator.Backend] =
    ChiselSimSettings(verilator.Backend.initializeFromProcessEnvironment())

  def verilatorBackend(
    traceStyle:                  TraceStyle = TraceStyle.NoTrace,
    resetWorkspace:              Boolean = false,
    randomlyInitializeRegisters: Boolean = true,
    enableAllAssertions:         Boolean = true,
    executionScriptEnabled:      Boolean = false,
    verilogPreprocessorDefines:  Seq[(String, Any)] = Seq()
  ): ChiselSimSettings[verilator.Backend] =
    ChiselSimSettings(
      verilator.Backend.initializeFromProcessEnvironment(),
      traceStyle = traceStyle,
      resetWorkspace = resetWorkspace,
      randomlyInitializeRegisters = randomlyInitializeRegisters,
      enableAllAssertions = enableAllAssertions,
      executionScriptEnabled = executionScriptEnabled,
      verilogPreprocessorDefines = verilogPreprocessorDefines
    )

  def defaultVcsSettings: ChiselSimSettings[vcs.Backend] =
    ChiselSimSettings(vcs.Backend.initializeFromProcessEnvironment().get)
}
