// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.scalatest

import chisel3.layers.Verification.Assert.Temporal
import chisel3.simulator.HasSimulator
import chisel3.simulator.scalatest.HasCliOptions.CliOption
import scala.util.control.NoStackTrace
import svsim.Backend.HarnessCompilationFlags.{
  enableFsdbTracingSupport,
  enableVcdTracingSupport,
  enableVpdTracingSupport
}
import svsim.{CommonCompilationSettings, CommonSettingsModifications}
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine

/** ChiselSim command line interface traits that can be added to Scalatest tests
  */
object Cli {

  /** Adds `-DchiselOpts=<space-delimited-chisel-options>`
    *
    * This allows for extra options to be passed directly to Chisel
    * elaboration. Options are space-delimited.  To pass multiple options use
    * single quotes, e.g.:
    *
    * {{{
    * -DchiselOpts='-foo -bar'
    * }}}
    */
  trait ChiselOpts { this: HasCliOptions =>

    addOption(
      CliOption[Seq[String]](
        name = "chiselOpts",
        convert = a => a.split(' ').toSeq,
        help = "additional options to pass to the Chisel elaboration",
        updateChiselOptions = (value, old) => old ++ value,
        updateFirtoolOptions = (_, a) => a,
        updateCommonSettings = (_, a) => a,
        updateBackendSettings = (_, a) => a,
        updateUnsetChiselOptions = (a: Array[String]) => a,
        updateUnsetFirtoolOptions = (a: Array[String]) => a,
        updateUnsetCommonSettings = (a: CommonCompilationSettings) => a,
        updateUnsetBackendSettings = (a: svsim.Backend.Settings) => a
      )
    )

  }

  /** Adds `-DemitFsdb=[1,true]`
    *
    * This causes a simulation to dump an FSDB wave starting at time zero.
    * Finer grained control can be achieved with
    * [[chisel3.simulator.ControlAPI.disableWaves]] and
    * [[chisel3.simulator.ControlAPI.enableWaves]].
    *
    * If the simulator does not support FSDB waves, then using this option will
    * throw an [[IllegalArgumentException]] when the option is used.
    */
  trait EmitFsdb { this: HasCliOptions =>

    addOption(
      CliOption
        .flag(
          name = "emitFsdb",
          help = "compile with FSDB waveform support and start dumping waves at time zero",
          updateCommonSettings = (options) => {
            options.copy(
              verilogPreprocessorDefines =
                options.verilogPreprocessorDefines :+ VerilogPreprocessorDefine(enableFsdbTracingSupport),
              simulationSettings = options.simulationSettings.copy(
                enableWavesAtTimeZero = true
              )
            )
          },
          updateBackendSettings = (options) =>
            options match {
              case options: svsim.vcs.Backend.CompilationSettings =>
                options.copy(
                  traceSettings = options.traceSettings.copy(fsdbSettings =
                    Some(
                      svsim.vcs.Backend.CompilationSettings.TraceSettings.FsdbSettings(
                        sys.env.getOrElse(
                          "VERDI_HOME",
                          throw new RuntimeException(
                            "Cannot enable FSDB support as the environment variable 'VERDI_HOME' was not set."
                          )
                        )
                      )
                    )
                  )
                )
              case options: svsim.verilator.Backend.CompilationSettings =>
                throw new IllegalArgumentException("Verilator does not support FSDB waveforms.")
            }
        )
    )

  }

  /** Adds `-DemitVcd=[1,true]`
    *
    * This causes a simulation to dump a VCD wave starting at time zero.  Finer
    * grained control can be achieved with
    * [[chisel3.simulator.ControlAPI.disableWaves]] and
    * [[chisel3.simulator.ControlAPI.enableWaves]].
    */
  trait EmitVcd { this: HasCliOptions =>

    addOption(
      CliOption
        .flag(
          name = "emitVcd",
          help = "compile with VCD waveform support and start dumping waves at time zero",
          updateCommonSettings = (options) => {
            options.copy(
              verilogPreprocessorDefines =
                options.verilogPreprocessorDefines :+ VerilogPreprocessorDefine(enableVcdTracingSupport),
              simulationSettings = options.simulationSettings.copy(
                enableWavesAtTimeZero = true
              )
            )
          },
          updateBackendSettings = (options) =>
            options match {
              case options: svsim.vcs.Backend.CompilationSettings =>
                options.copy(
                  traceSettings = options.traceSettings.copy(enableVcd = true)
                )
              case options: svsim.verilator.Backend.CompilationSettings =>
                options.withTraceStyle(
                  options.traceStyle match {
                    case None =>
                      Some(
                        svsim.verilator.Backend.CompilationSettings
                          .TraceStyle(svsim.verilator.Backend.CompilationSettings.TraceKind.Vcd)
                      )
                    case alreadySet => alreadySet
                  }
                )
            }
        )
    )

  }

  /** Adds `-DemitVpd=[1,true]`
    *
    * This causes a simulation to dump a VPD wave starting at time zero.  Finer
    * grained control can be achieved with
    * [[chisel3.simulator.ControlAPI.disableWaves]] and
    * [[chisel3.simulator.ControlAPI.enableWaves]].
    *
    * If the simulator does not support VPD waves, then using this option will
    * throw an [[IllegalArgumentException]] when the option is used.
    */
  trait EmitVpd { this: HasCliOptions =>

    addOption(
      CliOption
        .flag(
          name = "emitVpd",
          help = "compile with VPD waveform support and start dumping waves at time zero",
          updateCommonSettings = (options) => {
            options.copy(
              verilogPreprocessorDefines =
                options.verilogPreprocessorDefines :+ VerilogPreprocessorDefine(enableVpdTracingSupport),
              simulationSettings = options.simulationSettings.copy(
                enableWavesAtTimeZero = true
              )
            )
          },
          updateBackendSettings = (options) =>
            options match {
              case options: svsim.vcs.Backend.CompilationSettings =>
                options.copy(
                  traceSettings = options.traceSettings.copy(enableVpd = true)
                )
              case options: svsim.verilator.Backend.CompilationSettings =>
                throw new IllegalArgumentException("Verilator does not support VPD waveforms.")
            }
        )
    )

  }

  /** Adds `-DfirtoolOpts=<space-delimited-firtool-options>`
    *
    * This allows for extra options to be passed directly to `firtool`.  Options
    * are space-delimited.  To pass multiple options use single quotes, e.g.:
    *
    * {{{
    * -DfirtoolOpts='-foo -bar'
    * }}}
    */
  trait FirtoolOpts { this: HasCliOptions =>

    addOption(
      CliOption[Seq[String]](
        name = "firtoolOpts",
        convert = a => a.split(' ').toSeq,
        help = "additional options to pass to the firtool compiler",
        updateChiselOptions = (_, a) => a,
        updateFirtoolOptions = (value, old) => old ++ value,
        updateCommonSettings = (_, a) => a,
        updateBackendSettings = (_, a) => a,
        updateUnsetChiselOptions = (a: Array[String]) => a,
        updateUnsetFirtoolOptions = (a: Array[String]) => a,
        updateUnsetCommonSettings = (a: CommonCompilationSettings) => a,
        updateUnsetBackendSettings = (a: svsim.Backend.Settings) => a
      )
    )

  }

  /** Adds `-Dscale=<double>`
    *
    * This adds an option which can be used to control the "scaling factor" of
    * the test.  This option can be used by the test author to "scale" the test
    * to make it longer or shorter.
    *
    * This is inteded to be used with randomized testing or some testing which
    * does not have a natural end point or an end point which may be dependent
    * on execution context.  E.g., you can use this to make a test which runs
    * for briefly in a pre-merge Continuous Integration (CI) flow, but runs for
    * much longer in nightly CI.
    *
    * Note that the interpretation of the "scaling factor" is test dependent.
    * In order to document this and help end users, this trait has an abstract
    * member `scaleHelpText` which must be overridden to document what the scale
    * affects in the test.  E.g., if the scale is used to lengthen a test, a
    * test author could write:
    *
    * {{{
    * override protected def scaleHelpText = "scales the runtime of a test"
    * }}}
    *
    * While this adds the low-level `scale` option.  A user of this should use
    * the member functions that this trait adds, e.d., `scaled`.
    */
  trait Scale { this: HasCliOptions =>

    /** Help text that describes what the `-Dscale` parameter does for this test.
      *
      * This is used to populate the help text that an end user sees when
      * invoking this test with `-Dhelp=1`.
      */
    protected def scaleHelpText: String

    /** Scale an integer by the scaling factor, if set.
      *
      * If no scaling factor is set, this will scale by `1.0`, i.e., the input
      * is unchanged.
      */
    final def scaled(a: Int): Int = (a * getOption[Double]("scale").getOrElse(1.0)).toInt

    addOption(
      CliOption.double(
        name = "scale",
        help = scaleHelpText
      )
    )

  }

  /** Adds `-Dsimulator` to choose the simulator at runtime.
    *
    * This trait adds an option for controlling the simulator at runtime.
    *
    * If a user wants to add more simulators or simulators with different
    * options, they should override the `cliSimulatorMap` with their chosen
    * simulators.
    *
    * If a user wants to change the default simulator, they should override
    * `defaultCliSimulator`.  Making this a `None` will require the user to
    * always specify a simulator, i.e., there is no default.
    */
  trait Simulator { this: HasCliOptions =>

    /** A mapping of simulator names to simulators. */
    protected def cliSimulatorMap: Map[String, HasSimulator] = Map(
      "verilator" -> HasSimulator.simulators.verilator(),
      "vcs" -> HasSimulator.simulators.vcs()
    )

    /** Return a string showing legal simulator names. */
    private def legalValues: String = cliSimulatorMap.keys.toSeq.sorted.mkString("[", ", ", "]")

    /** An optional default simulator to use if the user does _not_ provide a simulator.
    *
    * If `Some` then the provided default will be used.  If `None`, then a
    * simulator must be provided.
    */
    protected def defaultCliSimulator: Option[HasSimulator] = Some(HasSimulator.simulators.verilator())

    implicit def cliSimulator: HasSimulator = configMap.getOptional[String]("simulator") match {
      case None =>
        defaultCliSimulator.getOrElse(
          throw new IllegalArgumentException(
            s"""a simulator must be provided to this test using '-Dsimulator=<simulator-name>' where <simulator-name> must be one of $legalValues"""
          )
        )
      case Some(simulator) =>
        cliSimulatorMap.getOrElse(
          simulator,
          throw new IllegalArgumentException(
            s"""illegal simulator '$simulator', must be one of $legalValues"""
          )
        )
    }

    protected def disableTemporalLayers: CommonSettingsModifications = (in: CommonCompilationSettings) =>
      in.copy(
        verilogPreprocessorDefines = in.verilogPreprocessorDefines.filter {
          case VerilogPreprocessorDefine(name, None) => !name.contains("$" + Temporal.name)
          case _                                     => true
        }
      )

    addOption(
      CliOption[String](
        name = "simulator",
        help = "sets the simulator for the test",
        convert = simulator => {
          if (cliSimulatorMap.contains(simulator)) {
            ()
          } else {
            throw new IllegalArgumentException(
              s"""illegal simulator '$simulator', must be one of $legalValues"""
            )
          }
          simulator
        },
        updateChiselOptions = (_, a) => a,
        updateFirtoolOptions = (_, a) => a,
        updateCommonSettings = (simulator, a) =>
          simulator match {
            case "verilator" => disableTemporalLayers(a)
            case _           => a
          },
        updateBackendSettings = (_, a) => a,
        updateUnsetChiselOptions = (a: Array[String]) => a,
        updateUnsetFirtoolOptions = (a: Array[String]) => a,
        updateUnsetCommonSettings = disableTemporalLayers,
        updateUnsetBackendSettings = (a: svsim.Backend.Settings) => a
      )
    )

  }

}
