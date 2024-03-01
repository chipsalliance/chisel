// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.stage.{
  ChiselCircuitAnnotation,
  ChiselGeneratorAnnotation,
  CircuitSerializationAnnotation,
  PrintFullStackTraceAnnotation,
  SourceRootAnnotation,
  ThrowOnFirstErrorAnnotation,
  UseLegacyShiftRightWidthBehavior,
  WarningConfigurationAnnotation,
  WarningConfigurationFileAnnotation,
  WarningsAsErrorsAnnotation
}
import firrtl.options.BareShell
import firrtl.options.TargetDirAnnotation
import logger.{ClassLogLevelAnnotation, LogClassNamesAnnotation, LogFileAnnotation, LogLevelAnnotation}

trait CLI { this: BareShell =>

  /** Include options for logging
    *
    * Defaults to true, override to false when mixing in to [[firrtl.options.Shell]] for use in a [[firrtl.options.Phase]]
    */
  protected def includeLoggerOptions: Boolean = true

  if (includeLoggerOptions) {
    parser.note("Logger options")
    Seq(LogLevelAnnotation, ClassLogLevelAnnotation, LogClassNamesAnnotation).foreach(_.addOptions(parser))
  }

  parser.note("Chisel options")
  Seq(
    ChiselGeneratorAnnotation,
    PrintFullStackTraceAnnotation,
    ThrowOnFirstErrorAnnotation,
    UseLegacyShiftRightWidthBehavior,
    WarningsAsErrorsAnnotation,
    WarningConfigurationAnnotation,
    WarningConfigurationFileAnnotation,
    SourceRootAnnotation,
    DumpFir
  ).foreach(_.addOptions(parser))

  parser.note("CIRCT (MLIR FIRRTL Compiler) options")
  Seq(
    CIRCTTargetAnnotation,
    PreserveAggregate,
    SplitVerilog,
    FirtoolBinaryPath
  ).foreach(_.addOptions(parser))
}

private trait LoggerOptions {}

/** Default Shell for [[ChiselStage]]
  */
private class Shell(applicationName: String) extends BareShell(applicationName) with CLI
