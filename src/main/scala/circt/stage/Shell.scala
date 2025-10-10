// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.stage.{
  ChiselCircuitAnnotation,
  ChiselGeneratorAnnotation,
  ChiselOutputFileAnnotation,
  CircuitSerializationAnnotation,
  IncludeInlineTestsForModule,
  IncludeInlineTestsWithName,
  IncludeUtilMetadata,
  PrintFullStackTraceAnnotation,
  RemapLayer,
  SourceRootAnnotation,
  SuppressSourceInfoAnnotation,
  ThrowOnFirstErrorAnnotation,
  UseLegacyWidthBehavior,
  UseSRAMBlackbox,
  WarningConfigurationAnnotation,
  WarningConfigurationFileAnnotation,
  WarningsAsErrorsAnnotation
}
import circt.stage.FirtoolOption
import firrtl.options.BareShell
import firrtl.options.TargetDirAnnotation
import logger.{ClassLogLevelAnnotation, LogClassNamesAnnotation, LogFileAnnotation, LogLevelAnnotation}

trait CLI extends BareShell { this: BareShell =>

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
    ChiselOutputFileAnnotation,
    PrintFullStackTraceAnnotation,
    ThrowOnFirstErrorAnnotation,
    UseLegacyWidthBehavior,
    WarningsAsErrorsAnnotation,
    WarningConfigurationAnnotation,
    WarningConfigurationFileAnnotation,
    SourceRootAnnotation,
    DumpFir,
    RemapLayer,
    IncludeUtilMetadata,
    UseSRAMBlackbox,
    IncludeInlineTestsForModule,
    IncludeInlineTestsWithName,
    SuppressSourceInfoAnnotation
  ).foreach(_.addOptions(parser))

  parser.note("CIRCT (MLIR FIRRTL Compiler) options")
  Seq(
    CIRCTTargetAnnotation,
    PreserveAggregate,
    SplitVerilog,
    FirtoolBinaryPath,
    FirtoolOption
  ).foreach(_.addOptions(parser))
}

private trait LoggerOptions {}

/** Default Shell for [[ChiselStage]]
  */
private class Shell(applicationName: String) extends BareShell(applicationName) with CLI
