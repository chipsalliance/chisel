// SPDX-License-Identifier: Apache-2.0

package svsim.verilator

import svsim._
import scala.sys.process._
import scala.collection.mutable

object Backend {
  object CompilationSettings {

    sealed trait TraceKind {
      private[svsim] def toCompileFlags: Seq[String]
    }

    object TraceKind {

      /**
        * VCD tracing
        */
      case object Vcd extends TraceKind {
        final def toCompileFlags = Seq("--trace")
      }

      private[Backend] sealed trait FstTraceKind { self: TraceKind =>
        protected val traceThreads: Int
        final def toCompileFlags: Seq[String] =
          Seq("--trace-fst") ++
            Option.when(traceThreads > 0)(Seq("--trace-threads", traceThreads.toString)).toSeq.flatten
      }

      /**
        * FST tracing
        * 
        * @param traceThreads If set to a positive value, enables FST waveform creation using `traceThreads` separate threads
        */
      case class Fst(traceThreads: Int) extends TraceKind with FstTraceKind

      case object Fst extends TraceKind with FstTraceKind { override val traceThreads = 0 }
    }

    case class TraceStyle(
      kind:            TraceKind,
      traceUnderscore: Boolean = false,
      traceStructs:    Boolean = true,
      traceParams:     Boolean = false,
      maxWidth:        Int = 0,
      maxArraySize:    Int = 0,
      traceDepth:      Int = 0
    ) {
      def toCompileFlags: Seq[String] = kind.toCompileFlags ++
        Option.when(traceUnderscore)("--trace-underscore") ++
        Option.when(traceStructs)("--trace-structs") ++
        Option.when(traceParams)("--trace-params") ++
        (
          Option.when(maxArraySize > 0)(Seq(s"--trace-max-array", maxArraySize.toString)) ++
            Option.when(maxWidth > 0)(Seq("--trace-max-width", maxWidth.toString)) ++
            Option.when(traceDepth > 0)(Seq("--trace-depth", traceDepth.toString))
        ).flatten
    }

    object Timing {
      sealed trait Type
      case object TimingEnabled extends Type
      case object TimingDisabled extends Type
    }
  }

  case class CompilationSettings(
    traceStyle:                 Option[CompilationSettings.TraceStyle] = None,
    outputSplit:                Option[Int] = None,
    outputSplitCFuncs:          Option[Int] = None,
    disabledWarnings:           Seq[String] = Seq(),
    disableFatalExitOnWarnings: Boolean = false,
    enableAllAssertions:        Boolean = false,
    timing:                     Option[CompilationSettings.Timing.Type] = None
  ) extends svsim.Backend.Settings

  def initializeFromProcessEnvironment() = {
    val output = mutable.ArrayBuffer.empty[String]
    val exitCode = List("which", "verilator").!(ProcessLogger(output += _))
    if (exitCode != 0) {
      throw new Exception(s"verilator not found on the PATH!\n${output.mkString("\n")}")
    }
    val executablePath = output.head.trim
    new Backend(executablePath = executablePath)
  }
}
final class Backend(executablePath: String) extends svsim.Backend {
  type CompilationSettings = Backend.CompilationSettings

  def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): svsim.Backend.Parameters = {
    import CommonCompilationSettings._
    import Backend.CompilationSettings._
    //format: off
    svsim.Backend.Parameters(
      compilerPath = executablePath,
      compilerInvocation = svsim.Backend.Parameters.Invocation(
        arguments = Seq[Seq[String]](
          Seq(
            "--cc", // "Create C++ output"
            "--exe", // "Link to create executable"
            "--build", // "Build model executable/library after Verilation"
            "-j", "0", // Parallelism for --build-jobs/--verilate-jobs, when 0 uses all available cores
            "-o", s"../$outputBinaryName", // "Name of final executable"
            "--top-module", topModuleName, // "Name of top-level input module"
            "--Mdir", "verilated-sources",  // "Name of output object directory"
            "--assert", // Enable assertions
          ),

          commonSettings.libraryExtensions match {
            case None => Seq()
            case Some(extensions) => Seq((Seq("+libext") ++ extensions).mkString("+"))
          },

          commonSettings.libraryPaths match {
            case None => Seq()
            case Some(paths) => paths.flatMap(Seq("-y", _))
          },

          commonSettings.includeDirs match {
            case None => Seq()
            case Some(dirs) => dirs.map(dir => s"+incdir+$dir")
          },

          commonSettings.defaultTimescale match {
            case Some(Timescale.FromString(value)) => Seq("--timescale", value)
            case None => Seq()
          },

          backendSpecificSettings.traceStyle match {
            case Some(traceStyle) => traceStyle.toCompileFlags
            case None => Seq()
          },

          backendSpecificSettings.timing match {
            case Some(Timing.TimingEnabled)   => Seq("--timing")
            case Some(Timing.TimingDisabled)  => Seq("--no-timing")
            case None                         => Seq()
          },

          Seq(
            ("-Wno-fatal", backendSpecificSettings.disableFatalExitOnWarnings),
            ("--assert", backendSpecificSettings.enableAllAssertions),
          ).collect {
            case (flag, true) => flag
          },

          backendSpecificSettings.disabledWarnings.map("-Wno-" + _),

          commonSettings.optimizationStyle match {
            case OptimizationStyle.Default => Seq()
            case OptimizationStyle.OptimizeForCompilationSpeed => Seq("-O1")
            case OptimizationStyle.OptimizeForSimulationSpeed =>
              Seq("-O3", "--x-assign", "fast", "--x-initial", "fast")
          },

          Seq[(String, Option[String])](
            ("--output-split", backendSpecificSettings.outputSplit.map(_.toString())), // "Split .cpp files into pieces"
            ("--output-split-cfuncs", backendSpecificSettings.outputSplitCFuncs.map(_.toString())), // "Split model functions"
          ).collect {
            /// Only include flags that have a value
            case (flag, Some(value)) => Seq(flag, value)
          }.flatten,

          Seq(
            ("-MAKEFLAGS", Seq(
              commonSettings.availableParallelism match {
                case AvailableParallelism.Default => Seq()
                case AvailableParallelism.UpTo(value) => Seq("-j", value.toString())
              },
            ).flatten),
            ("-CFLAGS", Seq(
              commonSettings.optimizationStyle match {
                case OptimizationStyle.Default => Seq()
                case OptimizationStyle.OptimizeForCompilationSpeed => Seq("-O1")
                case OptimizationStyle.OptimizeForSimulationSpeed =>
                  Seq("-O3", "-march=native", "-mtune=native")
              },

              Seq("-std=c++14"),

              additionalHeaderPaths.map { path => s"-I${path}" },

              Seq(
                // Use verilator support
                s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorSupport}",
              ),

              backendSpecificSettings.traceStyle match {
                case Some(_) => Seq(s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorTrace}")
                case None => Seq()
              },
            ).flatten)
          ).collect {
            /// Only include flags that have one or more values
            case (flag, value) if !value.isEmpty => {
              Seq(flag, value.mkString(" "))
            }
          }.flatten,

          Seq(
            commonSettings.verilogPreprocessorDefines,
            backendSpecificSettings.traceStyle match {
              case None => Seq()
              case Some(Backend.CompilationSettings.TraceStyle(Backend.CompilationSettings.TraceKind.Vcd, _, _, _, _, _, _) ) => Seq(
                VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport)
              )
              case Some(Backend.CompilationSettings.TraceStyle(_: Backend.CompilationSettings.TraceKind.FstTraceKind, _, _, _, _, _, _)) => Seq(
                VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.enableFstTracingSupport)
              )
            },
          ).flatten.map(_.toCommandlineArgument(this)),
        ).flatten,
        environment = Seq()
      ),
      simulationInvocation = svsim.Backend.Parameters.Invocation(commonSettings.simulationSettings.plusArgs.map(_.simulatorFlags), Seq())
    )
    //format: on
  }

  override def escapeDefine(string: String): String = string

  override val assertionFailed = "^.*Assertion failed in.*".r

}
