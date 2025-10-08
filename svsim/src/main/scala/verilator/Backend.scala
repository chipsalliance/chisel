// SPDX-License-Identifier: Apache-2.0

package svsim.verilator

import svsim._

import scala.collection.mutable
import scala.sys.process._

object Backend {
  object CompilationSettings {

    object TraceKind {

      sealed trait Type {
        private[Backend] def toCompileFlags: Seq[String]
      }

      /** VCD tracing */
      case object Vcd extends Type {
        final def toCompileFlags = Seq("--trace")
      }

      /** FST tracing
        *
        * @param traceThreads Enable FST waveform creation using `traceThreads` separate threads
        */
      case class Fst(traceThreads: Option[Int] = None) extends Type {
        final def toCompileFlags: Seq[String] =
          Seq("--trace-fst") ++ traceThreads.map(n => Seq("--trace-threads", n.toString)).toSeq.flatten
      }
    }

    /** Trace style options for verilator
      *
      * @param kind The format of the trace to generate, e.g., VCD or FST
      * @param traceUnderscore Whether to trace signals with names starting with an underscore
      * @param traceStructs Whether to trace structs
      * @param traceParams Whether to trace parameters
      * @param maxWidth The maximum bit width for tracing
      * @param maxArraySize The maximum array depth for tracing
      * @param traceDepth The maximum depth of tracing
      */
    case class TraceStyle(
      kind:            TraceKind.Type,
      traceUnderscore: Boolean = false,
      traceStructs:    Boolean = true,
      traceParams:     Boolean = false,
      maxWidth:        Option[Int] = None,
      maxArraySize:    Option[Int] = None,
      traceDepth:      Option[Int] = None
    ) {
      def toCompileFlags: Seq[String] = kind.toCompileFlags ++
        Option.when(traceUnderscore)("--trace-underscore") ++
        Option.when(traceStructs)("--trace-structs") ++
        Option.when(traceParams)("--trace-params") ++
        (
          maxArraySize.map(n => Seq(s"--trace-max-array", n.toString)) ++
            maxWidth.map(n => Seq("--trace-max-width", n.toString)) ++
            traceDepth.map(n => Seq("--trace-depth", n.toString))
        ).flatten
    }

    object Timing {
      sealed trait Type
      case object TimingEnabled extends Type
      case object TimingDisabled extends Type
    }

    /** Control job parallelism in verilator */
    object Parallelism {
      sealed trait Type {
        def toCompileFlags: Seq[String]
      }

      /** Apply uniform parallelism to Verilation.  This maps to `-j`. */
      case class Uniform(num: Int) extends Type {
        override def toCompileFlags = Seq("-j", num.toString)
      }

      /** Apply non-uniform parallelism to Verilation.  This allows control of
        * `--build-jobs` and `--verilate-jobs` separately.
        */
      case class Different(build: Option[Int] = None, verilate: Option[Int] = None) extends Type {
        override def toCompileFlags: Seq[String] = {
          val buildJobs:    Seq[String] = build.map(num => Seq("--build-jobs", num.toString)).toSeq.flatten
          val verilateJobs: Seq[String] = verilate.map(num => Seq("--verilate-jobs", num.toString)).toSeq.flatten
          buildJobs ++ verilateJobs
        }
      }
    }

  }

  case class CompilationSettings(
    traceStyle:                 Option[CompilationSettings.TraceStyle] = None,
    outputSplit:                Option[Int] = None,
    outputSplitCFuncs:          Option[Int] = None,
    disabledWarnings:           Seq[String] = Seq(),
    disableFatalExitOnWarnings: Boolean = false,
    enableAllAssertions:        Boolean = false,
    timing:                     Option[CompilationSettings.Timing.Type] = None,
    parallelism: Option[CompilationSettings.Parallelism.Type] = Some(CompilationSettings.Parallelism.Uniform(0))
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
    import Backend.CompilationSettings._
    import CommonCompilationSettings._
    //format: off

    val args = mutable.ArrayBuffer.empty[String]
    def addArg(xs: Iterable[String]): Unit = args ++= xs
    def addArgParts(flag: String, parts: Iterable[String]): Unit = {
      val v = parts.iterator.mkString(" ")
      if (v.nonEmpty) addArg(Seq(flag, v))
    }

    // Base verilator invocation
    addArg(Seq(
      "--cc",
      "--exe",
      "--build",
      "-o", s"../$outputBinaryName",
      "--top-module", topModuleName,
      "--Mdir", "verilated-sources",
      "--assert"
    ))

    backendSpecificSettings.parallelism match {
      case Some(parallelism) => addArg(parallelism.toCompileFlags)
      case None =>
    }

    commonSettings.libraryExtensions.foreach { extensions =>
      addArg(Seq((Seq("+libext") ++ extensions).mkString("+")))
    }

    commonSettings.libraryPaths.foreach { paths =>
      paths.foreach(p => addArg(Seq("-y", p)))
    }

    commonSettings.includeDirs.foreach { dirs =>
      addArg(dirs.map(dir => s"+incdir+$dir"))
    }

    commonSettings.defaultTimescale.foreach { value =>
      addArg(Seq("--timescale", value.toString))
    }

    backendSpecificSettings.traceStyle.foreach { ts => addArg(ts.toCompileFlags) }

    backendSpecificSettings.timing match {
      case Some(Timing.TimingEnabled)  => addArg(Seq("--timing"))
      case Some(Timing.TimingDisabled) => addArg(Seq("--no-timing"))
      case None                        =>
    }

    if (backendSpecificSettings.disableFatalExitOnWarnings) addArg(Seq("-Wno-fatal"))
    if (backendSpecificSettings.enableAllAssertions) addArg(Seq("--assert"))
    addArg(backendSpecificSettings.disabledWarnings.map("-Wno-" + _))

    commonSettings.optimizationStyle match {
      case OptimizationStyle.Default =>
      case OptimizationStyle.OptimizeForCompilationSpeed => addArg(Seq("-O1"))
      case OptimizationStyle.OptimizeForSimulationSpeed => addArg(Seq("-O3", "--x-assign", "fast", "--x-initial", "fast"))
    }

    backendSpecificSettings.outputSplit.foreach      { n => addArg(Seq("--output-split", n.toString)) }
    backendSpecificSettings.outputSplitCFuncs.foreach{ n => addArg(Seq("--output-split-cfuncs", n.toString)) }

    val makeflagsParts: Seq[String] = commonSettings.availableParallelism match {
      case AvailableParallelism.Default       => Seq()
      case AvailableParallelism.UpTo(value)   => Seq("-j", value.toString)
    }
    addArgParts("-MAKEFLAGS", makeflagsParts)

    val cflagsParts: Seq[String] = {
      val opt = commonSettings.optimizationStyle match {
        case OptimizationStyle.Default => Seq()
        case OptimizationStyle.OptimizeForCompilationSpeed => Seq("-O1")
        case OptimizationStyle.OptimizeForSimulationSpeed   => Seq("-O3", "-march=native", "-mtune=native")
      }
      val std = Seq("-std=c++17")
      val inc = additionalHeaderPaths.map(path => s"-I$path")
      val defs = Seq(s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorSupport}") ++ (
        backendSpecificSettings.traceStyle match {
          case Some(_) => Seq(s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorTrace}")
          case None    => Seq()
        }
      )
      opt ++ std ++ inc ++ defs
    }
    addArgParts("-CFLAGS", cflagsParts)

    val defineFlags: Seq[String] = {
      val base = commonSettings.verilogPreprocessorDefines
      val traceExtra = backendSpecificSettings.traceStyle match {
        case None => Seq()
        case Some(Backend.CompilationSettings.TraceStyle(Backend.CompilationSettings.TraceKind.Vcd, _, _, _, _, _, _)) =>
          Seq(VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport))
        case Some(Backend.CompilationSettings.TraceStyle(Backend.CompilationSettings.TraceKind.Fst(_), _, _, _, _, _, _)) =>
          Seq(VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.enableFstTracingSupport))
      }
      (base ++ traceExtra).map(_.toCommandlineArgument(this))
    }
    addArg(defineFlags)

    svsim.Backend.Parameters(
      compilerPath = executablePath,
      compilerInvocation = svsim.Backend.Parameters.Invocation(
        arguments = args.toSeq,
        environment = Seq()
      ),
      simulationInvocation = svsim.Backend.Parameters.Invocation(
        commonSettings.simulationSettings.plusArgs.map(_.simulatorFlags),
        Seq()
      )
    )
    //format: on
  }

  override def escapeDefine(string: String): String = string

  override val assertionFailed = "^.*Assertion failed in.*".r

}
