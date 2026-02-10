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

    /** Settings for controlling Verilator coverage instrumentation.
      *
      * These options map to Verilator coverage flags:
      *
      *   - `--coverage-line`
      *   - `--coverage-toggle`
      *   - `--coverage-user`
      *
      * If all coverage types are enabled, `--coverage` is used.
      *
      * @param line enable line coverage instrumentation
      * @param toggle enable toggle coverage instrumentation
      * @param user enable user coverage instrumentation for `cover` statements
      */
    final class CoverageSettings(
      val line:   Boolean = false,
      val toggle: Boolean = false,
      val user:   Boolean = false
    ) {
      private[Backend] def any: Boolean = line || toggle || user

      private[Backend] def toCompileFlags: Seq[String] = {
        if (line && toggle && user) {
          Seq("--coverage")
        } else {
          Seq(
            Option.when(line)("--coverage-line"),
            Option.when(toggle)("--coverage-toggle"),
            Option.when(user)("--coverage-user")
          ).flatten
        }
      }
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
      class Uniform private (num: Int) extends Type {
        override def toCompileFlags = Seq("-j", num.toString)

        private def copy(num: Int): Uniform = new Uniform(num = num)

        def withNum(num: Int): Uniform = copy(num = num)
      }

      object Uniform {
        def default: Uniform = new Uniform(num = 0)
      }

      /** Apply non-uniform parallelism to Verilation.  This allows control of
        * `--build-jobs` and `--verilate-jobs` separately.
        */
      class Different private (build: Option[Int], verilate: Option[Int]) extends Type {
        override def toCompileFlags: Seq[String] = {
          val buildJobs:    Seq[String] = build.map(num => Seq("--build-jobs", num.toString)).toSeq.flatten
          val verilateJobs: Seq[String] = verilate.map(num => Seq("--verilate-jobs", num.toString)).toSeq.flatten
          buildJobs ++ verilateJobs
        }

        private def copy(build: Option[Int] = this.build, verilate: Option[Int] = this.verilate): Different =
          new Different(
            build = build,
            verilate = verilate
          )

        def withBuild(build: Option[Int]): Different = copy(build = build)

        def withVerilate(verilate: Option[Int]): Different = copy(verilate = verilate)
      }

      object Different {
        def default = new Different(build = None, verilate = None)
      }
    }

    @deprecated("use 'CompilationSettings.default' and 'with<name>' helpers", "Chisel 7.1.0")
    def apply(
      traceStyle:                 Option[CompilationSettings.TraceStyle] = None,
      outputSplit:                Option[Int] = None,
      outputSplitCFuncs:          Option[Int] = None,
      disabledWarnings:           Seq[String] = Seq(),
      disableFatalExitOnWarnings: Boolean = false,
      enableAllAssertions:        Boolean = false,
      timing:                     Option[CompilationSettings.Timing.Type] = None
    ): CompilationSettings = CompilationSettings(
      traceStyle,
      outputSplit,
      outputSplitCFuncs,
      disabledWarnings,
      disableFatalExitOnWarnings,
      enableAllAssertions,
      timing,
      Some(CompilationSettings.Parallelism.Uniform.default),
      new CompilationSettings.CoverageSettings()
    )

    @deprecated("avoid use of unapply", "Chisel 7.1.0")
    def unapply(compilationSettings: CompilationSettings): Option[
      (
        Option[CompilationSettings.TraceStyle],
        Option[Int],
        Option[Int],
        Seq[String],
        Boolean,
        Boolean,
        Option[CompilationSettings.Timing.Type]
      )
    ] = Some(
      (
        compilationSettings.traceStyle,
        compilationSettings.outputSplit,
        compilationSettings.outputSplitCFuncs,
        compilationSettings.disabledWarnings,
        compilationSettings.disableFatalExitOnWarnings,
        compilationSettings.enableAllAssertions,
        compilationSettings.timing
      )
    )

    def default: CompilationSettings = new CompilationSettings(
      traceStyle = None,
      outputSplit = None,
      outputSplitCFuncs = None,
      disabledWarnings = Seq(),
      disableFatalExitOnWarnings = false,
      enableAllAssertions = false,
      timing = None,
      parallelism = Some(CompilationSettings.Parallelism.Uniform.default),
      coverageSettings = new CompilationSettings.CoverageSettings()
    )

  }

  case class CompilationSettings private (
    traceStyle:                 Option[CompilationSettings.TraceStyle],
    outputSplit:                Option[Int],
    outputSplitCFuncs:          Option[Int],
    disabledWarnings:           Seq[String],
    disableFatalExitOnWarnings: Boolean,
    enableAllAssertions:        Boolean,
    timing:                     Option[CompilationSettings.Timing.Type],
    parallelism:                Option[CompilationSettings.Parallelism.Type],
    coverageSettings:           CompilationSettings.CoverageSettings
  ) extends svsim.Backend.Settings {

    @deprecated("use 'CompilationSettings.default' and 'with<name>' helpers", "Chisel 7.1.0")
    def this(
      traceStyle:                 Option[CompilationSettings.TraceStyle] = None,
      outputSplit:                Option[Int] = None,
      outputSplitCFuncs:          Option[Int] = None,
      disabledWarnings:           Seq[String] = Seq(),
      disableFatalExitOnWarnings: Boolean = false,
      enableAllAssertions:        Boolean = false,
      timing:                     Option[CompilationSettings.Timing.Type] = None
    ) = this(
      traceStyle,
      outputSplit,
      outputSplitCFuncs,
      disabledWarnings,
      disableFatalExitOnWarnings,
      enableAllAssertions,
      timing,
      Some(CompilationSettings.Parallelism.Uniform.default),
      new CompilationSettings.CoverageSettings()
    )

    private def copyInternal(
      traceStyle:                 Option[CompilationSettings.TraceStyle] = this.traceStyle,
      outputSplit:                Option[Int] = this.outputSplit,
      outputSplitCFuncs:          Option[Int] = this.outputSplitCFuncs,
      disabledWarnings:           Seq[String] = this.disabledWarnings,
      disableFatalExitOnWarnings: Boolean = this.disableFatalExitOnWarnings,
      enableAllAssertions:        Boolean = this.enableAllAssertions,
      timing:                     Option[CompilationSettings.Timing.Type] = this.timing,
      parallelism:                Option[CompilationSettings.Parallelism.Type] = this.parallelism,
      coverageSettings:           CompilationSettings.CoverageSettings = this.coverageSettings
    ): CompilationSettings = CompilationSettings(
      traceStyle = traceStyle,
      outputSplit = outputSplit,
      outputSplitCFuncs = outputSplitCFuncs,
      disabledWarnings = disabledWarnings,
      disableFatalExitOnWarnings = disableFatalExitOnWarnings,
      enableAllAssertions = enableAllAssertions,
      timing = timing,
      parallelism = parallelism,
      coverageSettings = coverageSettings
    )

    // Keep this signature stable for binary compatibility with prior releases.
    def _copy(
      traceStyle:                 Option[CompilationSettings.TraceStyle] = this.traceStyle,
      outputSplit:                Option[Int] = this.outputSplit,
      outputSplitCFuncs:          Option[Int] = this.outputSplitCFuncs,
      disabledWarnings:           Seq[String] = this.disabledWarnings,
      disableFatalExitOnWarnings: Boolean = this.disableFatalExitOnWarnings,
      enableAllAssertions:        Boolean = this.enableAllAssertions,
      timing:                     Option[CompilationSettings.Timing.Type] = this.timing,
      parallelism:                Option[CompilationSettings.Parallelism.Type] = this.parallelism
    ): CompilationSettings = copyInternal(
      traceStyle = traceStyle,
      outputSplit = outputSplit,
      outputSplitCFuncs = outputSplitCFuncs,
      disabledWarnings = disabledWarnings,
      disableFatalExitOnWarnings = disableFatalExitOnWarnings,
      enableAllAssertions = enableAllAssertions,
      timing = timing,
      parallelism = parallelism,
      coverageSettings = this.coverageSettings
    )

    @deprecated("don't use the copy method, use 'with<name>' single setters", "Chisel 7.1.0")
    def copy(
      traceStyle:                 Option[CompilationSettings.TraceStyle] = this.traceStyle,
      outputSplit:                Option[Int] = this.outputSplit,
      outputSplitCFuncs:          Option[Int] = this.outputSplitCFuncs,
      disabledWarnings:           Seq[String] = this.disabledWarnings,
      disableFatalExitOnWarnings: Boolean = this.disableFatalExitOnWarnings,
      enableAllAssertions:        Boolean = this.enableAllAssertions,
      timing:                     Option[CompilationSettings.Timing.Type] = this.timing
    ): CompilationSettings = _copy(
      traceStyle = traceStyle,
      outputSplit = outputSplit,
      outputSplitCFuncs = outputSplitCFuncs,
      disabledWarnings = disabledWarnings,
      disableFatalExitOnWarnings = disableFatalExitOnWarnings,
      enableAllAssertions = enableAllAssertions,
      timing = timing,
      parallelism = this.parallelism
    )

    // Suppress generation of private copy with default arguments by Scala 3
    private def copy(
      traceStyle:                 Option[CompilationSettings.TraceStyle],
      outputSplit:                Option[Int],
      outputSplitCFuncs:          Option[Int],
      disabledWarnings:           Seq[String],
      disableFatalExitOnWarnings: Boolean,
      enableAllAssertions:        Boolean,
      timing:                     Option[CompilationSettings.Timing.Type],
      parallelism:                Option[CompilationSettings.Parallelism.Type],
      coverageSettings:           CompilationSettings.CoverageSettings
    ): CompilationSettings = copyInternal(
      traceStyle = traceStyle,
      outputSplit = outputSplit,
      outputSplitCFuncs = outputSplitCFuncs,
      disabledWarnings = disabledWarnings,
      disableFatalExitOnWarnings = disableFatalExitOnWarnings,
      enableAllAssertions = enableAllAssertions,
      timing = timing,
      parallelism = Some(CompilationSettings.Parallelism.Uniform.default),
      coverageSettings = coverageSettings
    )

    def withTraceStyle(traceStyle: Option[CompilationSettings.TraceStyle]) = _copy(traceStyle = traceStyle)

    def withOutputSplit(outputSplit: Option[Int]) = _copy(outputSplit = outputSplit)

    def withOutputSplitCFuncs(outputSplitCFuncs: Option[Int]) = _copy(outputSplitCFuncs = outputSplitCFuncs)

    def withDisabledWarnings(disabledWarnings: Seq[String]) = _copy(disabledWarnings = disabledWarnings)

    def withDisableFatalExitOnWarnings(disableFatalExitOnWarnings: Boolean) =
      _copy(disableFatalExitOnWarnings = disableFatalExitOnWarnings)

    def withEnableAllAssertions(enableAllAssertions: Boolean) = _copy(enableAllAssertions = enableAllAssertions)

    /** Configure coverage instrumentation for Verilator.
      *
      * @param coverageSettings coverage kinds to enable during compilation
      */
    def withCoverageSettings(coverageSettings: CompilationSettings.CoverageSettings) =
      copyInternal(coverageSettings = coverageSettings)

    def withTiming(timing: Option[CompilationSettings.Timing.Type]) = _copy(timing = timing)

    def withParallelism(parallelism: Option[CompilationSettings.Parallelism.Type]) = _copy(parallelism = parallelism)
  }

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
      case None => ()
    }

    commonSettings.libraryExtensions.foreach { extensions =>
      addArg(Seq((Seq("+libext") ++ extensions).mkString("+")))
    }

    commonSettings.libraryPaths.foreach { paths =>
      paths.foreach(p => addArg(Seq("-y", p)))
    }

    commonSettings.linkLibraryPaths.foreach(lib => addArg(Seq(lib)))

    commonSettings.includeDirs.foreach { dirs =>
      addArg(dirs.map(dir => s"+incdir+$dir"))
    }

    commonSettings.defaultTimescale.foreach { value =>
      addArg(Seq("--timescale", value.toString))
    }

    backendSpecificSettings.traceStyle.foreach { ts => addArg(ts.toCompileFlags) }

    addArg(backendSpecificSettings.coverageSettings.toCompileFlags)

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
      val defs = Seq(s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorSupport}") ++
        Option.when(backendSpecificSettings.traceStyle.nonEmpty)(
          s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorTrace}"
        ) ++
        Option.when(backendSpecificSettings.coverageSettings.any)(
          s"-D${svsim.Backend.HarnessCompilationFlags.enableVerilatorCoverage}"
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
