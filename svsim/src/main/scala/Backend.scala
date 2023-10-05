package svsim

// -- Compilation Settings

/** Settings supported by all svsim backends.
  */
case class CommonCompilationSettings(
  verilogPreprocessorDefines: Seq[CommonCompilationSettings.VerilogPreprocessorDefine] = Seq(),
  optimizationStyle:          CommonCompilationSettings.OptimizationStyle = CommonCompilationSettings.OptimizationStyle.Default,
  availableParallelism: CommonCompilationSettings.AvailableParallelism =
    CommonCompilationSettings.AvailableParallelism.Default,
  defaultTimescale:  Option[CommonCompilationSettings.Timescale] = None,
  libraryExtensions: Option[Seq[String]] = None,
  libraryPaths:      Option[Seq[String]] = None)
object CommonCompilationSettings {
  object VerilogPreprocessorDefine {
    def apply(name: String, value: String) = new VerilogPreprocessorDefine(name, Some(value))
    def apply(name: String) = new VerilogPreprocessorDefine(name, None)
  }
  case class VerilogPreprocessorDefine private (name: String, value: Option[String]) {
    private[svsim] def toCommandlineArgument: String = {
      value match {
        case Some(v) => s"+define+${name}=${v}"
        case None    => s"+define+${name}"
      }
    }
  }

  sealed trait OptimizationStyle
  object OptimizationStyle {

    /** Use the default optimization level specified by the backend tool (i.e. Verilator or VCS) being used.
      */
    object Default extends OptimizationStyle

    /** Optimize for compilation speed, which generally means disabling as many optimizations as possible.
      */
    object OptimizeForCompilationSpeed extends OptimizationStyle
  }

  sealed trait AvailableParallelism
  object AvailableParallelism {

    /** Use the default number of parallel processes specified by the backend tool (i.e. Verilator or VCS) being used.
      */
    object Default extends AvailableParallelism

    /** Use up to specified number of parallel processes.
      */
    case class UpTo(value: Int) extends AvailableParallelism
  }

  val default = CommonCompilationSettings()

  sealed trait Timescale
  object Timescale {
    case class FromString(value: String) extends Timescale
  }
}

trait Backend {
  type CompilationSettings
  def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): Backend.Parameters
}

final object Backend {

  final case class Parameters(
    private[svsim] val compilerPath:         String,
    private[svsim] val compilerInvocation:   Parameters.Invocation,
    private[svsim] val simulationInvocation: Parameters.Invocation)

  final object Parameters {

    /**
      * Parameters for the invocation of a command-line tool. The constituent properties are private to `svsim` and not meant for external consumption (we may change this representation in the future, for example to [add convenient tracing functionality to make-replay](https://github.com/chipsalliance/chisel/issues/3150)).
      */
    final case class Invocation(
      private[svsim] val arguments:   Seq[String],
      private[svsim] val environment: Seq[(String, String)])
  }

  /**
    * A namespace for flags affecting which code in the harness is compiled.
    */
  object HarnessCompilationFlags {

    /** Verilator support requires that we manually implement some SystemVerilog functions, such as `run_simulation` and `simulation_main`. These flags control the Verilator-specific code paths.
      */
    val enableVerilatorSupport = "SVSIM_ENABLE_VERILATOR_SUPPORT"
    val enableVerilatorTrace = "SVSIM_VERILATOR_TRACE_ENABLED"

    /** This flag controls if VCS-specifc code is compiled.
      */
    val enableVCSSupport = "SVSIM_ENABLE_VCS_SUPPORT"

    /** Flags enabling various tracing mechanisms.
      * Note: These flags do not cause tracing to occur, they simply support for these tracing mechanisms in the harness.
      */
    val enableVcdTracingSupport = "SVSIM_ENABLE_VCD_TRACING_SUPPORT"
    val enableVpdTracingSupport = "SVSIM_ENABLE_VPD_TRACING_SUPPORT"
    val enableFsdbTracingSupport = "SVSIM_ENABLE_FSDB_TRACING_SUPPORT"

    /** Verilator does not currently support delay (`#delay`) in DPI functions, so we omit the SystemVerilog definition of the `run_simulation` function and instead provide a C implementation.
      */
    val supportsDelayInPublicFunctions = "SVSIM_BACKEND_SUPPORTS_DELAY_IN_PUBLIC_FUNCTIONS"

    /** VCS first checks whether address-space layout randomization (ASLR) is enabled, and if it is, _helpfully_ relaunches this executable with ASLR disabled. Unfortunately, this causes code executed prior to `simulation_main` to be executed twice, which is problematic, especially since we redirect `stdin` and `stdout`.
      */
    val backendEngagesInASLRShenanigans = "SVSIM_BACKEND_ENGAGES_IN_ASLR_SHENANIGANS"
  }
}
