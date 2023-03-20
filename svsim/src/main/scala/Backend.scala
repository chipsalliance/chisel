package svsim

// -- Compilation Settings

/** Settings supported by all svsim backends.
  */
case class SvsimCompilationSettings(
  verilogPreprocessorDefines: Seq[SvsimCompilationSettings.VerilogPreprocessorDefine] = Seq(),
  optimizationStyle:          SvsimCompilationSettings.OptimizationStyle = SvsimCompilationSettings.OptimizationStyle.Default,
  availableParallelism: SvsimCompilationSettings.AvailableParallelism =
    SvsimCompilationSettings.AvailableParallelism.Default,
  defaultTimescale:  Option[SvsimCompilationSettings.Timescale] = None,
  libraryExtensions: Option[Seq[String]] = None,
  libraryPaths:      Option[Seq[String]] = None)
object SvsimCompilationSettings {
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

  val default = SvsimCompilationSettings()

  sealed trait Timescale
  object Timescale {
    case class FromString(value: String) extends Timescale
  }
}

trait Backend {
  type CompilationSettings
  private[svsim] def invocationSettings(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          SvsimCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): Backend.InvocationSettings
}

object Backend {

  private[svsim] final case class InvocationSettings(
    compilerPath:          String,
    compilerArguments:     Seq[String],
    compilerEnvironment:   Seq[(String, String)],
    simulationArguments:   Seq[String],
    simulationEnvironment: Seq[(String, String)])

  // -- Flags affecting shared code compilation

  /** Verilator support requires that we manually implement some SystemVerilog functions, such as `run_simulation` and `simulation_main`. These flags control the Verilator-specific code paths.
    */
  private[svsim] val enableVerilatorSupportFlag = "SVSIM_ENABLE_VERILATOR_SUPPORT"
  private[svsim] val enableVerilatorTraceFlag = "SVSIM_VERILATOR_TRACE_ENABLED"

  /** This flag controls if VCS-specifc code is compiled.
    */
  private[svsim] val enableVCSSupportFlag = "SVSIM_ENABLE_VCS_SUPPORT"

  /** Flags enabling various tracing mechanisms.
    */
  private[svsim] val enableVcdTracingFlag = "SVSIM_ENABLE_VCD_TRACING"
  private[svsim] val enableVpdTracingFlag = "SVSIM_ENABLE_VPD_TRACING"
  private[svsim] val enableFsdbTracingFlag = "SVSIM_ENABLE_FSDB_TRACING"

  /** Verilator does not currently support delay (`#delay`) in DPI functions, so we omit the SystemVerilog definition of the `run_simulation` function and instead provide a C implementation.
    */
  private[svsim] val supportsDelayInPublicFunctionsFlag = "SVSIM_BACKEND_SUPPORTS_DELAY_IN_PUBLIC_FUNCTIONS"

  /** VCS first checks whether address-space layout randomization (ASLR) is enabled, and if it is, _helpfully_ relaunches this executable with ASLR disabled. Unfortunately, this causes code executed prior to `simulation_main` to be executed twice, which is problematic, especially since we redirect `stdin` and `stdout`.
    */
  private[svsim] val engagesInASLRShenanigansFlag = "SVSIM_BACKEND_ENGAGES_IN_ASLR_SHENANIGANS"
}
