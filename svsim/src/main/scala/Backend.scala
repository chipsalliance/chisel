package svsim

import java.io.File
import scala.util.matching.Regex

trait CommonSettingsModifications extends (CommonCompilationSettings => CommonCompilationSettings)

object CommonSettingsModifications {

  implicit def unmodified: CommonSettingsModifications = identity(_)

}

// -- Compilation Settings

/** Backend-independent simulation runtime settings
  *
  * @note Use [[CommonSimulationSettings$]] methods to create objects of this
  * class.
  */
final class CommonSimulationSettings private[svsim] (
  val plusArgs:              Seq[PlusArg],
  val enableWavesAtTimeZero: Boolean,
  val traceFileStem:         String
) {

  require(traceFileStem.nonEmpty, "traceFileStem must be a non-empty String")

  /** Return a copy of this [[CommonSimulationSettings]] with some fields
    * modified.
    */
  def copy(
    plusArgs:              Seq[PlusArg] = plusArgs,
    enableWavesAtTimeZero: Boolean = enableWavesAtTimeZero,
    traceFileStem:         String = traceFileStem
  ) = new CommonSimulationSettings(
    plusArgs = plusArgs,
    enableWavesAtTimeZero = enableWavesAtTimeZero,
    traceFileStem = traceFileStem
  )
}

object CommonSimulationSettings {

  /** Return a [[CommonSimulationSettings]] with default values.
    *
    * @param plusArgs Verilog value or test plusargs to set at simulation
    * runtime
    */
  def default = new CommonSimulationSettings(
    plusArgs = Seq.empty,
    enableWavesAtTimeZero = false,
    traceFileStem = "trace"
  )

}

/** Settings supported by all svsim backends.
  */
case class CommonCompilationSettings(
  verilogPreprocessorDefines: Seq[CommonCompilationSettings.VerilogPreprocessorDefine] = Seq(),
  optimizationStyle: CommonCompilationSettings.OptimizationStyle = CommonCompilationSettings.OptimizationStyle.Default,
  availableParallelism: CommonCompilationSettings.AvailableParallelism =
    CommonCompilationSettings.AvailableParallelism.Default,
  defaultTimescale:   Option[CommonCompilationSettings.Timescale] = Some(CommonCompilationSettings.Timescale.default),
  libraryExtensions:  Option[Seq[String]] = None,
  libraryPaths:       Option[Seq[String]] = None,
  includeDirs:        Option[Seq[String]] = None,
  fileFilter:         PartialFunction[File, Boolean] = PartialFunction.empty,
  directoryFilter:    PartialFunction[File, Boolean] = PartialFunction.empty,
  simulationSettings: CommonSimulationSettings = CommonSimulationSettings.default
)
object CommonCompilationSettings {
  object VerilogPreprocessorDefine {
    def apply(name: String, value: String) = new VerilogPreprocessorDefine(name, Some(value))
    def apply(name: String) = new VerilogPreprocessorDefine(name, None)
  }
  case class VerilogPreprocessorDefine private (name: String, value: Option[String]) {
    final def toCommandlineArgument(backend: Backend): String = {
      value match {
        case Some(v) => s"+define+${backend.escapeDefine(name)}=${backend.escapeDefine(v)}"
        case None    => s"+define+${backend.escapeDefine(name)}"
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

    /** Optimize for execution speed, which generally means enabling as many optimizations as possible.
      */
    object OptimizeForSimulationSpeed extends OptimizationStyle
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

  /** Object that contains the implementation of SystemVerilog timescales.
    *
    * @see 1800-2023 Section 3.14
    */
  object Timescale {
    sealed trait Type

    /** Represents legal magnitudes for Verilog times.
      *
      * These must be exactly 1, 10, or 100.
      */
    object Magnitude {
      sealed trait Type { this: Singleton =>
        def toInt: Int
        override def toString = toInt.toString
      }
      case object One extends Type {
        override def toInt = 1
      }
      case object Ten extends Type {
        override def toInt = 10
      }
      case object Hundred extends Type {
        override def toInt = 100
      }
    }

    /** Represents legal units for Verilog times.
      *
      * These are allowed to be femtoseconds up to and including seconds in SI
      * units.
      */
    object Unit {
      sealed trait Type { this: Singleton =>
        override def toString = this.getClass.getSimpleName.dropRight(1)

        def exponent: Int
      }

      /** Seconds */
      case object s extends Type {
        override def exponent = 0
      }

      /** Milliseconds, i.e., `10^-3` seconds */
      case object ms extends Type {
        override def exponent = -3
      }

      /** Microseconds, i.e., `10^-6` seconds */
      case object us extends Type {
        override def exponent = -6
      }

      /** Nanoseconds, i.e., `10^-9` seconds */
      case object ns extends Type {
        override def exponent = -9
      }

      /** Picosedonds, i.e., `10^-12` seconds */
      case object ps extends Type {
        override def exponent = -12
      }

      /** Femtoseconds, i.e., `10^-15` seconds */
      case object fs extends Type {
        override def exponent = -15
      }
    }

    /** Represnts a Verilog notion of time.  This includes both a magnitude (the
      * time value) as well as the units.
      */
    case class Time(magnitude: Magnitude.Type, unit: Unit.Type) {
      override def toString = s"$magnitude$unit"

      /** Return the time in femtoseconds.
        *
        * Femtoseconds are the unit because this is the smallest unit of time
        * that Verilog represents.  I.e., this is sufficient to represent any
        * time that Verilator understands without resorting to imprecise floats.
        */
      final def fs: BigInt = magnitude.toInt * BigInt(10).pow(unit.exponent - Unit.fs.exponent)
    }

    /** A sane default timescale that allows for representing common, rounded clock
      * frequencies.
      */
    def default: Timescale =
      Timescale(unit = Time(Magnitude.One, Unit.ns), precision = Time(Magnitude.Hundred, Unit.ps))
  }

  /** A SystemVerilog timescale.
    *
    * This is something that can be legally used as the argument of a ``
    * `timescale `` directive.
    *
    * @see 1800-2023 Section 3.14
    */
  case class Timescale(unit: Timescale.Time, precision: Timescale.Time) {
    require(unit.fs >= precision.fs, "the timescale unit must be greater than or equal to the timescale precision")

    override def toString = s"$unit/$precision"

    /** Return a decimal represntation of one timescale _precision_ in the units of
      * one timescale _unit_.
      */
    final def tickPrecision: Double = precision.fs.toDouble / unit.fs.toDouble
  }

}

/** A key/value pair representing a Verilog plusarg
  *
  * When `value` is a `Some`, this acts like a `$value$plusargs`.  When `value`
  * is `None`, this acts like a `$test$plusargs`.  I.e., when `None`, the
  * plusarg is simply set.  When `Some`, the plusarg has a value.
  *
  * @param name the name of the plusarg
  * @param value an optional value for the plusarg
  */
final class PlusArg(
  val name:  String,
  val value: Option[String] = None
) {

  /** Return Verilator-compliant or VCS-compliant simulator options to set this
    * plusarg.
    */
  def simulatorFlags: String = (name +: value.toSeq).mkString("+", "=", "")

}

trait Backend {
  type CompilationSettings <: Backend.Settings
  def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): Backend.Parameters

  /** This function will be applied to all defines (both the keys and the values).
    * This can be used to workaround subtleties in how different simulators
    * parse defines and require different escaping.
    */
  def escapeDefine(string: String): String

  /** A regular expression that indicates lines in a log file which indicate
    * assertion failure.
    */
  def assertionFailed: Regex

}

/** Type class to modify backend comopilation settings */
trait BackendSettingsModifications extends (Backend.Settings => Backend.Settings)

object BackendSettingsModifications {

  implicit def unmodified: BackendSettingsModifications = identity(_)

}

final object Backend {

  /** The super type of all backend-specific settings. */
  trait Settings extends Product

  final case class Parameters(
    private[svsim] val compilerPath:         String,
    private[svsim] val compilerInvocation:   Parameters.Invocation,
    private[svsim] val simulationInvocation: Parameters.Invocation
  ) {

    /** Get the path to the compiler executable. */
    def getCompilerPath: String = compilerPath

    /** Get the compiler invocation arguments. */
    def getCompilerArguments: Seq[String] = compilerInvocation.arguments

    /** Get the compiler invocation environment variables. */
    def getCompilerEnvironment: Seq[(String, String)] = compilerInvocation.environment

    /** Get the simulation invocation arguments. */
    def getSimulationArguments: Seq[String] = simulationInvocation.arguments

    /** Get the simulation invocation environment variables. */
    def getSimulationEnvironment: Seq[(String, String)] = simulationInvocation.environment
  }

  final object Parameters {

    /**
      * Parameters for the invocation of a command-line tool. The constituent properties are private to `svsim` and not meant for external consumption (we may change this representation in the future, for example to [add convenient tracing functionality to make-replay](https://github.com/chipsalliance/chisel/issues/3150)).
      */
    final case class Invocation(
      private[svsim] val arguments:   Seq[String],
      private[svsim] val environment: Seq[(String, String)]
    ) {

      /** Get the invocation arguments. */
      def getArguments: Seq[String] = arguments

      /** Get the invocation environment variables. */
      def getEnvironment: Seq[(String, String)] = environment
    }
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
    val enableFstTracingSupport = "SVSIM_ENABLE_FST_TRACING_SUPPORT"
    val enableVpdTracingSupport = "SVSIM_ENABLE_VPD_TRACING_SUPPORT"
    val enableFsdbTracingSupport = "SVSIM_ENABLE_FSDB_TRACING_SUPPORT"

    /** Verilator does not currently support delay (`#delay`) in DPI functions, so we omit the SystemVerilog definition of the `run_simulation` function and instead provide a C implementation.
      */
    val supportsDelayInPublicFunctions = "SVSIM_BACKEND_SUPPORTS_DELAY_IN_PUBLIC_FUNCTIONS"
  }

  object Exceptions {

    /** Indicates that a backend failed to initialize. */
    final class FailedInitialization private[svsim] (message: String) extends RuntimeException(message)

  }
}
