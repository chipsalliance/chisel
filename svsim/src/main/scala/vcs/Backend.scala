// SPDX-License-Identifier: Apache-2.0

package svsim.vcs

import svsim._

object Backend {

  /** Utilities for working with VCS license files */
  object LicenseFile {

    /** The super type of all license files */
    sealed trait Type {

      /** The value of the environment variable */
      def value: String

      /** Convert this to name/value tuple that can be used to set an environment variable */
      def toEnvVar: (String, String)
    }

    /** A Synopsys-specific license file environment variable
      *
      * @param value the value of the environment variable
      */
    case class Synopsys(value: String) extends Type {
      override def toEnvVar = Synopsys.name -> value
    }
    object Synopsys {
      def name: String = "SNPSLMD_LICENSE_FILE"
    }

    /** A generic license file environment variable
      *
      * @param value the value of the environment value
      */
    case class Generic(value: String) extends Type {
      override def toEnvVar = Generic.name -> value
    }
    object Generic {
      def name: String = "LM_LICENSE_FILE"
    }
  }

  object CompilationSettings {
    sealed trait XProp
    object XProp {
      case object XMerge extends XProp
      case object TMerge extends XProp
    }

    object TraceSettings {
      final case class FsdbSettings(verdiHome: String)
    }
    final case class TraceSettings(
      enableVcd:    Boolean = false,
      enableVpd:    Boolean = false,
      fsdbSettings: Option[TraceSettings.FsdbSettings] = None
    ) {
      private def fsdbEnabled = fsdbSettings match {
        case Some(_) => true
        case None    => false
      }
      private[vcs] def compileFlags = Seq(
        if (enableVpd || fsdbEnabled) Seq("-debug_access+pp+dmptf") else Seq(),
        fsdbSettings match {
          case None => Seq()
          case Some(_) =>
            Seq(
              "-kdb"
            )
        }
      ).flatten
      private[vcs] def verilogPreprocessorDefines = Seq(
        (enableVcd, svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport),
        (enableVpd, svsim.Backend.HarnessCompilationFlags.enableVpdTracingSupport),
        (fsdbEnabled, svsim.Backend.HarnessCompilationFlags.enableFsdbTracingSupport)
      ).collect { case (true, value) =>
        svsim.CommonCompilationSettings.VerilogPreprocessorDefine(value)
      }
      private[vcs] def environment = fsdbSettings match {
        case None                                        => Seq()
        case Some(TraceSettings.FsdbSettings(verdiHome)) => Seq("VERDI_HOME" -> verdiHome)
      }
    }
  }

  sealed trait AssertionSettings
  case class AssertGlobalMaxFailCount(count: Int) extends AssertionSettings

  final case class SimulationSettings(
    customWorkingDirectory: Option[String] = None,
    assertionSettings:      Option[AssertionSettings] = None,
    coverageSettings:       CoverageSettings = CoverageSettings(),
    coverageDirectory:      Option[CoverageDirectory] = None,
    coverageName:           Option[CoverageName] = None
  )

  /** Trait that encodes a VCS "plus" option.
    *
    * VCS has a lot of options that take the form of:
    *
    *     -<name> <flag>[+<flag>...]
    *
    * This encapsulates the logic to generate these options from a `Product`
    * that consists of elements whose names are the flags.
    */
  sealed trait PlusSeparated { this: Product =>

    /** The name of the option. */
    def name: String

    /** Convert the option into command line flags */
    final def toFlags: Seq[String] = {
      val setFlags: Seq[String] = productElementNames
        .zip(productIterator)
        .flatMap {
          case (name, true) => Some(name)
          case _            => None
        }
        .toSeq

      if (setFlags.isEmpty) {
        Seq.empty
      } else {
        Seq(s"-$name", setFlags.mkString("+"))
      }

    }

    /** Return true if any of the flags are set. */
    final def any: Boolean = productIterator.collect(_.isInstanceOf[Boolean]).foldLeft(false)(_ || _)

  }

  /** Settings for controlling VCS coverage.
    *
    * These options map to the `-cm` option.  Each parameter turns on a specific
    * kind of coverage.  Consult the Synopsys VCS user guide for documentation.
    */
  // Note: This case class is being clever and re-using the names of the
  // elements as the names of the options.  This makes these very terse, but
  // they then match, exactly, the documentation in the VCS manual.
  final case class CoverageSettings(
    line:   Boolean = false,
    cond:   Boolean = false,
    fsm:    Boolean = false,
    tgl:    Boolean = false,
    obc:    Boolean = false,
    path:   Boolean = false,
    assert: Boolean = false,
    branch: Boolean = false,
    sdc:    Boolean = false
  ) extends PlusSeparated {

    override def name = "cm"

  }

  /** Settings for controlling the coverage directory
    *
    * This maps to the `-cm_dir` option.
    */
  final case class CoverageDirectory(
    directory: String
  ) {

    def toFlags: Seq[String] = Seq("-cm_dir", directory)

  }

  /** Sets a unique name used for this coverage run
    *
    * This maps to the `-cm_name` option.
    */
  final case class CoverageName(
    name: String
  ) {

    def toFlags: Seq[String] = Seq("-cm_name", name)

  }

  /** Settings for controlling VCS toggle coverage.
    *
    * These options map to the `-cm_tgl` option.  Consult the Synopsys VCS user
    * guide for documentation.
    */
  final case class ToggleCoverageSettings(
    assign:              Boolean = false,
    portsonly:           Boolean = false,
    fullintf:            Boolean = false,
    mda:                 Boolean = false,
    count:               Boolean = false,
    structarr:           Boolean = false,
    modportarr:          Boolean = false,
    unencrypted_signals: Boolean = false,
    old:                 Boolean = false
  ) extends PlusSeparated {

    override def name = "cm_tgl"

  }

  /** Settings for controlling VCS branch coverage.
    *
    * These options map to the `-cm_branch` option.  Consult the Synopsys VCS
    * user guide for documentation of this option.
    */
  final case class BranchCoverageSettings(
    values:               Boolean = false,
    ignoreMissingDefault: Boolean = false
  ) extends PlusSeparated {

    override def name = "cm_branch"

  }

  /** Container of all simple flag options to VCS.
    *
    * All these options take no arguments.
    */
  object Flag {

    /** A simple flag option
      *
      * This option only has a name and has no parameter.  The option name is
      * the class name.  Force this to be a `Singleton` so that it must be an
      * object/case object.  Because Scala will add a trailing `$` to objects,
      * drop that when generating the flag.
      */
    sealed trait Type { this: Singleton =>
      def toFlag: String = s"-${this.getClass.getSimpleName.dropRight(1)}"
    }

    case object cm_seqnoconst extends Type

  }

  case class CompilationSettings(
    xProp:                       Option[CompilationSettings.XProp] = None,
    randomlyInitializeRegisters: Boolean = false,
    traceSettings:               CompilationSettings.TraceSettings = CompilationSettings.TraceSettings(),
    simulationSettings:          SimulationSettings = SimulationSettings(),
    coverageSettings:            CoverageSettings = CoverageSettings(),
    coverageDirectory:           Option[CoverageDirectory] = None,
    toggleCoverageSettings:      ToggleCoverageSettings = ToggleCoverageSettings(),
    branchCoverageSettings:      BranchCoverageSettings = BranchCoverageSettings(),
    flags:                       Seq[Flag.Type] = Seq.empty,
    licenceExpireWarningTimeout: Option[Int] = None,
    archOverride:                Option[String] = None,
    waitForLicenseIfUnavailable: Boolean = false
  ) extends svsim.Backend.Settings

  def initializeFromProcessEnvironment() = {
    val vcsUserGuideNote =
      "Please consult the VCS User Guide for information on how to setup your environment to run VCS."

    // Extract VCS-specific environment variables.  VCS_HOME must be set.  Then
    // either SNPSLMD_LICENSE_FILE or LM_LICENSE_FILE must be set.
    val (vcsHome, lic) =
      (sys.env.get("VCS_HOME"), sys.env.get(LicenseFile.Synopsys.name), sys.env.get(LicenseFile.Generic.name)) match {
        case (None, _, _) =>
          throw new svsim.Backend.Exceptions.FailedInitialization(
            s"Unable to initialize VCS as the environment variable 'VCS_HOME' was not set.  $vcsUserGuideNote"
          )
        case (Some(vcsHome), None, None) =>
          throw new svsim.Backend.Exceptions.FailedInitialization(
            s"Unable to initialize VCS as neither the environment variable '${LicenseFile.Synopsys.name}' or '${LicenseFile.Generic.name}' was set.  $vcsUserGuideNote"
          )
        case (Some(vcsHome), Some(snpsLic), _)  => (vcsHome, LicenseFile.Synopsys(snpsLic))
        case (Some(vcsHome), None, Some(lmLic)) => (vcsHome, LicenseFile.Generic(lmLic))
      }

    new Backend(
      vcsHome,
      lic,
      defaultArchOverride = sys.env.get("VCS_ARCH_OVERRIDE"),
      defaultLicenseExpireWarningTimeout = sys.env.get("VCS_LIC_EXPIRE_WARNING")
    )

  }
}
final class Backend(
  vcsHome:                            String,
  licenseFile:                        Backend.LicenseFile.Type,
  defaultArchOverride:                Option[String] = None,
  defaultLicenseExpireWarningTimeout: Option[String] = None
) extends svsim.Backend {
  type CompilationSettings = Backend.CompilationSettings

  override def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): svsim.Backend.Parameters = {
    // These environment variables apply to both compilation and simulation
    val environment = Seq(
      backendSpecificSettings.archOverride
        .orElse(defaultArchOverride)
        .map("VCS_ARCH_OVERRIDE" -> _),
      backendSpecificSettings.licenceExpireWarningTimeout
        .map(_.toString)
        .orElse(defaultLicenseExpireWarningTimeout)
        .map("VCS_LIC_EXPIRE_WARNING" -> _)
    ).flatten

    //format: off
    import CommonCompilationSettings._
    import Backend.CompilationSettings._
    svsim.Backend.Parameters(
      compilerPath = s"$vcsHome/bin/vcs",
      compilerInvocation = svsim.Backend.Parameters.Invocation(
        arguments = Seq[Seq[String]](
          Seq(
            "-full64", // Enable 64-bit compilation
            "-sverilog", // Enable SystemVerilog
            "-nc", // Do not emit copyright notice
            // Specify resulting executable path
            "-o", outputBinaryName,
            // Rename `main` so we use the `main` provided by `simulation-driver.cpp`
            "-e", "simulation_main",
          ),

          Seq(
            ("-licqueue", backendSpecificSettings.waitForLicenseIfUnavailable),
            ("+vcs+initreg+random", backendSpecificSettings.randomlyInitializeRegisters)
          ).collect {
            case (flag, true) => flag
          },

          commonSettings.defaultTimescale match {
            case Some(timescale) => Seq(s"-timescale=$timescale")
            case None => Seq()
          },

          commonSettings.availableParallelism match {
            case AvailableParallelism.Default => Seq()
            case AvailableParallelism.UpTo(value) => Seq(s"-j${value}")
          },

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

          backendSpecificSettings.xProp match {
            case None => Seq()
            case Some(XProp.XMerge) => Seq("-xprop=xmerge")
            case Some(XProp.TMerge) => Seq("-xprop=tmerge")
          },

          Seq(
            ("-CFLAGS", Seq(
              commonSettings.optimizationStyle match {
                case OptimizationStyle.Default => Seq()
                case OptimizationStyle.OptimizeForCompilationSpeed => Seq("-O0")
                case OptimizationStyle.OptimizeForSimulationSpeed=> Seq("-O3")
              },

              additionalHeaderPaths.map { path => s"-I${path}" },

              Seq(
                // Enable VCS support
                s"-D${svsim.Backend.HarnessCompilationFlags.enableVCSSupport}",
              )
            )),
          ).collect {
            /// Only include flags that have more than one value
            case (flag, value) if !value.isEmpty => Seq(flag, value.flatten.mkString(" "))
          }.flatten,

          backendSpecificSettings.traceSettings.compileFlags,

          backendSpecificSettings.coverageSettings.toFlags,

          backendSpecificSettings.coverageDirectory.map(_.toFlags).getOrElse(Seq.empty),

          backendSpecificSettings.toggleCoverageSettings.toFlags,

          backendSpecificSettings.branchCoverageSettings.toFlags,

          backendSpecificSettings.flags.map(_.toFlag),

          Seq(
            commonSettings.verilogPreprocessorDefines,
            Seq(
              VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.supportsDelayInPublicFunctions)
            ),
            backendSpecificSettings.traceSettings.verilogPreprocessorDefines
          ).flatten.map(_.toCommandlineArgument(this)),
        ).flatten,
        environment = environment ++ Seq(
          "VCS_HOME" -> vcsHome,
          licenseFile.toEnvVar
        ) ++ backendSpecificSettings.traceSettings.environment
      ),
      simulationInvocation = svsim.Backend.Parameters.Invocation(
        arguments = Seq(
          backendSpecificSettings.simulationSettings.assertionSettings match {
            case None                                          => Seq()
            case Some(Backend.AssertGlobalMaxFailCount(count)) => Seq("-assert", s"global_finish_maxfail=$count")
          },
          backendSpecificSettings.simulationSettings.coverageSettings.toFlags,
          backendSpecificSettings.simulationSettings.coverageDirectory.map(_.toFlags).getOrElse(Seq.empty),
          backendSpecificSettings.simulationSettings.coverageName.map(_.toFlags).getOrElse(Seq.empty),
          commonSettings.simulationSettings.plusArgs.map(_.simulatorFlags),
          // In order to support save/restore functionality, VCS will detect if
          // Address Space Layout Randomization (ASLR) is ocurring when the
          // simulation starts.  If it is, then VCS will relaunch the simulation
          // with ASLR turned off.  This double-launch confuses svsim.  To avoid
          // this, and because svsim doesn't support save/restore functionality,
          // we turn off VCS save/restore features.  The simulation binary will
          // then only run once.
          Seq("-no_save")
        ).flatten,
        environment = environment
      )
    )
    //format: on
  }

  /** VCS seems to require that dollar signs in arguments are escaped.  This is
    * different from Verilator.
    */
  override def escapeDefine(string: String): String = string.replace("$", "\\$")

  override val assertionFailed =
    "^((Assertion failed:)|(Error: )|(Fatal: )|(.* started at .* failed at .*)|(.*Offending)).*$".r
}
