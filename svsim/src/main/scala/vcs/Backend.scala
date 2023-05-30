// SPDX-License-Identifier: Apache-2.0

package svsim.vcs

import svsim._

object Backend {
  object CompilationSettings {
    sealed trait XProp
    object XProp {
      case object XMerge extends XProp
      case object TMerge extends XProp
    }

    final object TraceSettings {
      final case class FsdbSettings(verdiHome: String)
    }
    final case class TraceSettings(
      enableVcd:    Boolean = false,
      enableVpd:    Boolean = false,
      fsdbSettings: Option[TraceSettings.FsdbSettings] = None) {
      private def fsdbEnabled = fsdbSettings match {
        case Some(_) => true
        case None    => false
      }
      private[vcs] def compileFlags = Seq(
        if (enableVpd || fsdbEnabled) Seq("-debug_acc+pp+dmptf") else Seq(),
        fsdbSettings match {
          case None => Seq()
          case Some(TraceSettings.FsdbSettings(verdiHome)) =>
            Seq(
              "-kdb",
              "-P",
              s"$verdiHome/share/PLI/VCS/LINUX64/novas.tab",
              s"$verdiHome/share/PLI/VCS/LINUX64/pli.a"
            )
        }
      ).flatten
      private[vcs] def verilogPreprocessorDefines = Seq(
        (enableVcd, svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport),
        (enableVpd, svsim.Backend.HarnessCompilationFlags.enableVpdTracingSupport),
        (fsdbEnabled, svsim.Backend.HarnessCompilationFlags.enableFsdbTracingSupport)
      ).collect {
        case (true, value) =>
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
    assertionSettings:      Option[AssertionSettings] = None)

  case class CompilationSettings(
    xProp:                       Option[CompilationSettings.XProp] = None,
    randomlyInitializeRegisters: Boolean = false,
    traceSettings:               CompilationSettings.TraceSettings = CompilationSettings.TraceSettings(),
    simulationSettings:          SimulationSettings = SimulationSettings(),
    licenceExpireWarningTimeout: Option[Int] = None,
    archOverride:                Option[String] = None,
    waitForLicenseIfUnavailable: Boolean = false)

  def initializeFromProcessEnvironment() = {
    (sys.env.get("VCS_HOME"), sys.env.get("LM_LICENSE_FILE")) match {
      case (Some(vcsHome), Some(lmLicenseFile)) =>
        Some(
          new Backend(
            vcsHome,
            lmLicenseFile,
            defaultArchOverride = sys.env.get("VCS_ARCH_OVERRIDE"),
            defaultLicenseExpireWarningTimeout = sys.env.get("VCS_LIC_EXPIRE_WARNING")
          )
        )
      case _ => None
    }
  }
}
final class Backend(
  vcsHome:                            String,
  lmLicenseFile:                      String,
  defaultArchOverride:                Option[String] = None,
  defaultLicenseExpireWarningTimeout: Option[String] = None)
    extends svsim.Backend {
  type CompilationSettings = Backend.CompilationSettings

  def generateParameters(
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
        .orElse(defaultLicenseExpireWarningTimeout)
        .map("VCS_LIC_EXPIRE_WARNING" -> _.toString)
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
            case Some(Timescale.FromString(value)) => Seq(s"-timescale=$value")
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
              },
              
              additionalHeaderPaths.map { path => s"-I${path}" },
              
              Seq(
                // Enable VCS support
                s"-D${svsim.Backend.HarnessCompilationFlags.enableVCSSupport}",
                // VCS engages in ASLR shenanigans
                s"-D${svsim.Backend.HarnessCompilationFlags.backendEngagesInASLRShenanigans}",
              )
            )),
          ).collect {
            /// Only include flags that have more than one value
            case (flag, value) if !value.isEmpty => Seq(flag, value.flatten.mkString(" "))
          }.flatten,

          backendSpecificSettings.traceSettings.compileFlags,
          
          Seq(
            commonSettings.verilogPreprocessorDefines,
            Seq(
              VerilogPreprocessorDefine(svsim.Backend.HarnessCompilationFlags.supportsDelayInPublicFunctions)
            ),
            backendSpecificSettings.traceSettings.verilogPreprocessorDefines
          ).flatten.map(_.toCommandlineArgument),
        ).flatten,
        environment = environment ++ Seq(
          "VCS_HOME" -> vcsHome,
          "LM_LICENSE_FILE" -> lmLicenseFile,
        ) ++ backendSpecificSettings.traceSettings.environment
      ),
      simulationInvocation = svsim.Backend.Parameters.Invocation(
        arguments = Seq(
          backendSpecificSettings.simulationSettings.assertionSettings match {
            case None                                          => Seq()
            case Some(Backend.AssertGlobalMaxFailCount(count)) => Seq("-assert", s"global_finish_maxfail=$count")
          },
        ).flatten,
        environment = environment
      )
    )
    //format: on
  }
}
