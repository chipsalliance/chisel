// SPDX-License-Identifier: Apache-2.0

package svsim.arcilator

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.net.URLClassLoader
import svsim._

object Backend {
  case class CompilationSettings(
    enableVcd:        Boolean = false,
    arcilatorOptions: Seq[String] = Seq.empty,
    frontendOptions:  Seq[String] = Seq.empty
  ) extends svsim.Backend.Settings

  def initializeFromProcessEnvironment(): Backend = {
    val executable = sys.env
      .get("ARCILATOR_PATH")
      .orElse {
        sys.env
          .get("PATH")
          .toSeq
          .flatMap(_.split(java.io.File.pathSeparator))
          .map { directory =>
            Paths.get(directory).resolve("arcilator")
          }
          .find(Files.isExecutable(_))
          .map(_.toString)
      }
      .getOrElse(Workspace.getProjectRootOrCwd.resolve("firtool-1.163.0/bin/arcilator").toString)
    new Backend(executable)
  }

  // Makefile compilation runs in a separate JVM, so include classloader URLs
  // as well as java.class.path (important when svsim is loaded by a build tool).
  private def compilerClasspath: String = {
    val loaderPaths = Iterator
      .iterate(Option(getClass.getClassLoader))(_.flatMap(loader => Option(loader.getParent)))
      .takeWhile(_.nonEmpty)
      .flatMap(_.get match {
        case loader: URLClassLoader =>
          loader.getURLs.toSeq.filter(_.getProtocol == "file").map(u => Paths.get(u.toURI).toString)
        case _ => Seq.empty
      })
    (sys.props("java.class.path").split(java.io.File.pathSeparator).toSeq ++ loaderPaths)
      .map(path => Paths.get(path).toAbsolutePath.toString)
      .distinct
      .mkString(java.io.File.pathSeparator)
  }
}

/** Compile SystemVerilog through CIRCT's frontend and arcilator, then link a native svsim bridge. */
final class Backend(executablePath: String) extends svsim.Backend {
  type CompilationSettings = Backend.CompilationSettings

  override def prepareWorkspace(moduleInfo: ModuleInfo, generatedSourcesPath: String): Unit = {
    val directory = Paths.get(generatedSourcesPath)
    val manifest = moduleInfo.name + "\n" + moduleInfo.ports.map { port =>
      s"${port.name}\t${port.isSettable}\t${port.isGettable}\n"
    }.mkString
    Files.write(directory.resolve("arcilator-ports.txt"), manifest.getBytes(StandardCharsets.UTF_8))
  }

  override def generateParameters(
    outputBinaryName:        String,
    topModuleName:           String,
    additionalHeaderPaths:   Seq[String],
    commonSettings:          CommonCompilationSettings,
    backendSpecificSettings: CompilationSettings
  ): svsim.Backend.Parameters = {
    val tools = Paths.get(executablePath).toAbsolutePath.getParent
    require(Files.isExecutable(tools.resolve("arcilator")), s"arcilator not executable: $executablePath")
    val unsupportedWaves = Set(
      svsim.Backend.HarnessCompilationFlags.enableFstTracingSupport,
      svsim.Backend.HarnessCompilationFlags.enableFsdbTracingSupport,
      svsim.Backend.HarnessCompilationFlags.enableVpdTracingSupport
    )
    require(
      !commonSettings.verilogPreprocessorDefines.exists(d => unsupportedWaves(d.name)),
      "Arcilator only supports VCD waveforms"
    )
    require(commonSettings.simulationSettings.plusArgs.isEmpty, "Arcilator does not support Verilog plusargs")
    val frontend = commonSettings.verilogPreprocessorDefines.map { define =>
      "-D" + define.name + define.value.map("=" + _).getOrElse("")
    } ++ commonSettings.includeDirs.toSeq.flatten.map("-I" + _) ++
      commonSettings.libraryPaths.toSeq.flatten.flatMap(p => Seq("-y", p)) ++
      commonSettings.libraryExtensions.toSeq.flatten.flatMap(e => Seq("-Y", e)) ++
      commonSettings.defaultTimescale.toSeq.map(t => s"--timescale=$t") ++
      backendSpecificSettings.frontendOptions
    val enableVcd = backendSpecificSettings.enableVcd || commonSettings.verilogPreprocessorDefines.exists(
      _.name == svsim.Backend.HarnessCompilationFlags.enableVcdTracingSupport
    )
    val optimization = commonSettings.optimizationStyle match {
      case CommonCompilationSettings.OptimizationStyle.OptimizeForCompilationSpeed => "-O0"
      case _                                                                       => "-O3"
    }
    svsim.Backend.Parameters(
      compilerPath = Paths.get(sys.props("java.home"), "bin", "java").toString,
      compilerInvocation = svsim.Backend.Parameters.Invocation(
        Seq(
          "-cp",
          Backend.compilerClasspath,
          "svsim.arcilator.Compiler",
          "--tools",
          tools.toString,
          "--output",
          outputBinaryName,
          "--optimization=" + optimization
        ) ++ Option.when(enableVcd)("--vcd") ++
          frontend.map("--frontend-option=" + _) ++
          backendSpecificSettings.arcilatorOptions.map("--arc-option=" + _) ++
          commonSettings.linkLibraryPaths.map("--link-library=" + _),
        Seq.empty
      ),
      simulationInvocation = svsim.Backend.Parameters.Invocation(Seq.empty, Seq.empty)
    )
  }

  override def escapeDefine(string: String): String = string
  override val assertionFailed = "^.*(Assertion failed|assertion failed|Fatal:|fatal error).*$".r
}
