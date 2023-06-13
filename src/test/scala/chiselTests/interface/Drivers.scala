// SPDX-License-Identifier: Apache-2.0
package chiselTests.interface

import java.io.File
import chisel3.RawModule
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{ChiselStage, FirtoolOption}
import firrtl.AnnotationSeq
import firrtl.options.{StageError, StageUtils}
import sys.process._

object Drivers {

  private val stage = new ChiselStage

  case class CompilationUnit(
    generator:   () => RawModule,
    args:        Array[String] = Array.empty,
    annotations: AnnotationSeq = Seq.empty)

  /** Compile one or more Chisel Modules into an output directory. The first
    * compile will be put into the output directory. All subsequent compiles
    * will be put in directories called "compile-n" with "n" starting at 0 and
    * incrementing.
    */
  def compile(
    dir:   java.io.File,
    main:  CompilationUnit,
    other: CompilationUnit*
  ) = {
    (main +: other).zipWithIndex.foreach {
      case (CompilationUnit(generator, args, annotations), i) =>
        stage.execute(
          Array(
            "--target",
            "systemverilog",
            "--target-dir",
            dir.getPath() + "/firrtl"
          ) ++ args,
          Seq(
            ChiselGeneratorAnnotation(generator),
            FirtoolOption("-split-verilog"),
            FirtoolOption("-o"),
            FirtoolOption(dir.getPath() + s"/compile-$i"),
            FirtoolOption("-disable-annotation-unknown"),
            FirtoolOption("-disable-all-randomization"),
            FirtoolOption("-strip-debug-info")
          ) ++ annotations
        )
    }
  }

  /** Perform a pseudo-link by running Verilator --lint-only on one or more
    * compiles done using the `compile` method above.
    */
  def link(dir: File, top: String) = {

    /** Generate includes from any subdirectories of "dir". */
    val includeDirs = dir
      .listFiles()
      .collect {
        case f if f.isDirectory && f.getName().startsWith("compile") => f
      }
    val includes = includeDirs
      .map(dir => s"-I${dir.getPath()}")

    /** Find any reference definition files, which need to be parsed by Verilator first */
    val refDefinitionFiles = includeDirs.flatMap {
      case f =>
        f.listFiles().collect {
          case f if f.getName().startsWith("ref_") => f.getPath()
        }
    }

    val cmd: Seq[String] = Seq(
      "verilator",
      "-lint-only"
    ) ++ refDefinitionFiles ++ Seq(new File(dir, top).getPath(), s"-I${dir.getPath()}") ++ includes

    val stdoutStream, stderrStream = new java.io.ByteArrayOutputStream
    val stdoutWriter = new java.io.PrintWriter(stdoutStream)
    val stderrWriter = new java.io.PrintWriter(stderrStream)
    val exitValue =
      (cmd).!(ProcessLogger(stdoutWriter.println, stderrWriter.println))

    stdoutWriter.close()
    stderrWriter.close()
    val result = stdoutStream.toString
    val errors = stderrStream.toString

    if (exitValue != 0) {
      StageUtils.dramaticError(
        s"${cmd} failed.\nExitCode:\n${exitValue}\nSTDOUT:\n${result}\nSTDERR:\n${errors}"
      )
      throw new StageError()
    }

  }

}
