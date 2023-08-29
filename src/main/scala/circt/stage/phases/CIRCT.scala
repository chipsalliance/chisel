// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import _root_.logger.LogLevel
import chisel3.BuildInfo.{firtoolVersion, version => chiselVersion}
import chisel3.InternalErrorException
import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
import chisel3.stage.{ChiselCircuitAnnotation, DesignAnnotation, SourceRootAnnotation}
import circt.stage.{CIRCTOptions, CIRCTTarget, EmittedMLIR, PreserveAggregate}
import firrtl.annotations.JsonProtocol
import firrtl.ir.CircuitWithAnnos
import firrtl.options.Viewer.view
import firrtl.options.{CustomFileEmission, Dependency, OptionsException, Phase, StageOptions, Unserializable}
import firrtl.stage.FirrtlOptions
import firrtl.{AnnotationSeq, EmittedVerilogCircuit, EmittedVerilogCircuitAnnotation}

import java.io.File
import scala.collection.mutable
import scala.util.control.NoStackTrace

private object Helpers {
  implicit class LogLevelHelpers(logLevel: LogLevel.Value) {
    def toCIRCTOptions: Seq[String] = logLevel match {
      case LogLevel.Error => Seq.empty
      case LogLevel.Warn  => Seq.empty
      case LogLevel.Info  => Seq("-verbose-pass-executions")
      case LogLevel.Debug => Seq("-verbose-pass-executions")
      case LogLevel.Trace => Seq("-verbose-pass-executions", "-print-ir-after-all")
      case LogLevel.None  => Seq.empty
    }
  }

  /** Extract the JSON-encoded Annotation region from a single file output.
    *
    * @todo This is very janky and should be changed to something more stable.
    */
  def extractAnnotationFile(string: String, filename: String): AnnotationSeq = {
    var inAnno = false
    val filtered: String = string.linesIterator.filter {
      case line if line.startsWith("// ----- 8< ----- FILE") && line.contains(filename) =>
        inAnno = true
        false
      case line if line.startsWith("// ----- 8< ----- FILE") =>
        inAnno = false
        false
      case line if inAnno =>
        true

      case _ => false
    }.toArray
      .mkString("\n")
    filtered.forall(_.isWhitespace) match {
      case false => JsonProtocol.deserialize(filtered, false)
      case true  => Seq.empty
    }
  }
}

private[this] object Exceptions {

  /** Wrap a message in colorful error text in the sytle of StageUtils.dramaticError.  That method prints to stdout and by
    * extracting this to just a method that does the string wrapping, this enables the message to be sent to another
    * file, e.g., stderr.
    * @todo Remove/unify this with StageUtils.dramaticError once SFC code is migrated into Chisel3.
    */
  def dramaticError(header: String, body: String): String = {
    s"""|$header
        |${"-" * 78}
        |$body
        |${"-" * 78}""".stripMargin
  }

  def versionAdvice: String =
    s"Note that this version of Chisel ($chiselVersion) was published against firtool version " +
      firtoolVersion.getOrElse("<unknown>") + "."

  /** Indicates that the firtool binary failed with a non-zero exit code.  This generally indicates a compiler error
    * either originating from a user error or from a crash.
    *
    * @param binary the path to the firtool binary
    * @param exitCode the numeric exit status code returned by the firtool binary
    * @param stdout the contents returned to standard out by the firtool binary
    * @param stderr the contents returned to standard error by the firtool binary
    */
  class FirtoolNonZeroExitCode(binary: String, exitCode: Int, stdout: String, stderr: String)
      extends RuntimeException(
        dramaticError(
          header = s"${binary} returned a non-zero exit code. $versionAdvice",
          body = s"ExitCode:\n${exitCode}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}"
        )
      )
      with NoStackTrace

  /** Indicates that the firtool binary was not found.  This likely indicates that the user didn't install
    * CIRCT/firtool.
    *
    * @param binary the path to the firtool binary
    */
  class FirtoolNotFound(binary: String)
      extends RuntimeException(
        dramaticError(
          header = s"$binary not found",
          body = """|Chisel requires that firtool, the MLIR-based FIRRTL Compiler (MFC), is installed
                    |and available on your $PATH.  (Did you forget to install it?)  You can download
                    |a binary release of firtool from the CIRCT releases webpage:
                    |  https://github.com/llvm/circt/releases""".stripMargin
        )
      )
      with NoStackTrace

}

/** A phase that calls and runs CIRCT, specifically `firtool`, while preserving an [[firrtl.AnnotationSeq AnnotationSeq]] API. */
class CIRCT extends Phase {

  import Helpers._

  import scala.sys.process._

  override def prerequisites = Seq(
    Dependency[circt.stage.phases.AddImplicitOutputFile]
  )
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val circtOptions = view[CIRCTOptions](annotations)

    // Early exit (do not run firtool) if the target is "CHIRRTL", i.e., specification FIRRTL.
    circtOptions.target match {
      case Some(CIRCTTarget.CHIRRTL) => return annotations
      case _                         =>
    }

    val firrtlOptions = view[FirrtlOptions](annotations)
    val stageOptions = view[StageOptions](annotations)

    var logLevel = _root_.logger.LogLevel.None
    var split = circtOptions.splitVerilog
    val includeDirs = mutable.ArrayBuffer.empty[String]

    // Partition the annotations into those that will be passed to CIRCT and
    // those that are not.  The annotations that are in the passhtrough set will
    // be returned without modification.
    val (passthroughAnnotations, circtAnnotations) = annotations.partition {
      case _: ImportDefinitionAnnotation[_] | _: DesignAnnotation[_] | _: ChiselCircuitAnnotation =>
        true
      case _ => false
    }

    val annotationsx: AnnotationSeq = circtAnnotations.flatMap {
      case a: CustomFileEmission => {
        val filename = a.filename(annotations)
        a.replacements(filename)
      }
      case _:    ImportDefinitionAnnotation[_] => Nil
      case anno: _root_.logger.LogLevelAnnotation =>
        logLevel = anno.globalLogLevel
        Nil
      case SourceRootAnnotation(dir) =>
        includeDirs += dir.toString
        Nil
      /* The following can be dropped. */
      case _: _root_.logger.ClassLogLevelAnnotation => Nil
      /* Default case: leave the annotation around and let firtool warn about it. */
      case a => Seq(a)
    }

    /* Filter the annotations to only those things which CIRCT should see. */
    val filteredAnnotations = annotationsx.flatMap {
      case _: ChiselCircuitAnnotation => None
      case _: Unserializable          => None
      case _: CustomFileEmission      => None
      case a => Some(a)
    }

    val (serialization: Iterable[String], circuitName: String) = firrtlOptions.firrtlCircuit match {
      case None => throw new OptionsException("No input file specified!")
      // TODO can we avoid converting, how else would we include filteredAnnos?
      case Some(circuit) =>
        val cwa = CircuitWithAnnos(circuit = circuit, annotations = filteredAnnotations)
        (firrtl.ir.Serializer.lazily(cwa), circuit.main)
    }

    // FIRRTL is serialized either in memory or to a file
    val input: Either[Iterable[String], os.Path] =
      if (circtOptions.dumpFir) {
        val td = os.Path(stageOptions.targetDir, os.pwd)
        val filename = firrtlOptions.outputFileName.getOrElse(circuitName)
        val firPath = td / s"$filename.fir"
        os.write.over(firPath, serialization, createFolders = true)
        Right(firPath)
      } else {
        Left(serialization)
      }

    val chiselAnnotationFilename: Option[String] =
      stageOptions.annotationFileOut.map(stageOptions.getBuildFileName(_, Some(".anno.json")))

    val circtAnnotationFilename = "circt.anno.json"

    val binary = circtOptions.firtoolBinaryPath.getOrElse("firtool")

    val cmd = // Only 1 of input or firFile will be Some
      Seq(binary, input.fold(_ => "-format=fir", _.toString)) ++
        Seq("-warn-on-unprocessed-annotations", "-dedup") ++
        Seq("-output-annotation-file", circtAnnotationFilename) ++
        circtOptions.firtoolOptions ++
        logLevel.toCIRCTOptions ++
        /* The following options are on by default, so we disable them if they are false. */
        (circtOptions.preserveAggregate match {
          case Some(PreserveAggregate.OneDimVec) => Seq("-preserve-aggregate=1d-vec")
          case Some(PreserveAggregate.Vec)       => Seq("-preserve-aggregate=vec")
          case Some(PreserveAggregate.All)       => Seq("-preserve-aggregate=all")
          case None                              => None
        }) ++
        circtOptions.preserveAggregate.map(_ => "-scalarize-top-module=0") ++
        includeDirs.flatMap(d => Seq("--include-dir", d.toString)) ++
        /* Convert the target to a firtool-compatible option. */
        ((circtOptions.target, split) match {
          case (Some(CIRCTTarget.FIRRTL), false)        => Seq("-ir-fir")
          case (Some(CIRCTTarget.HW), false)            => Seq("-ir-hw")
          case (Some(CIRCTTarget.Verilog), true)        => Seq("--split-verilog", s"-o=${stageOptions.targetDir}")
          case (Some(CIRCTTarget.Verilog), false)       => None
          case (Some(CIRCTTarget.SystemVerilog), true)  => Seq("--split-verilog", s"-o=${stageOptions.targetDir}")
          case (Some(CIRCTTarget.SystemVerilog), false) => None
          case (None, _) =>
            throw new Exception(
              "No 'circtOptions.target' specified. This should be impossible if dependencies are satisfied!"
            )
          case (_, true) =>
            throw new Exception(
              s"The circtOptions.target specified (${circtOptions.target}) does not support running with an EmitAllModulesAnnotation as CIRCT only supports one-file-per-module for Verilog or SystemVerilog targets."
            )
          case _ =>
            throw new Exception(
              s"Invalid combination of circtOptions.target ${circtOptions.target} and split ${split}"
            )
        })

    logger.info(s"""Running CIRCT: '${cmd.mkString(" ")}""" + input.fold(_ => " < $$input'", _ => "'"))
    val stdoutStream, stderrStream = new java.io.ByteArrayOutputStream
    val stdoutWriter = new java.io.PrintWriter(stdoutStream)
    val stderrWriter = new java.io.PrintWriter(stderrStream)
    val stdin: os.ProcessInput = input match {
      case Left(it) => (it: os.Source) // Static cast to apply implicit conversion
      case Right(_) => os.Pipe
    }
    val stdout = os.ProcessOutput.Readlines(stdoutWriter.println)
    val stderr = os.ProcessOutput.Readlines(stderrWriter.println)
    val exitValue =
      try {
        os.proc(cmd).call(check = false, stdin = stdin, stdout = stdout, stderr = stderr).exitCode
      } catch {
        case a: java.io.IOException if a.getMessage().startsWith("Cannot run program") =>
          throw new Exceptions.FirtoolNotFound(binary)
      }
    stdoutWriter.close()
    stderrWriter.close()
    val result = stdoutStream.toString
    logger.info(result)
    val errors = stderrStream.toString
    if (exitValue != 0)
      throw new Exceptions.FirtoolNonZeroExitCode(binary, exitValue, result, errors)
    val finalAnnotations = if (split) {
      logger.info(result)
      val file = new File(stageOptions.getBuildFileName(circtAnnotationFilename, Some(".anno.json")))
      file match {
        case file if !file.canRead() => Seq.empty
        case file => {
          val foo = JsonProtocol.deserialize(file, false)
          foo
        }
      }
    } else { // if split it has already been written out to the file system stdout not necessary.
      val outputFileName: String = stageOptions.getBuildFileName(firrtlOptions.outputFileName.get)
      val outputAnnotations = extractAnnotationFile(result, circtAnnotationFilename)
      outputAnnotations ++ (circtOptions.target match {
        case Some(CIRCTTarget.FIRRTL) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".fir.mlir")))
        case Some(CIRCTTarget.HW) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".hw.mlir")))
        case Some(CIRCTTarget.Verilog) =>
          Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(outputFileName, result, ".v")))
        case Some(CIRCTTarget.SystemVerilog) =>
          Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(outputFileName, result, ".sv")))
        case None =>
          throw new Exception(
            "No 'circtOptions.target' specified. This should be impossible if dependencies are satisfied!"
          )
        case unknown =>
          throw new InternalErrorException(s"Match Error: Unknon CIRCTTarget: $unknown")
      })
    }

    // Return the passthrough annotations and the output annotations from CIRCT.
    passthroughAnnotations ++ finalAnnotations
  }

}
