// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import _root_.logger.{LogLevel, Logger}
import chisel3.BuildInfo.{firtoolVersion, version => chiselVersion}
import chisel3.InternalErrorException
import chisel3.experimental.hierarchy.core.ImportDefinitionAnnotation
import chisel3.stage.{ChiselCircuitAnnotation, ChiselOptions, ChiselOptionsView, DesignAnnotation, SourceRootAnnotation}
import circt.stage.{CIRCTOptions, CIRCTOptionsView, CIRCTTarget, EmittedMLIR, PreserveAggregate}
import firrtl.annotations.JsonProtocol
import firrtl.ir.CircuitWithAnnos
import firrtl.options.Viewer.view
import firrtl.options.{
  CustomFileEmission,
  Dependency,
  OptionsException,
  Phase,
  StageOptions,
  StageOptionsView,
  Unserializable
}
import firrtl.options.StageUtils.dramaticMessage
import firrtl.stage.{FirrtlOptions, FirrtlOptionsView}
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq, EmittedVerilogCircuit, EmittedVerilogCircuitAnnotation}

import java.io.{BufferedReader, ByteArrayOutputStream, File, InputStreamReader, InputStream, PrintWriter}
import java.nio.file.{Files, Path, Paths}
import scala.collection.mutable
import scala.util.control.NoStackTrace
import firrtl.EmittedBtor2CircuitAnnotation
import firrtl.EmittedBtor2Circuit
import org.json4s.convertToJsonInput

private object Helpers {
  implicit class LogLevelHelpers(logLevel: LogLevel.Value) {
    def toCIRCTOptions: Seq[String] = logLevel match {
      case LogLevel.Error => Seq.empty
      case LogLevel.Warn  => Seq.empty
      case LogLevel.Info  => Seq("-verbose-pass-executions")
      case LogLevel.Debug => Seq("-verbose-pass-executions")
      case LogLevel.Trace => Seq("-verbose-pass-executions", "-mlir-print-ir-after-all")
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

  class LoggerShim(logger: Logger) extends firtoolresolver.Logger {
    def error(msg: String): Unit = logger.error(msg)
    def warn(msg:  String): Unit = logger.warn(msg)
    def info(msg:  String): Unit = logger.info(msg)
    def debug(msg: String): Unit = logger.debug(msg)
    def trace(msg: String): Unit = logger.trace(msg)
  }

  /** Create a thread that reads all lines from an InputStream and writes them to a PrintWriter.
    *
    * @param stream the input stream to read from
    * @param writer the writer to write lines to
    * @return a Thread that can be started to begin piping
    */
  def pipeStream(stream: InputStream, writer: PrintWriter): Thread = {
    new Thread(() => {
      val reader = new BufferedReader(new InputStreamReader(stream))
      try {
        var line: String = null
        while ({ line = reader.readLine(); line != null }) {
          writer.println(line)
        }
      } finally {
        reader.close()
      }
    })
  }
}

private[this] object Exceptions {

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
        dramaticMessage(
          header = Some(s"${binary} returned a non-zero exit code. $versionAdvice"),
          body = s"ExitCode:\n${exitCode}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}"
        )
      )
      with NoStackTrace

  /** Indicates that the firtool binary was not found. This likely indicates something is wrong with
    *  their firtool installation.
    *
    * @param msg the error message
    */
  class FirtoolNotFound(msg: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some(s"Error resolving firtool"),
          body = s"""|Chisel requires firtool, the MLIR-based FIRRTL Compiler (MFC), to generate Verilog.
                     |Something is wrong with your firtool installation, please see the following logging
                     |information.
                     |$msg""".stripMargin
        )
      )
      with NoStackTrace

}

/** A phase that calls and runs CIRCT, specifically `firtool`, while preserving an [[firrtl.AnnotationSeq AnnotationSeq]] API. */
// TODO this uses the converted FIRRTL circuit yet doesn't depend on anything--probably needs fixing and maybe we can stop using the FIRRTL CIRCT
class CIRCT extends Phase {

  import Helpers._

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

    val stageOptions = view[StageOptions](annotations)
    val firrtlOptions = view[FirrtlOptions](annotations)
    val chiselOptions = view[ChiselOptions](annotations)

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

    val (serialization: Iterable[String], circuitName: String) = chiselOptions.elaboratedCircuit match {
      case None => throw new OptionsException("No input file specified!")
      // TODO can we avoid converting, how else would we include filteredAnnos?
      case Some(circuit) =>
        (circuit.lazilySerialize(filteredAnnotations), circuit.name)
    }

    // FIRRTL is serialized either in memory or to a file
    val input: Either[Iterable[String], Path] =
      if (circtOptions.dumpFir) {
        val td = Paths.get(stageOptions.targetDir).toAbsolutePath
        Files.createDirectories(td)
        val filename = firrtlOptions.outputFileName.getOrElse(circuitName)
        val firPath = td.resolve(s"$filename.fir")
        val writer = new PrintWriter(firPath.toFile)
        try {
          serialization.foreach(writer.println)
        } finally {
          writer.close()
        }
        Right(firPath)
      } else {
        Left(serialization)
      }

    val chiselAnnotationFilename: Option[String] =
      stageOptions.annotationFileOut.map(stageOptions.getBuildFileName(_, Some(".anno.json")))

    val circtAnnotationFilename = "circt.anno.json"

    val binary = circtOptions.firtoolBinaryPath.getOrElse {
      // .get is safe, firtoolVersion is an Option for backwards compatibility
      val version = firtoolVersion.get
      val resolved = firtoolresolver.Resolve(new LoggerShim(logger), version)
      resolved match {
        case Left(msg) =>
          throw new Exceptions.FirtoolNotFound(msg)
        case Right(bin) =>
          bin.path.toString
      }
    }

    val cmd = // Only 1 of input or firFile will be Some
      Seq(binary, input.fold(_ => "-format=fir", _.toString)) ++
        Seq("-warn-on-unprocessed-annotations") ++
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
        circtOptions.preserveAggregate.map(_ => "-scalarize-public-modules=0") ++
        includeDirs.flatMap(d => Seq("--include-dir", d.toString)) ++
        /* Convert the target to a firtool-compatible option. */
        ((circtOptions.target, split) match {
          case (Some(CIRCTTarget.FIRRTL), false)        => Seq("-ir-fir")
          case (Some(CIRCTTarget.HW), false)            => Seq("-ir-hw")
          case (Some(CIRCTTarget.Verilog), true)        => Seq("--split-verilog", s"-o=${stageOptions.targetDir}")
          case (Some(CIRCTTarget.Verilog), false)       => None
          case (Some(CIRCTTarget.SystemVerilog), true)  => Seq("--split-verilog", s"-o=${stageOptions.targetDir}")
          case (Some(CIRCTTarget.SystemVerilog), false) => None
          case (Some(CIRCTTarget.Btor2), false)         => Seq("--btor2")
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

    logger.info(s"""Running CIRCT: '${cmd.mkString(" ")}""" + input.fold(_ => " < $$" + "input'", _ => "'"))
    val stdoutStream = new ByteArrayOutputStream
    val stderrStream = new ByteArrayOutputStream
    val stdoutWriter = new PrintWriter(stdoutStream)
    val stderrWriter = new PrintWriter(stderrStream)
    val exitValue =
      try {
        val pb = new ProcessBuilder(cmd: _*)
        val process = pb.start()

        // Handle stdin - write input to process if needed
        input match {
          case Left(it) =>
            val stdinWriter = new PrintWriter(process.getOutputStream)
            try {
              it.foreach(stdinWriter.println)
            } finally {
              stdinWriter.close()
            }
          case Right(_) =>
            // No stdin needed when reading from file
            process.getOutputStream.close()
        }

        // Read stdout and stderr in separate threads to prevent deadlock
        val stdoutThread = pipeStream(process.getInputStream, stdoutWriter)
        val stderrThread = pipeStream(process.getErrorStream, stderrWriter)

        stdoutThread.start()
        stderrThread.start()

        val code = process.waitFor()
        stdoutThread.join()
        stderrThread.join()
        code
      } catch {
        case e: java.io.IOException if e.getMessage.startsWith("Cannot run program") =>
          throw new Exceptions.FirtoolNotFound(e.getMessage)
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
        case Some(CIRCTTarget.Btor2) =>
          Seq(EmittedBtor2CircuitAnnotation(EmittedBtor2Circuit(outputFileName, result, ".btor2")))
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
