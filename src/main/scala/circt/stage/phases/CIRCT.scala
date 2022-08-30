// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.Implicits.BooleanImplicits
import circt.stage.{CIRCTOptions, CIRCTTarget, EmittedMLIR, PreserveAggregate}

import firrtl.{AnnotationSeq, EmittedVerilogCircuit, EmittedVerilogCircuitAnnotation}
import firrtl.options.{
  CustomFileEmission,
  Dependency,
  OptionsException,
  OutputAnnotationFileAnnotation,
  Phase,
  StageError,
  StageOptions,
  StageUtils
}
import firrtl.options.phases.WriteOutputAnnotations
import firrtl.options.Viewer.view
import firrtl.stage.{FirrtlOptions, RunFirrtlTransformAnnotation}
import _root_.logger.LogLevel

import scala.sys.process._

import java.io.File

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
}

/** A phase that calls and runs CIRCT, specifically `firtool`, while preserving an [[AnnotationSeq]] API.
  *
  * This is analogous to [[firrtl.stage.phases.Compiler]].
  */
class CIRCT extends Phase {

  import Helpers._

  override def prerequisites = Seq(
    Dependency[firrtl.stage.phases.AddDefaults],
    Dependency[firrtl.stage.phases.AddImplicitEmitter],
    Dependency[firrtl.stage.phases.AddImplicitOutputFile]
  )
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val circtOptions = view[CIRCTOptions](annotations)
    val firrtlOptions = view[FirrtlOptions](annotations)
    val stageOptions = view[StageOptions](annotations)

    var blackbox, inferReadWrite = false
    var dedup, imcp = true
    var logLevel = _root_.logger.LogLevel.None
    var split = false

    val annotationsx: AnnotationSeq = annotations.flatMap {
      case a: CustomFileEmission => {
        val filename = a.filename(annotations)
        a.replacements(filename)
      }
      case _: firrtl.EmitCircuitAnnotation => Nil
      case _: firrtl.EmitAllModulesAnnotation => {
        split = true
        Nil
      }
      case a @ RunFirrtlTransformAnnotation(transform) =>
        transform match {
          /* Inlining/Flattening happen by default, so these can be dropped. */
          case _: firrtl.passes.InlineInstances | _: firrtl.transforms.Flatten => Nil
          /* ReplSeqMem is converted to a firtool option */
          case _: firrtl.passes.memlib.ReplSeqMem =>
            blackbox = true
            Nil
          /* Any emitters should not be passed to firtool. */
          case _: firrtl.Emitter => Nil
          /* Default case: leave the annotation around and let firtool warn about it. */
          case _ => Seq(a)
        }
      case firrtl.passes.memlib.InferReadWriteAnnotation =>
        inferReadWrite = true
        Nil
      case _: firrtl.transforms.NoDedupAnnotation =>
        dedup = false
        Nil
      case firrtl.transforms.NoConstantPropagationAnnotation =>
        imcp = false
        Nil
      case anno: _root_.logger.LogLevelAnnotation =>
        logLevel = anno.globalLogLevel
        Nil
      /* The following can be dropped. */
      case _: firrtl.transforms.CombinationalPath   => Nil
      case _: _root_.logger.ClassLogLevelAnnotation => Nil
      /* Default case: leave the annotation around and let firtool warn about it. */
      case a => Seq(a)
    }

    /* Filter the annotations to only those things which CIRCT should see. */
    (new WriteOutputAnnotations).transform(annotationsx)

    val input: String = firrtlOptions.firrtlCircuit match {
      case None          => throw new OptionsException("No input file specified!")
      case Some(circuit) => circuit.serialize
    }

    val outputAnnotationFileName: Option[String] =
      stageOptions.annotationFileOut.map(stageOptions.getBuildFileName(_, Some(".anno.json")))

    val binary = "firtool"

    val cmd =
      Seq(binary, "-format=fir", "-warn-on-unprocessed-annotations", "-verify-each=false") ++
        circtOptions.firtoolOptions ++
        logLevel.toCIRCTOptions ++
        /* The following options are on by default, so we disable them if they are false. */
        (circtOptions.preserveAggregate match {
          case Some(PreserveAggregate.OneDimVec) => Seq("-preserve-aggregate=1d-vec")
          case Some(PreserveAggregate.Vec)       => Seq("-preserve-aggregate=vec")
          case Some(PreserveAggregate.All)       => Seq("-preserve-aggregate=all")
          case None                              => None
        }) ++
        circtOptions.preserveAggregate.map(_ => "-preserve-public-types=0") ++
        (!inferReadWrite).option("-infer-rw=0") ++
        (!imcp).option("-imcp=0") ++
        /* The following options are off by default, so we enable them if they are true. */
        (dedup).option("-dedup=1") ++
        (blackbox).option("-blackbox-memory") ++
        /* Communicate the annotation file through a file. */
        (outputAnnotationFileName.map(a => Seq("-annotation-file", a))).getOrElse(Seq.empty) ++
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

    try {
      logger.info(s"""Running CIRCT: '${cmd.mkString(" ")} < $$input'""")
      println(s"""Running CIRCT: '${cmd.mkString(" ")} < $$input'""")
      val stdoutStream, stderrStream = new java.io.ByteArrayOutputStream
      val stdoutWriter = new java.io.PrintWriter(stdoutStream)
      val stderrWriter = new java.io.PrintWriter(stderrStream)
      val exitValue = (cmd #< new java.io.ByteArrayInputStream(input.getBytes))
        .!(ProcessLogger(stdoutWriter.println, stderrWriter.println))
      stdoutWriter.close()
      stderrWriter.close()
      val result = stdoutStream.toString
      logger.info(result)
      val errors = stderrStream.toString
      if (exitValue != 0) {
        StageUtils.dramaticError(s"${binary} failed.\nExitCode:\n${exitValue}\nSTDOUT:\n${result}\nSTDERR:\n${errors}")
        throw new StageError()
      }
      if (split) {
        logger.info(result)
        Nil
      } else { // if split it has already been written out to the file system stdout not necessary.
        val outputFileName: String = stageOptions.getBuildFileName(firrtlOptions.outputFileName.get)
        circtOptions.target match {
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
        }
      }
    } catch {
      case a: java.io.IOException =>
        StageUtils.dramaticError(s"Binary '$binary' was not found on the $$PATH. (Do you have CIRCT installed?)")
        throw new StageError(cause = a)
    }

  }

}
