// See LICENSE for license details.

package firrtl

import scala.collection._
import scala.util.{Failure, Try}
import java.io.{File, FileNotFoundException}
import net.jcazevedo.moultingyaml._
import annotations._
import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.transforms._
import firrtl.Utils.throwInternalError
import firrtl.stage.{FirrtlExecutionResultView, FirrtlStage}
import firrtl.stage.phases.DriverCompatibility
import firrtl.options.{Dependency, Phase, PhaseManager, StageUtils, Viewer}
import firrtl.options.phases.DeletedWrapper


/**
  * The driver provides methods to access the firrtl compiler.
  * Invoke the compiler with either a FirrtlExecutionOption
  *
  * @example
  *          {{{
  *          val optionsManager = new ExecutionOptionsManager("firrtl")
  *          optionsManager.register(
  *              FirrtlExecutionOptionsKey ->
  *              new FirrtlExecutionOptions(topName = "Dummy", compilerName = "verilog"))
  *          firrtl.Driver.execute(optionsManager)
  *          }}}
  *  or a series of command line arguments
  * @example
  *          {{{
  *          firrtl.Driver.execute(Array("--top-name Dummy --compiler verilog".split(" +"))
  *          }}}
  * each approach has its own endearing aspects
  * @see firrtlTests/DriverSpec.scala in the test directory for a lot more examples
  * @see [[CompilerUtils.mergeTransforms]] to see how customTransformations are inserted
  */
@deprecated("Use firrtl.stage.FirrtlStage", "1.2")
object Driver {
  /** Print a warning message
    *
    * @param message error message
    */
  @deprecated("Use firrtl.options.StageUtils.dramaticWarning", "1.2")
  def dramaticWarning(message: String): Unit = StageUtils.dramaticWarning(message)

  /**
    * print the message in red
    *
    * @param message error message
    */
  @deprecated("Use firrtl.options.StageUtils.dramaticWarning", "1.2")
  def dramaticError(message: String): Unit = StageUtils.dramaticError(message)

  /** Load annotation file based on options
    * @param optionsManager use optionsManager config to load annotation file if it exists
    *                       update the firrtlOptions with new annotations if it does
    */
  @deprecated("Use side-effect free getAnnotation instead", "1.1")
  def loadAnnotations(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): Unit = {
    val msg = "Driver.loadAnnotations is deprecated, use Driver.getAnnotations instead"
    Driver.dramaticWarning(msg)
    optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(
      annotations = Driver.getAnnotations(optionsManager).toList
    )
  }

  /** Get annotations from specified files and options
    *
    * @param optionsManager use optionsManager config to load annotation files
    * @return Annotations read from files
    */
  //scalastyle:off cyclomatic.complexity method.length
  def getAnnotations(
      optionsManager: ExecutionOptionsManager with HasFirrtlOptions
  ): Seq[Annotation] = {
    val firrtlConfig = optionsManager.firrtlOptions

    //noinspection ScalaDeprecation
    val oldAnnoFileName = firrtlConfig.getAnnotationFileName(optionsManager)
    val oldAnnoFile = new File(oldAnnoFileName).getCanonicalFile

    val (annoFiles, usingImplicitAnnoFile) = {
      val afs = firrtlConfig.annotationFileNames.map { x =>
        new File(x).getCanonicalFile
      }
      // Implicit anno file could be included explicitly, only include it and
      // warn if it's not also explicit
      val use = oldAnnoFile.exists && !afs.contains(oldAnnoFile)
      if (use) (oldAnnoFile +: afs, true) else (afs, false)
    }

    // Warnings to get people to change to drop old API
    if (firrtlConfig.annotationFileNameOverride.nonEmpty) {
      val msg = "annotationFileNameOverride is deprecated! " +
                "Use annotationFileNames"
      dramaticWarning(msg)
    } else if (usingImplicitAnnoFile) {
      val msg = "Implicit .anno file from top-name is deprecated!\n" +
             (" "*9) + "Use explicit -faf option or annotationFileNames"
      Driver.dramaticWarning(msg)
    }

    val loadedAnnos = annoFiles.flatMap { file =>
      if (!file.exists) {
        throw new AnnotationFileNotFoundException(file)
      }
      // Try new protocol first
      JsonProtocol.deserializeTry(file).recoverWith { case jsonException =>
        // Try old protocol if new one fails
        Try {
          val yaml = FileUtils.getText(file).parseYaml
          val result = yaml.convertTo[List[LegacyAnnotation]]
          val msg = s"$file is a YAML file!\n" +
                    (" "*9) + "YAML Annotation files are deprecated! Use JSON"
          Driver.dramaticWarning(msg)
          result
        }.orElse { // Propagate original JsonProtocol exception if YAML also fails
          Failure(jsonException)
        }
      }.get
    }

    val targetDirAnno = List(BlackBoxTargetDirAnno(optionsManager.targetDirName))

    // Output Annotations
    val outputAnnos = firrtlConfig.getEmitterAnnos(optionsManager)

    val globalAnnos = Seq(TargetDirAnnotation(optionsManager.targetDirName)) ++
      (if (firrtlConfig.dontCheckCombLoops) Seq(DontCheckCombLoopsAnnotation) else Seq()) ++
      (if (firrtlConfig.noDCE) Seq(NoDCEAnnotation) else Seq())

    val annos = targetDirAnno ++ outputAnnos ++ globalAnnos ++
                firrtlConfig.annotations ++ loadedAnnos
    LegacyAnnotation.convertLegacyAnnos(annos)
  }

  private sealed trait FileExtension
  private case object FirrtlFile extends FileExtension
  private case object ProtoBufFile extends FileExtension

  private def getFileExtension(filename: String): FileExtension =
    filename.drop(filename.lastIndexOf('.')) match {
      case ".pb" => ProtoBufFile
      case _ => FirrtlFile // Default to FIRRTL File
    }

  // Useful for handling erros in the options
  case class OptionsException(message: String) extends Exception(message)

  /** Get the Circuit from the compile options
    *
    * Handles the myriad of ways it can be specified
    */
  def getCircuit(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): Try[ir.Circuit] = {
    val firrtlConfig = optionsManager.firrtlOptions
    Try {
      // Check that only one "override" is used
      val circuitSources = Map(
        "firrtlSource" -> firrtlConfig.firrtlSource.isDefined,
        "firrtlCircuit" -> firrtlConfig.firrtlCircuit.isDefined,
        "inputFileNameOverride" -> firrtlConfig.inputFileNameOverride.nonEmpty)
      if (circuitSources.values.count(x => x) > 1) {
        val msg = circuitSources.collect { case (s, true) => s }.mkString(" and ") +
          " are set, only 1 can be set at a time!"
        throw new OptionsException(msg)
      }
      firrtlConfig.firrtlCircuit.getOrElse {
        firrtlConfig.firrtlSource.map(x => Parser.parseString(x, firrtlConfig.infoMode)).getOrElse {
          if (optionsManager.topName.isEmpty && firrtlConfig.inputFileNameOverride.isEmpty) {
            val message = "either top-name or input-file-override must be set"
            throw new OptionsException(message)
          }
          if (
            optionsManager.topName.isEmpty &&
              firrtlConfig.inputFileNameOverride.nonEmpty &&
              firrtlConfig.outputFileNameOverride.isEmpty) {
            val message = "inputFileName set but neither top-name or output-file-override is set"
            throw new OptionsException(message)
          }
          val inputFileName = firrtlConfig.getInputFileName(optionsManager)
          try {
            // TODO What does InfoMode mean to ProtoBuf?
            getFileExtension(inputFileName) match {
              case ProtoBufFile => proto.FromProto.fromFile(inputFileName)
              case FirrtlFile => Parser.parseFile(inputFileName, firrtlConfig.infoMode)
            }
          }
          catch {
            case _: FileNotFoundException =>
              val message = s"Input file $inputFileName not found"
              throw new OptionsException(message)
          }
        }
      }
    }
  }

  /**
    * Run the firrtl compiler using the provided option
    *
    * @param optionsManager the desired flags to the compiler
    * @return a FirrtlExecutionResult indicating success or failure, provide access to emitted data on success
    *         for downstream tools as desired
    */
  def execute(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): FirrtlExecutionResult = {
    StageUtils.dramaticWarning("firrtl.Driver is deprecated since 1.2!\nPlease switch to firrtl.stage.FirrtlMain")

    val annos = optionsManager.firrtlOptions.toAnnotations ++ optionsManager.commonOptions.toAnnotations

    val phases: Seq[Phase] = {
      import DriverCompatibility._
      new PhaseManager(
        Seq( Dependency[AddImplicitFirrtlFile],
             Dependency[AddImplicitAnnotationFile],
             Dependency[AddImplicitOutputFile],
             Dependency[AddImplicitEmitter],
             Dependency[FirrtlStage] ))
        .transformOrder
        .map(DeletedWrapper(_))
    }

    val annosx = try {
      phases.foldLeft(annos)( (a, p) => p.transform(a) )
    } catch {
      case e: firrtl.options.OptionsException => return FirrtlExecutionFailure(e.message)
    }

    Viewer[FirrtlExecutionResult].view(annosx)
  }

  /**
    * this is a wrapper for execute that builds the options from a standard command line args,
    * for example, like strings passed to main()
    *
    * @param args  an Array of string s containing legal arguments
    * @return
    */
  def execute(args: Array[String]): FirrtlExecutionResult = {
    val optionsManager = new ExecutionOptionsManager("firrtl") with HasFirrtlOptions

    if(optionsManager.parse(args)) {
      execute(optionsManager) match {
        case success: FirrtlExecutionSuccess =>
          success
        case failure: FirrtlExecutionFailure =>
          optionsManager.showUsageAsError()
          failure
        case result =>
          throwInternalError(s"Error: Unknown Firrtl Execution result $result")
      }
    }
    else {
      FirrtlExecutionFailure("Could not parser command line options")
    }
  }

  def main(args: Array[String]): Unit = {
    execute(args)
  }
}
