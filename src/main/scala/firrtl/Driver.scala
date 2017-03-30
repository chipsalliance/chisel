// See LICENSE for license details.

package firrtl

import scala.collection._
import scala.io.Source
import scala.sys.process.{BasicIO,stringSeqToProcess}
import java.io.{File, FileNotFoundException}

import net.jcazevedo.moultingyaml._
import logger.Logger
import Parser.{IgnoreInfo, InfoMode}
import annotations._
import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.transforms.{BlackBoxSourceHelper, BlackBoxTargetDir}
import Utils.throwInternalError


/**
  * The driver provides methods to access the firrtl compiler.
  * Invoke the compiler with either a FirrtlExecutionOption
  *
  * @example
  *          {{{
  *          val optionsManager = ExecutionOptionsManager("firrtl")
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

object Driver {
  // Compiles circuit. First parses a circuit from an input file,
  //  executes all compiler passes, and writes result to an output
  //  file.
  @deprecated("Please use execute", "firrtl 1.0")
  def compile(
      input: String,
      output: String,
      compiler: Compiler,
      infoMode: InfoMode = IgnoreInfo,
      customTransforms: Seq[Transform] = Seq.empty,
      annotations: AnnotationMap = AnnotationMap(Seq.empty)
  ): String = {
    val parsedInput = Parser.parse(Source.fromFile(input).getLines(), infoMode)
    val outputBuffer = new java.io.CharArrayWriter
    compiler.compile(
      CircuitState(parsedInput, ChirrtlForm, Some(annotations)),
      outputBuffer,
      customTransforms)

    val outputFile = new java.io.PrintWriter(output)
    val outputString = outputBuffer.toString
    outputFile.write(outputString)
    outputFile.close()
    outputString
  }

  /**
    * print the message in red
    *
    * @param message error message
    */
  //scalastyle:off regex
  def dramaticError(message: String): Unit = {
    println(Console.RED + "-"*78)
    println(s"Error: $message")
    println("-"*78 + Console.RESET)
  }

  /**
    * Load annotation file based on options
    * @param optionsManager use optionsManager config to load annotation file if it exists
    *                       update the firrtlOptions with new annotations if it does
    */
  def loadAnnotations(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): Unit = {
    /*
     If firrtlAnnotations in the firrtlOptions are nonEmpty then these will be the annotations
     used by firrtl.
     To use the file annotations make sure that the annotations in the firrtlOptions are empty
     The annotation file if needed is found via
     s"$targetDirName/$topName.anno" or s"$annotationFileNameOverride.anno"
    */
    def firrtlConfig = optionsManager.firrtlOptions

    if (firrtlConfig.annotations.isEmpty || firrtlConfig.forceAppendAnnoFile) {
      val annotationFileName = firrtlConfig.getAnnotationFileName(optionsManager)
      val annotationFile = new File(annotationFileName)
      if (annotationFile.exists) {
        val annotationsYaml = io.Source.fromFile(annotationFile).getLines().mkString("\n").parseYaml
        val annotationArray = annotationsYaml.convertTo[Array[Annotation]]
        optionsManager.firrtlOptions = firrtlConfig.copy(annotations = firrtlConfig.annotations ++ annotationArray)
      }
    }

    if(firrtlConfig.annotations.nonEmpty) {
      val targetDirAnno = List(Annotation(
        CircuitName("All"),
        classOf[BlackBoxSourceHelper],
        BlackBoxTargetDir(optionsManager.targetDirName).serialize
      ))

      optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(
        annotations = firrtlConfig.annotations ++ targetDirAnno)
    }
  }

  /**
    * Run the firrtl compiler using the provided option
    *
    * @param optionsManager the desired flags to the compiler
    * @return a FirrtlExectionResult indicating success or failure, provide access to emitted data on success
    *         for downstream tools as desired
    */
  def execute(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): FirrtlExecutionResult = {
    def firrtlConfig = optionsManager.firrtlOptions

    Logger.setOptions(optionsManager)

    val firrtlSource = firrtlConfig.firrtlSource match {
      case Some(text) => text.split("\n").toIterator
      case None       =>
        if(optionsManager.topName.isEmpty && firrtlConfig.inputFileNameOverride.isEmpty) {
          val message = "either top-name or input-file-override must be set"
          dramaticError(message)
          return FirrtlExecutionFailure(message)
        }
        if(
          optionsManager.topName.isEmpty &&
            firrtlConfig.inputFileNameOverride.nonEmpty &&
            firrtlConfig.outputFileNameOverride.isEmpty) {
          val message = "inputFileName set but neither top-name or output-file-override is set"
          dramaticError(message)
          return FirrtlExecutionFailure(message)
        }
        val inputFileName = firrtlConfig.getInputFileName(optionsManager)
        try {
          io.Source.fromFile(inputFileName).getLines()
        }
        catch {
          case _: FileNotFoundException =>
            val message = s"Input file $inputFileName not found"
            dramaticError(message)
            return FirrtlExecutionFailure(message)
          }
        }

    loadAnnotations(optionsManager)

    val parsedInput = Parser.parse(firrtlSource, firrtlConfig.infoMode)

    // Does this need to be before calling compiler?
    optionsManager.makeTargetDir()

    // Output Annotations
    val outputAnnos = firrtlConfig.getEmitterAnnos(optionsManager)

    // Should these and outputAnnos be moved to loadAnnotations?
    val globalAnnos = Seq(TargetDirAnnotation(optionsManager.targetDirName))

    val finalState = firrtlConfig.compiler.compile(
      CircuitState(parsedInput,
                   ChirrtlForm,
                   Some(AnnotationMap(firrtlConfig.annotations ++ outputAnnos ++ globalAnnos))),
      firrtlConfig.customTransforms
    )

    // Do emission
    // Note: Single emission target assumption is baked in here
    // Note: FirrtlExecutionSuccess emitted is only used if we're emitting the whole Circuit
    val emittedRes = firrtlConfig.getOutputConfig(optionsManager) match {
      case SingleFile(filename) =>
        val emitted = finalState.getEmittedCircuit
        val outputFile = new java.io.PrintWriter(filename)
        outputFile.write(emitted.value)
        outputFile.close()
        emitted.value
      case OneFilePerModule(dirName) =>
        val emittedModules = finalState.emittedComponents collect { case x: EmittedModule => x }
        if (emittedModules.isEmpty) throwInternalError // There should be something
        emittedModules.foreach { case module =>
          val filename = optionsManager.getBuildFileName(firrtlConfig.outputSuffix, s"$dirName/${module.name}")
          val outputFile = new java.io.PrintWriter(filename)
          outputFile.write(module.value)
          outputFile.close()
        }
        "" // Should we return something different here?
    }

    FirrtlExecutionSuccess(firrtlConfig.compilerName, emittedRes)
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
          throw new Exception(s"Error: Unknown Firrtl Execution result $result")
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

object FileUtils {
  /**
    * recursive create directory and all parents
    *
    * @param directoryName a directory string with one or more levels
    * @return
    */
  def makeDirectory(directoryName: String): Boolean = {
    val dirFile = new java.io.File(directoryName)
    if(dirFile.exists()) {
      if(dirFile.isDirectory) {
        true
      }
      else {
        false
      }
    }
    else {
      dirFile.mkdirs()
    }
  }

  /**
    * recursively delete all directories in a relative path
    * DO NOT DELETE absolute paths
    *
    * @param directoryPathName a directory hierarchy to delete
    */
  def deleteDirectoryHierarchy(directoryPathName: String): Boolean = {
    deleteDirectoryHierarchy(new File(directoryPathName))
  }
  /**
    * recursively delete all directories in a relative path
    * DO NOT DELETE absolute paths
    *
    * @param file: a directory hierarchy to delete
    */
  def deleteDirectoryHierarchy(file: File, atTop: Boolean = true): Boolean = {
    if(file.getPath.split("/").last.isEmpty ||
      file.getAbsolutePath == "/" ||
      file.getPath.startsWith("/")) {
      Driver.dramaticError(s"delete directory ${file.getPath} will not delete absolute paths")
      false
    }
    else {
      val result = {
        if(file.isDirectory) {
          file.listFiles().forall( f => deleteDirectoryHierarchy(f)) && file.delete()
        }
        else {
          file.delete()
        }
      }
      result
    }
  }

  /** Indicate if an external command (executable) is available.
    *
    * @param cmd the command/executable
    * @return true if ```cmd``` is found in PATH.
    */
  def isCommandAvailable(cmd: String): Boolean = {
    // Eat any output.
    val sb = new StringBuffer
    val ioToDevNull = BasicIO(false, sb, None)

    Seq("bash", "-c", "which %s".format(cmd)).run(ioToDevNull).exitValue == 0
  }

  /** isVCSAvailable - flag indicating vcs is available (for Verilog compilation and testing. */
  lazy val isVCSAvailable: Boolean = isCommandAvailable("vcs")
}
