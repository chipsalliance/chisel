// See LICENSE for license details.

package firrtl

import scala.collection._
import scala.io.Source
import scala.sys.process.{BasicIO,stringSeqToProcess}
import scala.util.control.ControlThrowable
import java.io.{File, FileNotFoundException}

import net.jcazevedo.moultingyaml._
import logger.Logger
import Parser.{IgnoreInfo, InfoMode}
import annotations._
import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.passes.PassException
import firrtl.transforms._
import firrtl.Utils.throwInternalError


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

object Driver {
  //noinspection ScalaDeprecation
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
    val outputBuffer = new java.io.CharArrayWriter
    try {
      val parsedInput = Parser.parse(Source.fromFile(input).getLines(), infoMode)
      compiler.compile(
        CircuitState(parsedInput, ChirrtlForm, Some(annotations)),
        outputBuffer,
        customTransforms)
    }

    catch {
      // Rethrow the exceptions which are expected or due to the runtime environment (out of memory, stack overflow)
      case p: ControlThrowable => throw p
      case p: PassException  => throw p
      case p: FIRRTLException => throw p
      // Treat remaining exceptions as internal errors.
      case e: Exception => throwInternalError(exception = Some(e))
    }

    val outputFile = new java.io.PrintWriter(output)
    val outputString = outputBuffer.toString
    outputFile.write(outputString)
    outputFile.close()
    outputString
  }

  /** Print a warning message
    *
    * @param message error message
    */
  //scalastyle:off regex
  def dramaticWarning(message: String): Unit = {
    println(Console.YELLOW + "-"*78)
    println(s"Warning: $message")
    println("-"*78 + Console.RESET)
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
      Driver.dramaticWarning(msg)
    } else if (usingImplicitAnnoFile) {
      val msg = "Implicit .anno file from top-name is deprecated!\n" +
             (" "*9) + "Use explicit -faf option or annotationFileNames"
      Driver.dramaticWarning(msg)
    }

    val loadedAnnos = annoFiles.flatMap { file =>
      if (!file.exists) {
        throw new FileNotFoundException(s"Annotation file $file not found!")
      }
      val yaml = io.Source.fromFile(file).getLines().mkString("\n").parseYaml
      yaml.convertTo[List[Annotation]]

    }

    val targetDirAnno =
      List(Annotation(
        CircuitName("All"),
        classOf[BlackBoxSourceHelper],
        BlackBoxTargetDir(optionsManager.targetDirName).serialize
      ))

    // Output Annotations
    val outputAnnos = firrtlConfig.getEmitterAnnos(optionsManager)

    val globalAnnos = Seq(TargetDirAnnotation(optionsManager.targetDirName)) ++
      (if (firrtlConfig.dontCheckCombLoops) Seq(DontCheckCombLoopsAnnotation()) else Seq()) ++
      (if (firrtlConfig.noDCE) Seq(NoDCEAnnotation()) else Seq())

    targetDirAnno ++ outputAnnos ++ globalAnnos ++ firrtlConfig.annotations ++ loadedAnnos
  }

  /**
    * Run the firrtl compiler using the provided option
    *
    * @param optionsManager the desired flags to the compiler
    * @return a FirrtlExecutionResult indicating success or failure, provide access to emitted data on success
    *         for downstream tools as desired
    */
  //scalastyle:off cyclomatic.complexity method.length
  def execute(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): FirrtlExecutionResult = {
    def firrtlConfig = optionsManager.firrtlOptions

    Logger.makeScope(optionsManager) {
      val firrtlSource = firrtlConfig.firrtlSource match {
        case Some(text) => text.split("\n").toIterator
        case None =>
          if (optionsManager.topName.isEmpty && firrtlConfig.inputFileNameOverride.isEmpty) {
            val message = "either top-name or input-file-override must be set"
            dramaticError(message)
            return FirrtlExecutionFailure(message)
          }
          if (
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

      var maybeFinalState: Option[CircuitState] = None

      // Wrap compilation in a try/catch to present Scala MatchErrors in a more user-friendly format.
      try {
        val annos = getAnnotations(optionsManager)

        val parsedInput = Parser.parse(firrtlSource, firrtlConfig.infoMode)

        // Does this need to be before calling compiler?
        optionsManager.makeTargetDir()

        maybeFinalState = Some(firrtlConfig.compiler.compile(
          CircuitState(parsedInput,
            ChirrtlForm,
            Some(AnnotationMap(annos))),
          firrtlConfig.customTransforms
        ))
      }

    catch {
      // Rethrow the exceptions which are expected or due to the runtime environment (out of memory, stack overflow)
      case p: ControlThrowable => throw p
      case p: PassException  => throw p
      case p: FIRRTLException => throw p
      // Treat remaining exceptions as internal errors.
      case e: Exception => throwInternalError(exception = Some(e))
    }

      val finalState = maybeFinalState.get

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
          if (emittedModules.isEmpty) throwInternalError() // There should be something
          emittedModules.foreach { module =>
            val filename = optionsManager.getBuildFileName(firrtlConfig.outputSuffix, s"$dirName/${module.name}")
            val outputFile = new java.io.PrintWriter(filename)
            outputFile.write(module.value)
            outputFile.close()
          }
          "" // Should we return something different here?
      }

      // If set, emit final annotations to a file
      optionsManager.firrtlOptions.outputAnnotationFileName match {
        case "" =>
        case file =>
          val filename = optionsManager.getBuildFileName("anno", file)
          val outputFile = new java.io.PrintWriter(filename)
          finalState.annotations.foreach {
            finalAnnos => outputFile.write(finalAnnos.annotations.toYaml.prettyPrint)
          }
          outputFile.close()
      }

      FirrtlExecutionSuccess(firrtlConfig.compilerName, emittedRes)
    }
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
          throwInternalError(Some(s"Error: Unknown Firrtl Execution result $result"))
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
    val ioToDevNull = BasicIO(withIn = false, sb, None)

    Seq("bash", "-c", "which %s".format(cmd)).run(ioToDevNull).exitValue == 0
  }

  /** Flag indicating if vcs is available (for Verilog compilation and testing). */
  lazy val isVCSAvailable: Boolean = isCommandAvailable("vcs")
}
