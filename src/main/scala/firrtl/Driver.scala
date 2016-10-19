// See License

package firrtl

import java.io.FileNotFoundException

import scala.io.Source
import Annotations._

import Parser.{InfoMode, IgnoreInfo}
import scala.collection._

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
  * @see  firrtlTests.DriverSpec.scala in the test directory for a lot more examples
  */

object Driver {
  // Compiles circuit. First parses a circuit from an input file,
  //  executes all compiler passes, and writes result to an output
  //  file.
  def compile(
      input: String,
      output: String,
      compiler: Compiler,
      infoMode: InfoMode = IgnoreInfo,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)
  ): String = {
    val parsedInput = Parser.parse(Source.fromFile(input).getLines(), infoMode)
    val outputBuffer = new java.io.CharArrayWriter
    compiler.compile(parsedInput, annotations, outputBuffer)

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
  def dramaticError(message: String): Unit = {
    println(Console.RED + "-"*78)
    println(s"Error: $message")
    println("-"*78 + Console.RESET)
  }

  /**
    * Run the firrtl compiler using the provided option
    *
    * @param optionsManager the desired flags to the compiler
    * @return a FirrtlExectionResult indicating success or failure, provide access to emitted data on success
    *         for downstream tools as desired
    */
  def execute(optionsManager: ExecutionOptionsManager with HasFirrtlOptions): FirrtlExecutionResult = {
    val firrtlConfig = optionsManager.firrtlOptions

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
          case e: FileNotFoundException =>
            val message = s"Input file $inputFileName not found"
            dramaticError(message)
            return FirrtlExecutionFailure(message)
          }
        }

    val parsedInput = Parser.parse(firrtlSource, firrtlConfig.infoMode)
    val outputBuffer = new java.io.CharArrayWriter
    firrtlConfig.compiler.compile(parsedInput, new AnnotationMap(firrtlConfig.annotations), outputBuffer)

    val outputFileName = firrtlConfig.getOutputFileName(optionsManager)
    val outputFile     = new java.io.PrintWriter(outputFileName)
    val outputString   = outputBuffer.toString
    outputFile.write(outputString)
    outputFile.close()

    FirrtlExecutionSuccess(firrtlConfig.compilerName, outputBuffer.toString)
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

    optionsManager.parse(args) match {
      case true =>
        execute(optionsManager) match {
          case success: FirrtlExecutionSuccess =>
            success
          case failure: FirrtlExecutionFailure =>
            optionsManager.showUsageAsError()
            failure
          case result =>
            throw new Exception(s"Error: Unknown Firrtl Execution result $result")
        }
      case _ =>
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
  def deleteDirectoryHierarchy(directoryPathName: String): Unit = {
    if(directoryPathName.isEmpty || directoryPathName.startsWith("/")) {
      // don't delete absolute path
    }
    else {
      val directory = new java.io.File(directoryPathName)
      if(directory.isDirectory) {
        directory.delete()
        val directories = directoryPathName.split("/+").reverse.tail
        if (directories.nonEmpty) {
          deleteDirectoryHierarchy(directories.reverse.mkString("/"))
        }
      }
    }
  }
}