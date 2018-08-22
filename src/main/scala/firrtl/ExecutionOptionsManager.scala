// See LICENSE for license details.

package firrtl

import firrtl.annotations._
import firrtl.Parser._
import firrtl.ir.Circuit
import firrtl.passes.memlib.{InferReadWriteAnnotation, ReplSeqMemAnnotation}
import firrtl.passes.clocklist.ClockListAnnotation
import logger.LogLevel
import scopt.OptionParser

import scala.collection.Seq

/**
  * Use this trait to define an options class that can add its private command line options to a externally
  * declared parser.
  * '''NOTE''' In all derived trait/classes, if you intend on maintaining backwards compatibility,
  *  be sure to add new options at the end of the current ones and don't remove any existing ones.
  */
trait ComposableOptions

abstract class HasParser(applicationName: String) {
  final val parser = new OptionParser[Unit](applicationName) {
    var terminateOnExit = true
    override def terminate(exitState: Either[String, Unit]): Unit = {
      if(terminateOnExit) sys.exit(0)
    }
  }

  /**
    * By default scopt calls sys.exit when --help is in options, this defeats that
    */
  def doNotExitOnHelp(): Unit = {
    parser.terminateOnExit = false
  }
  /**
    * By default scopt calls sys.exit when --help is in options, this un-defeats doNotExitOnHelp
    */
  def exitOnHelp(): Unit = {
    parser.terminateOnExit = true
  }}

/**
  * Most of the chisel toolchain components require a topName which defines a circuit or a device under test.
  * Much of the work that is done takes place in a directory.
  * It would be simplest to require topName to be defined but in practice it is preferred to defer this.
  * For example, in chisel, by deferring this it is possible for the execute there to first elaborate the
  * circuit and then set the topName from that if it has not already been set.
  */
case class CommonOptions(
    topName:           String         = "",
    targetDirName:     String         = ".",
    globalLogLevel:    LogLevel.Value = LogLevel.None,
    logToFile:         Boolean        = false,
    logClassNames:     Boolean        = false,
    classLogLevels: Map[String, LogLevel.Value] = Map.empty,
    programArgs:    Seq[String]                 = Seq.empty
) extends ComposableOptions {

  def getLogFileName(optionsManager: ExecutionOptionsManager): String = {
    if(topName.isEmpty) {
      optionsManager.getBuildFileName("log", "firrtl")
    }
    else {
      optionsManager.getBuildFileName("log")
    }
  }
}

trait HasCommonOptions {
  self: ExecutionOptionsManager =>
  var commonOptions = CommonOptions()

  parser.note("common options")

  parser.opt[String]("top-name")
    .abbr("tn")
    .valueName("<top-level-circuit-name>")
    .foreach { x =>
      commonOptions = commonOptions.copy(topName = x)
    }
    .text("This options defines the top level circuit, defaults to dut when possible")

  parser.opt[String]("target-dir")
    .abbr("td").valueName("<target-directory>")
    .foreach { x =>
      commonOptions = commonOptions.copy(targetDirName = x)
    }
    .text(s"This options defines a work directory for intermediate files, default is ${commonOptions.targetDirName}")

  parser.opt[String]("log-level")
    .abbr("ll").valueName("<Error|Warn|Info|Debug|Trace>")
    .foreach { x =>
      val level = x.toLowerCase match {
        case "error" => LogLevel.Error
        case "warn"  => LogLevel.Warn
        case "info"  => LogLevel.Info
        case "debug" => LogLevel.Debug
        case "trace" => LogLevel.Trace
      }
      commonOptions = commonOptions.copy(globalLogLevel = level)
    }
    .validate { x =>
      if (Array("error", "warn", "info", "debug", "trace").contains(x.toLowerCase)) parser.success
      else parser.failure(s"$x bad value must be one of error|warn|info|debug|trace")
    }
    .text(s"This options defines a work directory for intermediate files, default is ${commonOptions.targetDirName}")

  parser.opt[Seq[String]]("class-log-level")
    .abbr("cll").valueName("<FullClassName:[Error|Warn|Info|Debug|Trace]>[,...]")
    .foreach { x =>
      val logAssignments = x.map { y =>
        val className :: levelName :: _ = y.split(":").toList

        val level = levelName.toLowerCase match {
          case "error" => LogLevel.Error
          case "warn" => LogLevel.Warn
          case "info" => LogLevel.Info
          case "debug" => LogLevel.Debug
          case "trace" => LogLevel.Trace
          case _ =>
            throw new Exception(s"Error: bad command line arguments for --class-log-level $x")
        }
        className -> level
      }

      commonOptions = commonOptions.copy(classLogLevels = commonOptions.classLogLevels ++ logAssignments)

    }
    .text(s"This options defines a work directory for intermediate files, default is ${commonOptions.targetDirName}")

  parser.opt[Unit]("log-to-file")
    .abbr("ltf")
    .foreach { _ =>
      commonOptions = commonOptions.copy(logToFile = true)
    }
    .text(s"default logs to stdout, this flags writes to topName.log or firrtl.log if no topName")

  parser.opt[Unit]("log-class-names")
    .abbr("lcn")
    .foreach { _ =>
      commonOptions = commonOptions.copy(logClassNames = true)
    }
    .text(s"shows class names and log level in logging output, useful for target --class-log-level")

  parser.help("help").text("prints this usage text")

  parser.arg[String]("<arg>...").unbounded().optional().action( (x, c) =>
    commonOptions = commonOptions.copy(programArgs = commonOptions.programArgs :+ x) ).text("optional unbounded args")

}

/** Firrtl output configuration specified by [[FirrtlExecutionOptions]]
  *
  * Derived from the fields of the execution options
  * @see [[FirrtlExecutionOptions.getOutputConfig]]
  */
sealed abstract class OutputConfig
final case class SingleFile(targetFile: String) extends OutputConfig
final case class OneFilePerModule(targetDir: String) extends OutputConfig

/**
  * The options that firrtl supports in callable component sense
  *
  * @param inputFileNameOverride  default is targetDir/topName.fir
  * @param outputFileNameOverride default is targetDir/topName.v  the .v is based on the compilerName parameter
  * @param compilerName           which compiler to use
  * @param annotations            annotations to pass to compiler
  */
case class FirrtlExecutionOptions(
    inputFileNameOverride:  String = "",
    outputFileNameOverride: String = "",
    compilerName:           String = "verilog",
    infoModeName:           String = "append",
    inferRW:                Seq[String] = Seq.empty,
    firrtlSource:           Option[String] = None,
    customTransforms:       Seq[Transform] = List.empty,
    annotations:            List[Annotation] = List.empty,
    annotationFileNameOverride: String = "",
    outputAnnotationFileName: String = "",
    emitOneFilePerModule:   Boolean = false,
    dontCheckCombLoops:     Boolean = false,
    noDCE:                  Boolean = false,
    annotationFileNames:    List[String] = List.empty,
    firrtlCircuit:          Option[Circuit] = None
)
extends ComposableOptions {

  require(!(emitOneFilePerModule && outputFileNameOverride.nonEmpty),
    "Cannot both specify the output filename and emit one file per module!!!")

  def infoMode: InfoMode = {
    infoModeName match {
      case "use" => UseInfo
      case "ignore" => IgnoreInfo
      case "gen" => GenInfo(inputFileNameOverride)
      case "append" => AppendInfo(inputFileNameOverride)
      case other => UseInfo
    }
  }

  def compiler: Compiler = {
    compilerName match {
      case "high"      => new HighFirrtlCompiler()
      case "low"       => new LowFirrtlCompiler()
      case "middle"    => new MiddleFirrtlCompiler()
      case "verilog"   => new VerilogCompiler()
      case "sverilog"  => new SystemVerilogCompiler()
    }
  }

  def outputSuffix: String = {
    compilerName match {
      case "verilog"   => "v"
      case "sverilog"  => "sv"
      case "low"       => "lo.fir"
      case "high"      => "hi.fir"
      case "middle"    => "mid.fir"
      case _ =>
        throw new Exception(s"Illegal compiler name $compilerName")
    }
  }

  /** Get the name of the input file
    *
    * @note Does not implicitly add a file extension to the input file
    * @param optionsManager this is needed to access build function and its common options
    * @return a properly constructed input file name
    */
  def getInputFileName(optionsManager: ExecutionOptionsManager): String = {
    if (inputFileNameOverride.nonEmpty) inputFileNameOverride
    else optionsManager.getBuildFileName("fir", inputFileNameOverride)
  }
  /** Get the user-specified [[OutputConfig]]
    *
    * @param optionsManager this is needed to access build function and its common options
    * @return the output configuration
    */
  def getOutputConfig(optionsManager: ExecutionOptionsManager): OutputConfig = {
    if (emitOneFilePerModule) OneFilePerModule(optionsManager.targetDirName)
    else SingleFile(optionsManager.getBuildFileName(outputSuffix, outputFileNameOverride))
  }
  /** Get the user-specified targetFile assuming [[OutputConfig]] is [[SingleFile]]
    *
    * @param optionsManager this is needed to access build function and its common options
    * @return the targetFile as a String
    */
  def getTargetFile(optionsManager: ExecutionOptionsManager): String = {
    getOutputConfig(optionsManager) match {
      case SingleFile(targetFile) => targetFile
      case other => throw new Exception("OutputConfig is not SingleFile!")
    }
  }
  /** Gives annotations based on the output configuration
    *
    * @param optionsManager this is needed to access build function and its common options
    * @return Annotations that will be consumed by emitter Transforms
    */
  def getEmitterAnnos(optionsManager: ExecutionOptionsManager): Seq[Annotation] = {
    // TODO should this be a public function?
    val emitter = compilerName match {
      case "high" => classOf[HighFirrtlEmitter]
      case "middle" => classOf[MiddleFirrtlEmitter]
      case "low" => classOf[LowFirrtlEmitter]
      case "verilog" => classOf[VerilogEmitter]
      case "sverilog" => classOf[VerilogEmitter]
    }
    getOutputConfig(optionsManager) match {
      case SingleFile(_) => Seq(EmitCircuitAnnotation(emitter))
      case OneFilePerModule(_) => Seq(EmitAllModulesAnnotation(emitter))
    }
  }
  /**
    * build the annotation file name, taking overriding parameters
    *
    * @param optionsManager this is needed to access build function and its common options
    * @return
    */
  @deprecated("Use FirrtlOptions.annotationFileNames instead", "1.1")
  def getAnnotationFileName(optionsManager: ExecutionOptionsManager): String = {
    optionsManager.getBuildFileName("anno", annotationFileNameOverride)
  }
}

trait HasFirrtlOptions {
  self: ExecutionOptionsManager =>
  var firrtlOptions = FirrtlExecutionOptions()

  parser.note("firrtl options")

  parser.opt[String]("input-file")
    .abbr("i")
    .valueName ("<firrtl-source>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(inputFileNameOverride = x)
    }.text {
      "use this to override the default input file name , default is empty"
    }

  parser.opt[String]("output-file")
    .abbr("o")
    .valueName("<output>")
    .validate { x =>
      if (firrtlOptions.emitOneFilePerModule)
        parser.failure("Cannot override output-file if split-modules is specified")
      else parser.success
    }
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(outputFileNameOverride = x)
    }.text {
    "use this to override the default output file name, default is empty"
  }

  parser.opt[String]("annotation-file")
    .abbr("faf")
    .unbounded()
    .valueName("<input-anno-file>")
    .foreach { x =>
      val annoFiles = x +: firrtlOptions.annotationFileNames
      firrtlOptions = firrtlOptions.copy(annotationFileNames = annoFiles)
    }.text("Used to specify annotation files (can appear multiple times)")

  parser.opt[Unit]("force-append-anno-file")
    .abbr("ffaaf")
    .hidden()
    .foreach { _ =>
      val msg = "force-append-anno-file is deprecated and will soon be removed\n" +
                (" "*9) + "(It does not do anything anymore)"
      Driver.dramaticWarning(msg)
    }

  parser.opt[String]("output-annotation-file")
    .abbr("foaf")
    .valueName ("<output-anno-file>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(outputAnnotationFileName = x)
    }.text {
    "use this to set the annotation output file"
  }

  parser.opt[String]("compiler")
    .abbr("X")
    .valueName ("<high|middle|low|verilog|sverilog>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(compilerName = x)
    }
    .validate { x =>
      if (Array("high", "middle", "low", "verilog", "sverilog").contains(x.toLowerCase)) parser.success
      else parser.failure(s"$x not a legal compiler")
    }.text {
      s"compiler to use, default is ${firrtlOptions.compilerName}"
    }

  parser.opt[String]("info-mode")
    .valueName ("<ignore|use|gen|append>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(infoModeName = x.toLowerCase)
    }
    .validate { x =>
      if (Array("ignore", "use", "gen", "append").contains(x.toLowerCase)) parser.success
      else parser.failure(s"$x bad value must be one of ignore|use|gen|append")
    }
    .text {
      s"specifies the source info handling, default is ${firrtlOptions.infoModeName}"
    }

  parser.opt[Seq[String]]("custom-transforms")
    .abbr("fct")
    .valueName ("<package>.<class>")
    .foreach { customTransforms: Seq[String] =>
      firrtlOptions = firrtlOptions.copy(
        customTransforms = firrtlOptions.customTransforms ++
          (customTransforms map { x: String =>
            Class.forName(x).asInstanceOf[Class[_ <: Transform]].newInstance()
          })
      )
    }
    .text {
      """runs these custom transforms during compilation."""
    }


  parser.opt[Seq[String]]("inline")
    .abbr("fil")
    .valueName ("<circuit>[.<module>[.<instance>]][,..],")
    .foreach { x =>
      val newAnnotations = x.map { value =>
        value.split('.') match {
          case Array(circuit) =>
            passes.InlineAnnotation(CircuitName(circuit))
          case Array(circuit, module) =>
            passes.InlineAnnotation(ModuleName(module, CircuitName(circuit)))
          case Array(circuit, module, inst) =>
            passes.InlineAnnotation(ComponentName(inst, ModuleName(module, CircuitName(circuit))))
        }
      }
      firrtlOptions = firrtlOptions.copy(
        annotations = firrtlOptions.annotations ++ newAnnotations,
        customTransforms = firrtlOptions.customTransforms :+ new passes.InlineInstances
      )
    }
    .text {
      """Inline one or more module (comma separated, no spaces) module looks like "MyModule" or "MyModule.myinstance"""
    }

  parser.opt[Unit]("infer-rw")
    .abbr("firw")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(
        annotations = firrtlOptions.annotations :+ InferReadWriteAnnotation,
        customTransforms = firrtlOptions.customTransforms :+ new passes.memlib.InferReadWrite
      )
    }.text {
      "Enable readwrite port inference for the target circuit"
    }

  parser.opt[String]("repl-seq-mem")
    .abbr("frsq")
    .valueName ("-c:<circuit>:-i:<filename>:-o:<filename>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(
        annotations = firrtlOptions.annotations :+ ReplSeqMemAnnotation.parse(x),
        customTransforms = firrtlOptions.customTransforms :+ new passes.memlib.ReplSeqMem
      )
    }
    .text {
      "Replace sequential memories with blackboxes + configuration file"
    }

  parser.opt[String]("list-clocks")
    .abbr("clks")
    .valueName ("-c:<circuit>:-m:<module>:-o:<filename>")
    .foreach { x =>
      firrtlOptions = firrtlOptions.copy(
        annotations = firrtlOptions.annotations :+ ClockListAnnotation.parse(x),
        customTransforms = firrtlOptions.customTransforms :+ new passes.clocklist.ClockListTransform
      )
    }
    .text {
      "List which signal drives each clock of every descendent of specified module"
    }

  parser.opt[Unit]("split-modules")
    .abbr("fsm")
    .validate { x =>
      if (firrtlOptions.outputFileNameOverride.nonEmpty)
        parser.failure("Cannot split-modules if output-file is specified")
      else parser.success
    }
    .foreach { _ =>
      firrtlOptions = firrtlOptions.copy(emitOneFilePerModule = true)
    }.text {
      "Emit each module to its own file in the target directory."
    }

  parser.opt[Unit]("no-check-comb-loops")
    .foreach { _ =>
      firrtlOptions = firrtlOptions.copy(dontCheckCombLoops = true)
    }.text {
      "Do NOT check for combinational loops (not recommended)"
    }

  parser.opt[Unit]("no-dce")
    .foreach { _ =>
      firrtlOptions = firrtlOptions.copy(noDCE = true)
    }.text {
      "Do NOT run dead code elimination"
    }

  parser.note("")
}

sealed trait FirrtlExecutionResult

object FirrtlExecutionSuccess {
  def apply(
    emitType    : String,
    emitted     : String,
    circuitState: CircuitState
  ): FirrtlExecutionSuccess = new FirrtlExecutionSuccess(emitType, emitted, circuitState)


  def unapply(arg: FirrtlExecutionSuccess): Option[(String, String)] = {
    Some((arg.emitType, arg.emitted))
  }
}
/**
  * Indicates a successful execution of the firrtl compiler, returning the compiled result and
  * the type of compile
  *
  * @param emitType  The name of the compiler used, currently "high", "middle", "low", "verilog", or "sverilog"
  * @param emitted   The emitted result of compilation
  */
class FirrtlExecutionSuccess(
  val emitType: String,
  val emitted : String,
  val circuitState: CircuitState
) extends FirrtlExecutionResult

/**
  * The firrtl compilation failed.
  *
  * @param message  Some kind of hint as to what went wrong.
  */
case class FirrtlExecutionFailure(message: String) extends FirrtlExecutionResult

/**
  *
  * @param applicationName  The name shown in the usage
  */
class ExecutionOptionsManager(val applicationName: String) extends HasParser(applicationName) with HasCommonOptions {

  def parse(args: Array[String]): Boolean = {
    parser.parse(args)
  }

  def showUsageAsError(): Unit = parser.showUsageAsError()

  /**
    * make sure that all levels of targetDirName exist
    *
    * @return true if directory exists
    */
  def makeTargetDir(): Boolean = {
    FileUtils.makeDirectory(commonOptions.targetDirName)
  }

  def targetDirName: String = commonOptions.targetDirName

  /**
    * this function sets the topName in the commonOptions.
    * It would be nicer to not need this but many chisel tools cannot determine
    * the name of the device under test until other options have been parsed.
    * Havin this function allows the code to set the TopName after it has been
    * determined
    *
    * @param newTopName  the topName to be used
    */
  def setTopName(newTopName: String): Unit = {
    commonOptions = commonOptions.copy(topName = newTopName)
  }
  def setTopNameIfNotSet(newTopName: String): Unit = {
    if(commonOptions.topName.isEmpty) {
      setTopName(newTopName)
    }
  }
  def topName: String = commonOptions.topName
  def setTargetDirName(newTargetDirName: String): Unit = {
    commonOptions = commonOptions.copy(targetDirName = newTargetDirName)
  }

  /**
    * return a file based on targetDir, topName and suffix
    * Will not add the suffix if the topName already ends with that suffix
    *
    * @param suffix suffix to add, removes . if present
    * @param fileNameOverride this will override the topName if nonEmpty, when using this targetDir is ignored
    * @return
    */
  def getBuildFileName(suffix: String, fileNameOverride: String = ""): String = {
    makeTargetDir()

    val baseName = if(fileNameOverride.nonEmpty) fileNameOverride else topName
    val directoryName = {
      if(fileNameOverride.nonEmpty) {
        ""
      }
      else if(baseName.startsWith("./") || baseName.startsWith("/")) {
        ""
      }
      else {
        if(targetDirName.endsWith("/")) targetDirName else targetDirName + "/"
      }
    }
    val normalizedSuffix = {
      val dottedSuffix = if(suffix.startsWith(".")) suffix else s".$suffix"
      if(baseName.endsWith(dottedSuffix)) "" else dottedSuffix
    }
    val path = directoryName + baseName.split("/").dropRight(1).mkString("/")
    FileUtils.makeDirectory(path)
    s"$directoryName$baseName$normalizedSuffix"
  }
}
