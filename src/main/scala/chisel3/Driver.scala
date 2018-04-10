// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.Emitter
import chisel3.experimental.{RawModule, RunFirrtlTransform, ChiselAnnotation}

import java.io._
import net.jcazevedo.moultingyaml._

import internal.firrtl._
import firrtl._
import firrtl.annotations.{Annotation, JsonProtocol}
import firrtl.util.{ BackendCompilationUtilities => FirrtlBackendCompilationUtilities }

import _root_.firrtl.annotations.AnnotationYamlProtocol._

/**
  * The Driver provides methods to invoke the chisel3 compiler and the firrtl compiler.
  * By default firrtl is automatically run after chisel.  an [[ExecutionOptionsManager]]
  * is needed to manage options.  It can parser command line arguments or coordinate
  * multiple chisel toolchain tools options.
  *
  * @example
  *          {{{
  *          val optionsManager = new ExecutionOptionsManager("chisel3")
  *              with HasFirrtlOptions
  *              with HasChiselExecutionOptions {
  *            commonOptions = CommonOption(targetDirName = "my_target_dir")
  *            chiselOptions = ChiselExecutionOptions(runFirrtlCompiler = false)
  *          }
  *          chisel3.Driver.execute(optionsManager, () => new Dut)
  *          }}}
  * or via command line arguments
  * @example {{{
  *          args = "--no-run-firrtl --target-dir my-target-dir".split(" +")
  *          chisel3.execute(args, () => new DUT)
  *          }}}
  */
import BuildInfo._

trait BackendCompilationUtilities extends FirrtlBackendCompilationUtilities {
  /** Compile Chirrtl to Verilog by invoking Firrtl inside the same JVM
    *
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def compileFirrtlToVerilog(prefix: String, dir: File): Boolean = {
    val optionsManager = new ExecutionOptionsManager(
      "chisel3",
      Array("--top-name", prefix,
            "--target-dir", dir.getAbsolutePath,
            "--compiler", "verilog")) with HasChiselExecutionOptions with HasFirrtlOptions

    firrtl.Driver.execute(optionsManager) match {
      case _: FirrtlExecutionSuccess => true
      case _: FirrtlExecutionFailure => false
    }
  }
}

/**
  * This family provides return values from the chisel3 and possibly firrtl compile steps
  */
trait ChiselExecutionResult

/**
  *
  * @param circuitOption  Optional circuit, has information like circuit name
  * @param emitted            The emitted Chirrrl text
  * @param firrtlResultOption Optional Firrtl result, @see ucb-bar/firrtl for details
  */
case class ChiselExecutionSuccess(
                                  circuitOption: Option[Circuit],
                                  emitted: String,
                                  firrtlResultOption: Option[FirrtlExecutionResult]
                                  ) extends ChiselExecutionResult

/**
  * Getting one of these indicates failure of some sort
 *
  * @param message  a clue perhaps will be provided in the here
  */
case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult

object Driver extends BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting Chisel IR in the form of a Circuit (TODO: Should be FIRRTL IR)
    */
  def elaborate[T <: RawModule](gen: () => T): Circuit = internal.Builder.build(Module(gen()))

  def emit[T <: RawModule](gen: () => T): String = Emitter.emit(elaborate(gen))

  def emit[T <: RawModule](ir: Circuit): String = Emitter.emit(ir)

  /** Elaborates the Module specified in the gen function into Verilog
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting String containing the design in Verilog
    */
  def emitVerilog[T <: RawModule](gen: => T): String = {
    execute(Array[String](), { () => gen }) match {
      case ChiselExecutionSuccess(_, _, Some(firrtl.FirrtlExecutionSuccess(_, verilog))) => verilog
      case _ => sys.error("Cannot get Verilog!")
    }
  }

  def dumpFirrtl(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".fir"))
    val w = new FileWriter(f)
    w.write(Emitter.emit(ir))
    w.close()
    f
  }

  def dumpAnnotations(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".json"))
    val w = new FileWriter(f)
    w.write(JsonProtocol.serialize(ir.annotations.map(_.toFirrtl)))
    w.close()
    f
  }

  private var target_dir: Option[String] = None
  def parseArgs(args: Array[String]): Unit = {
    for (i <- 0 until args.size) {
      if (args(i) == "--targetDir") {
        target_dir = Some(args(i + 1))
      }
    }
  }

  def targetDir(): String = { target_dir getOrElse new File(".").getCanonicalPath }

  private def customTransformsArg(annos: Seq[ChiselAnnotation]): Array[String] = annos
    .collect { case anno: RunFirrtlTransform => anno.transformClass }
    .distinct
    .filterNot(_ == classOf[firrtl.Transform])
    .map{ transformClass: Class[_ <: Transform] => transformClass.getName } match {
      case a: Seq[String] if a.nonEmpty => Array("--custom-transforms") ++ a
      case a: Seq[String] => Array("")
      case _ => throw new Exception("Shouldn't be here...") }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified.
    *
    * @param optionsManager The options specified
    * @param dut            The device under test
    * @return               An execution result with useful stuff, or failure with message
    */
  @deprecated("use Driver.execute(args: Array[String], dut: () => RawModule)", "3.2.0")
  def execute(optionsManager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions,
              dut: () => RawModule): ChiselExecutionResult = {
    val circuit = elaborate(dut)
    val firrtlString = Emitter.emit(circuit)
    val firrtlAnnos = circuit.annotations.map(_.toFirrtl)

    val optionsManagerX = new ExecutionOptionsManager(
      optionsManager.applicationName,
      Array("--firrtl-source", firrtlString) ++ customTransformsArg(circuit.annotations),
      optionsManager.firrtlOptions.annotations) with HasFirrtlOptions with HasChiselExecutionOptions

    val (chiselOptions, firrtlOptions) = (optionsManagerX.chiselOptions, optionsManagerX.firrtlOptions)

    if (chiselOptions.saveChirrtl)
      dumpFirrtl(circuit, Some(new File(firrtlOptions.getInputFileName(optionsManagerX))))

    if (chiselOptions.saveAnnotations)
      dumpAnnotations(circuit, Some(new File(optionsManagerX.getBuildFileName("anno.json"))))

    val firrtlExecutionResult = if (chiselOptions.runFirrtlCompiler)
      Some(firrtl.Driver.execute(optionsManagerX))
    else
      None

    ChiselExecutionSuccess(Some(circuit), firrtlString, firrtlExecutionResult)
  }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified via an array of Strings
    *
    * @param args   The options specified, command line style
    * @param dut    The device under test
    * @return       An execution result with useful stuff, or failure with message
    */
  def execute(args: Array[String], dut: () => RawModule): ChiselExecutionResult = {
    val circuit = elaborate(dut)
    val firrtlString = Emitter.emit(circuit)
    val firrtlAnnos = circuit.annotations.map(_.toFirrtl)

    val argsx = args ++ Array("--firrtl-source", firrtlString) ++ customTransformsArg(circuit.annotations)

    val optionsManager = new ExecutionOptionsManager("chisel3", argsx, firrtlAnnos)
        with HasChiselExecutionOptions with HasFirrtlOptions

    val (chiselOptions, firrtlOptions) = (optionsManager.chiselOptions, optionsManager.firrtlOptions)

    if (chiselOptions.saveChirrtl)
      dumpFirrtl(circuit, Some(new File(firrtlOptions.getInputFileName(optionsManager))))

    if (chiselOptions.saveAnnotations)
      dumpAnnotations(circuit, Some(new File(optionsManager.getBuildFileName("anno.json"))))

    val firrtlExecutionResult = if (chiselOptions.runFirrtlCompiler)
      Some(firrtl.Driver.execute(optionsManager))
    else
      None

    ChiselExecutionSuccess(Some(circuit), firrtlString, firrtlExecutionResult)
  }

  /**
    * This is just here as command line way to see what the options are
    * It will not successfully run
    * TODO: Look into dynamic class loading as way to make this main useful
    *
    * @param args unused args
    */
  def main(args: Array[String]) {
    execute(Array("--help"), null)
  }

  val version = BuildInfo.version
  val chiselVersionString = BuildInfo.toString
}
