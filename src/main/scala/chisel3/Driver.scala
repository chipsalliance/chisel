// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.Converter
import chisel3.experimental.{RawModule, RunFirrtlTransform}

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
    val optionsManager = new ExecutionOptionsManager("chisel3") with HasChiselExecutionOptions with HasFirrtlOptions {
      commonOptions = CommonOptions(topName = prefix, targetDirName = dir.getAbsolutePath)
      firrtlOptions = FirrtlExecutionOptions(compilerName = "verilog")
    }

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

  def toFirrtl(ir: Circuit): firrtl.ir.Circuit = Converter.convert(ir)

  def emit[T <: RawModule](gen: () => T): String = Driver.emit(elaborate(gen))

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

  /** Dumps the elaborated Circuit to FIRRTL
    *
    * If no File is given as input, it will dump to a default filename based on the name of the
    * Top Module
    *
    * @param c Elaborated Chisel Circuit
    * @param optName Optional File to dump to
    * @return The File the circuit was dumped to
    */
  def dumpFirrtl(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".fir"))
    val w = new FileWriter(f)
    w.write(Driver.emit(ir))
    w.close()
    f
  }

  /** Dumps the elaborated Circuit to ProtoBuf
    *
    * If no File is given as input, it will dump to a default filename based on the name of the
    * Top Module
    *
    * @param c Elaborated Chisel Circuit
    * @param optFile Optional File to dump to
    * @return The File the circuit was dumped to
    */
  def dumpProto(c: Circuit, optFile: Option[File]): File = {
    val f = optFile.getOrElse(new File(c.name + ".pb"))
    val ostream = new java.io.FileOutputStream(f)
    // Lazily convert modules to make intermediate objects garbage collectable
    val modules = c.components.map(m => () => Converter.convert(m))
    firrtl.proto.ToProto.writeToStreamFast(ostream, ir.NoInfo, modules, c.name)
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

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified
 *
    * @param optionsManager The options specified
    * @param dut                    The device under test
    * @return                       An execution result with useful stuff, or failure with message
    */
  def execute(
      optionsManager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions,
      dut: () => RawModule): ChiselExecutionResult = {
    val circuit = elaborate(dut)

    // this little hack let's us set the topName with the circuit name if it has not been set from args
    optionsManager.setTopNameIfNotSet(circuit.name)

    val firrtlOptions = optionsManager.firrtlOptions
    val chiselOptions = optionsManager.chiselOptions

    val firrtlCircuit = Converter.convert(circuit)

    // Still emit to leave an artifact (and because this always has been the behavior)
    val firrtlString = Driver.emit(circuit)
    val firrtlFileName = firrtlOptions.getInputFileName(optionsManager)
    val firrtlFile = new File(firrtlFileName)

    val w = new FileWriter(firrtlFile)
    w.write(firrtlString)
    w.close()

    // Emit the annotations because it has always been the behavior
    val annotationFile = new File(optionsManager.getBuildFileName("anno.json"))
    val af = new FileWriter(annotationFile)
    val firrtlAnnos = circuit.annotations.map(_.toFirrtl)
    af.write(JsonProtocol.serialize(firrtlAnnos))
    af.close()

    /** Find the set of transform classes associated with annotations then
      * instantiate an instance of each transform
      * @note Annotations targeting firrtl.Transform will not result in any
      *   transform being instantiated
      */
    val transforms = circuit.annotations
                       .collect { case anno: RunFirrtlTransform => anno.transformClass }
                       .distinct
                       .filterNot(_ == classOf[firrtl.Transform])
                       .map { transformClass: Class[_ <: Transform] =>
                         transformClass.newInstance()
                       }
    /* This passes the firrtl source and annotations directly to firrtl */
    optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(
      firrtlCircuit = Some(firrtlCircuit),
      annotations = optionsManager.firrtlOptions.annotations ++ firrtlAnnos,
      customTransforms = optionsManager.firrtlOptions.customTransforms ++ transforms.toList)

    val firrtlExecutionResult = if(chiselOptions.runFirrtlCompiler) {
      Some(firrtl.Driver.execute(optionsManager))
    }
    else {
      None
    }
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
    val optionsManager = new ExecutionOptionsManager("chisel3") with HasChiselExecutionOptions with HasFirrtlOptions

    optionsManager.parse(args) match {
      case true =>
        execute(optionsManager, dut)
      case _ =>
        ChiselExecutionFailure("could not parse results")
    }
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
