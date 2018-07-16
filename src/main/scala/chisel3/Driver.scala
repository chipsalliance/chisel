// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.Converter
import chisel3.experimental.{RawModule, RunFirrtlTransform}

import java.io._
import net.jcazevedo.moultingyaml._

import internal.firrtl._
import firrtl.{ HasFirrtlExecutionOptions, FirrtlExecutionSuccess, FirrtlExecutionFailure, FirrtlExecutionResult,
  Transform, FirrtlExecutionOptions, AnnotationSeq, FirrtlCircuitAnnotation, ir => fir }
import firrtl.options.ExecutionOptionsManager
import firrtl.annotations.JsonProtocol
import firrtl.util.{ BackendCompilationUtilities => FirrtlBackendCompilationUtilities }
import firrtl.options.Viewer._
import firrtl.FirrtlViewer._
import chisel3.ChiselViewer._

import java.io._
import BuildInfo._

trait BackendCompilationUtilities extends FirrtlBackendCompilationUtilities {
  /**
    * Compile Chirrtl to Verilog by invoking Firrtl inside the same JVM
    *
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def compileFirrtlToVerilog(prefix: String, dir: File): Boolean = {
    val args = Array("--top-name", prefix, "--target-dir", dir.getAbsolutePath, "--compiler", "verilog")

    firrtl.Driver.execute(args) match {
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
  * Getting one of these indicates a successful Chisel run
  *
  * @param circuitOption  Optional circuit, has information like circuit name
  * @param emitted            The emitted Chirrrl text
  * @param firrtlResultOption Optional Firrtl result, @see ucb-bar/firrtl for details
  */
case class ChiselExecutionSuccess(
  circuitOption: Option[Circuit],
  emitted: String,
  firrtlResultOption: Option[FirrtlExecutionResult]) extends ChiselExecutionResult

/**
  * Getting one of these indicates failure of some sort
  *
  * @param message  a clue perhaps will be provided in the here
  */
case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult

/**
  * The Driver provides methods to invoke the Chisel3 compiler and the
  * FIRRTL compiler. By default FIRRTL is automatically run after chisel.
  *
  * @example
  * {{{
  * val args = Array(
  *   "--top-name", "MyTopModule",     // The name of the top module
  *   "--target-dir", "my_target_dir", // The work directory
  *   "--compiler", "low" )            // The FIRRTL compiler to use
  * Driver.execute(args, () => new MyTopModule)
  * }}}
  */
object Driver extends BackendCompilationUtilities {
  val optionsManager = new ExecutionOptionsManager("chisel3") with HasChiselExecutionOptions
      with HasFirrtlExecutionOptions

  /**
    * Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting Chisel IR in the form of a Circuit (TODO: Should be FIRRTL IR)
    */
  def elaborate[T <: RawModule](gen: () => T): Circuit = internal.Builder.build(Module(gen()))

  def toFirrtl(ir: Circuit): fir.Circuit = Converter.convert(ir)

  def emit[T <: RawModule](gen: () => T): String = Driver.emit(elaborate(gen))

  def emit[T <: RawModule](ir: Circuit): String = Emitter.emit(ir)

  /**
    * Elaborates the Module specified in the gen function into Verilog
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting String containing the design in Verilog
    */
  def emitVerilog[T <: RawModule](gen: => T): String = {
    execute(Array[String](), { () => gen }) match {
      case ChiselExecutionSuccess(_, _, Some(FirrtlExecutionSuccess(_, verilog))) => verilog
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

  /**
    * Emit the annotations of a circuit
    *
    * @param ir The circuit containing annotations to be emitted
    * @param optName An optional filename (will use s"${ir.name}.json" otherwise)
    */
  def dumpAnnotations(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".json"))
    val w = new FileWriter(f)
    w.write(JsonProtocol.serialize(ir.annotations.map(_.toFirrtl)))
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
    firrtl.proto.ToProto.writeToStreamFast(ostream, fir.NoInfo, modules, c.name)
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
    * Determine custom transforms FIRRTL Driver command line arguments,
    * e.g., "--custom-transforms ..."
    *
    * @param ir A circuit that may contain annotations
    * @return An array of command line arguments
    */
  private def customTransformsArg(ir: Circuit): Array[String] = ir.annotations
    .collect { case anno: RunFirrtlTransform => anno.transformClass }
    .distinct
    .filterNot(_ == classOf[firrtl.Transform])
    .map{ transformClass: Class[_ <: Transform] => transformClass.getName } match {
      case a: Seq[String] if a.nonEmpty => Array("--custom-transforms") ++ a
      case _ => Array("") }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified via an array of Strings
    *
    * @param args The options specified, command line style
    * @param dut The device under test
    * @param initAnnos Initial annotations (an alternative to args)
    * @return An execution result with useful stuff, or failure with message
    */
  def execute(args: Array[String], dut: () => RawModule, initAnnos: AnnotationSeq = Seq.empty): ChiselExecutionResult = {
    val circuit = elaborate(dut)
    val firrtlCircuit = Converter.convert(circuit)
    val firrtlAnnos = circuit.annotations.map(_.toFirrtl)
    val annotations = optionsManager.parse(args ++ customTransformsArg(circuit),
                                           initAnnos ++ firrtlAnnos :+ FirrtlCircuitAnnotation(firrtlCircuit))

    val firrtlOptions = view[FirrtlExecutionOptions](annotations).getOrElse{
      throw new Exception("Unable to parse Firrtl options") }
    val chiselOptions = view[ChiselExecutionOptions](annotations).getOrElse{
      throw new Exception("Unable to parse Chisel options") }

    // Still emit to leave an artifact (and because this always has been the behavior)
    val firrtlString = Driver.emit(circuit)
    if (chiselOptions.saveChirrtl) {
      val w = new FileWriter(new File(firrtlOptions.getInputFileName))
      w.write(firrtlString)
      w.close()
    }

    if (chiselOptions.saveAnnotations)
      dumpAnnotations(circuit, Some(new File(firrtlOptions.getBuildFileName("anno.json"))))

    val firrtlExecutionResult = if(chiselOptions.runFirrtlCompiler) {
      Some(firrtl.Driver.execute(Array.empty, annotations))
    }
    else {
      None
    }
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
