// See LICENSE for license details.

package chisel3

import chisel3.internal.ErrorLog
import internal.firrtl._
import firrtl._
import firrtl.options.{Dependency, Phase, PhaseManager, StageError}
import firrtl.options.phases.DeletedWrapper
import firrtl.options.Viewer.view
import firrtl.annotations.JsonProtocol
import firrtl.util.{BackendCompilationUtilities => FirrtlBackendCompilationUtilities}
import chisel3.stage.{ChiselExecutionResultView, ChiselGeneratorAnnotation, ChiselStage}
import chisel3.stage.phases.DriverCompatibility
import java.io._


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
@deprecated("This will be removed in Chisel 3.5", "Chisel3 3.4")
trait ChiselExecutionResult

/**
  *
  * @param circuitOption  Optional circuit, has information like circuit name
  * @param emitted            The emitted Chirrrl text
  * @param firrtlResultOption Optional Firrtl result, @see freechipsproject/firrtl for details
  */
@deprecated("This will be removed in Chisel 3.5", "Chisel 3.4")
case class ChiselExecutionSuccess(
                                  circuitOption: Option[Circuit],
                                  emitted: String,
                                  firrtlResultOption: Option[FirrtlExecutionResult]
                                  ) extends ChiselExecutionResult

/**
  * Getting one of these indicates failure of some sort.
  *
  * @param message A clue might be provided here.
  */
@deprecated("This will be removed in Chisel 3.5", "Chisel 3.4")
case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult

@deprecated("Please switch to chisel3.stage.ChiselStage. Driver will be removed in 3.4.", "3.2.4")
object Driver extends BackendCompilationUtilities {

  /**
    * Elaborate the Module specified in the gen function into a Chisel IR Circuit.
    *
    * @param gen A function that creates a Module hierarchy.
    * @return The resulting Chisel IR in the form of a Circuit. (TODO: Should be FIRRTL IR)
    */
  @deprecated("Use ChiselStage.elaborate or use a ChiselStage class. This will be removed in 3.4.", "3.2.4")
  def elaborate[T <: RawModule](gen: () => T): Circuit = internal.Builder.build(Module(gen()))._1

  /**
    * Convert the given Chisel IR Circuit to a FIRRTL Circuit.
    *
    * @param ir Chisel IR Circuit, generated e.g. by elaborate().
    */
  @deprecated("Use ChiselStage.convert or use a ChiselStage class. This will be removed in 3.4.", "3.2.4")
  def toFirrtl(ir: Circuit): firrtl.ir.Circuit = Converter.convert(ir)

  /**
    * Emit the Module specified in the gen function directly as a FIRRTL string without
    * invoking FIRRTL.
    *
    * @param gen A function that creates a Module hierarchy.
    */
  @deprecated("Use (new chisel3.stage.ChiselStage).emitChirrtl. This will be removed in 3.4.", "3.2.2")
  def emit[T <: RawModule](gen: () => T): String = Driver.emit(elaborate(gen))

  /**
    * Emit the given Chisel IR Circuit as a FIRRTL string, without invoking FIRRTL.
    *
    * @param ir Chisel IR Circuit, generated e.g. by elaborate().
    */
  @deprecated("Use (new chisel3.stage.ChiselStage).emitChirrtl", "3.2.2")
  def emit[T <: RawModule](ir: Circuit): String = Emitter.emit(ir)

  /**
    * Elaborate the Module specified in the gen function into Verilog.
    *
    * @param gen A function that creates a Module hierarchy.
    * @return A String containing the design in Verilog.
    */
  @deprecated("Use (new chisel3.stage.ChiselStage).emitVerilog. This will be removed in 3.4.", "3.2.2")
  def emitVerilog[T <: RawModule](gen: => T): String = {
    execute(Array[String](), { () => gen }) match {
      case ChiselExecutionSuccess(_, _, Some(firrtl.FirrtlExecutionSuccess(_, verilog))) => verilog
      case _ => sys.error("Cannot get Verilog!")
    }
  }

  /**
    * Dump the elaborated Chisel IR Circuit as a FIRRTL String, without invoking FIRRTL.
    *
    * If no File is given as input, it will dump to a default filename based on the name of the
    * top Module.
    *
    * @param c Elaborated Chisel Circuit.
    * @param optName File to dump to. If unspecified, defaults to "<topmodule>.fir".
    * @return The File the circuit was dumped to.
    */
  @deprecated("Migrate to chisel3.stage.ChiselStage. This will be removed in 3.4.", "3.2.4")
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
    * @param optName An optional filename (will use s"\${ir.name}.json" otherwise)
    */
  @deprecated("Migrate to chisel3.stage.ChiselStage. This will be removed in 3.4.", "3.2.4")
  def dumpAnnotations(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".anno.json"))
    val w = new FileWriter(f)
    w.write(JsonProtocol.serialize(ir.annotations.map(_.toFirrtl)))
    w.close()
    f
  }

  /**
    * Dump the elaborated Circuit to ProtoBuf.
    *
    * If no File is given as input, it will dump to a default filename based on the name of the
    * top Module.
    *
    * @param c Elaborated Chisel Circuit.
    * @param optFile Optional File to dump to. If unspecified, defaults to "<topmodule>.pb".
    * @return The File the circuit was dumped to.
    */
  @deprecated("Migrate to chisel3.stage.ChiselStage. This will be removed in 3.4.", "3.2.4")
  def dumpProto(c: Circuit, optFile: Option[File]): File = {
    val f = optFile.getOrElse(new File(c.name + ".pb"))
    val ostream = new java.io.FileOutputStream(f)
    // Lazily convert modules to make intermediate objects garbage collectable
    val modules = c.components.map(m => () => Converter.convert(m))
    firrtl.proto.ToProto.writeToStreamFast(ostream, ir.NoInfo, modules, c.name)
    f
  }

  private var target_dir: Option[String] = None
  @deprecated("Use chisel3.stage.ChiselStage with '--target-directory'. This will be removed in 3.4.", "3.2.2")
  def parseArgs(args: Array[String]): Unit = {
    for (i <- 0 until args.size) {
      if (args(i) == "--targetDir") {
        target_dir = Some(args(i + 1))
      }
    }
  }

  @deprecated("This has no effect on Chisel3 Driver! This will be removed in 3.4.", "3.2.2")
  def targetDir(): String = { target_dir getOrElse new File(".").getCanonicalPath }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified
    *
    * @param optionsManager The options specified
    * @param dut                    The device under test
    * @return                       An execution result with useful stuff, or failure with message
    */
  @deprecated("Use chisel3.stage.ChiselStage.execute. This will be removed in 3.4.", "3.2.2")
  def execute(
      optionsManager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions,
      dut: () => RawModule): ChiselExecutionResult = {

    val annos: AnnotationSeq =
      Seq(DriverCompatibility.OptionsManagerAnnotation(optionsManager), ChiselGeneratorAnnotation(dut)) ++
        optionsManager.chiselOptions.toAnnotations ++
        optionsManager.firrtlOptions.toAnnotations ++
        optionsManager.commonOptions.toAnnotations

    val targets =
      Seq( Dependency[DriverCompatibility.AddImplicitOutputFile],
           Dependency[DriverCompatibility.AddImplicitOutputAnnotationFile],
           Dependency[DriverCompatibility.DisableFirrtlStage],
           Dependency[ChiselStage],
           Dependency[DriverCompatibility.MutateOptionsManager],
           Dependency[DriverCompatibility.ReEnableFirrtlStage],
           Dependency[DriverCompatibility.FirrtlPreprocessing],
           Dependency[chisel3.stage.phases.MaybeFirrtlStage] )
    val currentState =
      Seq( Dependency[firrtl.stage.phases.DriverCompatibility.AddImplicitFirrtlFile],
           Dependency[chisel3.stage.phases.Convert] )

    val phases: Seq[Phase] = new PhaseManager(targets, currentState) {
      override val wrappers = Seq( DeletedWrapper(_: Phase) )
    }.transformOrder

    val annosx = try {
      phases.foldLeft(annos)( (a, p) => p.transform(a) )
    } catch {
      /* ChiselStage and FirrtlStage can throw StageError. Since Driver is not a StageMain, it cannot catch these. While
       * Driver is deprecated and removed in 3.2.1+, the Driver catches all errors.
       */
      case e: StageError => annos
    }

    view[ChiselExecutionResult](annosx)
  }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified via an array of Strings
 *
    * @param args   The options specified, command line style
    * @param dut    The device under test
    * @return       An execution result with useful stuff, or failure with message
    */
  @deprecated("Use chisel3.stage.ChiselStage.execute. This will be removed in 3.4.", "3.2.2")
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
  @deprecated("Use chisel3.stage.ChiselMain. This will be removed in 3.4.", "3.2.2")
  def main(args: Array[String]) {
    execute(Array("--help"), null)
  }

  val version = BuildInfo.version
  val chiselVersionString = BuildInfo.toString
}
