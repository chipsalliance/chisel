// See LICENSE for license details.

package chisel3

import scopt.OptionParser

import scala.sys.process._
import java.io._

import internal.firrtl._
import firrtl.{FirrtlExecutionFailure, FirrtlExecutionSuccess, FirrtlExecutionResult, FirrtlExecutionOptions}

/**
  * The Driver provides methods to invoke the chisel3 compiler and the firrtl compuiler.
  * By default firrtl is automatically run after chisel.  The `ChiselExecutionOptions` extends
  * 'FirrtlExecutionOptions' which extends 'CommonOptions'
  * Like firrtl the Driver defines two execute methods, the first uses a ChiselExecutionOption
  * @example
  *          {{{
  *          val options = new ChiselExecutionOptions()
  *          options.runFirrtlCompiler = false
  *          options.targetDir = "my-target-dir"
  *          chisel3.Driver.execute(options, () => new Dut)
  *          }}}
  * or via command line arguments
  * @example {{{
  *          args = "--no-run-firrtl --target-dir my-target-dir".split(" +")
  *          chisel3.execute(args, () => new DUT)
  *          }}}
  */

trait BackendCompilationUtilities {
  /** Create a temporary directory with the prefix name. Exists here because it doesn't in Java 6.
    */
  def createTempDirectory(prefix: String): File = {
    val temp = File.createTempFile(prefix, "")
    if (!temp.delete()) {
      throw new IOException(s"Unable to delete temp file '$temp'")
    }
    if (!temp.mkdir()) {
      throw new IOException(s"Unable to create temp directory '$temp'")
    }
    temp
  }

  def makeHarness(template: String => String, post: String)(f: File): File = {
    val prefix = f.toString.split("/").last
    val vf = new File(f.toString + post)
    val w = new FileWriter(vf)
    w.write(template(prefix))
    w.close()
    vf
  }

  /**
    * like 'firrtlToVerilog' except it runs the process inside the same JVM
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def compileFirrtlToVerilog(prefix: String, dir: File): Boolean = {
    val firrtlExecutionOptions = new FirrtlExecutionOptions(
      compilerName = "verilog")
    firrtlExecutionOptions.topName = prefix
    firrtlExecutionOptions.targetDirName = dir.getAbsolutePath

    firrtl.Driver.execute(firrtlExecutionOptions) match {
      case _: FirrtlExecutionSuccess => true
      case _: FirrtlExecutionFailure => false
    }
  }

  /**
    * compule chirrtl to verilog by using a separate process
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def firrtlToVerilog(prefix: String, dir: File): ProcessBuilder = {
    Process(
      Seq("firrtl",
          "-i", s"$prefix.fir",
          "-o", s"$prefix.v",
          "-X", "verilog"),
      dir)
  }

  /** Generates a Verilator invocation to convert Verilog sources to C++
    * simulation sources.
    *
    * The Verilator prefix will be V$dutFile, and running this will generate
    * C++ sources and headers as well as a makefile to compile them.
    *
    * @param dutFile name of the DUT .v without the .v extension
    * @param topModule of the top-level module in the design
    * @param dir output directory
    * @param vSources list of additional Verilog sources to compile
    * @param cppHarness C++ testharness to compile/link against
    */
  def verilogToCpp(
      dutFile: String,
      topModule: String,
      dir: File,
      vSources: Seq[File],
      cppHarness: File
                  ): ProcessBuilder = {
    val command = Seq("verilator",
      "--cc", s"$dutFile.v") ++
      vSources.map(file => Seq("-v", file.toString)).flatten ++
      Seq("--assert",
        "-Wno-fatal",
        "-Wno-WIDTH",
        "-Wno-STMTDLY",
        "--trace",
        "-O2",
        "--top-module", topModule,
        "+define+TOP_TYPE=V" + dutFile,
        s"+define+PRINTF_COND=!$topModule.reset",
        s"+define+STOP_COND=!$topModule.reset",
        "-CFLAGS",
        s"""-Wno-undefined-bool-conversion -O2 -DTOP_TYPE=V$dutFile -include V$dutFile.h""",
        "-Mdir", dir.toString,
        "--exe", cppHarness.toString)
    System.out.println(s"${command.mkString(" ")}") // scalastyle:ignore regex
    command
  }

  def cppToExe(prefix: String, dir: File): ProcessBuilder =
    Seq("make", "-C", dir.toString, "-j", "-f", s"V${prefix}.mk", s"V${prefix}")

  def executeExpectingFailure(
      prefix: String,
      dir: File,
      assertionMsg: String = ""): Boolean = {
    var triggered = false
    val assertionMessageSupplied = assertionMsg != ""
    val e = Process(s"./V${prefix}", dir) !
      ProcessLogger(line => {
        triggered = triggered || (assertionMessageSupplied && line.contains(assertionMsg))
        System.out.println(line) // scalastyle:ignore regex
      })
    // Fail if a line contained an assertion or if we get a non-zero exit code
    //  or, we get a SIGABRT (assertion failure) and we didn't provide a specific assertion message
    triggered || (e != 0 && (e != 134 || !assertionMessageSupplied))
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir)
  }
}

/**
  * Options that are specific to chisel.
  * @param runFirrtlCompiler when true just run chisel, when false run chisel then compile its output with firrtl
  * @note this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  */
class ChiselExecutionOptions(
                              var runFirrtlCompiler: Boolean = true
                              // var runFirrtlAsProcess: Boolean = false
                            ) extends FirrtlExecutionOptions {

  //TODO: provide support for running firrtl as separate process, could alternatively be controlled by external driver
  //TODO: provide option for not saving chirrtl file, instead calling firrtl with in memory chirrtl
  override def addOptions(parser: OptionParser[Unit]): Unit = {
    super.addOptions(parser)

    parser.note("chisel3 options")

    parser.opt[Unit]("no-run-firrtl").abbr("chnrf").foreach(_ => runFirrtlCompiler = false)
      .text("Stop after chisel emits chirrtl file")

    // parser.opt[Unit]("run-firrtl-as-process").abbr("chrfap").foreach(_ => runFirrtlCompiler = false)
    //  .text("Run firrtl as a separate process, not within same JVM as chisel")
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
case class ChiselExecutionSucccess(
                                  circuitOption: Option[Circuit],
                                  emitted: String,
                                  firrtlResultOption: Option[FirrtlExecutionResult]
                                  ) extends ChiselExecutionResult

/**
  * Getting one of these indicates failure of some sort
  * @param message  a clue perhaps will be provided in the here
  */
case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult

object Driver extends BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting Chisel IR in the form of a Circuit (TODO: Should be FIRRTL IR)
    */
  def elaborate[T <: Module](gen: () => T): Circuit = internal.Builder.build(Module(gen()))

  def emit[T <: Module](gen: () => T): String = Emitter.emit(elaborate(gen))

  def emit[T <: Module](ir: Circuit): String = Emitter.emit(ir)

  def dumpFirrtl(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".fir"))
    val w = new FileWriter(f)
    w.write(Emitter.emit(ir))
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

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified
    * @param chiselExecutionOptions The options specified
    * @param dut                    The device under test
    * @return                       An execution result with useful stuff, or failure with message
    */
  def execute(chiselExecutionOptions: ChiselExecutionOptions, dut: () => Module): ChiselExecutionResult = {
    val circuit = elaborate(dut)

    // this little hack let's us set the topName with the circuit name if it has not been set from args
    chiselExecutionOptions.setTopNameIfUnset(circuit.name)

    // use input because firrtl will be reading this
    val firrtlString = Emitter.emit(circuit)
    val firrtlFileName = chiselExecutionOptions.inputFileName
    val firrtlFile = new File(firrtlFileName)

    val w = new FileWriter(firrtlFile)
    w.write(firrtlString)
    w.close()

    val firrtlExecutionResult = if(chiselExecutionOptions.runFirrtlCompiler) {
      Some(firrtl.Driver.execute(chiselExecutionOptions))
    }
    else {
      None
    }
    ChiselExecutionSucccess(Some(circuit), firrtlString, firrtlExecutionResult)
  }

  /**
    * Run the chisel3 compiler and possibly the firrtl compiler with options specified via an array of Strings
    * @param args   The options specified, command line style
    * @param dut    The device under test
    * @return       An execution result with useful stuff, or failure with message
    */
  def execute(args: Array[String], dut: () => Module): ChiselExecutionResult = {
    val chiselExecutionOptions = new ChiselExecutionOptions()

    val parser = new OptionParser[Unit]("chisel3"){}
    chiselExecutionOptions.addOptions(parser)

    parser.parse(args) match {
      case true =>
        execute(chiselExecutionOptions, dut)
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
}
