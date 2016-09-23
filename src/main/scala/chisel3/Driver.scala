// See LICENSE for license details.

package chisel3

import chisel3._
import scopt.OptionParser

import scala.sys.process._
import java.io._

import internal.{firrtl => ifirrtl,_}

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
      assertionMsg: String = "Assertion failed"): Boolean = {
    var triggered = false
    val e = Process(s"./V${prefix}", dir) !
      ProcessLogger(line => {
        triggered = triggered || line.contains(assertionMsg)
        System.out.println(line) // scalastyle:ignore regex
      })
    triggered
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir)
  }
}


case class Emitted(circuit: ifirrtl.Circuit, firrtlString: String, annotationString: String)

object Driver extends BackendCompilationUtilities {
  val FirrtlSuffix = ".fir"
  case object ChiselConfigKey extends firrtl.ExecutionModel.ExecutionKey

  /** Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting Chisel IR in the form of a Circuit (TODO: Should be FIRRTL IR)
    */
  def elaborate[T <: Module](gen: () => T): ifirrtl.Circuit = Builder.build(Module(gen()))

  def emit[T <: Module](gen: () => T): String = ifirrtl.Emitter.emit(elaborate(gen))

  def emit[T <: Module](ir: ifirrtl.Circuit): String = ifirrtl.Emitter.emit(ir)


  def dumpFirrtl(ir: ifirrtl.Circuit, optName: Option[File] = None): File = {
    val (directory: File, fileName: String) = optName match {
      case Some(file) =>
        val fileDirectory = new File(file.getParent)
        val fileName = file.getName
        if(fileName.endsWith(FirrtlSuffix)) {
          (fileDirectory, fileName.dropRight(FirrtlSuffix.length))
        }
        else {
          (fileDirectory, fileName)
        }
      case _ =>
        (new File("."), ir.name)
    }
    dumpFirrtlWithAnnotations(ir, Some(directory), Some(fileName))
  }

  def dumpFirrtlWithAnnotations(ir: ifirrtl.Circuit,
                                directoryOpt: Option[File] = None,
                                nameOpt: Option[String] = None
                               ): File = {
    val directory = directoryOpt.getOrElse(new File("."))
    if(!directory.isDirectory) {
      throwException(s"dumpFirrtlWithAnnotations, directory name ${directory.getPath} is not a directory")
    }
    val name = nameOpt.getOrElse(ir.name)

    val firrtlFile = new File(directory, name + FirrtlSuffix)
    val firrtlWriter = new FileWriter(firrtlFile)
    firrtlWriter.write(ifirrtl.Emitter.emit(ir))
    firrtlWriter.close()

    if(ir.annotations.nonEmpty) {
      val annotationFile = new File(directory, name + ".anno")
      val annotationWriter = new FileWriter(annotationFile)
      annotationWriter.write(ir.annotations.mkString("\n"))
      annotationWriter.close()
    }


    firrtlFile
  }

  private var target_dir: Option[String] = None
  def parseArgs(args: Array[String]): Unit = {
    for (i <- 0 until args.length) {
      if (args(i) == "--targetDir") {
        target_dir = Some(args(i + 1))
      }
    }
  }

  def targetDir(): String = { target_dir getOrElse new File(".").getCanonicalPath }

  case class ExecuteConfig(
                         writeFiles:       Boolean = true,
                         targetDir:        String  = ""
                       )

  case class ExecuteResult(
                          circuit: ifirrtl.Circuit,
                          emittedString: String
                          )

  val parser = new OptionParser[ExecuteConfig]("scopt") {
    head("scopt", "3.x")
    //    opt[Boolean]('w', "write-files") action { (x, c) =>
    //      c.copy(writeFiles = x)
    //    } text { "write chisel files to target-dir" }

    opt[String]('t', "target-dir") action { (x, c) =>
      c.copy(targetDir = x)
    } text { "directory to write firrtl and annotation files" }
    opt[String]('f', "run-firrtl") action { (x, c) =>
      c.copy(targetDir = x)
    } text { "directory to write firrtl and annotation files" }
  }

  trait ChiselFileParser { self: OptionParser[ExecuteConfig] =>
    head("scopt", "3.x")
    //    opt[Boolean]('w', "write-files") action { (x, c) =>
    //      c.copy(writeFiles = x)
    //    } text { "write chisel files to target-dir" }

    opt[String]('t', "target-dir") action { (x, c) =>
      c.copy(targetDir = x)
    } text { "directory to write firrtl and annotation files" }
  }

  def execute[T <: Module](dut: () => T, executeConfig: ExecuteConfig): ExecuteResult = {
    val circuit = elaborate(dut)
    val emittedString = emit(circuit)

    if(executeConfig.targetDir.nonEmpty) {
      dumpFirrtlWithAnnotations(circuit, Some(new File(executeConfig.targetDir)))
    }

    ExecuteResult(circuit, emittedString)
  }

  def execute[T <: Module](dut: () => T, args: Array[String]): ExecuteResult = {
    parser.parse(args, ExecuteConfig()) match {
      case Some(execConfig) =>
        val result = execute(dut, execConfig)
        result
      case _ =>
        throwException(s"Whoops no execution configuration generated for args: ${args.mkString(" ")}")
    }
  }
}
