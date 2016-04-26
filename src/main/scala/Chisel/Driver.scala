// See LICENSE for license details.

package Chisel

import scala.sys.process._
import java.io._

import internal._
import internal.firrtl._

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
    * Verilator will automatically locate the top-level module as the one among
    * all the files which are not included elsewhere. If multiple ones exist,
    * the compilation will fail.
    *
    * @param dutFile name of the DUT .v without the .v extension
    * @param dir output directory
    * @param vSources list of additional Verilog sources to compile
    * @param cppHarness C++ testharness to compile/link against
    */
  def verilogToCpp(
      dutFile: String,
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
        "+define+TOP_TYPE=V" + dutFile,
        s"+define+PRINTF_COND=!$dutFile.reset",
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

object Driver extends BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *  @return the resulting Chisel IR in the form of a Circuit (TODO: Should be FIRRTL IR)
    */
  def elaborate[T <: Module](gen: () => T): Circuit = Builder.build(Module(gen()))

  def emit[T <: Module](gen: () => T): String = elaborate(gen).emit

  def dumpFirrtl(ir: Circuit, optName: Option[File]): File = {
    val f = optName.getOrElse(new File(ir.name + ".fir"))
    val w = new FileWriter(f)
    w.write(ir.emit)
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

  def targetDir(): String = { target_dir.get }
}
