// See LICENSE for license details.

package Chisel

import scala.sys.process._
import java.io._

import internal._
import firrtl._

trait FileSystemUtilities {
  def writeTempFile(pre: String, post: String, contents: String): File = {
    val t = File.createTempFile(pre, post)
    val w = new FileWriter(t)
    w.write(contents)
    w.close()
    t
  }

  // This "fire-and-forgets" the method, which can be lazily read through
  // a Stream[String], and accumulates all errors on a StringBuffer
  def sourceFilesAt(baseDir: String): (Stream[String], StringBuffer) = {
    val buffer = new StringBuffer()
    val cmd = Seq("find", baseDir, "-name", "*.scala", "-type", "f")
    val lines = cmd lines_! ProcessLogger(buffer append _)
    (lines, buffer)
  }
}

trait BackendCompilationUtilities {
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
    * @param prefix output class name
    * @param dir output directory
    * @oaran vDut .v file containing the top-level DUR
    * @param vSources list of additional Verilog sources to compile
    * @param cppHarness C++ testharness to compile/link against
    * @param vH .h file to generate
    */
  def verilogToCpp(
      prefix: String,
      dir: File,
      vDut: File,
      vSources: Seq[File],
      cppHarness: File,
      vH: File): ProcessBuilder =
    Seq("verilator",
        "--cc", vDut.toString) ++
        vSources.map(file => Seq("-v", file.toString)).flatten ++
        Seq("--assert",
            "--Wno-fatal",
            "--trace",
            "-O2",
            "+define+TOP_TYPE=V" + prefix,
            "-CFLAGS", s"""-Wno-undefined-bool-conversion -O2 -DTOP_TYPE=V$prefix -include ${vH.toString}""",
            "-Mdir", dir.toString,
            "--exe", cppHarness.toString)

  def cppToExe(prefix: String, dir: File): ProcessBuilder =
    Seq("make", "-C", dir.toString, "-j", "-f", s"V${prefix}.mk", s"V${prefix}")

  def executeExpectingFailure(
      prefix: String,
      dir: File,
      assertionMsg: String = "Assertion failed"): Boolean = {
    var triggered = false
    val e = Process(s"./V${prefix}", dir) ! ProcessLogger(line =>
      triggered = triggered || line.contains(assertionMsg))
    triggered
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir)
  }

}

object Driver extends FileSystemUtilities with BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit
    *
    *  @param gen a function that creates a Module hierarchy
    *
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
}
