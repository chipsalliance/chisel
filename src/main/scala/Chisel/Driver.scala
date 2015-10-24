// See LICENSE for license details.

package Chisel

import scala.sys.process._
import java.io._

trait FileSystemUtilities {
  def createTempOutputFile(pre: String, post: String, contents: String): File = {
    val t = File.createTempFile(pre, post)
    val w = new FileWriter(t)
    w.write(contents)
    w.close()
    t
  }

  def createOutputFile(name: String, contents: String): ProcessBuilder =
    s"cat <<EOF $contents EOF" #>> new File(name)

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

  def firrtlToVerilog(prefix: String, dir: String): ProcessBuilder =
    Seq("firrtl", "-i", s"$prefix.fir", "-o", s"$prefix.v", "-X", "verilog")

  def verilogToCpp(dir: File, vDUT: File, vHarness: File, cppHarness: File): ProcessBuilder =
    Seq("verilator", "-Wall", "--cc", vDUT.toString, vHarness.toString, s"-M$dir", "--exe", cppHarness.toString)

  def cppToExe(prefix: String, dir: File): ProcessBuilder =
    Seq("make", "-C", dir.toString, "-j", "-f", "V$prefix.mk", "V$prefix")

  def executeExpectingFailure(prefix: String, dir: File, assertionMsg: String): Boolean = {
    var triggered = false
    val assertionFinder = ProcessLogger(line =>
                            triggered = triggered || line.contains(assertionMsg))
    Process(s"V$prefix", dir) ! assertionFinder
    triggered
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir, "Assertion Failed")
  }

}

object Driver extends FileSystemUtilities with BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit 
    *
    *  @param gen a function that creates a Module hierarchy
    *
    *  @return the resulting Chisel IR in the form of a Circuit
    */
  def elaborate[T <: Module](gen: () => T): Circuit = Builder.build(Module(gen()))
  
  def emit[T <: Module](gen: () => T): String = elaborate(gen).emit

  def dumpFirrtl(ir: Circuit, optName: Option[String]): ProcessBuilder = {
    val prefix = optName.getOrElse(ir.name)
    createOutputFile(s"$prefix.fir", ir.emit)
  }
}
