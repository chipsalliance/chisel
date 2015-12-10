// See LICENSE for license details.

package Chisel.testers
import Chisel._
import scala.sys.process._
import java.io.File

object TesterDriver extends BackendCompilationUtilities with FileSystemUtilities {
  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executeables with assertions. */
  def execute(t: () => BasicTester): Boolean = {
    // Invoke the chisel compiler to get the circuit's IR
    val circuit = Driver.elaborate(t)

    // Set up a bunch of file handlers based on a random temp filename,
    // plus the quirks of Verilator's naming conventions
    val target = circuit.name
    val fname = File.createTempFile(target, "")
    val path = fname.getParentFile.toString
    val prefix = fname.toString.split("/").last
    val dir = new File(System.getProperty("java.io.tmpdir"))
    val vDut = new File(fname.toString + ".v")
    val vH = new File(path + "/V" + prefix + ".h")
    val cppHarness = new File(fname.toString + ".cpp")

    // For now, dump the IR out to a file
    Driver.dumpFirrtl(circuit, Some(new File(fname.toString + ".fir")))

    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if (((new File(System.getProperty("user.dir") + "/src/main/resources/top.cpp") #> cppHarness) #&&
        firrtlToVerilog(prefix, dir) #&&
        verilogToCpp(prefix, dir, vDut, cppHarness, vH) #&&
        cppToExe(prefix, dir)).! == 0) {
      executeExpectingSuccess(prefix, dir)
    } else {
      false
    }
  }
}
