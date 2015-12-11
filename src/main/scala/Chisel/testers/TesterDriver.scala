// See LICENSE for license details.

package Chisel.testers
import Chisel._
import scala.sys.process._
import java.io.File

object TesterDriver extends BackendCompilationUtilities {
  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executeables with assertions. */
  def execute(t: () => BasicTester, additionalVSources: Seq[File] = Seq()): Boolean = {
    // Invoke the chisel compiler to get the circuit's IR
    val circuit = Driver.elaborate(t)

    // Set up a bunch of file handlers based on a random temp filename,
    // plus the quirks of Verilator's naming conventions
    val target = circuit.name

    val path = createTempDirectory(target)
    val fname = File.createTempFile(target, "", path)
    val prefix = fname.toString.split("/").last
    val cppHarness = new File(System.getProperty("user.dir") + "/src/main/resources/top.cpp")

    // For now, dump the IR out to a file
    Driver.dumpFirrtl(circuit, Some(new File(fname.toString + ".fir")))

    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if ((firrtlToVerilog(prefix, path) #&&
        verilogToCpp(prefix, path, additionalVSources, cppHarness) #&&
        cppToExe(prefix, path)).! == 0) {
      executeExpectingSuccess(prefix, path)
    } else {
      false
    }
  }
}
