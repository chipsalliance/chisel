// See LICENSE for license details.

package Chisel.testers
import Chisel._
import scala.io.Source
import scala.sys.process._
import java.io._

object TesterDriver extends BackendCompilationUtilities {
  /** Copy the contents of a resource to a destination file.
    */
  def copyResourceToFile(name: String, file: File) {
    val in = getClass().getResourceAsStream(name)
    if (in == null) {
      throw new FileNotFoundException(s"Resource '$name'")
    }
    val out = new FileOutputStream(file)
    Iterator.continually(in.read).takeWhile(-1 !=).foreach(out.write)
    out.close()
  }

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t: () => BasicTester, additionalVResources: Seq[String] = Seq()): Boolean = {
    // Invoke the chisel compiler to get the circuit's IR
    val circuit = Driver.elaborate(finishWrapper(t))

    // Set up a bunch of file handlers based on a random temp filename,
    // plus the quirks of Verilator's naming conventions
    val target = circuit.name

    val path = createTempDirectory(target)
    val fname = File.createTempFile(target, "", path)
    val prefix = fname.toString.split("/").last

    // For now, dump the IR out to a file
    Driver.dumpFirrtl(circuit, Some(new File(fname.toString + ".fir")))

    // Copy CPP harness and other Verilog sources from resources into files
    val cppHarness =  new File(path, "top.cpp")
    copyResourceToFile("/top.cpp", cppHarness)
    val additionalVFiles = additionalVResources.map((name: String) => {
      val mangledResourceName = name.replace("/", "_")
      val out = new File(path, mangledResourceName)
      copyResourceToFile(name, out)
      out
    })

    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if ((firrtlToVerilog(prefix, path) #&&
        verilogToCpp(prefix, path, additionalVFiles, cppHarness) #&&
        cppToExe(prefix, path)).! == 0) {
      executeExpectingSuccess(prefix, path)
    } else {
      false
    }
  }
  /*
   * provides a hook for testers to implement necessary control logic for tests after the
   * implementation of the users test definition has been completed.
   * typically the finish method will inspect the users circuit and connect the tester
   * to the device under test
   */
  def finishWrapper(test: () => BasicTester): () => BasicTester = {
    () => {
      val tester = test()
      tester.finish()
      tester
    }
  }
}
