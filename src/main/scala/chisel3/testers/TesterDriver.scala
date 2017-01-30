// See LICENSE for license details.

package chisel3.testers

import chisel3._
import java.io._

import firrtl.{Driver => _, _}
import firrtl.util.BackendCompilationUtilities

object TesterDriver extends BackendCompilationUtilities {

  /**
    * like 'firrtlToVerilog' except it runs the process inside the same JVM
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

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t: () => BasicTester,
              additionalVResources: Seq[String] = Seq()): Boolean = {
    // Invoke the chisel compiler to get the circuit's IR
    val circuit = Driver.elaborate(finishWrapper(t))

    // Set up a bunch of file handlers based on a random temp filename,
    // plus the quirks of Verilator's naming conventions
    val target = circuit.name

    val path = createTempDirectory(target)
    val fname = new File(path, target)

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

    // Compile firrtl
    if (!compileFirrtlToVerilog(target, path)) {
      return false
    }
    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if ((verilogToCpp(target, path, additionalVFiles, cppHarness) #&&
        cppToExe(target, path)).! == 0) {
      executeExpectingSuccess(target, path)
    } else {
      false
    }
  }
  /**
    * Calls the finish method of an BasicTester or a class that extends it.
    * The finish method is a hook for code that augments the circuit built in the constructor.
    */
  def finishWrapper(test: () => BasicTester): () => BasicTester = {
    () => {
      val tester = test()
      tester.finish()
      tester
    }
  }
}
