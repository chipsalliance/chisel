// See LICENSE for license details.

package chisel3.testers

import chisel3._
import java.io._

import chisel3.experimental.RunFirrtlTransform
import firrtl.{Transform, RunFirrtlTransformAnnotation, FirrtlCircuitAnnotation, FirrtlExecutionFailure}
import firrtl.options.ExecutionOptionsManager

object TesterDriver extends BackendCompilationUtilities {

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t: () => BasicTester,
              additionalVResources: Seq[String] = Seq()): Boolean = {
    // Invoke the chisel compiler to get the circuit's IR
    val circuit = Driver.elaborate(finishWrapper(t))

    // Set up a bunch of file handlers based on a random temp filename,
    // plus the quirks of Verilator's naming conventions
    val target = circuit.name

    val path = createTestDirectory(target)
    val fname = new File(path, target)

    // For now, dump the IR out to a file
    Driver.dumpFirrtl(circuit, Some(new File(fname.toString + ".fir")))
    val firrtlCircuit = Driver.toFirrtl(circuit)

    // Copy CPP harness and other Verilog sources from resources into files
    val cppHarness =  new File(path, "top.cpp")
    copyResourceToFile("/chisel3/top.cpp", cppHarness)
    val additionalVFiles = additionalVResources.map((name: String) => {
      val mangledResourceName = name.replace("/", "_")
      val out = new File(path, mangledResourceName)
      copyResourceToFile(name, out)
      out
    })

    // Compile firrtl
    val transforms = circuit.annotations.collect { case anno: RunFirrtlTransform => anno.transformClass }.distinct
      .filterNot(_ == classOf[Transform])
      .map( RunFirrtlTransformAnnotation(_) )
    val annotations = circuit.annotations.map(_.toFirrtl).toList ++ transforms
    val args = Array("--top-name", target,
                     "--target-dir", path.getAbsolutePath,
                     "--compiler", "verilog")
    firrtl.Driver.execute(args, FirrtlCircuitAnnotation(firrtlCircuit) +: annotations) match {
      case _: FirrtlExecutionFailure => return false
      case _ =>
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
