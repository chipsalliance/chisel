// See LICENSE for license details.

package chisel3.tester

import java.io.{File, FileWriter, IOException}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

import chisel3.tester.VerilatorTesterBackend.getTopModule
import chisel3.{ChiselExecutionResult, ChiselExecutionSuccess, Module}
import firrtl.{ChirrtlForm, CircuitState}
import firrtl.transforms.BlackBoxTargetDirAnno

object CSimulator {

  /**
    * Copies the necessary header files used for verilator compilation to the specified destination folder
    */
  def copyHeaderFiles(destinationDirPath: String): Unit = {
    val dir = new File(destinationDirPath)
    dir.mkdirs()
    val fileNames = Array("sim_api.h", "veri_api.h") ++ Array("chisel3_tester_JNITestAPI.h", "chisel3_tester_JNITestAPI.cpp")
    try {
      for (fileName <- fileNames) {
        val path = Paths.get(destinationDirPath, fileName)
        Files.copy(getClass.getResourceAsStream("/chisel3/tester/" + fileName), path, REPLACE_EXISTING)
      }
    } catch {
      case _: FileAlreadyExistsException =>
        System.out.format("")
      case x: IOException =>
        System.err.format("Files.copy error: %s%n", x)
    }
  }

  def generateVerilog[T <: Module](dutGen: => T, options: TesterOptionsManager): (ChiselExecutionResult, TesterOptionsManager) = {
    val optionsManager = options
    // We need to intercept the CHIRRTL output and tweak it.
    optionsManager.chiselOptions = optionsManager.chiselOptions.copy(runFirrtlCompiler = false)
    optionsManager.makeTargetDir()
    val dir = new File(optionsManager.targetDirName)

    val result = chisel3.Driver.execute(optionsManager, () => dutGen)
    result match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val chirrtl = firrtl.Parser.parse(emitted)

        // This makes sure annotations for command line options get created
        firrtl.Driver.loadAnnotations(optionsManager)

        /*
        The following block adds an annotation that tells the black box helper where the
        current build directory is, so that it can copy verilog resource files into the right place
         */
        val annotations = optionsManager.firrtlOptions.annotations ++
          List(BlackBoxTargetDirAnno(optionsManager.targetDirName))

        val transforms = optionsManager.firrtlOptions.customTransforms

        // Generate Verilog
        val verilogFile = new File(dir, s"${circuit.name}.v")
        val verilogWriter = new FileWriter(verilogFile)

        val compileResult = (new firrtl.VerilogCompiler).compileAndEmit(
          CircuitState(chirrtl, ChirrtlForm, annotations),
          customTransforms = transforms
        )
        val compiledStuff = compileResult.getEmittedCircuit
        verilogWriter.write(compiledStuff.value)
        verilogWriter.close()

        copyHeaderFiles(optionsManager.targetDirName)
      case _ =>
        throw new Exception("Problem with compilation")
    }
    (result, optionsManager)
  }
}
