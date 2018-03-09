// See LICENSE for license details.

package chisel3.tester

import java.io.{File, FileWriter, IOException}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.nio.file.{FileAlreadyExistsException, Files, Paths}

import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}
import firrtl.transforms.{BlackBoxSourceHelper, BlackBoxTargetDirAnno}

class VerilatorProcessBackend[T <: Module](
      dut: T,
      cmd: Seq[String],
      rnd: scala.util.Random
  ) extends InterProcessBackend[T](dut, cmd, rnd) {
}

object VerilatorTesterShim {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  /**
    * Copies the necessary header files used for verilator compilation to the specified destination folder
    */
  def copyHeaderFiles(destinationDirPath: String): Unit = {
    new File(destinationDirPath).mkdirs()
    val simApiHFilePath = Paths.get(destinationDirPath + "/sim_api.h")
    val verilatorApiHFilePath = Paths.get(destinationDirPath + "/veri_api.h")
    try {
      Files.createFile(simApiHFilePath)
      Files.createFile(verilatorApiHFilePath)
    } catch {
      case _: FileAlreadyExistsException =>
        System.out.format("")
      case x: IOException =>
        System.err.format("createFile error: %s%n", x)
    }

    Files.copy(getClass.getResourceAsStream("/sim_api.h"), simApiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/veri_api.h"), verilatorApiHFilePath, REPLACE_EXISTING)
  }

  /**
    * Generates the Module specific verilator harness cpp file for verilator compilation
    */
  def generateHarness(dut: chisel3.Module, vcdFilePath: String): String = {
    val codeBuffer = new StringBuilder

    def pushBack(vector: String, pathName: String, width: BigInt) {
      if (width == 0) {
        // Do nothing- 0 width wires are removed
      } else if (width <= 8) {
        codeBuffer.append(s"        sim_data.$vector.push_back(new VerilatorCData(&($pathName)));\n")
      } else if (width <= 16) {
        codeBuffer.append(s"        sim_data.$vector.push_back(new VerilatorSData(&($pathName)));\n")
      } else if (width <= 32) {
        codeBuffer.append(s"        sim_data.$vector.push_back(new VerilatorIData(&($pathName)));\n")
      } else if (width <= 64) {
        codeBuffer.append(s"        sim_data.$vector.push_back(new VerilatorQData(&($pathName)));\n")
      } else {
        val numWords = (width-1)/32 + 1
        codeBuffer.append(s"        sim_data.$vector.push_back(new VerilatorWData($pathName, $numWords));\n")
      }
    }

    val (inputs, outputs) = getIOPorts(dut)
    val dutName = dut.name
    val dutApiClassName = dutName + "_api_t"
    val dutVerilatorClassName = "V" + dutName
    codeBuffer.append("#include \"%s.h\"\n".format(dutVerilatorClassName))
    codeBuffer.append("#include \"verilated.h\"\n")
    codeBuffer.append("#include \"veri_api.h\"\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("#include \"verilated_vcd_c.h\"\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("#include <iostream>\n")
    codeBuffer.append(s"class $dutApiClassName: public sim_api_t<VerilatorDataWrapper*> {\n")
    codeBuffer.append("public:\n")
    codeBuffer.append(s"    $dutApiClassName($dutVerilatorClassName* _dut) {\n")
    codeBuffer.append("        dut = _dut;\n")
    codeBuffer.append("        main_time = 0L;\n")
    codeBuffer.append("        is_exit = false;\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("        tfp = NULL;\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    void init_sim_data() {\n")
    codeBuffer.append("        sim_data.inputs.clear();\n")
    codeBuffer.append("        sim_data.outputs.clear();\n")
    codeBuffer.append("        sim_data.signals.clear();\n")
    // Replace the "." in the top-level io reference with the C pointer dereference operator.
    val dutIODotRef = dutName + "."
    val dutIOPtrRef = "dut->"
    inputs.toList foreach { case (node, name) =>
      // replaceFirst used here in case port name contains the dutName
      pushBack("inputs", name replaceFirst (dutIODotRef, dutIOPtrRef), node.getWidth)
    }
    outputs.toList foreach { case (node, name) =>
      // replaceFirst used here in case port name contains the dutName
      pushBack("outputs", name replaceFirst (dutIODotRef, dutIOPtrRef), node.getWidth)
    }
    pushBack("signals", "dut->reset", 1)
    codeBuffer.append(s"""        sim_data.signal_map["%s"] = 0;\n""".format(dut.reset.pathName))
    codeBuffer.append("    }\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("     void init_dump(VerilatedVcdC* _tfp) { tfp = _tfp; }\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("    inline bool exit() { return is_exit; }\n")

    // required for sc_time_stamp()
    codeBuffer.append("    virtual inline double get_time_stamp() {\n")
    codeBuffer.append("        return main_time;\n")
    codeBuffer.append("    }\n")

    codeBuffer.append("private:\n")
    codeBuffer.append(s"    $dutVerilatorClassName* dut;\n")
    codeBuffer.append("    bool is_exit;\n")
    codeBuffer.append("    vluint64_t main_time;\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("    VerilatedVcdC* tfp;\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("    virtual inline size_t put_value(VerilatorDataWrapper* &sig, uint64_t* data, bool force=false) {\n")
    codeBuffer.append("        return sig->put_value(data);\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline size_t get_value(VerilatorDataWrapper* &sig, uint64_t* data) {\n")
    codeBuffer.append("        return sig->get_value(data);\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline size_t get_chunk(VerilatorDataWrapper* &sig) {\n")
    codeBuffer.append("        return sig->get_num_words();\n")
    codeBuffer.append("    } \n")
    codeBuffer.append("    virtual inline void reset() {\n")
    codeBuffer.append("        dut->reset = 1;\n")
    codeBuffer.append("        step();\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline void start() {\n")
    codeBuffer.append("        dut->reset = 0;\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline void finish() {\n")
    codeBuffer.append("        dut->eval();\n")
    codeBuffer.append("        is_exit = true;\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline void step() {\n")
    codeBuffer.append("        dut->clock = 0;\n")
    codeBuffer.append("        dut->eval();\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("        if (tfp) tfp->dump(main_time);\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("        main_time++;\n")
    codeBuffer.append("        dut->clock = 1;\n")
    codeBuffer.append("        dut->eval();\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("        if (tfp) tfp->dump(main_time);\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("        main_time++;\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("    virtual inline void update() {\n")
    codeBuffer.append("        dut->_eval_settle(dut->__VlSymsp);\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("};\n")

    // The following isn't strictly required unless we emit (possibly indirectly) something
    // requiring a time-stamp (such as an assert).
    codeBuffer.append(s"static $dutApiClassName * _Top_api;\n")
    codeBuffer.append("double sc_time_stamp () { return _Top_api->get_time_stamp(); }\n")

    codeBuffer.append("int main(int argc, char **argv, char **env) {\n")
    codeBuffer.append("    Verilated::commandArgs(argc, argv);\n")
    codeBuffer.append(s"    $dutVerilatorClassName* top = new $dutVerilatorClassName;\n")
    codeBuffer.append("    std::string vcdfile = \"%s\";\n".format(vcdFilePath))
    codeBuffer.append("    std::vector<std::string> args(argv+1, argv+argc);\n")
    codeBuffer.append("    std::vector<std::string>::const_iterator it;\n")
    codeBuffer.append("    for (it = args.begin() ; it != args.end() ; it++) {\n")
    codeBuffer.append("      if (it->find(\"+waveform=\") == 0) vcdfile = it->c_str()+10;\n")
    codeBuffer.append("    }\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("    Verilated::traceEverOn(true);\n")
    codeBuffer.append("    VL_PRINTF(\"Enabling waves..\");\n")
    codeBuffer.append("    VerilatedVcdC* tfp = new VerilatedVcdC;\n")
    codeBuffer.append("    top->trace(tfp, 99);\n")
    codeBuffer.append("    tfp->open(vcdfile.c_str());\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append(s"    $dutApiClassName api(top);\n")
    codeBuffer.append("    _Top_api = &api; /* required for sc_time_stamp() */\n")
    codeBuffer.append("    api.init_sim_data();\n")
    codeBuffer.append("    api.init_channels();\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("    api.init_dump(tfp);\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("    while(!api.exit()) api.tick();\n")
    codeBuffer.append("#if VM_TRACE\n")
    codeBuffer.append("    if (tfp) tfp->close();\n")
    codeBuffer.append("    delete tfp;\n")
    codeBuffer.append("#endif\n")
    codeBuffer.append("    delete top;\n")
    codeBuffer.append("    exit(0);\n")
    codeBuffer.append("}\n")
    codeBuffer.toString()
  }

  def start[T <: Module](dutGen: => T, options: TesterOptionsManager): BackendInstance[T] = {
    val optionsManager = options
    // We need to intercept the CHIRRTL output and tweak it.
    optionsManager.chiselOptions = optionsManager.chiselOptions.copy(runFirrtlCompiler = false)
    optionsManager.makeTargetDir()
    val dir = new File(optionsManager.targetDirName)

    chisel3.Driver.execute(optionsManager, () => dutGen) match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val chirrtl = firrtl.Parser.parse(emitted)
        val dut = getTopModule(circuit).asInstanceOf[T]

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

        // Generate Harness
        val cppHarnessFileName = s"${circuit.name}-harness.cpp"
        val cppHarnessFile = new File(dir, cppHarnessFileName)
        val cppHarnessWriter = new FileWriter(cppHarnessFile)
        val vcdFile = new File(dir, s"${circuit.name}.vcd")
        val emittedStuff = generateHarness(
          dut, vcdFile.toString
        )
        cppHarnessWriter.append(emittedStuff)
        cppHarnessWriter.close()

        assert(
          chisel3.Driver.verilogToCpp(
            circuit.name,
            dir,
            vSources = Seq(),
            cppHarnessFile
          ).! == 0
        )
        assert(chisel3.Driver.cppToExe(circuit.name, dir).! == 0)

        val command = if(optionsManager.testerOptions.testCmd.nonEmpty) {
          optionsManager.testerOptions.testCmd
        }
        else {
          Seq(new File(dir, s"V${circuit.name}").toString)
        }
        val seed = optionsManager.testerOptions.testerSeed
        val rnd = new scala.util.Random(seed)
        new VerilatorProcessBackend(dut, command, rnd)
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
