// See LICENSE for license details.
package chisel3.iotesters

import chisel3.internal.InstanceId

import scala.util.Random
import java.io.{File, Writer, FileWriter, PrintStream, IOException}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

/**
  * Copies the necessary header files used for verilator compilation to the specified destination folder
  */
object copyVerilatorHeaderFiles {
  def apply(destinationDirPath: String): Unit = {
    new File(destinationDirPath).mkdirs()
    val simApiHFilePath = Paths.get(destinationDirPath + "/sim_api.h")
    val verilatorApiHFilePath = Paths.get(destinationDirPath + "/veri_api.h")
    try {
      Files.createFile(simApiHFilePath)
      Files.createFile(verilatorApiHFilePath)
    } catch {
      case x: FileAlreadyExistsException =>
        System.out.format("")
      case x: IOException => {
        System.err.format("createFile error: %s%n", x)
      }
    }

    Files.copy(getClass.getResourceAsStream("/sim_api.h"), simApiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/veri_api.h"), verilatorApiHFilePath, REPLACE_EXISTING)
  }
}

/**
  * Generates the Module specific verilator harness cpp file for verilator compilation
  */
class GenVerilatorCppHarness(writer: Writer, dut: Chisel.Module,
    nodes: Seq[InstanceId], vcdFilePath: String) extends firrtl.Transform {
  import firrtl._
  import firrtl.ir._
  import firrtl.Mappers._
  import firrtl.Annotations.AnnotationMap
  import firrtl.Utils.create_exps
  import firrtl.passes.bitWidth

  private def findWidths(m: DefModule) = {
    type WidthMap = collection.mutable.ArrayBuffer[(InstanceId, BigInt)]
    val modNodes = nodes filter (_.parentModName == m.name)
    val widthMap = new WidthMap

    /* Sadly, ports disappear in verilator ...
    def loop_port(port: Port) = {
      widthMap ++= (create_exps(port.name, port.tpe) flatMap (exp =>
        modNodes filter (_.instanceName == exp.serialize) map (_ -> bitWidth(exp.tpe)))
      port
    }
    */

    def loop(s: Statement): Statement = {
      s match {
        /* Sadly, wires disappear in verilator...
        case s: DefWire if s.name.slice(0, 2) != "T_" && s.name.slice(0, 4) != "GEN_" =>
          widthMap ++= (create_exps(s.name, s.tpe) flatMap (exp =>
            modNodes filter (_.instanceName == exp.serialize) map (_ -> bitWidth(exp.tpe)))
        */
        case s: DefRegister if s.name.slice(0, 2) != "T_" && s.name.slice(0, 4) != "GEN_" =>
          widthMap ++= (create_exps(s.name, s.tpe) flatMap (exp =>
            modNodes filter (_.instanceName == exp.serialize) map (_ -> bitWidth(exp.tpe))))
        case s: DefNode if s.name.slice(0, 2) != "T_" && s.name.slice(0, 4) != "GEN_" =>
          widthMap ++= (create_exps(s.name, s.value.tpe) flatMap (exp =>
            modNodes filter (_.instanceName == exp.serialize) map (_ -> bitWidth(exp.tpe))))
        case s: DefMemory if s.name.slice(0, 2) != "T_" && s.name.slice(0, 4) != "GEN_" =>
          widthMap ++= (modNodes filter (_.instanceName == s.name) map (_ -> bitWidth(s.dataType)))
        case _ =>
      }
      s map loop
    }

    m map loop
    widthMap.toSeq
  }

  private def pushBack(vector: String, pathName: String, width: BigInt) {
    if (width <= 8) {
      writer.write(s"        sim_data.$vector.push_back(new VerilatorCData(&(${pathName})));\n")
    } else if (width <= 16) {
      writer.write(s"        sim_data.$vector.push_back(new VerilatorSData(&(${pathName})));\n")
    } else if (width <= 32) {
      writer.write(s"        sim_data.$vector.push_back(new VerilatorIData(&(${pathName})));\n")
    } else if (width <= 64) {
      writer.write(s"        sim_data.$vector.push_back(new VerilatorQData(&(${pathName})));\n")
    } else {
      val numWords = (width-1)/32 + 1
      writer.write(s"        sim_data.$vector.push_back(new VerilatorWData(${pathName}, ${numWords}));\n")
    }
  }

  def execute(circuit: Circuit, annotationMap: AnnotationMap): TransformResult = {
    val (inputs, outputs) = getPorts(dut, "->")
    val dutName = dut.name
    val dutApiClassName = dutName + "_api_t"
    val dutVerilatorClassName = "V" + dutName
    val widthMap = (circuit.modules flatMap findWidths).toMap
    writer.write("#include \"%s.h\"\n".format(dutVerilatorClassName))
    writer.write("#include \"verilated.h\"\n")
    writer.write("#include \"veri_api.h\"\n")
    writer.write("#if VM_TRACE\n")
    writer.write("#include \"verilated_vcd_c.h\"\n")
    writer.write("#endif\n")
    writer.write("#include <iostream>\n")
    writer.write(s"class ${dutApiClassName}: public sim_api_t<VerilatorDataWrapper*> {\n")
    writer.write("public:\n")
    writer.write(s"    ${dutApiClassName}(${dutVerilatorClassName}* _dut) {\n")
    writer.write("        dut = _dut;\n")
    writer.write("        main_time = 0L;\n")
    writer.write("        is_exit = false;\n")
    writer.write("#if VM_TRACE\n")
    writer.write("        tfp = NULL;\n")
    writer.write("#endif\n")
    writer.write("    }\n")
    writer.write("    void init_sim_data() {\n")
    writer.write("        sim_data.inputs.clear();\n")
    writer.write("        sim_data.outputs.clear();\n")
    writer.write("        sim_data.signals.clear();\n")
    inputs.toList foreach { case (node, name) =>
      pushBack("inputs", name replace (dutName, "dut"), node.getWidth)
    }
    outputs.toList foreach { case (node, name) =>
      pushBack("outputs", name replace (dutName, "dut"), node.getWidth)
    }
    pushBack("signals", "dut->reset", 1)
    writer.write(s"""        sim_data.signal_map["%s"] = 0;\n""".format(dut.reset.pathName))
    (nodes foldLeft 1){ (id, node) =>
      val instanceName = s"%s.%s".format(node.parentPathName, validName(node.instanceName))
      val pathName = instanceName replace (".", "__DOT__") replace ("$", "__024")
      try {
        node match {
          case mem: Chisel.MemBase[_] =>
            writer.write(s"        for (size_t i = 0 ; i < ${mem.length} ; i++) {\n")
            pushBack("signals", s"dut->${pathName}[i]", widthMap(node))
            writer.write(s"          ostringstream oss;\n")
            writer.write(s"""          oss << "${instanceName}" << "[" << i << "]";\n""")
            writer.write(s"          sim_data.signal_map[oss.str()] = $id + i;\n")
            writer.write(s"        }\n")
            id + mem.length
          case _ =>
//            pushBack("signals", s"dut->$pathName", widthMap(node))
//            writer.write(s"""        sim_data.signal_map["${instanceName}"] = $id;\n""")
            id + 1
        }
      } catch {
        // For debugging
        case e: java.util.NoSuchElementException =>
          println(s"error with $id: $instanceName")
          throw e
      }
    }
    writer.write("    }\n")
    writer.write("#if VM_TRACE\n")
    writer.write("     void init_dump(VerilatedVcdC* _tfp) { tfp = _tfp; }\n")
    writer.write("#endif\n")
    writer.write("    inline bool exit() { return is_exit; }\n")

    // required for sc_time_stamp()
    writer.write("    virtual inline double get_time_stamp() {\n")
    writer.write("        return main_time;\n")
    writer.write("    }\n")

    writer.write("private:\n")
    writer.write(s"    ${dutVerilatorClassName}* dut;\n")
    writer.write("    bool is_exit;\n")
    writer.write("    vluint64_t main_time;\n")
    writer.write("#if VM_TRACE\n")
    writer.write("    VerilatedVcdC* tfp;\n")
    writer.write("#endif\n")
    writer.write("    virtual inline size_t put_value(VerilatorDataWrapper* &sig, uint64_t* data, bool force=false) {\n")
    writer.write("        return sig->put_value(data);\n")
    writer.write("    }\n")
    writer.write("    virtual inline size_t get_value(VerilatorDataWrapper* &sig, uint64_t* data) {\n")
    writer.write("        return sig->get_value(data);\n")
    writer.write("    }\n")
    writer.write("    virtual inline size_t get_chunk(VerilatorDataWrapper* &sig) {\n")
    writer.write("        return sig->get_num_words();\n")
    writer.write("    } \n")
    writer.write("    virtual inline void reset() {\n")
    writer.write("        dut->reset = 1;\n")
    writer.write("        step();\n")
    writer.write("    }\n")
    writer.write("    virtual inline void start() {\n")
    writer.write("        dut->reset = 0;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void finish() {\n")
    writer.write("        dut->eval();\n")
    writer.write("        is_exit = true;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void step() {\n")
    writer.write("        dut->clock = 0;\n")
    writer.write("        dut->eval();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("        if (tfp) tfp->dump(main_time);\n")
    writer.write("#endif\n")
    writer.write("        main_time++;\n")
    writer.write("        dut->clock = 1;\n")
    writer.write("        dut->eval();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("        if (tfp) tfp->dump(main_time);\n")
    writer.write("#endif\n")
    writer.write("        main_time++;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void update() {\n")
    writer.write("        dut->_eval_settle(dut->__VlSymsp);\n")
    writer.write("    }\n")
    writer.write("};\n")

    // The following isn't strictly required unless we emit (possibly indirectly) something
    // requiring a time-stamp (such as an assert).
    writer.write(s"static ${dutApiClassName} * _Top_api;\n")
    writer.write("double sc_time_stamp () { return _Top_api->get_time_stamp(); }\n")

    writer.write("int main(int argc, char **argv, char **env) {\n")
    writer.write("    Verilated::commandArgs(argc, argv);\n")
    writer.write(s"    ${dutVerilatorClassName}* top = new ${dutVerilatorClassName};\n")
    writer.write("    std::string vcdfile = \"%s\";\n".format(vcdFilePath))
    writer.write("    std::vector<std::string> args(argv+1, argv+argc);\n")
    writer.write("    std::vector<std::string>::const_iterator it;\n")
    writer.write("    for (it = args.begin() ; it != args.end() ; it++) {\n")
    writer.write("      if (it->find(\"+waveform=\") == 0) vcdfile = it->c_str()+10;\n")
    writer.write("    }\n")
    writer.write("#if VM_TRACE\n")
    writer.write("    Verilated::traceEverOn(true);\n")
    writer.write("    VL_PRINTF(\"Enabling waves..\");\n")
    writer.write("    VerilatedVcdC* tfp = new VerilatedVcdC;\n")
    writer.write("    top->trace(tfp, 99);\n")
    writer.write("    tfp->open(vcdfile.c_str());\n")
    writer.write("#endif\n")
    writer.write(s"    ${dutApiClassName} api(top);\n")
    writer.write("    _Top_api = &api; /* required for sc_time_stamp() */\n")
    writer.write("    api.init_sim_data();\n")
    writer.write("    api.init_channels();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("    api.init_dump(tfp);\n")
    writer.write("#endif\n")
    writer.write("    while(!api.exit()) api.tick();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("    if (tfp) tfp->close();\n")
    writer.write("    delete tfp;\n")
    writer.write("#endif\n")
    writer.write("    delete top;\n")
    writer.write("    exit(0);\n")
    writer.write("}\n")
    writer.close()
    TransformResult(circuit)
  }
}

class VerilatorCppHarnessCompiler(dut: Chisel.Module,
    nodes: Seq[InstanceId], vcdFilePath: String) extends firrtl.Compiler {
  def transforms(w: Writer) = Seq(
    new firrtl.Chisel3ToHighFirrtl,
    new firrtl.IRToWorkingIR,
    new firrtl.ResolveAndCheck,
    new GenVerilatorCppHarness(w, dut, nodes, vcdFilePath)
  )
}

private[iotesters] object setupVerilatorBackend {
  def apply[T <: chisel3.Module](dutGen: () => T, optionsManager: TesterOptionsManager): (T, Backend) = {
    optionsManager.makeTargetDir()
    val dir = new File(optionsManager.targetDirName)

    // Generate CHIRRTL
    val circuit = chisel3.Driver.elaborate(dutGen)
    val chirrtl = firrtl.Parser.parse(chisel3.Driver.emit(circuit))
    val dut = getTopModule(circuit).asInstanceOf[T]
    val nodes = getChiselNodes(circuit)

    // Generate Verilog
    val verilogFile = new File(dir, s"${circuit.name}.v")
    val verilogWriter = new FileWriter(verilogFile)
    val annotation = new firrtl.Annotations.AnnotationMap(Seq(
      new firrtl.passes.memlib.InferReadWriteAnnotation(circuit.name, firrtl.Annotations.TransID(-1))))
    (new firrtl.VerilogCompiler).compile(chirrtl, annotation, verilogWriter)
    verilogWriter.close

    val cppHarnessFileName = s"${circuit.name}-harness.cpp"
    val cppHarnessFile = new File(dir, cppHarnessFileName)
    val cppHarnessWriter = new FileWriter(cppHarnessFile)
    val vcdFile = new File(dir, s"${circuit.name}.vcd")
    val harnessCompiler = new VerilatorCppHarnessCompiler(dut, nodes, vcdFile.toString)
    copyVerilatorHeaderFiles(dir.toString)
    harnessCompiler.compile(chirrtl, annotation, cppHarnessWriter)
    cppHarnessWriter.close
    assert(chisel3.Driver.verilogToCpp(circuit.name, circuit.name, dir, Seq(), new File(cppHarnessFileName)).! == 0)
    assert(chisel3.Driver.cppToExe(circuit.name, dir).! == 0)

    (dut, new VerilatorBackend(dut, Seq((new File(dir, s"V${circuit.name}")).toString)))
  }
}

private[iotesters] class VerilatorBackend(dut: Chisel.Module, 
                                          cmd: Seq[String],
                                          _seed: Long = System.currentTimeMillis) extends Backend(_seed) {

  private[iotesters] val simApiInterface = new SimApiInterface(dut, cmd)

  def poke(signal: InstanceId, value: BigInt, off: Option[Int])
          (implicit logger: PrintStream, verbose: Boolean, base: Int) {
    val idx = off map (x => s"[$x]") getOrElse ""
    val path = s"${signal.parentPathName}.${validName(signal.instanceName)}$idx"
    poke(path, value)
  }

  def peek(signal: InstanceId, off: Option[Int])
          (implicit logger: PrintStream, verbose: Boolean, base: Int): BigInt = {
    val idx = off map (x => s"[$x]") getOrElse ""
    val path = s"${signal.parentPathName}.${validName(signal.instanceName)}$idx"
    peek(path)
  }

  def expect(signal: InstanceId, expected: BigInt, msg: => String)
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean = {
    val path = s"${signal.parentPathName}.${validName(signal.instanceName)}"
    expect(path, expected, msg)
  }

  def poke(path: String, value: BigInt)
          (implicit logger: PrintStream, verbose: Boolean, base: Int) {
    if (verbose) logger println s"  POKE ${path} <- ${bigIntToStr(value, base)}"
    simApiInterface.poke(path, value)
  }

  def peek(path: String)
          (implicit logger: PrintStream, verbose: Boolean, base: Int): BigInt = {
    val result = simApiInterface.peek(path) getOrElse BigInt(rnd.nextInt)
    if (verbose) logger println s"  PEEK ${path} -> ${bigIntToStr(result, base)}"
    result
  }

  def expect(path: String, expected: BigInt, msg: => String = "")
            (implicit logger: PrintStream, verbose: Boolean, base: Int): Boolean = {
    val got = simApiInterface.peek(path) getOrElse BigInt(rnd.nextInt)
    val good = got == expected
    if (verbose) logger println (
      s"""${msg}  EXPECT ${path} -> ${bigIntToStr(got, base)} == """ +
      s"""${bigIntToStr(expected, base)} ${if (good) "PASS" else "FAIL"}""")
    good
  }

  def step(n: Int)(implicit logger: PrintStream): Unit = {
    simApiInterface.step(n)
  }

  def reset(n: Int = 1): Unit = {
    simApiInterface.reset(n)
  }

  def finish(implicit logger: PrintStream): Unit = {
    simApiInterface.finish
  }
}

