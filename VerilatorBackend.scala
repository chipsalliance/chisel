// See LICENSE for license details.
package chisel.iotesters

import chisel.internal.HasId

import scala.collection.mutable.HashMap
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
class GenVerilatorCppHarness(writer: Writer, dut: Chisel.Module, vcdFilePath: String) extends firrtl.Transform {
  import firrtl._
  import firrtl.ir._
  import firrtl.Mappers._

  def loweredName(e: Expression): String = e match {
    case e: WRef => e.name
    case e: WSubField => loweredName(e.exp) + "_" + e.name
    case e: WSubIndex => loweredName(e.exp) + "_" + e.value
  }

  private def getWidth(tpe: Type): Int = tpe match {
    case UIntType(w) => w match {
      case IntWidth(width) => width.toInt
      case UnknownWidth => throw new Exception("Can't be unknown width")
    }
    case SIntType(w) => w match {
      case IntWidth(width) => width.toInt
      case UnknownWidth => throw new Exception("Can't be unknown width")
    }
    case BundleType(fields) => (fields foldLeft 0)((w, f) => w + getWidth(f.tpe))
    case VectorType(tpe, size) => size * getWidth(tpe)
    case ClockType => 1
    case UnknownType => throw new Exception("Can't be unknown type")
  }

  private def findWidths(m: DefModule) = {
    val nodes = CircuitGraph getNodes m.name
    val widthMap = HashMap[HasId, Int]()

    /* Sadly, ports disappear in verilator ...
    def getPortWidth(port: Port) = Utils.create_exps(port.name, port.tpe) map { exp =>
      nodes find (x => (CircuitGraph getName x) == loweredName(exp)) match {
        case None =>
        case Some(node) => widthMap(node) = getWidth(Utils.tpe(exp))
      } 
    } 
    */

    def loop(s: Statement): Statement = s map loop match {
      /* Sadly, wires disappear in verilator...
      case wire: DefWire if wire.name.slice(0, 2) != "T_" && wire.name.slice(0, 4) != "GEN_" =>
        Utils.create_exps(wire.name, wire.tpe) map { exp =>
          nodes find (x => (CircuitGraph getName x) == validName(loweredName(exp))) match {
            case None =>
            case Some(node) => widthMap(node) = getWidth(Utils.tpe(exp))
          }
        }
        wire
      */
      case reg: DefRegister if reg.name.slice(0, 2) != "T_" && reg.name.slice(0, 4) != "GEN_" =>
        Utils.create_exps(reg.name, reg.tpe) map { exp =>
          nodes find (x => (CircuitGraph getName x) == validName(loweredName(exp))) match {
            case None =>
            case Some(node) => widthMap(node) = getWidth(Utils.tpe(exp))
          }
        }
        reg
      case prim: DefNode if prim.name.slice(0, 2) != "T_" && prim.name.slice(0, 4) != "GEN_" =>
        nodes find (x => (CircuitGraph getName x) == validName(prim.name)) match {
          case None =>
          case Some(node) => widthMap(node) = getWidth(Utils.tpe(prim.value))
        }
        prim
      case mem: DefMemory if mem.name.slice(0, 2) != "T_" && mem.name.slice(0, 4) != "GEN_" => mem.dataType match {
        case _: UIntType | _: SIntType =>
          nodes find (x => (CircuitGraph getName x) == validName(mem.name)) match {
            case None =>
            case Some(node) => widthMap(node) = getWidth(mem.dataType)
          }
          mem
        case _ => mem
        }
      case _ => s
    }

    m match {
      case m: ExtModule =>
      case m: Module => loop(m.body)
    }

    widthMap.toMap
  }

  private def pushBack(vector: String, pathName: String, width: Int) {
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

  def execute(circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult = {
    val (inputs, outputs) = getPorts(dut)
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
    inputs.toList foreach { node =>
      val pathName = CircuitGraph getPathName (node, "->") replace (dutName, "dut") replace ("$", "__024")
      pushBack("inputs", pathName, node.getWidth)
    }
    outputs.toList foreach { node =>
      val pathName = CircuitGraph getPathName (node, "->") replace (dutName, "dut") replace ("$", "__024")  
      pushBack("outputs", pathName, node.getWidth)
    }
    (CircuitGraph.nodes foldLeft 0){ (id, node) =>
      val pathName = CircuitGraph getPathName (node, "__DOT__") replace (dutName, "v") replace ("$", "__024")
      val signalName = CircuitGraph getPathName (node, ".")
      try {
        node match {
          case mem: Chisel.MemBase[_] =>
            writer.write(s"        for (size_t i = 0 ; i < ${mem.length} ; i++) {\n")
            pushBack("signals", s"dut->${pathName}[i]", widthMap(node))
            writer.write(s"          ostringstream oss;\n")
            writer.write(s"""          oss << "${signalName}" << "[" << i << "]";\n""")
            writer.write(s"          sim_data.signal_map[oss.str()] = $id + i;\n")
            writer.write(s"        }\n")
            id + mem.length
          case _ =>
            pushBack("signals", s"dut->$pathName", widthMap(node))
            writer.write(s"""        sim_data.signal_map["${signalName}"] = $id;\n""")
            id + 1
        }
      } catch {
        // For debugging
        case e: java.util.NoSuchElementException =>
          println(pathName)
          throw e
      }
    }
    writer.write("    }\n")
    writer.write("#if VM_TRACE\n")
    writer.write("     void init_dump(VerilatedVcdC* _tfp) { tfp = _tfp; }\n")
    writer.write("#endif\n")
    writer.write("    inline bool exit() { return is_exit; }\n")
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
    writer.write("        dut->clk = 1;\n")
    writer.write("        dut->eval();\n")
    writer.write("        dut->reset = 0;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void start() { }\n")
    writer.write("    virtual inline void finish() {\n")
    writer.write("        dut->eval();\n")
    writer.write("        is_exit = true;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void step() {\n")
    writer.write("        dut->clk = 0;\n")
    writer.write("        dut->eval();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("        if (tfp) tfp->dump(main_time);\n")
    writer.write("#endif\n")
    writer.write("        main_time++;\n")
    writer.write("        dut->clk = 1;\n")
    writer.write("        dut->eval();\n")
    writer.write("#if VM_TRACE\n")
    writer.write("        if (tfp) tfp->dump(main_time);\n")
    writer.write("#endif\n")
    writer.write("        main_time++;\n")
    writer.write("    }\n")
    writer.write("    virtual inline void update() {\n")
    writer.write("        dut->clk = 0;\n")
    writer.write("        dut->eval();\n")
    writer.write("    }\n")
    writer.write("};\n")
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

class VerilatorCppHarnessCompiler(
    dut: Chisel.Module, vcdFilePath: String) extends firrtl.Compiler {
  def transforms(w: Writer) = Seq(
    new firrtl.Chisel3ToHighFirrtl,
    new firrtl.IRToWorkingIR,
    new firrtl.ResolveAndCheck,
    new GenVerilatorCppHarness(w, dut, vcdFilePath)
  )
}

private[iotesters] object setupVerilatorBackend {
  def apply(dutGen: ()=> chisel.Module): Backend = {
    val rootDirPath = new File(".").getCanonicalPath()
    val testDirPath = s"${rootDirPath}/test_run_dir"
    val dir = new File(testDirPath)
    dir.mkdirs()

    CircuitGraph.clear
    val circuit = chisel.Driver.elaborate(dutGen)
    val dut = CircuitGraph construct circuit

    // Dump FIRRTL for debugging
    val firrtlIRFilePath = s"${testDirPath}/${circuit.name}.ir"
    chisel.Driver.dumpFirrtl(circuit, Some(new File(firrtlIRFilePath)))
    // Generate Verilog
    val verilogFilePath = s"${testDirPath}/${circuit.name}.v"
    firrtl.Driver.compile(firrtlIRFilePath, verilogFilePath, new firrtl.VerilogCompiler)

    val verilogFileName = verilogFilePath.split("/").last
    val cppHarnessFileName = "classic_tester_top.cpp"
    val cppHarnessFilePath = s"${testDirPath}/${cppHarnessFileName}"
    val cppBinaryPath = s"${testDirPath}/V${circuit.name}"
    val vcdFilePath = s"${testDirPath}/${circuit.name}.vcd"

    // Generate Harness
    copyVerilatorHeaderFiles(testDirPath)
    firrtl.Driver.compile(firrtlIRFilePath, cppHarnessFilePath, new VerilatorCppHarnessCompiler(dut, vcdFilePath))

    chisel.Driver.verilogToCpp(verilogFileName.split("\\.")(0), dut.name, new File(testDirPath), Seq(), new File(cppHarnessFileName)).!
    chisel.Driver.cppToExe(verilogFileName.split("\\.")(0), new File(testDirPath)).!

    new VerilatorBackend(dut, List(cppBinaryPath))
  }
}

private[iotesters] class VerilatorBackend(
                                          dut: Chisel.Module, 
                                          cmd: List[String],
                                          verbose: Boolean = true,
                                          logger: PrintStream = System.out,
                                          _base: Int = 16,
                                          _seed: Long = System.currentTimeMillis) extends Backend(_seed) {

  val simApiInterface = new SimApiInterface(dut, cmd, logger)

  def poke(signal: HasId, value: BigInt, off: Option[Int]) {
    val idx = off map (x => s"[$x]") getOrElse ""
    val path = "%s%s".format(CircuitGraph getPathName (signal, "."), idx)
    poke(path, value)
  }

  def peek(signal: HasId, off: Option[Int]) = {
    val idx = off map (x => s"[$x]") getOrElse ""
    val path = "%s%s".format(CircuitGraph getPathName (signal, "."), idx)
    peek(path)
  }

  def poke(path: String, value: BigInt) {
    if (verbose) logger println s"  POKE ${path} <- ${bigIntToStr(value, _base)}"
    simApiInterface.poke(path, value)
  }

  def peek(path: String) = {
    val result = simApiInterface.peek(path) getOrElse BigInt(rnd.nextInt)
    if (verbose) logger println s"  PEEK ${path} -> ${bigIntToStr(result, _base)}"
    result
  }

  def expect(signal: HasId, expected: BigInt, msg: => String = "") = {
    val name = CircuitGraph getPathName (signal, ".")
    val got = simApiInterface.peek(name) getOrElse BigInt(rnd.nextInt)
    val good = got == expected
    if (verbose) logger println s"""${msg}  EXPECT ${name} -> ${bigIntToStr(got, _base)} == ${bigIntToStr(expected, _base)} ${if (good) "PASS" else "FAIL"}"""
    good
  }

  def step(n: Int): Unit = {
    simApiInterface.step(n)
  }

  def reset(n: Int = 1): Unit = {
    simApiInterface.reset(n)
  }

  def finish: Unit = {
    simApiInterface.finish
  }
}

