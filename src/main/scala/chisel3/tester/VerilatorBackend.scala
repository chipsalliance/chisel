// See LICENSE for license details.

package chisel3.tester

import java.io.{File, FileWriter}

import firrtl.util.BackendCompilationUtilities._
import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}

class VerilatorProcessBackend[T <: Module](
      dut: T,
      cmd: Seq[String],
      rnd: scala.util.Random
  ) extends InterProcessBackend[T](dut, cmd, rnd) {
}

object VerilatorTesterBackend {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  /**
    * Generates the Module specific verilator harness cpp file for verilator compilation
    */
  def generateHarness(dut: chisel3.Module, circuitName: String, dir: File, options: TesterOptionsManager): File = {
    // Generate Harness
    val cppHarnessFileName = s"${circuitName}-harness.cpp"
    val cppHarnessFile = new File(dir, cppHarnessFileName)
    val cppHarnessWriter = new FileWriter(cppHarnessFile)
    val vcdFile = new File(dir, s"${circuitName}.vcd")
    val vcdFilePath = getPosixCompatibleAbsolutePath(vcdFile.toString)
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
    codeBuffer.append(s"""
#include "${dutVerilatorClassName}.h"
#include "verilated.h"
#include "veri_api.h"
#if VM_TRACE
#include "verilated_vcd_c.h"
#endif
#include <iostream>
class $dutApiClassName: public sim_api_t<VerilatorDataWrapper*> {
    public:
    $dutApiClassName($dutVerilatorClassName* _dut) {
        dut = _dut;
        main_time = 0L;
        is_exit = false;
#if VM_TRACE
        tfp = NULL;
#endif
    }
    void init_sim_data() {
        sim_data.inputs.clear();
        sim_data.outputs.clear();
        sim_data.signals.clear();

""")
    inputs.toList foreach { case (node, name) =>
      // replaceFirst used here in case port name contains the dutName
      pushBack("inputs", s"dut->$name", node.getWidth)
    }
    outputs.toList foreach { case (node, name) =>
      // replaceFirst used here in case port name contains the dutName
      pushBack("outputs", s"dut->$name", node.getWidth)
    }
    pushBack("signals", "dut->reset", 1)
    codeBuffer.append(s"""        sim_data.signal_map["${dut.reset.pathName}"] = 0;
    }
#if VM_TRACE
     void init_dump(VerilatedVcdC* _tfp) { tfp = _tfp; }
#endif
    inline bool exit() { return is_exit; }

    // required for sc_time_stamp()
    virtual inline double get_time_stamp() {
        return main_time;
    }

    private:
    friend void sim_finish(${dutApiClassName}* api, VCDInfo& vcd);
    ${dutVerilatorClassName}* dut;
    bool is_exit;
    vluint64_t main_time;
#if VM_TRACE
    VerilatedVcdC* tfp;
#endif
    virtual inline size_t put_value(VerilatorDataWrapper* &sig, uint64_t* data, bool force=false) {
//        fprintf(stderr, "api_sim_t::put_value(0x%lx, 0x%lx, r %d)\\n", sig, *data, dut->reset);
        return sig->put_value(data);
    }
    virtual inline size_t get_value(VerilatorDataWrapper* &sig, uint64_t* data) {
        size_t nw = sig->get_value(data);
//        fprintf(stderr, "api_sim_t::get_value(0x%lx, 0x%lx, r %d)\\n", sig, *data, dut->reset);
        return nw;
    }
    virtual inline size_t get_chunk(VerilatorDataWrapper* &sig) {
        return sig->get_num_words();
    }
    virtual inline void reset() {
//        fprintf(stderr, "api_sim_t::reset (r %d)\\n", dut->reset);
        dut->reset = 1;
        step();
    }
    virtual inline void start() {
//        fprintf(stderr, "api_sim_t::start (r %d)\\n", dut->reset);
        dut->reset = 0;
    }
    virtual inline void finish() {
        dut->eval();
        is_exit = true;
    }
    virtual inline void step() {
//        fprintf(stderr, "api_sim_t::step (r %d)\\n", dut->reset);
        dut->clock = 0;
        dut->eval();
#if VM_TRACE
        if (tfp) tfp->dump(main_time);
#endif
        main_time++;
        dut->clock = 1;
        dut->eval();
#if VM_TRACE
        if (tfp) tfp->dump(main_time);
#endif
        main_time++;
    }
    virtual inline void update() {
//        fprintf(stderr, "api_sim_t::update (r %d)\\n", dut->reset);
        dut->_eval_settle(dut->__VlSymsp);
    }
    virtual inline int search(sim_data_map_key path) {
        int result = -1;
        std::map<sim_data_map_key, size_t>::iterator it = sim_data.signal_map.find(path);
        if (it != sim_data.signal_map.end()) {
            result = it->second;
        }
        return result;
    }
};

// The following isn't strictly required unless we emit (possibly indirectly) something
// requiring a time-stamp (such as an assert).
static ${dutApiClassName} * _Top_api;
double sc_time_stamp () { return _Top_api->get_time_stamp(); }

// Initialize the simulation data structures.
${dutApiClassName}* sim_init(VCDInfo& vcd) {
    ${dutVerilatorClassName}* top = new ${dutVerilatorClassName};
#if VM_TRACE
    Verilated::traceEverOn(true);
    VL_PRINTF(\"Enabling waves..\");
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open(vcd.fileName);
    vcd.vcdFP = tfp;
#endif
    static ${dutApiClassName} api(top);
    _Top_api = &api; /* required for sc_time_stamp() */
    api.init_sim_data();
#if VM_TRACE
    api.init_dump(tfp);
#endif
    return &api;
}

void sim_finish(${dutApiClassName}* api, VCDInfo& vcd) {
    ${dutVerilatorClassName}* top = api->dut;
#if VM_TRACE
    VerilatedVcdC* tfp = vcd.vcdFP;
    vcd.vcdFP = NULL;
    if (tfp) tfp->close();
    delete tfp;
#endif
    delete top;
}

void sim_peekn() {
}

// Entry point for shared execution with JVM
${dutApiClassName}* sim_jninit(VCDInfo& vcd) {
    ${dutApiClassName}* api = sim_init(vcd);
    // We want to allocate the buffers in C++ land so they are susceptible to the vagaries of GC.
    size_t inputSize = api->getInputSize() + channel_data_offset_64bw;
    size_t outputSize = api->getOutputSize() + channel_data_offset_64bw;
    ChannelBuffers *cb = new ChannelBuffers(inputSize, outputSize, channel_data_offset_64bw);
    api->init_channels(cb);
    return api;
}

// Called from Java with a pointer.
void sim_jnfinish(SIM_API* apip, VCDInfo& vcd) {
    ${dutApiClassName}* api = (${dutApiClassName}*) apip;
    sim_finish(api, vcd);
}

int main(int argc, char **argv, char **env) {
    VCDInfo vcdInfo;
    Verilated::commandArgs(argc, argv);
    std::string vcdfile = "${vcdFilePath}";
    std::vector<std::string> args(argv+1, argv+argc);
    std::vector<std::string>::const_iterator it;
    for (it = args.begin() ; it != args.end() ; it++) {
      if (it->find("+waveform=") == 0)
          vcdfile = it->c_str()+10;
    }
    vcdInfo.fileName = vcdfile.c_str();
    ${dutApiClassName}* api = sim_init(vcdInfo);
    IPCChannels ipc(api->getInputSize(), api->getOutputSize());
    api->init_channels(ipc.in_sharedMap->mapped_buffer, ipc.out_sharedMap->mapped_buffer, ipc.cmd_sharedMap->mapped_buffer);
    ipc.announce();
    while(!api->exit()) api->tick();
    sim_finish(api, vcdInfo);
    exit(0);
}
""")
    val emittedStuff = codeBuffer.toString()
    cppHarnessWriter.append(emittedStuff)
    cppHarnessWriter.close()
    cppHarnessFile
  }

  def buildSharedLibrary(circuitName: String, dir: File, cppHarnessFile: File, options: TesterOptionsManager, shimPieces: Seq[String], extraObjects: Seq[String] = Seq.empty): String = {
    assert(
      chisel3.Driver.verilogToCpp(
        circuitName,
        dir,
        vSources = Seq(),
        cppHarnessFile
      ).! == 0
    )
    assert(chisel3.Driver.cppToLib(circuitName, dir, shimPieces, extraObjects).! == 0)
    s"V${circuitName}.${sharedLibraryExtension}"
  }

  def start[T <: Module](dutGen: => T, options: TesterOptionsManager): BackendInstance[T] = {
    val (result, updatedOptions) = CSimulator.generateVerilog(dutGen, options)
    result match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val dut = getTopModule(circuit).asInstanceOf[T]
        val dir = new File(updatedOptions.targetDirName)
        val cppHarnessFile = generateHarness(
          dut, circuit.name, dir, updatedOptions
        )

        assert(
          chisel3.Driver.verilogToCpp(
            circuit.name,
            dir,
            vSources = Seq(),
            cppHarnessFile
          ).! == 0
        )
        assert(chisel3.Driver.cppToExe(circuit.name, dir).! == 0)

        val command = if(updatedOptions.testerOptions.testCmd.nonEmpty) {
          updatedOptions.testerOptions.testCmd
        }
        else {
	  val exe = if (osVersion == OSVersion.Windows) ".exe" else ""
          Seq(new File(dir, s"V${circuit.name}${exe}").toString)
        }
        val seed = updatedOptions.testerOptions.testerSeed
        val rnd = new scala.util.Random(seed)
        new VerilatorProcessBackend(dut, command, rnd)
    }
  }
}
