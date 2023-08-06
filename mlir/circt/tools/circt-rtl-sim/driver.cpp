//===- driver.cpp - Verilator software driver -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fairly standard, boilerplate Verilator C++ simulation driver. Assumes the
// top level exposes just two signals: 'clk' and 'rst'.
//
//===----------------------------------------------------------------------===//

#include "Vtop.h"

#include "verilated_vcd_c.h"

#include "signal.h"
#include <iostream>

vluint64_t timeStamp;

// Stop the simulation gracefully on ctrl-c.
volatile bool stopSimulation = false;
void handle_sigint(int) { stopSimulation = true; }

// Called by $time in Verilog.
double sc_time_stamp() { return timeStamp; }

int main(int argc, char **argv) {
  // Register graceful exit handler.
  signal(SIGINT, handle_sigint);

  Verilated::commandArgs(argc, argv);

  size_t numCyclesToRun = 0;
  bool runForever = true;
  // Search the command line args for those we are sensitive to.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--cycles") {
      if (i + 1 < argc) {
        numCyclesToRun = std::strtoull(argv[++i], nullptr, 10);
        runForever = false;
      } else {
        std::cerr << "--cycles must be followed by number of cycles."
                  << std::endl;
        return 1;
      }
    }
  }

  // Construct the simulated module's C++ model.
  auto &dut = *new Vtop();
  char *waveformFile = getenv("SAVE_WAVE");

  VerilatedVcdC *tfp = nullptr;
  if (waveformFile) {
#ifdef TRACE
    tfp = new VerilatedVcdC();
    Verilated::traceEverOn(true);
    dut.trace(tfp, 99); // Trace 99 levels of hierarchy
    tfp->open(waveformFile);
    std::cout << "[driver] Writing trace to " << waveformFile << std::endl;
#else
    std::cout
        << "[driver] Warning: waveform file specified, but not a debug build"
        << std::endl;
#endif
  }

  std::cout << "[driver] Starting simulation" << std::endl;

  // Reset.
  dut.rst = 1;
  dut.clk = 0;

  // Run for a few cycles with reset held.
  for (timeStamp = 0; timeStamp < 8 && !Verilated::gotFinish(); timeStamp++) {
    dut.eval();
    dut.clk = !dut.clk;
    if (tfp)
      tfp->dump(timeStamp);
  }

  // Take simulation out of reset.
  dut.rst = 0;

  // Run for the specified number of cycles out of reset.
  vluint64_t endTime = timeStamp + (numCyclesToRun * 2);
  for (; (runForever || timeStamp <= endTime) && !Verilated::gotFinish() &&
         !stopSimulation;
       timeStamp++) {
    dut.eval();
    dut.clk = !dut.clk;
    if (tfp)
      tfp->dump(timeStamp);
  }

  // Tell the simulator that we're going to exit. This flushes the output(s) and
  // frees whatever memory may have been allocated.
  dut.final();
  if (tfp)
    tfp->close();

  std::cout << "[driver] Ending simulation at tick #" << timeStamp << std::endl;
  return 0;
}
