#include <verilated.h>
#include <iostream>

#if VM_TRACE
# include <verilated_vcd_c.h>	// Trace file format header
#endif

// Override Verilator definition so first $finish ends simulation
// Note: VL_USER_FINISH needs to be defined when compiling Verilator code
void vl_finish(const char* filename, int linenum, const char* hier) {
  Verilated::flushCall();
  exit(0);
}

using namespace std;

//VGCDTester *top;
TOP_TYPE *top;

vluint64_t main_time = 0;       // Current simulation time
        // This is a 64-bit integer to reduce wrap over issues and
        // allow modulus.  You can also use a double, if you wish.

double sc_time_stamp () { // Called by $time in Verilog
  return main_time;       // converts to double, to match
                          // what SystemC does
}

// TODO Provide command-line options like vcd filename, timeout count, etc.
const long timeout = 100000000L;

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);   // Remember args
  top = new TOP_TYPE;

#if VM_TRACE			// If verilator was invoked with --trace
    Verilated::traceEverOn(true);	// Verilator must compute traced signals
    VL_PRINTF("Enabling waves...\n");
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace (tfp, 99);	// Trace 99 levels of hierarchy
    tfp->open ("dump.vcd");	// Open the dump file
#endif


  top->reset = 1;

  cout << "Starting simulation!\n";

  while (!Verilated::gotFinish() && main_time < timeout) {
    // Deassert reset on timestep 10.  This needs to occur before the clock
    // asserts on timestep 11 because there is a single call to top->eval() in
    // this loop.  Verilator evaluates sequential logic (always blocks) before
    // combinational logic during top->eval().  Staggering the reset update is
    // necessary to produce the same simulation behavior independent of whether
    // or not the generated Verilog puts synchronous reset logic inside or
    // outside its associated always block.
    if (main_time == 10) {
      top->reset = 0;
    }
    if ((main_time % 10) == 1) {
      top->clock = 1;       // Toggle clock
    }
    if ((main_time % 10) == 6) {
      top->clock = 0;
    }
    top->eval();               // Evaluate model
#if VM_TRACE
	if (tfp) tfp->dump (main_time);	// Create waveform trace for this timestamp
#endif
    main_time++;               // Time passes...
  }

  if (main_time >= timeout) {
      cout << "Simulation terminated by timeout at time " << main_time <<
              " (cycle " << main_time / 10 << ")"<< endl;
      return -1;
  } else {
      cout << "Simulation completed at time " << main_time <<
              " (cycle " << main_time / 10 << ")"<< endl;
  }

  // Run for 10 more clocks
  vluint64_t end_time = main_time + 100;
  while (main_time < end_time) {
    if ((main_time % 10) == 1) {
      top->clock = 1;       // Toggle clock
    }
    if ((main_time % 10) == 6) {
      top->clock = 0;
    }
    top->eval();               // Evaluate model
#if VM_TRACE
	if (tfp) tfp->dump (main_time);	// Create waveform trace for this timestamp
#endif
    main_time++;               // Time passes...
  }

#if VM_TRACE
    if (tfp) tfp->close();
#endif
}
