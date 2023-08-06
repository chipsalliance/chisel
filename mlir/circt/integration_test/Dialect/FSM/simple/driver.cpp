#include "Vtop.h"
#include "verilated.h"
#include <iostream>

int main(int argc, char **argv) {

  Verilated::commandArgs(argc, argv);
  auto *tb = new Vtop;

  // Post-reset start time.
  int t0 = 2;

  for (int i = 0; i < 10; i++) {
    if (i > t0)
      std::cout << "out: " << char('A' + tb->out0) << std::endl;

    // Rising edge
    tb->clk = 1;
    tb->eval();

    // Testbench
    tb->rst = i < t0;

    // t0:   Starts in A,
    // t0+1: Default transition to B

    if (i == t0 + 2) {
      // B -> C
      tb->in0 = 1;
      tb->in1 = 1;
    }

    if (i == t0 + 3) {
      // C -> B
      tb->in0 = 0;
      tb->in1 = 0;
    }

    if (i == t0 + 4 || i == t0 + 5) {
      // B -> C, C-> A
      tb->in0 = 1;
      tb->in1 = 1;
    }

    // t0+6: Default transition to B

    // Falling edge
    tb->clk = 0;
    tb->eval();
  }

  exit(EXIT_SUCCESS);
}
