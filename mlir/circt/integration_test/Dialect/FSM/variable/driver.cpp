#include "Vtop.h"
#include "verilated.h"
#include <iostream>

int main(int argc, char **argv) {

  Verilated::commandArgs(argc, argv);
  auto *tb = new Vtop;

  // Post-reset start time.
  int t0 = 2;

  for (int i = 0; i < 11; i++) {
    tb->rst = i < t0;
    if (i > t0)
      std::cout << "out: " << tb->out0 << std::endl;

    // go?
    tb->in0 = i == t0;

    // Rising edge
    tb->clk = 1;
    tb->eval();

    // Falling edge
    tb->clk = 0;
    tb->eval();
  }

  exit(EXIT_SUCCESS);
}
