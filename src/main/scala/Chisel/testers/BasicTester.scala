// See LICENSE for license details.

package Chisel.testers
import Chisel._
import scala.sys.process.ProcessBuilder
import java.io.File

object BasicTester extends FileSystemUtilities {
  def makeVerilogHarness(prefix: String): File = {
    val template = s"""
module ${prefix}Harness;
  initial begin
    $$display("$prefix!");
    $$finish;
  end
endmodule
"""
    createTempOutputFile(prefix, "harness.v", template)
  }

  def makeCppHarness(prefix: String): File = {
    val template = s"""
#include "V$prefix.h"
#include "verilated.h"
int main(int argc, char **argv, char **env) {
    Verilated::commandArgs(argc, argv);
    V${prefix}Harness* top = new V${prefix}Harness;
    while (!Verilated::gotFinish()) { top->eval(); }
    delete top;
    exit(0);
}
"""
    createTempOutputFile(prefix, ".cpp", template)
  }

}

class BasicTester extends Module {
  val io = new Bundle {
    val done = Bool()
    val error = UInt(width = 4)
  }
  io.done := Bool(false)
  io.error := UInt(0)
}
