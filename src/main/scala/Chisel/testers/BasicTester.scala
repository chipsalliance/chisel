// See LICENSE for license details.

package Chisel.testers
import Chisel._
import scala.sys.process.ProcessBuilder
import java.io._

object BasicTester extends FileSystemUtilities {
  def makeHarness(template: String => String, post: String)(f: File): File = {
    val prefix = f.toString.split("/").last
    val vf = new File(f.toString + post)
    val w = new FileWriter(vf)
    w.write(template(prefix))
    w.close()
    vf
  }

  def makeVerilogHarness = makeHarness((prefix: String) => s"""
module ${prefix}Harness;
  initial begin
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", "Harness.v") _

  def makeCppHarness = makeHarness((prefix: String) => s"""
#include "V${prefix}Harness.h"
#include "verilated.h"

vluint64_t main_time = 0;
double sc_time_stamp () { return main_time; }

int main(int argc, char **argv, char **env) {
    Verilated::commandArgs(argc, argv);
    V${prefix}Harness* top = new V${prefix}Harness;
    while (!Verilated::gotFinish()) { top->eval(); }
    delete top;
    exit(0);
}
""", ".cpp") _

}

class BasicTester extends Module {
  val io = new Bundle {
    val done = Bool()
    val error = UInt(width = 4)
  }
  io.done := Bool(false)
  io.error := UInt(0)
}
