// See LICENSE for license details.

package chiselTests

import chisel3.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._
import java.io.File
import firrtl.util.BackendCompilationUtilities

class HarnessSpec extends ChiselPropSpec
  with BackendCompilationUtilities {

  def makeTrivialVerilog: (File => File) = makeHarness((prefix: String) => s"""
module ${prefix};
  initial begin
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", ".v") _

  def makeFailingVerilog: (File => File) = makeHarness((prefix: String) => s"""
module $prefix;
  initial begin
    assert (1 == 0) else $$error("My specific, expected error message!");
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", ".v") _

  def makeCppHarness: (File => File) = makeHarness((prefix: String) => s"""
#include "V$prefix.h"
#include "verilated.h"

vluint64_t main_time = 0;
double sc_time_stamp () { return main_time; }

int main(int argc, char **argv, char **env) {
    Verilated::commandArgs(argc, argv);
    V${prefix}* top = new V${prefix};
    while (!Verilated::gotFinish()) { top->eval(); }
    delete top;
    exit(0);
}

void vl_finish(const char* filename, int linenum, const char* hier) {
  Verilated::flushCall();
  exit(0);
}
""", ".cpp") _

  /** Compiles a C++ emulator from Verilog and returns the path to the
    * executable and the executable filename as a tuple.
    */
  def simpleHarnessBackend(make: File => File): (File, String) = {
    val target = "test"
    val path = createTestDirectory(target)
    val fname = new File(path, target)

    val cppHarness = makeCppHarness(fname)

    make(fname)
    verilogToCpp(target, path, Seq(), cppHarness).!
    cppToExe(target, path).!
    (path, target)
  }

  property("Test making trivial verilog harness and executing") {
    val (path, target) = simpleHarnessBackend(makeTrivialVerilog)

    assert(executeExpectingSuccess(target, path))
  }

  property("Test that assertion failues in Verilog are caught") {
    val (path, target) = simpleHarnessBackend(makeFailingVerilog)

    assert(!executeExpectingSuccess(target, path))
    assert(executeExpectingFailure(target, path))
    assert(executeExpectingFailure(target, path, "My specific, expected error message!"))
    assert(!executeExpectingFailure(target, path, "A string that doesn't match any test output"))
  }
}

