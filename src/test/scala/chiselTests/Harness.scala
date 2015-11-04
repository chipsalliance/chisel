package chiselTests
import Chisel.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._
import java.io.File

class HarnessSpec extends ChiselPropSpec 
  with Chisel.BackendCompilationUtilities {

  def makeTrivialVerilog = makeHarness((prefix: String) => s"""
module ${prefix};
  initial begin
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", ".v") _

  def makeFailingVerilog = makeHarness((prefix: String) => s"""
module $prefix;
  initial begin
    assert (1 == 0) else $$error("My specific, expected error message!");
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", ".v") _

  def makeCppHarness = makeHarness((prefix: String) => s"""
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
""", ".cpp") _

  val dir = new File(System.getProperty("java.io.tmpdir"))

  def simpleHarnessBackend(make: File => File): String = {
    val target = "test"
    val fname = File.createTempFile(target, "")
    val path = fname.getParentFile.toString
    val prefix = fname.toString.split("/").last
    val vDut = make(fname)
    val vH = new File(path + "/V" + prefix + ".h")
    val cppHarness = makeCppHarness(fname)
    verilogToCpp(target, dir, vDut, cppHarness, vH).!
    cppToExe(prefix, dir).!
    prefix
  }

  property("Test making trivial verilog harness and executing") {
    val prefix = simpleHarnessBackend(makeTrivialVerilog)

    assert(executeExpectingSuccess(prefix, dir))
  }

  property("Test that assertion failues in Verilog are caught") {
    val prefix = simpleHarnessBackend(makeFailingVerilog)

    assert(!executeExpectingSuccess(prefix, dir))
    assert(executeExpectingFailure(prefix, dir))
    assert(executeExpectingFailure(prefix, dir, "My specific, expected error message!"))
    assert(!executeExpectingFailure(prefix, dir, "A string that doesn't match any test output"))
  }
}
 
