package chiselTests
import Chisel.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._
import java.io.File

class HarnessSpec extends ChiselPropSpec 
  with Chisel.BackendCompilationUtilities
  with Chisel.FileSystemUtilities {

  def makeTrivialVerilogHarness = BasicTester.makeHarness((prefix: String) => s"""
module ${prefix}Harness;
  initial begin
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", "Harness.v") _

  def makeFailingVerilogHarness = BasicTester.makeHarness((prefix: String) => s"""
module ${prefix}Harness;
  initial begin
    assert (1 == 0) else $$error("It's gone wrong");
    $$display("$prefix!");
    $$finish;
  end
endmodule
""", "Harness.v") _

  property("Test making trivial verilog harness and executing") {
    val fname = File.createTempFile("our", "")
    val prefix = fname.toString.split("/").last
    val dir = new File(System.getProperty("java.io.tmpdir"))
    val vHarness = makeTrivialVerilogHarness(fname)
    val cppHarness = BasicTester.makeCppHarness(fname)
    verilogToCpp(dir, Seq(vHarness), cppHarness).!
    cppToExe(prefix, dir).!

    assert(executeExpectingSuccess(prefix, dir))
  }

  property("Test that assertion failues in Verilog are caught") {
    val fname = File.createTempFile("our", "")
    val prefix = fname.toString.split("/").last
    val dir = new File(System.getProperty("java.io.tmpdir"))
    val vHarness = makeFailingVerilogHarness(fname)
    val cppHarness = BasicTester.makeCppHarness(fname)
    verilogToCpp(dir, Seq(vHarness), cppHarness).!
    cppToExe(prefix, dir).!

    assert(!executeExpectingSuccess(prefix, dir))
    assert(executeExpectingFailure(prefix, dir))
    assert(executeExpectingFailure(prefix, dir, "It's gone wrong"))
    assert(!executeExpectingFailure(prefix, dir, "XXXXXXXXXXXXXX"))
  }
}
 
