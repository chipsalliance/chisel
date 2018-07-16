// See LICENSE for license details.

package chiselTests
package experimental

import chisel3._
import chisel3.experimental.MultiIOModule

// NOTE This is currently an experimental API and subject to change
// Example using a private port
class PrivatePort extends NamedModuleTester {
  private val port = expectName(IO(Input(UInt(8.W))), "foo")
  port.suggestName("foo")
}

// Example of using composition to add ports to a Module
class CompositionalPort(module: NamedModuleTester, name: String) {
  import chisel3.experimental.IO
  val foo = module.expectName(IO(Output(Bool())), name)
  foo.suggestName(name)
  foo := true.B
}

class CompositionalPortTester extends NamedModuleTester {
  val a = new CompositionalPort(this, "cheese")
  val b = new CompositionalPort(this, "tart")
}

class PortsWinTester extends NamedModuleTester {
  val wire = expectName(Wire(UInt()), "wire_1")
  val foo = expectName(Wire(UInt()).suggestName("wire"), "wire_2")
  val output = expectName(IO(Output(UInt())).suggestName("wire"), "wire")
}

class ProgrammaticPortsSpec extends ChiselFlatSpec {

  private def doTest(testMod: => NamedModuleTester): Unit = {
    var module: NamedModuleTester = null
    elaborate { module = testMod; module }
    assert(module.getNameFailures() == Nil)
  }

  "Programmatic port creation" should "be supported" in {
    doTest(new PrivatePort)
  }

  "Calling IO outside of a Module definition" should "be supported" in {
    doTest(new CompositionalPortTester)
  }

  "Ports" should "always win over internal components in naming" in {
    doTest(new PortsWinTester)
  }

  "LegacyModule" should "ignore suggestName on ports" in {
    doTest(new Module with NamedModuleTester {
      val io = IO(new Bundle {
        val foo = Output(UInt(8.W))
      })
      expectName(io.suggestName("cheese"), "io")
      expectName(clock.suggestName("tart"), "clock")
      expectName(reset.suggestName("teser"), "reset")
    })
  }

  "SuggestName collisions on ports" should "be illegal" in {
    a [ChiselException] should be thrownBy {
      elaborate(new MultiIOModule {
        val foo = IO(UInt(8.W)).suggestName("apple")
        val bar = IO(UInt(8.W)).suggestName("apple")
      })
    }
  }
}

