// See LICENSE for license details.

package chiselTests
package experimental

import chisel3._

// NOTE This is currently an experimental API and subject to change
class PrivatePort extends NamedModuleTester {
  private val port = expectName(IO(Input(UInt(8.W))), "foo")
  port.suggestName("foo")
}

class PortAdder(module: NamedModuleTester, name: String) {
  import chisel3.experimental.IO
  val foo = module.expectName(IO(Output(Bool())), name)
  foo.suggestName(name)
  foo := true.B
}

class CompositionalPorts extends NamedModuleTester {
  val a = new PortAdder(this, "cheese")
  val b = new PortAdder(this, "tart")
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
    doTest(new CompositionalPorts)
  }
}
