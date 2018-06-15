// See LICENSE for license details.

package chiselTests

import chisel3._

class PrivatePort extends NamedModuleTester {
  // Use the ValName macro
  private val foo = experimental.IO(Input(UInt(8.W)))
  expectName(foo, "foo")
}

class PortAdder(module: NamedModuleTester, name: String) {
  import chisel3.experimental.{IO, ValName}
  // Explicitly set the name
  val foo = IO(Output(Bool()))(ValName(name))
  module.expectName(foo, name)
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
