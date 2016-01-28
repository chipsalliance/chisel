// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class ParameterizedModule(invert: Boolean) extends Module {
  val io = new Bundle {
    val in = new Bool(INPUT)
    val out = new Bool(OUTPUT)
  }
  if (invert) {
    io.out := !io.in
  } else {
    io.out := io.in
  }
}

/** A simple test to check Module deduplication doesn't affect correctness (two
  * modules with the same name but different contents aren't aliased). Doesn't
  * check that deduplication actually happens, though.
  */
class ParameterizedModuleTester() extends BasicTester {
  val invert = Module(new ParameterizedModule(true))
  val noninvert = Module(new ParameterizedModule(false))

  invert.io.in := Bool(true)
  noninvert.io.in := Bool(true)
  assert(invert.io.out === Bool(false))
  assert(noninvert.io.out === Bool(true))

  stop()
}

class ParameterizedModuleSpec extends ChiselFlatSpec {
  "Different parameterized modules" should "have different behavior" in {
    assert(execute{ new ParameterizedModuleTester() })
  }
}
