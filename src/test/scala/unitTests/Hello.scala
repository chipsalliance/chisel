package unitTests


import Chisel._
import Chisel.testers.{UnitTestRunners, UnitTester}
import chiselTests.ChiselFlatSpec

class Hello extends Module {
  val io = new Bundle {
    val out = UInt(OUTPUT, 8)
  }
  io.out := UInt(42)
}

class HelloTester extends UnitTester {
  val c = Module( new Hello )

  step(1)
  expect(c.io.out, 42)

  install(c)
}

object Hello extends UnitTestRunners {
  def main(args: Array[String]): Unit = {
    execute { new HelloTester }
  }
}

class HelloSpec extends ChiselFlatSpec {
  "a" should "b" in {
    assert( execute { new HelloTester } )
  }
}
