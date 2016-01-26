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
  val device_under_test = Module( new Hello )

  testBlock {
    step(1)
    expect(device_under_test.io.out, 42)
  }
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
