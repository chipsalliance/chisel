package unitTests

import chiselTests.ChiselFlatSpec
import Chisel.testers.{UnitTester, BasicTester}
import Chisel.{UInt, Bundle}


class BundleTester extends UnitTester {
  val x = new Bundle {
    val a = UInt(7)
    val b = UInt(8)
  }
}

class TestConstantBundles extends ChiselFlatSpec {
  "a" should "b" in {
    execute(new BundleTester)
  }
}