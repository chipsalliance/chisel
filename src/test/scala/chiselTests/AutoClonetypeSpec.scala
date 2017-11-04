// See LICENSE for license details.

package chiselTests

import chisel3._

import chisel3.testers.BasicTester

class BundleWithIntArg(val i: Int) extends Bundle {
  val out = Output(UInt(i.W))
}

class ModuleWithInner extends Module {
  class InnerBundle(val i: Int) extends Bundle {
    val out = Output(UInt(i.W))
  }

  val io = IO(new InnerBundle(14))
  io.out := 1.U
}


class AutoClonetypeSpec extends ChiselFlatSpec {
  "Bundles with Scala args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new BundleWithIntArg(8))
      io.out := 1.U
    } }
  }

  "Inner bundles with Scala args" should "not need clonetype" in {
    elaborate { new ModuleWithInner }
  }
}
