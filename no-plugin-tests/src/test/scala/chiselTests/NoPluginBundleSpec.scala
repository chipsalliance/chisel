// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester



trait BundleSpecUtils {
  
  class BundleBaz(val w: Int) extends Bundle {
    val baz = UInt(w.W)
    // (if we don't have the val on the val w: Int then it is an Exception)
    // Check that we get a runtime deprecation warning if we don't have this:
    // override def cloneType = (new BundleBaz(w)).asInstanceOf[this.type]
  }

}

class NoPluginBundleSpec extends ChiselFlatSpec with BundleSpecUtils with Utils {

 "No override def cloneType" should "give a runtime deprecation warning without compiler plugin" in {
    class MyModule extends MultiIOModule {
      val in = IO(Input(new BundleBaz(w = 3)))
      val out = IO(Output(in.cloneType))
    }
    val (log, _) = grabLog(
        ChiselStage.elaborate(new MyModule())
    )
    log should include ("warn")
    log should include ("deprecated")
    log should include ("The runtime reflection inference for cloneType")
  }
}
