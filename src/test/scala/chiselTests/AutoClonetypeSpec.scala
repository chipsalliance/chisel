// See LICENSE for license details.

package chiselTests

import chisel3._

import chisel3.testers.BasicTester
//
//var clazz = classOf[BundleWithIntArg]
//var mirror = runtimeMirror(clazz.getClassLoader)
//var classSymbol = mirror.classSymbol(clazz)
//    

//class M2I {
//  class InnerBundle(val i: Int) extends Bundle {
//    val out = Output(UInt(i.W))
//  }
//  
//  def getNew = new InnerBundle(8)
//}
//

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

class BundleWithNoArg() extends Bundle {
  val out = Output(Bool())
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
