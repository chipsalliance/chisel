// See LICENSE for license details.

package chiselTests

import chisel3._

import chisel3.testers.BasicTester

class BundleWithIntArg(val i: Int) extends Bundle {
  val out = Output(UInt(i.W))
}

class BundleWithImplicit()(implicit val ii: Int) extends Bundle {
  val out = Output(UInt(ii.W))
}

class BundleWithArgAndImplicit(val i: Int)(implicit val ii: Int) extends Bundle {
  val out1 = Output(UInt(i.W))
  val out2 = Output(UInt(ii.W))
}

class ModuleWithInner extends Module {
  class InnerBundle(val i: Int) extends Bundle {
    val out = Output(UInt(i.W))
  }

  val io = IO(new Bundle{})
  
  val myWire = Wire(new InnerBundle(14))
  require(myWire.i == 14)
}


class AutoClonetypeSpec extends ChiselFlatSpec {
  "Bundles with Scala args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})
      
      val myWire = Wire(new BundleWithIntArg(8))
      assert(myWire.i == 8)
    } }
  }

  "Bundles with Scala implicit args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})
      
      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithImplicit())
      
      assert(myWire.ii == 4)
    } }
  }
  
  "Bundles with Scala explicit and impicit args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})
      
      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithArgAndImplicit(8))
      
      assert(myWire.i == 8)
      assert(myWire.ii == 4)
    } }
  }
  
  "Inner bundles with Scala args" should "not need clonetype" in {
    elaborate { new ModuleWithInner }
  }
}
