// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._

case class BundleWithUInt(x: UInt = UInt()) extends Bundle {
  override def cloneType = BundleWithUInt(x.cloneType).asInstanceOf[this.type]
}

case class BundleWithBundleWithUInt(x: BundleWithUInt = BundleWithUInt()) extends Bundle {
  override def cloneType = BundleWithBundleWithUInt(x.cloneType).asInstanceOf[this.type]
}

case class BundleWithUIntAndBadCloneType(x: UInt = UInt()) extends Bundle


class BundleWithUIntModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithUInt(UInt(4.W)))
  })
  io.out := BundleWithUInt(10.U)
}

class BundleWithCloneTypeModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithUInt(9.U)) // will call cloneType
  })
  io.out := BundleWithUInt(5.U)
}

class BundleWithBadCloneTypeModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithUIntAndBadCloneType(3.U))
  })
  io.out := BundleWithUIntAndBadCloneType(0.U)
}

class BundleWithBundleWithUIntModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithBundleWithUInt(BundleWithUInt(UInt(4.W))))
  })
  io.out := BundleWithBundleWithUInt(BundleWithUInt(7.U))
}

class BundleLiteralsSpec extends FlatSpec with Matchers {
  behavior of "Bundle literals"

  it should "build the module without crashing" in {
    println(chisel3.Driver.emitVerilog( new BundleWithUIntModule ))
  }

  it should "work with a correct cloneType implementation" in {
    println(chisel3.Driver.emitVerilog( new BundleWithCloneTypeModule ))
  }

  it should "throw an exception if cloneType is bad" in {
    assertThrows[IllegalArgumentException] {
      println(chisel3.Driver.emitVerilog( new BundleWithBadCloneTypeModule ))
    }
  }

  behavior of "Bundle literals with bundle literals inside"

  it should "build the module without crashing" in {
    println(chisel3.Driver.emitVerilog( new BundleWithBundleWithUIntModule ))
  }
}
