// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._

// Simple bundle which might be a literal.
class MyPair(xLit: Option[Int] = None, yLit: Option[Int] = None) extends Bundle {
  val x: UInt = xLit match {
    case Some(l) => l.U(4.W)
    case _ => UInt(4.W)
  }
  val y: UInt = yLit match {
    case Some(l) => l.U(4.W)
    case _ => UInt(4.W)
  }

  def +(that: MyPair): MyPair = {
    val output = Wire(MyPair())
    output.x := x + that.x
    output.y := y + that.y
    output
  }

  override def cloneType = (new MyPair(xLit, yLit)).asInstanceOf[this.type]
}

object MyPair {
  // Create an unbound pair
  def apply(): MyPair = new MyPair()

  // Create a literal pair
  def apply(x: Int, y: Int): MyPair = {
    new MyPair(Some(x), Some(y))
  }
}

// Example usecase of a user might use the MyPair type
class PairModule extends Module {
  val io = IO(new Bundle{
      val in = Input(MyPair())
      val plus_one = Output(MyPair())
      val vanilla = Output(MyPair())
  })
  val not_a_lit = new MyPair(Some(1))
  assert(!not_a_lit.isLit)
  val only_a_type = new MyPair()
  assert(!only_a_type.isLit)

  val one = MyPair(1, 1)
  assert(one.isLit)
  io.vanilla := io.in
  io.plus_one := io.in + one
}

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

  it should "check that all elements are bound and work as expected" in {
    println(chisel3.Driver.emitVerilog( new PairModule ))
  }

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
