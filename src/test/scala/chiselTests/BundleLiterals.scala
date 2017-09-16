// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._

class BundleWithLit(lit: Option[Int]=None) extends Bundle {
  val internal = lit.map(_.U).getOrElse(UInt(4.W))
  override def cloneType = new BundleWithLit(lit).asInstanceOf[this.type]
}

object BundleWithLit {
  def apply(x: Int) = {
    val ret = new BundleWithLit(Some(x))
    Aggregate.LitBind(ret)
    ret
  }
  def apply() = {
    new BundleWithLit()
  }
}

class BundleWithBundleWithLit(lit: Option[Int]=None) extends Bundle {
  val internal = lit.map(BundleWithLit(_)).getOrElse(BundleWithLit())
  override def cloneType = new BundleWithBundleWithLit(lit).asInstanceOf[this.type]
}

object BundleWithBundleWithLit {
  def apply(x: Int) = {
    val ret = new BundleWithBundleWithLit(Some(x))
    Aggregate.LitBind(ret)
    ret
  }
  def apply() = {
    new BundleWithBundleWithLit()
  }
}

class BundleWithLitModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithLit())
  })
  io.out := BundleWithLit(3)
}

class BundleWithBundleWithLitModule extends Module {
  val io = IO(new Bundle {
    val out = Output(BundleWithBundleWithLit())
  })
  io.out := BundleWithBundleWithLit(3)
}

class BundleLiteralsSpec extends FlatSpec with Matchers {
  behavior of "Bundle literals"

  it should "build the module without crashing" in {
    println(chisel3.Driver.emitVerilog( new BundleWithLitModule ))
  }

  behavior of "Bundle literals with bundle literals inside"

  it should "build the module without crashing" in {
    println(chisel3.Driver.emitVerilog( new BundleWithBundleWithLitModule ))
  }
}
