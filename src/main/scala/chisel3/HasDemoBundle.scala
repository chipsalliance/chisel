// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage
import chisel3.util.Decoupled

/* ### How do I create a Bundle from a UInt?
 *
 * On an instance of the Bundle, call the method fromBits with the UInt as the argument
 */
class HasDemoBundle extends Module {
  // Example

  trait Varmint {
    val varmint = Bool()
  }
  class AnimalBundle(w1: Int, w2: Int) extends Bundle {
    val dog = SInt(w1.W)
    val fox = UInt(w2.W)
  }
  class DemoBundle[T <: Data](gen: T, gen2: => T) extends Bundle with Varmint {
    val foo = gen
    val bar = Bool()
    val qux = gen2
    val bad = 4
    val baz = Decoupled(UInt(16.W))
    val animals = new AnimalBundle(4, 8)
  }

  val out = IO(Output(new DemoBundle(UInt(4.W), FixedPoint(10.W, 4.BP))))

  out := DontCare

  println(s"HasDemoBundle.elements:\n" + out.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object BundleShower  {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new HasDemoBundle)
    println("done!")
  }
}

