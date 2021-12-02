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

  abstract class AbstractBundle extends Bundle {
    def doNothing: Unit

    val fromAbstractBundle = UInt(22.W)
  }

  class OneFieldBundle extends Bundle {
    val fieldOne = SInt(8.W)
  }

  class TwoFieldBundle extends OneFieldBundle {
    val fieldTwo = SInt(8.W)
    val fieldThree = Vec(4, UInt(12.W))
  }
  class AnimalBundle(w1: Int, w2: Int) extends Bundle {
    val dog = SInt(w1.W)
    val fox = UInt(w2.W)
  }

  class DemoBundle[T <: Data](gen: T, gen2: => T) extends TwoFieldBundle with Varmint {
    val foo = gen
    val bar = Bool()
    val qux = gen2
    val bad = 4
//     val baz = Decoupled(UInt(16.W))
    val animals = new AnimalBundle(4, 8)
  }

  val out = IO(Output(new DemoBundle(UInt(4.W), FixedPoint(10.W, 4.BP))))

  val out2 = IO(Output(new AbstractBundle {
    override def doNothing: Unit = println("ugh")
    val notAbstract: Bool = Bool()
  }))

  val out4 = IO(Output(new AnimalBundle(99, 100)))
  val out5 = IO(Output(new TwoFieldBundle))

  out := DontCare
  out5 := DontCare

  println(s"TwoFieldBundle.elements: \n" + out5.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"DemoBundle.elements:\n" + out.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"AbstractBundle.elements:\n" + out2.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"Animal.elements:\n" + out4.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))

  // The following block does not work, suggesting that ParamIsField is not a case we need to solve
//  class ParamIsField(val paramField: UInt) extends Bundle
//  val out3 = IO(Output(new ParamIsField(UInt(10.W))))
//  println(s"ParamsIsField.elements:\n" + out3.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
//  out3.paramField := 7.U
}

object BundleShower  {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new HasDemoBundle)
    println("done!!!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem1 extends Module {

  class BadSeqBundle extends Bundle {
    val bar = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
  }

  val out = IO(Output(new Bundle { val b = new BadSeqBundle}))

  out := DontCare

  println(s"out.elements:\n" + out.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem1  {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem1)
    println("done!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem2 extends Module {

  class SimpleBundle2 extends Bundle {
    val bar = UInt(16.W)
  }

  val out = IO(Output(new SimpleBundle2))

  out := DontCare

  println(s"out.elements:\n" + out.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem2  {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem2)
    println("done!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem3 extends Module {

  class SimpleBundle1 extends Bundle {
    val barbar = UInt(16.W)
  }

  class SimpleBundle2 extends SimpleBundle1 {
    val bar = UInt(16.W)
  }

  val out1 = IO(Output(new SimpleBundle1))
  val out2 = IO(Output(new SimpleBundle2))

  out1 := DontCare
  out2 := DontCare

  println(s"out1.elements:\n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"out2.elements:\n" + out2.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem3  {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem3)
    println("done!")
  }
}


