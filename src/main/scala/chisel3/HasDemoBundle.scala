// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.FixedPoint
import chisel3.stage.ChiselStage
import chisel3.util.Decoupled

object BundleComparator {
  println(f"${"New Field Name"}%30s ${"Old Field Name"}%30s")
  def compare(bundle: Bundle): Unit = {
    val newElements = bundle.elements.toList
//    val oldElements = bundle.oldElements.toList
//
//    newElements.zipAll(oldElements, "Oops" -> Bool(), "Oops" -> Bool()).foreach { case ((a, b), (c, d)) =>
//      val color = if(a == c) { Console.RESET } else { Console.RED }
//      println(f"$color$a%30s $c%30s${Console.RESET}")
//    }
    println(s"BundleComparator:\n" + newElements.mkString("\n"))
  }
}

/* Demo stuff
 */
class BpipHasDemoBundle extends Module {
  // Example

  trait BpipVarmint {
    val varmint = Bool()
    def vermin = Bool()
    private val puppy = Bool()
  }

  abstract class BpipAbstractBundle extends Bundle {
    def doNothing: Unit

    val fromAbstractBundle = UInt(22.W)
  }

  class BpipOneField extends Bundle {
    val fieldOne = SInt(8.W)
  }

  class BpipTwoField extends BpipOneField {
    val fieldTwo = SInt(8.W)
    val fieldThree = Vec(4, UInt(12.W))
  }
  class BpipAnimalBundle(w1: Int, w2: Int) extends Bundle {
    val dog = SInt(w1.W)
    val fox = UInt(w2.W)
  }

  class DemoBundle[T <: Data](gen: T, gen2: => T) extends BpipTwoField with BpipVarmint {
    val foo = gen
    val bar = Bool()
    val qux = gen2
    val bad = 44
    //TODO: This line is breaking things,, not sure why, error is AliasedAggregateField
//     val baz = Decoupled(UInt(16.W))
    val animals = new BpipAnimalBundle(4, 8)
  }

  val out = IO(Output(new DemoBundle(UInt(4.W), FixedPoint(10.W, 4.BP))))

  val out2 = IO(Output(new BpipAbstractBundle {
    override def doNothing: Unit = println("ugh")
    val notAbstract:        Bool = Bool()
  }))

  val out4 = IO(Output(new BpipAnimalBundle(99, 100)))
  val out5 = IO(Output(new BpipTwoField))

  out := DontCare
  out5 := DontCare

  println(s"BpipTwoField.elements: \n" + out5.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"\n\nDemoBundle.elements:\n" + out.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"\nBpipAbstractBundle.elements:\n" + out2.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"\nAnimal.elements:\n" + out4.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
  println(s"\nTwoField.elements:\n" + out5.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))

  // The following block does not work, suggesting that ParamIsField is not a case we need to solve
//  class ParamIsField(val paramField: UInt) extends Bundle
//  val out3 = IO(Output(new ParamIsField(UInt(10.W))))
//  println(s"ParamsIsField.elements:\n" + out3.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
//  out3.paramField := 7.U
}

/* Rich and complicated bundle example
 *
 */
object DebugProblem1 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new BpipHasDemoBundle)
    println("done!")
  }
}

trait BpipSuperTraitWithField {
  val fromTraitOne = SInt(17.W)
  def notFieldOne = SInt(22.W)
}

trait BpipTraitWithField extends BpipSuperTraitWithField {
  val fromTraitTwo = SInt(17.W)
  def notFieldTwo = SInt(22.W)
}

class BpipOneField extends Bundle with BpipTraitWithField {
  val fieldZero = SInt(8.W)
  val fieldOne = SInt(8.W)
}

class BpipTwoField extends BpipOneField {
  val fieldTwo = SInt(8.W)
  val fieldThree = Vec(4, UInt(12.W))
  val myInt = 7
//  val baz = Decoupled(UInt(16.W))
}

class BpipDecoupled extends BpipOneField {
  val fieldTwo = SInt(8.W)
  val fieldThree = Vec(4, UInt(12.W))
  val baz = Decoupled(UInt(16.W))
}

class BpipBadBundleWithHardware extends Bundle {
  val noHardwareField = SInt(8.W)
  val isHardwareField = 244.U(16.W)
}

class BpipExtendsBadBundleWithHardware extends BpipBadBundleWithHardware {
  val anotherField = SInt(8.W)
}

class DebugProblem2 extends Module {
  val out1 = IO(Output(new BpipDecoupled))
  out1 := DontCare
  println(s"out1.elements:\n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

/* plugin should work with decoupled
 *
 */
object DebugProblem2 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem2)
    println("done!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem3 extends Module {
  val out1 = IO(Output(new BpipTwoField))
  out1 := DontCare
  println(s"out1.elements:\n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem3 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem3)
    println("done!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem4 extends Module {
  val out1 = IO(Output(new BpipBadBundleWithHardware))
  out1 := DontCare
  println(s"out1.elements:\n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem4 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem4)
    println("done!")
  }
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem5 extends Module {
  val out1 = IO(Output(new BpipExtendsBadBundleWithHardware))
  out1 := DontCare
  println(s"out1.elements:\n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem5 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem5)
    println("done!")
  }
}

//TODO: If you comment out this block and compile, there will be a compiler
//      compiler error at the badSeqField
//class BpipBadSeqBundle extends Bundle {
//  val goodField = UInt(999.W)
//  val badSeqField = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
//}
//
///* plugin should not affect the seq detection
// *
// */
//class DebugProblem6 extends Module {
//  val out1 = IO(Output(new BpipBadSeqBundle))
//  out1 := DontCare
//  println(s"out1.elements: \n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
//}
//
//object DebugProblem6 {
//  def main(args: Array[String]): Unit = {
//    ChiselStage.emitFirrtl(new DebugProblem6)
//    println("done!")
//  }
//}

class BpipBadSeqBundleWithIgnore extends Bundle with IgnoreSeqInBundle {
  val goodFieldWithIgnore = UInt(999.W)
  val badSeqFieldWithIgnore = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
}

/* plugin should not affect the seq detection
 *
 */
class DebugProblem7 extends Module {
  val out1 = IO(Output(new BpipBadSeqBundleWithIgnore))
  out1 := DontCare
  println(s"out1.elements: \n" + out1.elements.map(e => s"${e._1} (${e._2})").mkString("\n"))
}

object DebugProblem7 {
  def main(args: Array[String]): Unit = {
    ChiselStage.emitFirrtl(new DebugProblem7)
    println("done!")
  }
}
