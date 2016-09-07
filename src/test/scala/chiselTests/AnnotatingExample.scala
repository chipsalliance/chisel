// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import org.scalatest._

//scalastyle:off magic.number

/**
  * This Spec file illustrates use of Donggyu's component name API, it currently only
  * uses three methods .signalName, .parentModName and .pathName
  *
  * This is also an illustration of how to implement an annotation system in chisel3
  * A local (my) Driver and Builder are created to provide thread-local access to
  * an annotation map, and then a post elaboration annotation processor can resolve
  * the keys and could serialize the annotations to a file for use by firrtl passes
  */
case class ExampleRelative(component: InstanceId, value: String) extends Annotation with Annotation.Scope.General
case class ExampleAbsolute(component: InstanceId, value: String) extends Annotation with Annotation.Scope.Specific

class SomeSubMod(param1: Int, param2: Int) extends Module {
  val io = new Bundle {
    val in = UInt(INPUT, 16)
    val out = SInt(OUTPUT, 32)
  }
  annotate(ExampleAbsolute(this, s"SomeSubMod($param1, $param2)"))
  annotate(ExampleRelative(io.in, "sub mod io.in"))
  annotate(ExampleAbsolute(io.out, "sub mod io.out"))
}

class AnnotatingExample extends Module {
  val io = new Bundle {
    val a  = UInt(INPUT, 32)
    val b  = UInt(INPUT, 32)
    val e  = Bool(INPUT)
    val z  = UInt(OUTPUT, 32)
    val v  = Bool(OUTPUT)
    val bun = new Bundle {
      val nested_1 = UInt(INPUT, 12)
      val nested_2 = Bool(OUTPUT)
    }
  }
  val x = Reg(UInt(width = 32))
  val y = Reg(UInt(width = 32))

  val subModule1 = Module(new SomeSubMod(1, 2))
  val subModule2 = Module(new SomeSubMod(3, 4))


  annotate(ExampleRelative(subModule2, "SomeSubMod was used"))

  annotate(ExampleAbsolute(x, "I am register X"))
  annotate(ExampleRelative(y, "I am register Y"))
  annotate(ExampleAbsolute(io.a, "I am io.a"))
  annotate(ExampleRelative(io.bun.nested_1, "I am io.bun.nested_1"))
  annotate(ExampleAbsolute(io.bun.nested_2, "I am io.bun.nested_2"))
}

class AnnotatingExampleTester extends BasicTester {
  val dut = Module(new AnnotatingExample)

  stop()
}

class AnnotatingExampleSpec extends FlatSpec with Matchers {
  behavior of "Annotating components of a circuit"

  def hasComponent(name: String, annotations: Seq[Annotation]): Boolean = {
    annotations.exists { annotation =>
      annotation.firrtlInstanceName == name }
  }
  def valueOf(name: String, annotations: Seq[Annotation]): Option[String] = {
    annotations.find { annotation =>
      annotation.firrtlInstanceName == name
    } match {
      case Some(ExampleAbsolute(_, value1)) => Some(value1)
      case Some(ExampleRelative(_, value2)) => Some(value2)
      case _ => None
    }
  }

  it should "contain the following relative keys" in {
    val circuit = Driver.elaborate{ () => new AnnotatingExampleTester }
    Driver.emit(circuit)

    val annotations = circuit.annotations

    hasComponent("SomeSubMod.io.in", annotations) should be (true)
    hasComponent("AnnotatingExample.y", annotations) should be (true)

    valueOf("SomeSubMod.io.in", annotations) should be (Some("sub mod io.in"))
  }
  it should "contain the following absolute keys" in {
    val circuit = Driver.elaborate{ () => new AnnotatingExampleTester }
    Driver.emit(circuit)

    val annotations = circuit.annotations

    hasComponent("AnnotatingExampleTester.dut.subModule2.io.out", annotations) should be (true)
    hasComponent("AnnotatingExampleTester.dut.x", annotations) should be (true)

    assert(valueOf("AnnotatingExampleTester.dut.subModule2.io.out", annotations) === Some("sub mod io.out"))
  }
}